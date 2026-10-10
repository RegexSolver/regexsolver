//! Neural synthesis end to end, with a model of the real format but tiny and
//! untrained (`tests/data/kleene_tiny`, written by
//! `kleene::tests::write_tiny_model`): its proposals are mostly wrong, which
//! the conversions must catch, but it runs the whole inference, from a single
//! automaton to the batches of decomposition.
//!
//! The model is embedded rather than read, so the tests need no filesystem:
//! CI runs them on `wasm32-unknown-unknown` in a module with no imports at
//! all, so with no clock and no source of randomness.

#![cfg(feature = "neural-synthesis")]

use regexsolver::fast_automaton::FastAutomaton;
use regexsolver::neural_synthesis::{Device, NeuralSynthesizer};
use regexsolver::regex::RegularExpression;

fn synthesizer() -> NeuralSynthesizer {
    NeuralSynthesizer::from_bytes(
        include_bytes!("data/kleene_tiny/config.json"),
        include_bytes!("data/kleene_tiny/model.safetensors").to_vec(),
        Device::Cpu,
    )
    .unwrap()
}

fn automaton(pattern: &str) -> FastAutomaton {
    RegularExpression::new(pattern)
        .unwrap()
        .to_automaton()
        .unwrap()
}

fn assert_keeps_the_language(regex: &RegularExpression, automaton: &FastAutomaton) {
    assert!(
        regex.to_automaton().unwrap().equivalent(automaton).unwrap(),
        "{regex}"
    );
}

#[test]
fn proposes_for_automata_the_model_takes() {
    let sampled = synthesizer();
    let greedy = synthesizer().with_candidates(1);
    assert_eq!(sampled.model_name(), "kleene-tiny");
    for pattern in ["abc", "(ab|cd)*e", "[a-c]*d[e-g]*", "(a|b)*abb"] {
        let automaton = automaton(pattern);
        for synthesizer in [&sampled, &greedy] {
            if let Some(proposed) = synthesizer.propose(&automaton).unwrap() {
                assert_keeps_the_language(&proposed, &automaton);
            }
            let regex = automaton.to_regex_with(synthesizer).unwrap();
            assert_keeps_the_language(&regex, &automaton);
        }
    }

    let automata: Vec<FastAutomaton> = ["x*y", "(xy)+", "x|yz"].map(automaton).into();
    let batch: Vec<&FastAutomaton> = automata.iter().collect();
    for (proposed, automaton) in sampled.propose_batch(&batch).unwrap().iter().zip(&automata) {
        if let Some(proposed) = proposed {
            assert_keeps_the_language(proposed, automaton);
        }
    }
}

#[test]
fn decomposes_automata_beyond_the_model() {
    let synthesizer = synthesizer();
    let mut intersection = automaton(".*abc.*");
    for other in [".*def.*", ".*gh.*"] {
        intersection = intersection.intersection(&automaton(other)).unwrap();
    }
    let mut dfa = intersection.determinize().unwrap().into_owned();
    dfa.minimize().unwrap();
    assert!(dfa.number_of_states() > synthesizer.max_states());

    let regex = dfa.to_regex_with(&synthesizer).unwrap();
    assert_keeps_the_language(&regex, &dfa);
    // State elimination's regex is a candidate, so the result is no worse.
    assert!(regex.evaluate_complexity() <= dfa.to_regex().unwrap().evaluate_complexity());
}
