//! Compares [`FastAutomaton::to_regex_with`] with and without decomposition,
//! and with state elimination, on DFAs beyond the model's limits:
//! `KLEENE_MODEL_DIR=<release dir> cargo test --release --features neural-synthesis decompose::bench -- --ignored --nocapture`

use std::time::{Duration, Instant};

use super::*;
use crate::{execution_profile::ExecutionProfileBuilder, neural_synthesis::Device};

/// DFAs per set operation.
const PER_OPERATION: usize = 10;

#[test]
#[ignore = "benchmark; needs the model weights (KLEENE_MODEL_DIR)"]
fn bench_decompose() {
    let load = || {
        NeuralSynthesizer::from_dir(std::env::var("KLEENE_MODEL_DIR").unwrap(), Device::Cpu)
            .unwrap()
    };
    let whole = load().with_decomposition(false);
    let decomposing = load();

    let reference: Vec<serde_json::Value> = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/kleene_parity.json"
    )))
    .unwrap();
    let atoms: Vec<RegularExpression> = reference
        .iter()
        .map(|case| RegularExpression::new(case["pattern"].as_str().unwrap()).unwrap())
        .collect();

    for (operation, dfas) in large_dfas(&atoms, whole.max_states()) {
        let mut ratios = (0.0, 0.0);
        let mut simpler = 0;
        let mut times = (Duration::ZERO, Duration::ZERO);
        for dfa in &dfas {
            let eliminated = dfa.to_regex().unwrap().evaluate_complexity();
            let start = Instant::now();
            let without = dfa.to_regex_with(&whole).unwrap();
            times.0 += start.elapsed();
            let start = Instant::now();
            let with = dfa.to_regex_with(&decomposing).unwrap();
            times.1 += start.elapsed();
            assert!(
                with.to_automaton().unwrap().equivalent(dfa).unwrap(),
                "{with}"
            );

            ratios.0 += (without.evaluate_complexity() / eliminated).ln();
            ratios.1 += (with.evaluate_complexity() / eliminated).ln();
            simpler += usize::from(with.evaluate_complexity() < eliminated);
        }
        let count = dfas.len();
        println!(
            "{operation:<13} {count} DFAs: complexity vs state elimination (geometric mean) {:.3} without decomposition, {:.3} with; simpler for {simpler}; {:?} and {:?} per DFA",
            (ratios.0 / count as f64).exp(),
            (ratios.1 / count as f64).exp(),
            times.0 / count as u32,
            times.1 / count as u32,
        );
    }
}

/// A set operation on three regexes.
type Operation = fn(&[&RegularExpression; 3]) -> Result<FastAutomaton, EngineError>;

const OPERATIONS: [(&str, Operation); 5] = [
    ("concatenation", |[a, b, c]| {
        a.concat(b, true).concat(c, true).to_automaton()
    }),
    ("star", |[a, b, c]| {
        a.union(b).repeat(0, None).concat(c, true).to_automaton()
    }),
    ("difference", |[a, b, c]| {
        a.concat(b, true)
            .to_automaton()?
            .difference(&containing(c)?)
    }),
    ("intersection", |[a, b, c]| {
        a.union(b)
            .repeat(0, None)
            .to_automaton()?
            .intersection(&containing(c)?)
    }),
    ("union", |[a, b, c]| a.union(b).union(c).to_automaton()),
];

/// The strings containing a match of `regex`.
fn containing(regex: &RegularExpression) -> Result<FastAutomaton, EngineError> {
    let total = RegularExpression::new_total();
    total
        .concat(regex, true)
        .concat(&total, true)
        .to_automaton()
}

/// Minimal DFAs of more than `max_states` states (and at most 80) from set
/// operations on `atoms`, by operation.
fn large_dfas(
    atoms: &[RegularExpression],
    max_states: usize,
) -> Vec<(&'static str, Vec<FastAutomaton>)> {
    let profile = ExecutionProfileBuilder::new()
        .execution_timeout(300)
        .max_number_of_states(5000)
        .build();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut random = move |bound: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % bound as u64) as usize
    };
    OPERATIONS
        .iter()
        .map(|(name, operation)| {
            let mut dfas = vec![];
            for _ in 0..1000 {
                if dfas.len() == PER_OPERATION {
                    break;
                }
                let operands = [0, 1, 2].map(|_| &atoms[random(atoms.len())]);
                let dfa = profile.run(|| {
                    let dfa = minimal_dfa(&operation(&operands)?)?;
                    // State elimination must finish too, for the comparison.
                    let _ = dfa.to_regex()?;
                    Ok::<_, EngineError>(dfa)
                });
                if let Ok(dfa) = dfa
                    && (max_states + 1..=80).contains(&dfa.number_of_states())
                {
                    dfas.push(dfa);
                }
            }
            (*name, dfas)
        })
        .collect()
}
