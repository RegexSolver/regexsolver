//! Compares [`FastAutomaton::to_regex_with`] with and without decomposition,
//! and with state elimination, on DFAs beyond the model's limits:
//! `KLEENE_MODEL_DIR=<release dir> cargo test --release --features neural-synthesis decompose::bench -- --ignored --nocapture`

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    ops::RangeInclusive,
    time::{Duration, Instant},
};

use super::*;
use crate::{
    execution_profile::ExecutionProfileBuilder,
    neural_synthesis::{test_device, test_patterns},
};

/// DFAs per set operation.
const PER_OPERATION: usize = 10;

#[test]
#[ignore = "benchmark; needs the model weights (KLEENE_MODEL_DIR)"]
fn bench_decompose() {
    let load = || {
        NeuralSynthesizer::from_dir(std::env::var("KLEENE_MODEL_DIR").unwrap(), test_device())
            .unwrap()
    };
    let whole = load().with_decomposition(false);
    let decomposing = load();

    let atoms = test_patterns();

    for (operation, dfas) in large_dfas(&atoms, whole.max_states()) {
        let mut ratios = (0.0, 0.0);
        let mut simpler = 0;
        let mut times = (Duration::ZERO, Duration::ZERO);
        // Of the decomposition without the model: the splits, state
        // elimination and the combination.
        let mut alone = Duration::ZERO;
        // Of the regexes, so an optimization can be checked not to change them.
        let mut checksum = DefaultHasher::new();
        for dfa in &dfas {
            let eliminated = dfa.to_regex().unwrap().evaluate_complexity();
            let start = Instant::now();
            let without = dfa.to_regex_with(&whole).unwrap();
            times.0 += start.elapsed();
            let start = Instant::now();
            let with = dfa.to_regex_with(&decomposing).unwrap();
            times.1 += start.elapsed();
            let start = Instant::now();
            let _ = Decomposer::new(None).convert(dfa).unwrap();
            alone += start.elapsed();
            assert!(
                with.to_automaton().unwrap().equivalent(dfa).unwrap(),
                "{with}"
            );
            (without.to_string(), with.to_string()).hash(&mut checksum);

            ratios.0 += (without.evaluate_complexity() / eliminated).ln();
            ratios.1 += (with.evaluate_complexity() / eliminated).ln();
            simpler += usize::from(with.evaluate_complexity() < eliminated);
        }
        let count = dfas.len();
        println!(
            "{operation:<13} {count} DFAs: complexity vs state elimination (geometric mean) {:.3} without decomposition, {:.3} with; simpler for {simpler}; {:?} and {:?} per DFA ({:?} without the model); checksum {:016x}",
            (ratios.0 / count as f64).exp(),
            (ratios.1 / count as f64).exp(),
            times.0 / count as u32,
            times.1 / count as u32,
            alone / count as u32,
            checksum.finish(),
        );
    }
}

/// Quality and time of [`FastAutomaton::to_regex_with`] over the number of
/// candidates and the minimum success rate, on DFAs the model takes whole and
/// on larger ones it takes in pieces:
/// `KLEENE_MODEL_DIR=<release dir> cargo test --release --features neural-synthesis decompose::bench::bench_settings -- --ignored --nocapture`
///
/// Per setting: the geometric mean of the complexity relative to state
/// elimination's, the DFAs made simpler, and the mean and slowest time per
/// DFA. Decomposition without the model is the reference the model improves
/// on.
#[test]
#[ignore = "benchmark; needs the model weights (KLEENE_MODEL_DIR)"]
fn bench_settings() {
    let load = || {
        NeuralSynthesizer::from_dir(std::env::var("KLEENE_MODEL_DIR").unwrap(), test_device())
            .unwrap()
    };
    let max_states = load().max_states();
    let atoms = test_patterns();
    let corpora = [
        ("whole", dfas(&atoms, 2..=max_states, 20)),
        ("decomposed", dfas(&atoms, max_states + 1..=80, PER_OPERATION)),
    ];
    for (corpus, by_operation) in corpora {
        let dfas: Vec<FastAutomaton> = by_operation.into_iter().flat_map(|(_, d)| d).collect();
        let eliminated: Vec<f64> = dfas
            .iter()
            .map(|dfa| dfa.to_regex().unwrap().evaluate_complexity())
            .collect();
        // The geometric mean of the complexities relative to state
        // elimination's, the DFAs made simpler, the mean and slowest times.
        let measure = |convert: &dyn Fn(&FastAutomaton) -> RegularExpression| {
            let (mut log_ratio, mut simpler) = (0.0, 0);
            let (mut total, mut slowest) = (Duration::ZERO, Duration::ZERO);
            for (dfa, eliminated) in dfas.iter().zip(&eliminated) {
                let start = Instant::now();
                let regex = convert(dfa);
                let elapsed = start.elapsed();
                total += elapsed;
                slowest = slowest.max(elapsed);
                let complexity = regex.evaluate_complexity();
                log_ratio += (complexity / eliminated).ln();
                simpler += usize::from(complexity < *eliminated);
            }
            format!(
                "{:>6.3} {:>4}/{:<3} {:>10.1?} {:>10.1?}",
                (log_ratio / dfas.len() as f64).exp(),
                simpler,
                dfas.len(),
                total / dfas.len() as u32,
                slowest,
            )
        };
        let states: Vec<usize> = dfas.iter().map(FastAutomaton::number_of_states).collect();
        println!(
            "\n{corpus}: {} DFAs of {} to {} states",
            dfas.len(),
            states.iter().min().unwrap(),
            states.iter().max().unwrap()
        );
        println!("candidates min_success  ratio  simpler       mean    slowest");
        println!(
            "{:<22} {}",
            "no model",
            measure(&|dfa| Decomposer::new(None).convert(dfa).unwrap())
        );
        for candidates in [1, 2, 4, 8, 16] {
            for min_success in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95] {
                let synthesizer = load()
                    .with_candidates(candidates)
                    .with_min_success(min_success);
                println!(
                    "{candidates:>10} {min_success:>11} {}",
                    measure(&|dfa| dfa.to_regex_with(&synthesizer).unwrap())
                );
            }
        }
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
    dfas(atoms, max_states + 1..=80, PER_OPERATION)
}

/// Up to `per_operation` minimal DFAs of `states` states from each set
/// operation on `atoms`, by operation.
fn dfas(
    atoms: &[RegularExpression],
    states: RangeInclusive<usize>,
    per_operation: usize,
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
                if dfas.len() == per_operation {
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
                    && states.contains(&dfa.number_of_states())
                {
                    dfas.push(dfa);
                }
            }
            (*name, dfas)
        })
        .collect()
}
