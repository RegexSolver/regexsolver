//! Measures the quality of automaton→regex conversion (state elimination)
//! over the shared corpus. Not a pass/fail test of absolute numbers — it
//! prints aggregate metrics so a heuristic change can be compared before/after
//! (`cargo test --test state_elimination_quality -- --ignored --nocapture`),
//! while still asserting that every conversion round-trips (correctness).

use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use regexsolver::regex::RegularExpression;

#[test]
#[ignore = "measurement harness; run explicitly with --ignored --nocapture"]
fn measure_state_elimination_quality() {
    let file = File::open("tests/data/regex.txt").unwrap();
    let reader = BufReader::new(file);

    let mut count = 0usize;
    let mut total_complexity_nfa = 0.0f64;
    let mut total_len_nfa = 0usize;
    let mut total_complexity_dfa = 0.0f64;
    let mut total_len_dfa = 0usize;

    for line in reader.lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let input = match RegularExpression::parse(&line, true) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let automaton = input.to_automaton().unwrap();

        // NFA-derived conversion.
        let out_nfa = automaton.to_regex();
        assert!(
            automaton
                .equivalent(&out_nfa.to_automaton().unwrap())
                .unwrap(),
            "NFA round-trip mismatch for {line:?} -> {out_nfa}"
        );
        total_complexity_nfa += out_nfa.evaluate_complexity();
        total_len_nfa += out_nfa.to_string().chars().count();

        // DFA-derived conversion.
        let dfa = automaton.determinize().unwrap();
        let out_dfa = dfa.to_regex();
        assert!(
            dfa.equivalent(&out_dfa.to_automaton().unwrap()).unwrap(),
            "DFA round-trip mismatch for {line:?} -> {out_dfa}"
        );
        total_complexity_dfa += out_dfa.evaluate_complexity();
        total_len_dfa += out_dfa.to_string().chars().count();

        count += 1;
    }

    println!("=== state elimination quality over {count} patterns ===");
    println!("NFA: total_complexity = {total_complexity_nfa:.3}, total_len = {total_len_nfa}");
    println!("DFA: total_complexity = {total_complexity_dfa:.3}, total_len = {total_len_dfa}");
    println!(
        "SUM: total_complexity = {:.3}, total_len = {}",
        total_complexity_nfa + total_complexity_dfa,
        total_len_nfa + total_len_dfa
    );
}
