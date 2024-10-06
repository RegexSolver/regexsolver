use std::{fs::File, io::{BufRead, BufReader}};

use regex::Regex;
use regexsolver::regex::RegularExpression;

fn assert_regex(regex: &str) {
    let re = Regex::new(&format!("(?s)^{}$", regex)).unwrap();

    let regex = RegularExpression::new(regex).unwrap();
    let automaton = regex.to_automaton().unwrap();
    let strings = automaton.generate_strings(500).unwrap();
    for string in strings {
        assert!(re.is_match(&string), "'{string}'");
    }

    assert_eq!(automaton.get_number_of_states(), regex.get_number_of_states_in_nfa());

    let determinized_automaton = automaton.determinize().unwrap();
    let strings = determinized_automaton.generate_strings(500).unwrap();
    for string in strings {
        assert!(re.is_match(&string), "'{string}'");
    }

    assert!(automaton.is_subset_of(&determinized_automaton).unwrap());
    assert!(determinized_automaton.is_subset_of(&automaton).unwrap());
    assert!(automaton.is_equivalent_of(&determinized_automaton).unwrap());

    let regex_from_automaton = automaton.to_regex().unwrap();
    let automaton_from_regex = regex_from_automaton.to_automaton().unwrap();
    assert!(automaton.is_equivalent_of(&automaton_from_regex).unwrap());
}

#[test]
fn test_regular_expression_parsing() {
    let file = File::open("tests/data/regex.txt").unwrap();
    let reader = BufReader::new(file);
    for regex in reader.lines() {
        let regex = regex.unwrap();
        println!("{}", &regex);
        assert_regex(&regex);
    }
}
