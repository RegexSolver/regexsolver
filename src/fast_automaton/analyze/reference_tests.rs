//! `equivalent`, `subset`, `cardinality` and `difference`, checked against
//! the textbook definitions (determinize, then complement and intersect, or
//! count paths).

use proptest::prelude::*;
use regex_charclass::char::Char;

use crate::CharRange;
use crate::cardinality::Cardinality;
use crate::error::EngineError;
use crate::execution_profile::ExecutionProfileBuilder;
use crate::fast_automaton::FastAutomaton;
use crate::regex::RegularExpression;

fn automaton(pattern: &str) -> FastAutomaton {
    RegularExpression::new(pattern)
        .unwrap()
        .to_automaton()
        .unwrap()
}

/// The textbook check: `a ∩ ¬b` is empty.
fn subset_by_complement(a: &FastAutomaton, b: &FastAutomaton) -> bool {
    let mut complement = b.determinize().unwrap().into_owned();
    complement.complement().unwrap();
    !a.has_intersection(&complement).unwrap()
}

fn equivalent_by_complement(a: &FastAutomaton, b: &FastAutomaton) -> bool {
    subset_by_complement(a, b) && subset_by_complement(b, a)
}

/// The textbook difference: `a ∩ ¬b`.
fn difference_by_complement(a: &FastAutomaton, b: &FastAutomaton) -> FastAutomaton {
    let mut complement = b.determinize().unwrap().into_owned();
    complement.complement().unwrap();
    a.intersection(&complement).unwrap()
}

fn assert_difference_matches_reference(a: &FastAutomaton, b: &FastAutomaton) {
    let difference = a.difference(b).unwrap();
    let reference = difference_by_complement(a, b);
    assert!(
        equivalent_by_complement(&difference, &reference),
        "difference disagrees on\n{a}\n{b}"
    );
    // Both build the pairs `a` reaches with the sets of states of `b`, so
    // the result is no larger than the reference, and deterministic when
    // `a` is.
    assert!(
        difference.number_of_states() <= reference.number_of_states(),
        "difference is larger on\n{a}\n{b}"
    );
    assert!(!a.is_deterministic() || difference.is_deterministic());
}

/// The check on two deterministic automata, on the determinized operands
/// (which keep their dead states and have missing transitions) and on a
/// minimized one (which has neither, and another spanning set).
fn assert_deterministic_equivalence_matches_reference(a: &FastAutomaton, b: &FastAutomaton) {
    let (a_dfa, b_dfa) = (a.determinize().unwrap(), b.determinize().unwrap());
    let mut a_minimal = a_dfa.clone().into_owned();
    a_minimal.minimize().unwrap();
    let expected = equivalent_by_complement(a, b);
    for (left, right) in [
        (a_dfa.as_ref(), b_dfa.as_ref()),
        (b_dfa.as_ref(), a_dfa.as_ref()),
        (&a_minimal, b_dfa.as_ref()),
        (b_dfa.as_ref(), &a_minimal),
    ] {
        assert_eq!(
            expected,
            left.equivalent_deterministic(right).unwrap(),
            "equivalent disagrees on the deterministic\n{left}\n{right}"
        );
    }
    assert!(a_dfa.equivalent_deterministic(&a_minimal).unwrap());
    assert!(a_minimal.equivalent_deterministic(&a_dfa).unwrap());
}

/// Both checks, in both directions, against the complement-based
/// reference.
fn assert_matches_reference(a: &FastAutomaton, b: &FastAutomaton) {
    assert_eq!(
        equivalent_by_complement(a, b),
        a.equivalent_over_state_sets(b).unwrap(),
        "equivalent disagrees on\n{a}\n{b}"
    );
    assert_eq!(
        subset_by_complement(a, b),
        a.subset_over_state_sets(b).unwrap(),
        "subset disagrees on\n{a}\n{b}"
    );
    assert_eq!(
        subset_by_complement(b, a),
        b.subset_over_state_sets(a).unwrap(),
        "subset disagrees on\n{b}\n{a}"
    );
}

#[test]
fn matches_the_reference_on_patterns() {
    let patterns = [
        "",
        "a",
        "a*",
        ".*",
        ".+",
        "(a|b)*",
        "(a*b*)*",
        "(a|b)*abc(a|b)*",
        "(a*b*)*abc(b*a*)*",
        "(a|b)*a(a|b){6}",
        "(a*b*)*a(a|b){6}",
        "(a|b)*a(a|b){5}",
        "(a|b)*a(a|b)*",
        "(.*a.{4})|(.*b.{4})",
        ".*[ab].{4}",
        ".*abc.*def.*",
        ".*abc.*(def).*x?",
        "abc(de|fg){1,3}",
        "abc.*",
        "test.{0,9}other",
        "test.{0,8}other",
        r"\w+@\w+\.\w{2,3}",
        r"[\w]+@\w*\w\.\w{2,3}",
        r"\d+",
        "[0-9]+|x",
    ];
    let automata: Vec<FastAutomaton> = patterns.iter().map(|p| automaton(p)).collect();
    let empty = FastAutomaton::new_empty();
    for a in automata.iter().chain([&empty]) {
        let determinized = a.determinize().unwrap();
        assert!(a.equivalent_over_state_sets(&determinized).unwrap());
        assert!(determinized.equivalent_over_state_sets(a).unwrap());
        for b in automata.iter().chain([&empty]) {
            assert_matches_reference(a, b);
            assert_matches_reference(&determinized, b);
            assert_difference_matches_reference(a, b);
            assert_deterministic_equivalence_matches_reference(a, b);
        }
    }
}

// The 2^(n+1) reachable sets of states all accept different languages, so
// nothing can be pruned, most recorded sets are contained in the queried one
// (the normal form search runs out of budget) and, once determinized, every
// recorded set is a singleton hanging off one wide trie node.
#[test]
fn matches_the_reference_when_nothing_can_be_pruned() {
    let a = automaton("(a|b)*a(a|b){11}");
    let b = automaton("(a*b*)*a(a|b){11}");
    let c = automaton("(a|b)*a(a|b){11}|b{20}");
    assert!(a.equivalent_over_state_sets(&b).unwrap());
    assert!(!a.equivalent_over_state_sets(&c).unwrap());
    assert!(a.subset_over_state_sets(&c).unwrap());
    assert!(!c.subset_over_state_sets(&a).unwrap());

    let (a, b) = (a.determinize().unwrap(), b.determinize().unwrap());
    assert!(
        a.equivalent_over_state_sets(&c)
            .is_ok_and(|equivalent| !equivalent)
    );
    assert!(a.equivalent_over_state_sets(&b).unwrap());
    assert!(a.subset_over_state_sets(&b).unwrap());
    assert!(
        c.equivalent_over_state_sets(&b)
            .is_ok_and(|equivalent| !equivalent)
    );
}

// Determinizing either operand takes at least 2^9 states, the walk fewer
// than 32 pairs.
#[test]
fn skips_what_the_recorded_pairs_imply() {
    // United as automata: as a pattern, the union is simplified into `b`.
    let a = FastAutomaton::union_all(&[
        automaton(".*a.{8}"),
        automaton(".*b.{8}"),
        automaton(".*c.{8}"),
    ])
    .unwrap();
    let b = automaton(".*[abc].{8}");
    assert_ne!(a, b);
    assert!(a.determinize().unwrap().number_of_states() >= 512);
    assert!(b.determinize().unwrap().number_of_states() >= 512);
    ExecutionProfileBuilder::new()
        .max_number_of_states(32)
        .build()
        .run(|| {
            assert!(a.equivalent(&b).unwrap());
            assert!(a.subset(&b).unwrap());
            assert!(b.subset(&a).unwrap());
        });
}

// The minuend only reaches a few of the sets of the subtrahend, which
// cannot be determinized under the same cap.
#[test]
fn difference_only_builds_the_pairs_a_word_reaches() {
    let a = automaton("(a|b){0,6}");
    let b = automaton("(a|b)*a(a|b){11}");
    ExecutionProfileBuilder::new()
        .max_number_of_states(200)
        .build()
        .run(|| {
            assert_eq!(
                b.determinize().unwrap_err(),
                EngineError::AutomatonHasTooManyStates
            );
            assert!(a.difference(&b).unwrap().equivalent(&a).unwrap());
            assert_eq!(
                automaton("(a|b)*").difference(&b).unwrap_err(),
                EngineError::AutomatonHasTooManyStates
            );
        });
}

#[test]
fn respects_the_state_budget_and_the_timeout() {
    let a = automaton("(a|b)*a(a|b){11}");
    let b = automaton("(a*b*)*a(a|b){11}");
    let (a_dfa, b_dfa) = (a.determinize().unwrap(), b.determinize().unwrap());
    ExecutionProfileBuilder::new()
        .max_number_of_states(100)
        .build()
        .run(|| {
            let err = EngineError::AutomatonHasTooManyStates;
            assert_eq!(a.equivalent(&b).unwrap_err(), err);
            assert_eq!(a.subset(&b).unwrap_err(), err);
            assert_eq!(a_dfa.equivalent(&b_dfa).unwrap_err(), err);
        });
    ExecutionProfileBuilder::new()
        .execution_timeout(0)
        .clock(|| {
            use std::sync::atomic::{AtomicU64, Ordering};
            static READINGS: AtomicU64 = AtomicU64::new(0);
            std::time::Duration::from_millis(READINGS.fetch_add(1, Ordering::Relaxed))
        })
        .build()
        .run(|| {
            let err = EngineError::OperationTimeOutError;
            assert_eq!(a.equivalent(&b).unwrap_err(), err);
            assert_eq!(a.subset(&b).unwrap_err(), err);
        });
}

/// A random automaton of up to `max_states` states over ranges cut from
/// the first `letters` letters. Unreachable and dead states, empty languages
/// and non-determinism all occur.
fn arb_automaton(max_states: usize, letters: u8) -> impl Strategy<Value = FastAutomaton> {
    arb_automaton_with(max_states, letters, false)
}

/// [`arb_automaton`], with every transition going to a higher state when
/// `acyclic` is set: the language is then finite.
fn arb_automaton_with(
    max_states: usize,
    letters: u8,
    acyclic: bool,
) -> impl Strategy<Value = FastAutomaton> {
    (1..=max_states).prop_flat_map(move |states| {
        (
            proptest::collection::vec(any::<bool>(), states),
            proptest::collection::vec(
                (0..states, 0..states, 0..letters, 0..letters),
                0..3 * states,
            ),
        )
            .prop_map(move |(accept, transitions)| {
                let mut automaton = FastAutomaton::new_empty();
                for _ in 1..states {
                    automaton.new_state();
                }
                for (state, accept) in accept.into_iter().enumerate() {
                    if accept {
                        automaton.accept(state);
                    }
                }
                for (from, to, x, y) in transitions {
                    let (from, to) = if acyclic {
                        (from.min(to), from.max(to))
                    } else {
                        (from, to)
                    };
                    if acyclic && from == to {
                        continue;
                    }
                    let (low, high) = (b'a' + x.min(y), b'a' + x.max(y));
                    let range =
                        CharRange::new_from_range(Char::new(low as char)..=Char::new(high as char));
                    automaton
                        .add_transition_from_range(from, to, &range)
                        .unwrap();
                }
                automaton
            })
    })
}

fn assert_matches_reference_and_itself(a: &FastAutomaton, b: &FastAutomaton) {
    assert_matches_reference(a, b);
    assert_difference_matches_reference(a, b);
    assert_difference_matches_reference(b, a);
    assert_deterministic_equivalence_matches_reference(a, b);
    let determinized = a.determinize().unwrap();
    assert!(a.equivalent_over_state_sets(&determinized).unwrap());
    assert!(determinized.subset_over_state_sets(a).unwrap());
    // Same language as `a`, over the states of both.
    let doubled = a.union(&determinized).unwrap();
    assert!(a.equivalent_over_state_sets(&doubled).unwrap());
    assert!(doubled.subset_over_state_sets(a).unwrap());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    // Many letters: the operands rarely share a spanning set.
    #[test]
    fn matches_the_reference_on_random_automata(
        a in arb_automaton(6, 6),
        b in arb_automaton(6, 6),
    ) {
        assert_matches_reference_and_itself(&a, &b);
    }

    // Two letters and more states: long walks over large sets, where
    // pairs are skipped without having been met before.
    #[test]
    fn matches_the_reference_on_dense_random_automata(
        a in arb_automaton(8, 2),
        b in arb_automaton(8, 2),
    ) {
        assert_matches_reference_and_itself(&a, &b);
    }

    // Finite languages over few letters: many paths spell the same string.
    #[test]
    fn cardinality_matches_the_determinized_count(a in arb_automaton_with(9, 3, true)) {
        let determinized = a.determinize().unwrap();
        prop_assert!(determinized.is_deterministic());
        prop_assert_eq!(a.cardinality().unwrap(), determinized.cardinality().unwrap());
    }

    // Cycles included: the count is only reached when the language is finite.
    #[test]
    fn cardinality_matches_the_determinized_count_with_cycles(a in arb_automaton(5, 3)) {
        prop_assert_eq!(
            a.cardinality().unwrap(),
            a.determinize().unwrap().cardinality().unwrap()
        );
    }
}

/// The number of strings over `a`, `b` and `c` of at most `max_length`
/// letters that `automaton` matches.
fn count_matches(automaton: &FastAutomaton, max_length: usize) -> u32 {
    let mut strings = vec![String::new()];
    let mut count = 0;
    for _ in 0..=max_length {
        count += strings.iter().filter(|s| automaton.is_match(s)).count() as u32;
        strings = strings
            .iter()
            .flat_map(|s| ["a", "b", "c"].map(|letter| format!("{s}{letter}")))
            .collect();
    }
    count
}

#[test]
fn cardinality_counts_each_string_once() {
    // Several paths spell the same string in each of these: the count is
    // checked against matching every string that is short enough.
    for (pattern, max_length) in [
        ("(ab|a)(bc|c)?", 4),
        ("(a|b|ab|ba){3}", 6),
        ("(a|b){0,4}a(a|b){3}", 8),
        ("a{2,5}b?a{0,3}", 9),
    ] {
        let a = automaton(pattern);
        assert!(!a.is_deterministic(), "{pattern}");
        let expected = Cardinality::Integer(count_matches(&a, max_length));
        assert_eq!(a.cardinality().unwrap(), expected, "{pattern}");
        assert_eq!(
            a.determinize().unwrap().cardinality().unwrap(),
            expected,
            "{pattern}"
        );
    }
    assert_eq!(
        automaton("(ab|a)(bc|c)?").cardinality().unwrap(),
        Cardinality::Integer(5)
    );

    for (pattern, expected) in [
        (
            r"((25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(25[0-5]|2[0-4]\d|[01]?\d?\d)",
            Cardinality::BigInteger,
        ),
        ("(a|b){0,4}a(a|b){40}", Cardinality::BigInteger),
        ("(ab|a)(bc|c)*", Cardinality::Infinite),
    ] {
        let a = automaton(pattern);
        assert!(!a.is_deterministic(), "{pattern}");
        assert_eq!(a.cardinality().unwrap(), expected, "{pattern}");
    }
}

// The count overflows long before every set is visited: determinizing
// exceeds the same cap.
#[test]
fn cardinality_stops_at_the_first_overflow() {
    let a = automaton("(a|b){0,20}a(a|b){40}");
    ExecutionProfileBuilder::new()
        .max_number_of_states(1000)
        .build()
        .run(|| {
            assert_eq!(a.cardinality().unwrap(), Cardinality::BigInteger);
            assert_eq!(
                a.determinize().unwrap_err(),
                EngineError::AutomatonHasTooManyStates
            );
        });
}

#[test]
fn cardinality_respects_the_state_budget() {
    let a = automaton("(a|b){0,12}a(a|b){12}");
    ExecutionProfileBuilder::new()
        .max_number_of_states(100)
        .build()
        .run(|| {
            assert_eq!(
                a.cardinality().unwrap_err(),
                EngineError::AutomatonHasTooManyStates
            );
        });
}
