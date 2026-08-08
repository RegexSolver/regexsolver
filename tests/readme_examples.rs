//! Keeps the README's examples honest: these tests are the README snippets,
//! verbatim. If one fails, update the README.

use regexsolver::Term;
use regexsolver::error::EngineError;
use regexsolver::execution_profile::ExecutionProfileBuilder;
use regexsolver::fast_automaton::{GenerationOptions, PathOrder};

#[test]
fn readme_automaton_building_example() -> Result<(), EngineError> {
    use regexsolver::CharRange;
    use regexsolver::fast_automaton::FastAutomaton;
    use regexsolver::regex_charclass::char::Char;

    // Build an automaton matching "[a-c][0-9]*" by hand:
    let mut automaton = FastAutomaton::new_empty();
    let s1 = automaton.new_state();
    automaton.accept(s1);

    let a_to_c = CharRange::new_from_range(Char::new('a')..=Char::new('c'));
    let digits = CharRange::new_from_range(Char::new('0')..=Char::new('9'));
    automaton.add_transition_from_range(0, s1, &a_to_c)?;
    automaton.add_transition_from_range(s1, s1, &digits)?;

    assert!(automaton.is_match("b42"));
    assert!(!automaton.is_match("4b"));
    assert_eq!(automaton.to_regex()?.to_string(), "[a-c][0-9]*");

    Ok(())
}

#[test]
fn readme_hero_example() -> Result<(), EngineError> {
    let a: Term = "(ab|xy){2}".parse()?;
    let b: Term = ".*xy".parse()?;

    // Which strings match BOTH patterns? Get the answer as a regex:
    let both = a.intersection([&b])?;
    assert_eq!(both.to_pattern()?, "(ab|xy)xy");

    // Test a concrete string against the result (matching is anchored):
    assert!(both.matches("abxy")?);

    // ...and sample them:
    assert_eq!(
        both.generate_strings(2, 0, PathOrder::Sweep)?,
        ["xyxy", "abxy"]
    );

    Ok(())
}

#[test]
fn readme_regular_expression_example() -> Result<(), EngineError> {
    use regexsolver::regex::RegularExpression;

    // A validation pattern for an order id, e.g. "ORD-2024-12345".
    let pattern = RegularExpression::new("ORD-20[0-9]{2}-[0-9]{4,6}")?;

    // How long can matching ids get? Size your database column accordingly.
    assert_eq!(pattern.length(), (Some(13), Some(15)));

    // The AST is a plain enum: walk it to lint patterns, e.g. reject
    // validation rules that accept unboundedly long input.
    fn has_unbounded_repetition(regex: &RegularExpression) -> bool {
        match regex {
            RegularExpression::Character(_) => false,
            RegularExpression::Repetition(inner, _, max) => {
                max.is_none() || has_unbounded_repetition(inner)
            }
            RegularExpression::Concat(parts) => parts.iter().any(has_unbounded_repetition),
            RegularExpression::Alternation(parts) => parts.iter().any(has_unbounded_repetition),
        }
    }
    assert!(!has_unbounded_repetition(&pattern));
    assert!(has_unbounded_repetition(&RegularExpression::new(
        ".*@example\\.com"
    )?));

    Ok(())
}

// Timing-dependent in one direction only: the assertion needs the
// 100-million-string generation to *not* finish within 50ms, which no
// machine can do, so the test cannot flake.
#[test]
fn readme_time_bounded_execution_example() -> Result<(), EngineError> {
    let term = Term::from_pattern(".*abc.*cdef.*sqdsqf.*")?;

    let execution_profile = ExecutionProfileBuilder::new()
        .execution_timeout(50) // limit in milliseconds
        .build();

    // Asking for 100 million strings cannot finish within the budget, so the
    // generation aborts instead of running to completion.
    execution_profile.run(|| {
        assert_eq!(
            EngineError::OperationTimeOutError,
            term.generate_strings(100_000_000, 0, GenerationOptions::new())
                .unwrap_err()
        );
    });

    Ok(())
}

#[test]
fn readme_state_limited_execution_example() -> Result<(), EngineError> {
    let term1 = Term::from_pattern(".*abcdef.*")?;
    let term2 = Term::from_pattern(".*defabc.*")?;

    let execution_profile = ExecutionProfileBuilder::new()
        .max_number_of_states(5) // we set the limit
        .build();

    // We run the operation with the defined limitation
    execution_profile.run(|| {
        assert_eq!(
            EngineError::AutomatonHasTooManyStates,
            term1.intersection(&[term2]).unwrap_err()
        );
    });

    Ok(())
}

#[test]
fn readme_disabling_implicit_determinization_example() -> Result<(), EngineError> {
    // Any non-deterministic FastAutomaton; ".*abc" compiles to one.
    let nfa = Term::from_pattern(".*abc")?.to_automaton()?.into_owned();
    assert!(!nfa.is_deterministic());

    let execution_profile = ExecutionProfileBuilder::new()
        .implicit_determinization(false) // default is true
        .build();

    execution_profile.run(|| {
        let mut cannot_minimize = nfa.clone();
        assert_eq!(
            EngineError::DeterministicAutomatonRequired,
            cannot_minimize.minimize().unwrap_err()
        );

        // Determinizing explicitly is always allowed.
        let mut dfa = nfa.determinize().unwrap().into_owned();
        assert!(dfa.minimize().is_ok());
    });

    Ok(())
}
