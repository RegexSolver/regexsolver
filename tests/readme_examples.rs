//! Keeps the README's examples honest: these tests are the README snippets,
//! verbatim. If one fails, update the README.

use regexsolver::Term;
use regexsolver::error::EngineError;

#[test]
fn readme_automaton_building_example() -> Result<(), EngineError> {
    use regex_charclass::char::Char;
    use regexsolver::CharRange;
    use regexsolver::fast_automaton::FastAutomaton;

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
    assert_eq!(automaton.to_regex().to_string(), "[a-c][0-9]*");

    Ok(())
}

#[test]
fn readme_hero_example() -> Result<(), EngineError> {
    let a = Term::from_pattern("(ab|xy){2}")?;
    let b = Term::from_pattern(".*xy")?;

    // Which strings match BOTH patterns? Get the answer as a regex:
    let both = a.intersection(&[b])?;
    assert_eq!(both.to_pattern(), "(ab|xy)xy");

    // ...and sample them:
    assert_eq!(both.generate_strings(2, 0)?, ["xyxy", "abxy"]);

    Ok(())
}

#[test]
fn readme_regular_expression_example() -> Result<(), EngineError> {
    use regexsolver::cardinality::Cardinality;
    use regexsolver::regex::RegularExpression;

    // A validation pattern for an order id, e.g. "ORD-2024-12345".
    let pattern = RegularExpression::new("ORD-20[0-9]{2}-[0-9]{4,6}")?;

    // How long can matching ids get? Size your database column accordingly.
    assert_eq!(pattern.get_length(), (Some(13), Some(15)));

    // How many distinct ids does the pattern allow?
    assert_eq!(pattern.get_cardinality(), Cardinality::Integer(111_000_000));

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
