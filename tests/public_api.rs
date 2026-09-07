//! Tests driving `regexsolver` through its public API, the way a dependent
//! crate sees it: the conversions and argument shapes the type system
//! promises, and the properties that hold across the crate rather than inside
//! one module.

use std::borrow::Cow;
use std::collections::HashSet;

use regexsolver::error::EngineError;
use regexsolver::fast_automaton::{CharacterOrder, FastAutomaton, GenerationOptions, PathOrder};
use regexsolver::regex::RegularExpression;
use regexsolver::regex_charclass::CharacterClass;
use regexsolver::{CharRange, Term};

fn term(pattern: &str) -> Term {
    Term::from_pattern(pattern).unwrap()
}

/// The Unicode tables `regex-charclass` compiles in have to match the ones
/// `regex-syntax` parses with, or a named class survives a parse/print round
/// trip as a raw list of ranges. `Cargo.toml` pins the `ucd-16` feature for
/// this; a `regex-syntax` bump to a newer Unicode release fails here.
#[test]
fn unicode_tables_agree_with_the_parser() {
    assert_eq!("16.0.0", regexsolver::regex_charclass::UCD_VERSION);

    // A class is parsed into ranges by `regex-syntax` and named back by
    // `regex-charclass`, so it only survives if both know the same set.
    for class in [
        "\\p{Greek}",
        "\\p{Latin}",
        "\\p{Cyrillic}",
        "\\p{Han}",
        "\\p{Letter}",
        "\\p{Alphabetic}",
        "\\w",
        "\\d",
        "\\s",
    ] {
        assert_eq!(
            class,
            RegularExpression::new(class).unwrap().to_string(),
            "`{class}` came back as a range list rather than its name; the `ucd-*` \
             feature and `regex-syntax` disagree about the Unicode version"
        );
    }

    // An alias comes back under the canonical name of the same set.
    assert_eq!(
        "\\p{Uppercase_Letter}",
        RegularExpression::new("\\p{Lu}").unwrap().to_string()
    );
    assert_eq!(
        "\\d",
        RegularExpression::new("\\p{Decimal_Number}")
            .unwrap()
            .to_string()
    );
}

#[test]
fn default_is_the_empty_language() {
    assert_eq!(Term::new_empty(), Term::default());
    assert!(Term::default().is_empty().unwrap());
}

#[test]
fn parses_through_from_str_and_from_pattern_alike() {
    let parsed: Term = "(ab|xy){2}".parse().unwrap();
    assert_eq!(Term::from_pattern("(ab|xy){2}").unwrap(), parsed);

    let error = "a(".parse::<Term>().unwrap_err();
    assert!(
        matches!(error, EngineError::RegexSyntaxError(_)),
        "{error:?}"
    );
}

#[test]
fn converts_from_both_representations() {
    let regex = RegularExpression::new("abc").unwrap();
    let automaton = regex.to_automaton().unwrap();

    let from_regex: Term = regex.clone().into();
    let from_automaton: Term = automaton.clone().into();

    assert_eq!(Term::from_regex(regex), from_regex);
    assert_eq!(Term::from_automaton(automaton), from_automaton);
    assert!(from_regex.equivalent(&from_automaton).unwrap());
}

/// The set operations take `impl IntoIterator<Item = impl Borrow<Term>>`, so a
/// slice, an array of references, an owned `Vec` and a bare iterator all work
/// without cloning anything.
#[test]
fn set_operations_accept_every_argument_shape() {
    let a = term("abc");
    let b = term("abcd");
    let expected = term("abc|abcd");

    assert!(a.union([&b]).unwrap().equivalent(&expected).unwrap());
    assert!(
        a.union(std::slice::from_ref(&b))
            .unwrap()
            .equivalent(&expected)
            .unwrap()
    );
    assert!(
        a.union(vec![b.clone()])
            .unwrap()
            .equivalent(&expected)
            .unwrap()
    );
    assert!(
        a.union(std::iter::once(&b))
            .unwrap()
            .equivalent(&expected)
            .unwrap()
    );

    // With no operands at all, a union is just the term itself.
    assert!(
        a.union(Vec::<&Term>::new())
            .unwrap()
            .equivalent(&a)
            .unwrap()
    );
}

/// `repeat` takes any `RangeBounds<u32>`, so the bound forms below all denote
/// the languages their range syntax reads as.
#[test]
fn repeat_accepts_every_range_form() {
    let a = term("ab");

    assert_eq!("(ab){2}", a.repeat(2..3).unwrap().to_pattern().unwrap());
    assert_eq!("(ab){2}", a.repeat(2..=2).unwrap().to_pattern().unwrap());
    assert_eq!("(ab){2,3}", a.repeat(2..=3).unwrap().to_pattern().unwrap());
    assert_eq!("(ab){2,}", a.repeat(2..).unwrap().to_pattern().unwrap());
    assert_eq!("(ab)*", a.repeat(..).unwrap().to_pattern().unwrap());

    // An empty range gives the empty language rather than an error, the way
    // Rust reads such a range. The ranges go through a variable because
    // `clippy::reversed_empty_ranges` fires on a literal `3..2`.
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = 3..2;
    assert!(a.repeat(reversed).unwrap().is_empty().unwrap());
    #[allow(clippy::reversed_empty_ranges)]
    let empty = 0..0;
    assert!(a.repeat(empty).unwrap().is_empty().unwrap());
}

/// The `to_*` conversions borrow when the term already holds the representation
/// asked for, which is what returning `Cow` is for.
#[test]
fn conversions_borrow_when_they_can() {
    let regex_backed = term("abc");
    assert!(matches!(regex_backed.to_regex().unwrap(), Cow::Borrowed(_)));
    assert!(matches!(
        regex_backed.to_automaton().unwrap(),
        Cow::Owned(_)
    ));

    let automaton_backed = Term::from_automaton(
        RegularExpression::new("abc")
            .unwrap()
            .to_automaton()
            .unwrap(),
    );
    assert!(matches!(
        automaton_backed.to_automaton().unwrap(),
        Cow::Borrowed(_)
    ));
}

/// Every operation is available on both representations, and which one the
/// `Term` layer picks may not change the answer.
#[test]
fn both_representations_give_the_same_answers() {
    let regex_backed = term("(ab|xy){2}");
    let automaton_backed = Term::from_automaton(regex_backed.to_automaton().unwrap().into_owned());

    assert_eq!(
        regex_backed.length(),
        automaton_backed.length(),
        "length disagrees"
    );
    assert_eq!(
        regex_backed.cardinality().unwrap(),
        automaton_backed.cardinality().unwrap(),
        "cardinality disagrees"
    );
    assert_eq!(
        regex_backed.is_empty().unwrap(),
        automaton_backed.is_empty().unwrap()
    );
    assert_eq!(
        regex_backed.is_total().unwrap(),
        automaton_backed.is_total().unwrap()
    );
    assert_eq!(
        regex_backed.is_finite().unwrap(),
        automaton_backed.is_finite().unwrap()
    );
    assert_eq!(
        regex_backed.matches("abxy").unwrap(),
        automaton_backed.matches("abxy").unwrap()
    );
    assert!(regex_backed.equivalent(&automaton_backed).unwrap());
}

/// `iter_strings` is the lazy form of `generate_strings`; over the same options
/// the two enumerate the same language in the same order.
#[test]
fn eager_and_lazy_generation_agree() {
    let term = term("[a-c][0-9]");

    for options in [
        GenerationOptions::from(PathOrder::Sweep),
        GenerationOptions::from(PathOrder::Interleave),
        GenerationOptions::from((PathOrder::Shuffled, CharacterOrder::Shuffled)),
    ] {
        let eager = term.generate_strings(10, 0, options.clone()).unwrap();
        let lazy = term
            .iter_strings(options.clone())
            .take(10)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(eager, lazy);

        // Paging is a window over that same sequence.
        let paged = term.generate_strings(4, 6, options).unwrap();
        assert_eq!(&eager[6..], paged.as_slice());
    }
}

/// The two enumeration axes reorder a language without changing it: over a
/// finite language, every mode yields exactly the same set of strings.
#[test]
fn every_generation_order_enumerates_the_same_language() {
    let term = term("[a-c][0-9]");
    let total = 30;

    let reference: HashSet<String> = term
        .generate_strings(total, 0, PathOrder::Sweep)
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(total, reference.len());

    for path in [PathOrder::Sweep, PathOrder::Interleave, PathOrder::Shuffled] {
        for characters in [CharacterOrder::Ascending, CharacterOrder::Shuffled] {
            let generated: HashSet<String> = term
                .generate_strings(total, 0, (path, characters))
                .unwrap()
                .into_iter()
                .collect();
            assert_eq!(reference, generated, "{path:?} / {characters:?} differs");
        }
    }
}

/// The shuffled modes repeat under the same seed and differ under another.
#[test]
fn shuffled_generation_is_reproducible_by_seed() {
    let term = term("[a-z]{4}");
    let shuffled = GenerationOptions::from((PathOrder::Shuffled, CharacterOrder::Shuffled));

    let first = term
        .generate_strings(20, 0, shuffled.clone().with_seed(42))
        .unwrap();
    let again = term
        .generate_strings(20, 0, shuffled.clone().with_seed(42))
        .unwrap();
    let other_seed = term
        .generate_strings(20, 0, shuffled.with_seed(43))
        .unwrap();

    assert_eq!(first, again);
    assert_ne!(first, other_seed);
}

/// The bounds are constraints on what may come out, not filters applied after
/// the fact: nothing outside them is generated, and `offset` does not count it.
#[test]
fn generation_bounds_constrain_what_is_generated() {
    let term = term(".*abc.*");

    let options = GenerationOptions::new()
        .with_charset(CharRange::new_from_range_char('a'..='z'))
        .with_min_length(4)
        .with_max_length(6);

    for string in term.generate_strings(50, 0, options).unwrap() {
        assert!(string.contains("abc"), "{string:?} is not in the language");
        assert!((4..=6).contains(&string.chars().count()), "{string:?}");
        assert!(
            string.chars().all(|c| c.is_ascii_lowercase()),
            "{string:?} leaves the charset"
        );
    }
}

/// A language bounded above in length is finite, so its lazy iterator ends
/// rather than running forever over an infinite one.
#[test]
fn a_max_length_makes_an_infinite_language_finite() {
    let strings = term("[ab]*")
        .iter_strings(GenerationOptions::new().with_max_length(3))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // The empty string, then every string of a and b up to length 3.
    assert_eq!(1 + 2 + 4 + 8, strings.len());
    assert!(strings.iter().all(|s| s.chars().count() <= 3));
}

/// The identities that hold for any language, checked against the crate's own
/// notion of equivalence.
#[test]
fn set_identities_hold() {
    let a = term("(ab|xy){2}");
    let b = term(".*xy");

    let complement = a.complement().unwrap();
    assert!(a.intersection([&complement]).unwrap().is_empty().unwrap());
    assert!(a.union([&complement]).unwrap().is_total().unwrap());
    assert!(complement.complement().unwrap().equivalent(&a).unwrap());

    // difference(a, b) == intersection(a, complement(b))
    assert!(
        a.difference(&b)
            .unwrap()
            .equivalent(&a.intersection([&b.complement().unwrap()]).unwrap())
            .unwrap()
    );

    // Both operands are subsets of their union, and their intersection is a
    // subset of both.
    let union = a.union([&b]).unwrap();
    let intersection = a.intersection([&b]).unwrap();
    assert!(a.subset(&union).unwrap());
    assert!(b.subset(&union).unwrap());
    assert!(intersection.subset(&a).unwrap());
    assert!(intersection.subset(&b).unwrap());
}

/// `to_pattern` has to denote the same language as the term it came from.
/// Reparsing its output is the check.
#[test]
fn patterns_round_trip_through_their_language() {
    for pattern in [
        "(ab|xy){2}",
        ".*abc.*def.*",
        "[a-c][0-9]{2,4}",
        "a(bcfe|bcdg|mkv)*",
        "[]",
        "",
    ] {
        let original = term(pattern);
        let printed = original.to_pattern().unwrap();
        let reparsed = term(&printed);
        assert!(
            original.equivalent(&reparsed).unwrap(),
            "`{pattern}` printed as `{printed}`, which is a different language"
        );
    }
}

/// The empty language and the language of the empty string are different.
#[test]
fn the_empty_language_is_not_the_empty_string() {
    let empty = Term::new_empty();
    let empty_string = Term::new_empty_string();

    assert!(empty.is_empty().unwrap());
    assert!(!empty.is_empty_string().unwrap());
    assert!(!empty_string.is_empty().unwrap());
    assert!(empty_string.is_empty_string().unwrap());

    assert!(!empty.equivalent(&empty_string).unwrap());
    assert_eq!("[]", empty.to_pattern().unwrap());
    assert_eq!("", empty_string.to_pattern().unwrap());

    assert!(!empty.matches("").unwrap());
    assert!(empty_string.matches("").unwrap());
}

/// Matching is anchored: a pattern describes whole strings.
#[test]
fn matching_is_anchored_and_dot_matches_a_line_feed() {
    let abc = term("abc");
    assert!(abc.matches("abc").unwrap());
    assert!(!abc.matches("xabc").unwrap());
    assert!(!abc.matches("abcx").unwrap());

    assert!(term(".").matches("\n").unwrap());
}

/// Constructs a regular language cannot express are refused rather than applied
/// incorrectly; redundant anchors are the documented exception.
#[test]
fn unsupported_constructs_are_refused() {
    for pattern in ["(a)\\1", "a(?=b)", "a(?<=b)", "(?i)abc", "a\\bc", "a^b"] {
        assert!(
            Term::from_pattern(pattern).is_err(),
            "`{pattern}` was accepted"
        );
    }

    // A leading `^` and a trailing `$` are no-ops, since matching is already
    // full-string.
    assert!(term("^abc$").equivalent(&term("abc")).unwrap());
}

/// A hand-built automaton is a first-class input: it goes through the same
/// operations as a parsed one.
#[test]
fn a_hand_built_automaton_is_a_term() {
    use regexsolver::regex_charclass::char::Char;

    let mut automaton = FastAutomaton::new_empty();
    let state = automaton.new_state();
    automaton.accept(state);
    automaton
        .add_transition_from_range(
            0,
            state,
            &CharRange::new_from_range(Char::new('a')..=Char::new('c')),
        )
        .unwrap();

    let built = Term::from_automaton(automaton);
    assert!(built.equivalent(&term("[a-c]")).unwrap());
    assert_eq!("[a-c]", built.to_pattern().unwrap());
}

/// `EngineError` is comparable and cloneable across the crate boundary, and
/// `#[non_exhaustive]`, so a dependent has to match it with a wildcard arm.
#[test]
fn errors_are_comparable_cloneable_and_non_exhaustive() {
    // A hand-built repetition whose maximum is below its minimum denotes no
    // valid language, and is refused when it is converted.
    let malformed =
        RegularExpression::Repetition(Box::new(RegularExpression::new("ab").unwrap()), 3, Some(1));
    let error = Term::from_regex(malformed).to_automaton().unwrap_err();
    assert_eq!(EngineError::InvalidRepetitionBounds(3, 1), error);

    assert_eq!(error, error.clone());
    assert!(!error.to_string().is_empty());

    let described = match &error {
        EngineError::InvalidRepetitionBounds(..) => "bounds",
        _ => "something else",
    };
    assert_eq!("bounds", described);
}
