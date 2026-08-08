use regex_charclass::irange::range::AnyRange;
use regex_syntax::ParserBuilder;

use super::*;

impl RegularExpression {
    /// Parses and simplifies the provided pattern and returns the resulting [`RegularExpression`].
    pub fn new(pattern: &str) -> Result<Self, EngineError> {
        Self::parse(pattern, true)
    }

    /// Parses the provided pattern and returns the resulting [`RegularExpression`]. If `simplify` is `true`, the expression is simplified during parsing.
    #[tracing::instrument(level = "debug", skip(pattern), fields(pattern_len = pattern.len()))]
    pub fn parse(pattern: &str, simplify: bool) -> Result<Self, EngineError> {
        if pattern.is_empty() {
            return Ok(RegularExpression::new_empty_string());
        }
        if pattern == "[]" {
            return Ok(RegularExpression::new_empty());
        }
        // Inline flags such as `(?i)` change matching semantics the engine
        // cannot represent (it operates uniformly over character ranges).
        // Silently ignoring them would diverge from every mainstream engine
        // (e.g. `(?i)abc` would not match `ABC`), so they are rejected instead.
        Self::reject_inline_flags(pattern)?;
        match ParserBuilder::new()
            .dot_matches_new_line(true)
            .build()
            .parse(pattern)
        {
            // The whole pattern is matched against the full input (anchored),
            // so a leading start-of-text anchor and a trailing end-of-text
            // anchor are accepted as redundant no-ops; anchors elsewhere and
            // word boundaries are rejected (see `convert_look`).
            Ok(hir) => Self::convert_to_regex(&hir, simplify, true, true),
            Err(err) => Err(EngineError::RegexSyntaxError(err.to_string())),
        }
    }

    /// Rejects patterns that set inline flags, either as a `(?flags)` directive
    /// or a `(?flags:...)` group. Parsing to the AST (rather than scanning the
    /// string) means flag-lookalikes inside character classes (e.g. `[a(?i)]`)
    /// are correctly *not* treated as flags.
    fn reject_inline_flags(pattern: &str) -> Result<(), EngineError> {
        use regex_syntax::ast::parse::Parser;

        let ast = Parser::new()
            .parse(pattern)
            .map_err(|err| EngineError::RegexSyntaxError(err.to_string()))?;
        if Self::ast_sets_flags(&ast) {
            return Err(EngineError::UnsupportedRegexFeature(
                "inline flags such as (?i), (?m), (?s) and (?x) are not supported".to_string(),
            ));
        }
        Ok(())
    }

    /// Returns `true` if the AST sets any inline flag. Non-capturing groups
    /// without flags (`(?:...)`) are allowed.
    fn ast_sets_flags(ast: &regex_syntax::ast::Ast) -> bool {
        use regex_syntax::ast::{Ast, GroupKind};

        match ast {
            Ast::Flags(_) => true,
            Ast::Group(group) => {
                matches!(&group.kind, GroupKind::NonCapturing(flags) if !flags.items.is_empty())
                    || Self::ast_sets_flags(&group.ast)
            }
            Ast::Repetition(repetition) => Self::ast_sets_flags(&repetition.ast),
            Ast::Alternation(alternation) => alternation.asts.iter().any(Self::ast_sets_flags),
            Ast::Concat(concat) => concat.asts.iter().any(Self::ast_sets_flags),
            _ => false,
        }
    }

    /// Creates a regular expression that matches all possible strings.
    pub fn new_total() -> Self {
        RegularExpression::Repetition(
            Box::new(RegularExpression::Character(CharRange::total())),
            0,
            None,
        )
    }

    /// Creates a regular expression that matches the empty language.
    pub fn new_empty() -> Self {
        RegularExpression::Character(CharRange::empty())
    }

    /// Creates a regular expression that matches only the empty string `""`.
    pub fn new_empty_string() -> Self {
        RegularExpression::Concat(VecDeque::new())
    }

    /// Converts a parsed HIR node into a [`RegularExpression`].
    ///
    /// `at_start`/`at_end` track whether this node sits at the very start/end
    /// of the overall match (nothing can be consumed before/after it). They
    /// govern which anchors are accepted as redundant no-ops; see
    /// [`convert_look`](Self::convert_look).
    fn convert_to_regex(
        hir: &Hir,
        simplify: bool,
        at_start: bool,
        at_end: bool,
    ) -> Result<Self, EngineError> {
        match hir.kind() {
            HirKind::Empty => Ok(RegularExpression::new_empty_string()),
            HirKind::Literal(literal) => {
                if let Ok(string) = std::str::from_utf8(&literal.0) {
                    let mut elements = VecDeque::new();
                    for char in string.chars() {
                        RegularExpression::push_concat_element(
                            &mut elements,
                            RegularExpression::Character(CharRange::new_from_range(
                                Char::new(char)..=Char::new(char),
                            )),
                        );
                    }
                    Ok(match elements.len() {
                        1 => elements.pop_front().expect("len() == 1"),
                        _ => RegularExpression::Concat(elements),
                    })
                } else {
                    Err(EngineError::InvalidCharacterInRegex)
                }
            }
            HirKind::Class(class) => match class {
                Class::Unicode(class_unicode) => {
                    let range = Self::to_range_unicode(class_unicode);
                    Ok(RegularExpression::Character(range))
                }
                Class::Bytes(class_bytes) => {
                    let range = Self::to_range_bytes(class_bytes);
                    Ok(RegularExpression::Character(range))
                }
            },
            HirKind::Look(look) => Self::convert_look(look, at_start, at_end),
            HirKind::Repetition(repetition) => {
                let (min, max) = (repetition.min, repetition.max);
                // The body can repeat, so it is not at the overall boundary in
                // general; an anchor inside it is rejected.
                let regex = Self::convert_to_regex(&repetition.sub, simplify, false, false)?;
                Ok(if simplify {
                    regex.repeat(min, max)
                } else {
                    RegularExpression::Repetition(Box::new(regex), min, max)
                })
            }
            // A capture group does not consume input, so it inherits the
            // surrounding boundary context unchanged.
            HirKind::Capture(capture) => {
                Self::convert_to_regex(&capture.sub, simplify, at_start, at_end)
            }
            HirKind::Concat(concat) => {
                let len = concat.len();
                let mut concat_regex = RegularExpression::Concat(VecDeque::with_capacity(len));
                for (i, c) in concat.iter().enumerate() {
                    // Only the first element can be at the overall start, and
                    // only the last at the overall end.
                    let child_at_start = at_start && i == 0;
                    let child_at_end = at_end && i + 1 == len;
                    let concat_value =
                        Self::convert_to_regex(c, simplify, child_at_start, child_at_end)?;
                    if simplify {
                        concat_regex = concat_regex.concat(&concat_value, true);
                    } else if let RegularExpression::Concat(mut values) = concat_regex {
                        values.push_back(concat_value);
                        concat_regex = RegularExpression::Concat(values);
                    }
                }
                Ok(concat_regex)
            }
            HirKind::Alternation(alternation) => {
                let mut branches = Vec::with_capacity(alternation.len());
                for a in alternation {
                    // Each branch occupies the alternation's own position, so
                    // it inherits the boundary context unchanged.
                    branches.push(Self::convert_to_regex(a, simplify, at_start, at_end)?);
                }
                if simplify {
                    // Folds the branches through the incremental accumulator
                    // (folding via pairwise `union` re-cloned the accumulated
                    // alternation per branch, quadratic in the branch count)
                    // and honors the execution deadline per branch, so a
                    // pathological alternation cannot burn CPU outside the
                    // profile.
                    RegularExpression::union_all_bounded(branches.iter())
                } else {
                    Ok(RegularExpression::Alternation(branches))
                }
            }
        }
    }

    /// Interprets a look-around assertion under the engine's full-string
    /// (anchored) matching model.
    ///
    /// A start-of-text anchor (`^`, `\A`) at the very start of the match and an
    /// end-of-text anchor (`$`, `\z`) at the very end are redundant, so they
    /// are accepted as the empty string. Anchors anywhere else would constrain
    /// matching in a way the engine cannot represent (e.g. `ab$cd`), and word
    /// boundaries (`\b`, `\B`) never can, so both are rejected rather than
    /// silently changing the language.
    fn convert_look(look: &Look, at_start: bool, at_end: bool) -> Result<Self, EngineError> {
        match look {
            Look::Start | Look::StartLF | Look::StartCRLF => {
                if at_start {
                    Ok(RegularExpression::new_empty_string())
                } else {
                    Err(EngineError::UnsupportedRegexFeature(
                        "a start-of-text anchor (^ or \\A) is only supported at the start of the \
                         pattern; matching is implicitly anchored to the full string"
                            .to_string(),
                    ))
                }
            }
            Look::End | Look::EndLF | Look::EndCRLF => {
                if at_end {
                    Ok(RegularExpression::new_empty_string())
                } else {
                    Err(EngineError::UnsupportedRegexFeature(
                        "an end-of-text anchor ($ or \\z) is only supported at the end of the \
                         pattern; matching is implicitly anchored to the full string"
                            .to_string(),
                    ))
                }
            }
            _ => Err(EngineError::UnsupportedRegexFeature(
                "word boundaries (\\b, \\B) and look-around assertions are not supported"
                    .to_string(),
            )),
        }
    }

    fn to_range_unicode(class_unicode: &ClassUnicode) -> CharRange {
        let mut new_range = Vec::with_capacity(class_unicode.ranges().len());
        for range in class_unicode.ranges() {
            new_range.push(AnyRange::from(
                Char::new(range.start())..=Char::new(range.end()),
            ));
        }
        CharRange::new_from_ranges(&new_range)
    }

    fn to_range_bytes(class_bytes: &ClassBytes) -> CharRange {
        let mut new_range = Vec::with_capacity(class_bytes.ranges().len());
        for range in class_bytes.ranges() {
            new_range.push(AnyRange::from(
                Char::new(range.start() as char)..=Char::new(range.end() as char),
            ));
        }
        CharRange::new_from_ranges(&new_range)
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    // Folding an alternation with many shared-affix branches is quadratic in
    // the branch count; the parse must honor the execution deadline instead
    // of burning CPU outside the profile.
    #[test]
    fn parse_honors_the_execution_deadline() {
        use crate::error::EngineError;
        use crate::execution_profile::ExecutionProfileBuilder;

        let branches: Vec<String> = (0..20000).map(|i| format!("x{i:05}y")).collect();
        let pattern = format!("({})", branches.join("|"));

        ExecutionProfileBuilder::new()
            .execution_timeout(10)
            .build()
            .run(|| {
                assert_eq!(
                    EngineError::OperationTimeOutError,
                    RegularExpression::new(&pattern).unwrap_err()
                );
            });
    }

    // Inline flags are rejected (the engine cannot honor them); non-capturing
    // groups `(?:...)` are still accepted, and flag-lookalikes inside a
    // character class are not mistaken for flags.
    #[test]
    fn inline_flags_are_rejected() {
        use crate::error::EngineError;

        for pattern in ["(?i)a", "a(?m-s)b", "a(?-s)b", "(?i:abc)", "(?x) a b c"] {
            assert!(
                matches!(
                    RegularExpression::new(pattern),
                    Err(EngineError::UnsupportedRegexFeature(_))
                ),
                "pattern {pattern:?} with inline flags should be rejected"
            );
        }

        // Non-capturing groups without flags are fine.
        assert!(RegularExpression::new("(?:ab|c)d").is_ok());
        // `(?i)` inside a character class is a set of literal members, not a
        // flag directive, so it must not be rejected.
        assert!(RegularExpression::new("[a(?i)]").is_ok());
    }

    // Anchors are accepted only where they are redundant under full-string
    // matching (leading `^`, trailing `$`); elsewhere they and word
    // boundaries are rejected rather than silently changing the language.
    #[test]
    fn anchors_and_boundaries() {
        use crate::error::EngineError;

        // Redundant anchors are no-ops: `^abc$` == `abc`.
        let anchored = RegularExpression::new("^abc$").unwrap();
        let plain = RegularExpression::new("abc").unwrap();
        assert!(
            anchored
                .to_automaton()
                .unwrap()
                .equivalent(&plain.to_automaton().unwrap())
                .unwrap()
        );
        assert!(RegularExpression::new("^abc").is_ok());
        assert!(RegularExpression::new("abc$").is_ok());
        // Alternation branches carry the boundary context.
        assert!(RegularExpression::new("^a|b$").is_ok());

        // Mid-pattern anchors and word boundaries are rejected.
        for pattern in ["ab$cd", "a^b", r"a\bc", r"a\Bc", r"a(^b)c"] {
            assert!(
                matches!(
                    RegularExpression::new(pattern),
                    Err(EngineError::UnsupportedRegexFeature(_))
                ),
                "pattern {pattern:?} should be rejected"
            );
        }
    }

    // A hand-built tree nested past `MAX_NESTING_DEPTH` is rejected at the
    // conversion boundary instead of overflowing the stack.
    #[test]
    fn to_automaton_rejects_too_deeply_nested() {
        use crate::error::EngineError;
        use regex_charclass::char::Char;

        let mut regex = RegularExpression::Character(crate::CharRange::new_from_range(
            Char::new('a')..=Char::new('a'),
        ));
        for _ in 0..(RegularExpression::MAX_NESTING_DEPTH + 10) {
            regex = RegularExpression::Repetition(Box::new(regex), 1, Some(1));
        }
        assert!(matches!(
            regex.to_automaton(),
            Err(EngineError::RegexTooDeeplyNested(_))
        ));

        // A shallow tree still converts fine.
        assert!(
            RegularExpression::new("(a(b(c)))")
                .unwrap()
                .to_automaton()
                .is_ok()
        );
    }

    // The variants are freely constructible (open enum); invalid bounds are
    // rejected at the conversion boundary instead.
    #[test]
    fn to_automaton_rejects_invalid_repetition_bounds() {
        use crate::error::EngineError;

        let a = RegularExpression::new("a").unwrap();
        let invalid = RegularExpression::Repetition(Box::new(a.clone()), 5, Some(2));
        assert_eq!(
            invalid.to_automaton().unwrap_err(),
            EngineError::InvalidRepetitionBounds(5, 2)
        );

        // Nested invalid repetitions are caught by the recursion.
        let nested = RegularExpression::Concat([a.clone(), invalid].into());
        assert_eq!(
            nested.to_automaton().unwrap_err(),
            EngineError::InvalidRepetitionBounds(5, 2)
        );

        // The simplifying combinators must not panic on invalid trees either
        // (e.g. the affix factoring of `r{1,0}` must not underflow).
        let degenerate = RegularExpression::Repetition(Box::new(a.clone()), 1, Some(0));
        let _ = a.union(&degenerate);
        let _ = a.concat(&degenerate, true);
    }

    // Singleton Alternation/Concat wrappers print transparently, so a
    // quantifier applied to one must be parenthesized by looking through the
    // wrapper: `((.a))*` must print as `(.a)*`, not `.a*` (a different
    // language).
    #[test]
    fn display_parenthesizes_through_singleton_wrappers() {
        use regex_charclass::char::Char;

        let dot = RegularExpression::Character(crate::CharRange::total());
        let a = RegularExpression::Character(crate::CharRange::new_from_range(
            Char::new('a')..=Char::new('a'),
        ));
        let wrapped =
            RegularExpression::Alternation(vec![RegularExpression::Concat([dot, a].into())]);
        let star = RegularExpression::Repetition(Box::new(wrapped), 0, None);
        assert_eq!(star.to_string(), "(.a)*");

        // The printed pattern must denote the same language as the tree.
        let reparsed = RegularExpression::parse(&star.to_string(), false).unwrap();
        assert!(
            star.to_automaton()
                .unwrap()
                .equivalent(&reparsed.to_automaton().unwrap())
                .unwrap()
        );
    }

    #[test]
    fn test_parse() -> Result<(), String> {
        assert_parse("abc+");
        assert_parse("(abc){3,129}");
        assert_parse("a?");
        assert_parse("\\d");
        assert_parse("\\D");
        assert_parse("\\s");
        assert_parse("\\S");
        assert_parse("\\w");
        assert_parse("\\W");
        assert_parse("\\n");
        assert_parse("\\r");
        assert_parse("\\t");
        assert_parse("\\v");

        assert_parse("\\p{Common}");
        assert_parse("\\p{Arabic}");
        assert_parse("\\p{Armenian}");
        assert_parse("\\p{Bengali}");
        assert_parse("\\p{Bopomofo}");
        assert_parse("\\p{Braille}");
        assert_parse("\\p{Buhid}");
        assert_parse("\\p{Canadian_Aboriginal}");
        assert_parse("\\p{Cherokee}");
        assert_parse("\\p{Cyrillic}");
        assert_parse("\\p{Devanagari}");
        assert_parse("\\p{Ethiopic}");
        assert_parse("\\p{Georgian}");
        assert_parse("\\p{Greek}");
        assert_parse("\\p{Gujarati}");
        assert_parse("\\p{Gurmukhi}");
        assert_parse("\\p{Han}");
        assert_parse("\\p{Hangul}");
        assert_parse("\\p{Hanunoo}");
        assert_parse("\\p{Hebrew}");
        assert_parse("\\p{Hiragana}");
        assert_parse("\\p{Inherited}");
        assert_parse("\\p{Kannada}");
        assert_parse("\\p{Katakana}");
        assert_parse("\\p{Khmer}");
        assert_parse("\\p{Lao}");
        assert_parse("\\p{Latin}");
        assert_parse("\\p{Limbu}");
        assert_parse("\\p{Malayalam}");
        assert_parse("\\p{Mongolian}");
        assert_parse("\\p{Myanmar}");
        assert_parse("\\p{Ogham}");
        assert_parse("\\p{Oriya}");
        assert_parse("\\p{Runic}");
        assert_parse("\\p{Sinhala}");
        assert_parse("\\p{Syriac}");
        assert_parse("\\p{Tagalog}");
        assert_parse("\\p{Tagbanwa}");
        assert_parse("\\p{Tai_Le}");
        assert_parse("\\p{Tamil}");
        assert_parse("\\p{Telugu}");
        assert_parse("\\p{Thaana}");
        assert_parse("\\p{Thai}");
        assert_parse("\\p{Tibetan}");
        assert_parse("\\p{Yi}");

        assert_parse("\\p{Letter}");
        assert_parse("\\p{Lowercase_Letter}");
        assert_parse("\\p{Uppercase_Letter}");
        assert_parse("\\p{Titlecase_Letter}");
        assert_parse("\\p{Modifier_Letter}");
        assert_parse("\\p{Other_Letter}");

        assert_parse("\\p{Mark}");
        assert_parse("\\p{Nonspacing_Mark}");
        assert_parse("\\p{Spacing_Mark}");
        assert_parse("\\p{Enclosing_Mark}");

        assert_parse("\\p{Separator}");
        assert_parse("\\p{Space_Separator}");
        assert_parse("\\p{Line_Separator}");
        assert_parse("\\p{Paragraph_Separator}");

        assert_parse("\\p{Symbol}");
        assert_parse("\\p{Math_Symbol}");
        assert_parse("\\p{Currency_Symbol}");
        assert_parse("\\p{Modifier_Symbol}");
        assert_parse("\\p{Other_Symbol}");

        assert_parse("\\p{Number}");
        assert_parse("\\p{Letter_Number}");
        assert_parse("\\p{Other_Number}");

        assert_parse("\\p{Punctuation}");
        assert_parse("\\p{Dash_Punctuation}");
        assert_parse("\\p{Open_Punctuation}");
        assert_parse("\\p{Close_Punctuation}");
        assert_parse("\\p{Initial_Punctuation}");
        assert_parse("\\p{Final_Punctuation}");
        assert_parse("\\p{Connector_Punctuation}");
        assert_parse("\\p{Other_Punctuation}");

        assert_parse("\\p{Other}");
        assert_parse("\\p{Control}");
        assert_parse("\\p{Format}");
        assert_parse("\\p{Private_Use}");
        assert_parse("\\p{Unassigned}");

        Ok(())
    }

    fn assert_parse(regex: &str) {
        let regex_parsed = RegularExpression::new(regex).unwrap();
        assert_eq!(regex, regex_parsed.to_string());
    }

    #[test]
    fn test_match() -> Result<(), String> {
        let regex_parsed = RegularExpression::new(".").unwrap();
        let automaton = regex_parsed.to_automaton().unwrap();

        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("\t"));
        assert!(automaton.is_match("\n"));
        assert!(automaton.is_match("\r"));

        // Inline flags are rejected rather than silently stripped.
        assert!(RegularExpression::new("(?i)a").is_err());
        assert!(RegularExpression::new("a(?i)a(?-s).").is_err());

        assert!(RegularExpression::new("\\1").is_err());

        let two_chars = RegularExpression::new("..")
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(two_chars.is_match("aé"));
        assert!(two_chars.is_match("éa"));
        assert!(two_chars.is_match("éé"));
        assert!(!two_chars.is_match("é"));
        assert!(!two_chars.is_match("aéa"));

        Ok(())
    }

    /*#[test]
    fn test_parse_1() -> Result<(), String> {
        let regex_parsed = RegularExpression::new("abc(?=def)").unwrap();

        println!("{:?}", regex_parsed);

        Ok(())
    }*/
}
