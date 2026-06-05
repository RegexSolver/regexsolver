use regex_charclass::irange::range::AnyRange;
use regex_syntax::ParserBuilder;

use super::*;

impl RegularExpression {
    /// Parses and simplifies the provided pattern and returns the resulting [`RegularExpression`].
    pub fn new(pattern: &str) -> Result<Self, EngineError> {
        Self::parse(pattern, true)
    }

    /// Parses the provided pattern and returns the resulting [`RegularExpression`]. If `simplify` is `true`, the expression is simplified during parsing.
    pub fn parse(pattern: &str, simplify: bool) -> Result<Self, EngineError> {
        if pattern.is_empty() {
            return Ok(RegularExpression::new_empty_string());
        }
        if pattern == "[]" {
            return Ok(RegularExpression::new_empty());
        }
        match ParserBuilder::new()
            .dot_matches_new_line(true)
            .build()
            .parse(&Self::remove_flags(pattern))
        {
            Ok(hir) => Self::convert_to_regex(&hir, simplify),
            Err(err) => Err(EngineError::RegexSyntaxError(err.to_string())),
        }
    }

    /// Strips inline flag groups like `(?i)`, `(?m-s)` or `(?-s)` from the
    /// pattern: the engine treats all characters uniformly, so the flags are
    /// meaningless here. Equivalent to deleting every match of
    /// `\(\?[imsx]*-?[imsx]*\)`; anything else — including non-capturing
    /// groups `(?:...)` — is left untouched.
    fn remove_flags(regex: &str) -> String {
        let bytes = regex.as_bytes();
        let mut result = String::with_capacity(regex.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'(' && i + 1 < bytes.len() && bytes[i + 1] == b'?' {
                let mut j = i + 2;
                while j < bytes.len() && matches!(bytes[j], b'i' | b'm' | b's' | b'x') {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'-' {
                    j += 1;
                    while j < bytes.len() && matches!(bytes[j], b'i' | b'm' | b's' | b'x') {
                        j += 1;
                    }
                }
                if j < bytes.len() && bytes[j] == b')' {
                    // a flag group: skip it entirely
                    i = j + 1;
                    continue;
                }
            }
            // not a flag group: copy the whole character (UTF-8 safe)
            let char_len = regex[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
            result.push_str(&regex[i..i + char_len]);
            i += char_len;
        }
        result
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

    fn convert_to_regex(hir: &Hir, simplify: bool) -> Result<Self, EngineError> {
        match hir.kind() {
            HirKind::Empty => Ok(RegularExpression::new_empty_string()),
            HirKind::Literal(literal) => {
                let mut regex_concat = RegularExpression::new_empty_string();
                if let Ok(string) = String::from_utf8(literal.0.clone().into_vec()) {
                    for char in string.chars() {
                        regex_concat = regex_concat.concat(
                            &RegularExpression::Character(CharRange::new_from_range(
                                Char::new(char)..=Char::new(char),
                            )),
                            true,
                        );
                    }
                    Ok(regex_concat)
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
            HirKind::Look(_) => Ok(RegularExpression::new_empty_string()),
            HirKind::Repetition(repetition) => {
                let (min, max) = (repetition.min, repetition.max);
                let regex = Self::convert_to_regex(&repetition.sub, simplify)?;
                Ok(if simplify {
                    regex.repeat(min, max)
                } else {
                    RegularExpression::Repetition(Box::new(regex), min, max)
                })
            }
            HirKind::Capture(capture) => Self::convert_to_regex(&capture.sub, simplify),
            HirKind::Concat(concat) => {
                let mut concat_regex =
                    RegularExpression::Concat(VecDeque::with_capacity(concat.len()));
                for c in concat {
                    let concat_value = Self::convert_to_regex(c, simplify)?;
                    if simplify {
                        concat_regex = concat_regex.concat(&concat_value, true);
                    } else if let RegularExpression::Concat(values) = concat_regex {
                        let mut values = values.clone();
                        values.push_back(concat_value);
                        concat_regex = RegularExpression::Concat(values);
                    }
                }
                Ok(concat_regex)
            }
            HirKind::Alternation(alternation) => {
                let mut alternation_regex =
                    RegularExpression::Alternation(Vec::with_capacity(alternation.len()));
                for a in alternation {
                    let alternation_value = Self::convert_to_regex(a, simplify)?;
                    if simplify {
                        alternation_regex = alternation_regex.union(&alternation_value);
                    } else if let RegularExpression::Alternation(values) = alternation_regex {
                        let mut values = values.clone();
                        values.push(alternation_value);
                        alternation_regex = RegularExpression::Alternation(values);
                    }
                }
                Ok(alternation_regex)
            }
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

    // The hand-rolled flag stripper must delete exactly the matches of
    // `\(\?[imsx]*-?[imsx]*\)` (the regex it replaced) and nothing else.
    #[test]
    fn remove_flags_strips_flag_groups_only() {
        let strip = RegularExpression::remove_flags;

        assert_eq!(strip("(?i)a"), "a");
        assert_eq!(strip("a(?m-s)b"), "ab");
        assert_eq!(strip("a(?-s)b"), "ab");
        assert_eq!(strip("(?imsx)(?)a(?i-)"), "a");

        // Non-flag constructs are untouched.
        assert_eq!(strip("(?:ab|c)d"), "(?:ab|c)d");
        assert_eq!(strip("(a?)b"), "(a?)b");
        assert_eq!(strip("a(?i-s"), "a(?i-s"); // unterminated: not a flag group
        assert_eq!(strip("héllo(?i)é"), "hélloé"); // multi-byte safe
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
        // (regression: the affix factoring of `r{1,0}` used to underflow).
        let degenerate = RegularExpression::Repetition(Box::new(a.clone()), 1, Some(0));
        let _ = a.union(&degenerate);
        let _ = a.concat(&degenerate, true);
    }

    // Regression (found by the proptest generators): singleton
    // Alternation/Concat wrappers print transparently, so quantified
    // expressions must be parenthesized by looking through them —
    // `((.a))*` used to print as `.a*` instead of `(.a)*`, changing the
    // language.
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

        let regex_parsed = RegularExpression::new("(?i)a").unwrap();
        let automaton = regex_parsed.to_automaton().unwrap();

        assert!(automaton.is_match("a"));
        assert!(!automaton.is_match("A"));

        let regex_parsed = RegularExpression::new("a(?i)a(?-s).").unwrap();
        let automaton = regex_parsed.to_automaton().unwrap();

        assert!(automaton.is_match("aa\n"));
        assert!(!automaton.is_match("aAb"));

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
