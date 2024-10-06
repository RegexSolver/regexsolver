use ::regex::Regex;
use lazy_static::lazy_static;
use regex_charclass::irange::range::AnyRange;
use regex_syntax::ParserBuilder;

use super::*;

lazy_static! {
    static ref RE_FLAG_DETECTION: Regex =
        Regex::new(r"\(\?[imsx]*-?[imsx]*\)").expect("Can not compile flag detection regex.");
}

impl RegularExpression {
    pub fn new(regex: &str) -> Result<Self, EngineError> {
        if regex.is_empty() {
            return Ok(RegularExpression::new_empty_string());
        }
        if regex == "[]" {
            return Ok(RegularExpression::new_empty());
        }
        match ParserBuilder::new()
            .dot_matches_new_line(true)
            .build()
            .parse(&Self::remove_flags(regex))
        {
            Ok(hir) => Self::convert_to_regex(&hir),
            Err(err) => Err(EngineError::RegexSyntaxError(err.to_string())),
        }
    }

    fn remove_flags(regex: &str) -> String {
        RE_FLAG_DETECTION.replace_all(regex, "").to_string()
    }

    pub fn new_total() -> Self {
        RegularExpression::Repetition(
            Box::new(RegularExpression::Character(Range::total())),
            0,
            None,
        )
    }

    pub fn new_empty() -> Self {
        RegularExpression::Character(Range::empty())
    }

    pub fn new_empty_string() -> Self {
        RegularExpression::Concat(VecDeque::new())
    }

    fn convert_to_regex(hir: &Hir) -> Result<Self, EngineError> {
        match hir.kind() {
            HirKind::Empty => Ok(RegularExpression::new_empty_string()),
            HirKind::Literal(literal) => {
                let mut regex_concat = RegularExpression::new_empty_string();
                if let Ok(string) = String::from_utf8(literal.0.clone().into_vec()) {
                    for char in string.chars() {
                        regex_concat = regex_concat.concat(
                            &RegularExpression::Character(Range::new_from_range(
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
                Self::convert_to_regex(&repetition.sub).map(|v| v.repeat(min, max))
            }
            HirKind::Capture(capture) => Self::convert_to_regex(&capture.sub),
            HirKind::Concat(concat) => {
                let mut concat_regex =
                    RegularExpression::Concat(VecDeque::with_capacity(concat.len()));
                for c in concat {
                    let concat_value = Self::convert_to_regex(c)?;
                    concat_regex = concat_regex.concat(&concat_value, true);
                }
                Ok(concat_regex)
            }
            HirKind::Alternation(alternation) => {
                let mut alternation_regex =
                    RegularExpression::Alternation(Vec::with_capacity(alternation.len()));
                for a in alternation {
                    let alternation_value = Self::convert_to_regex(a)?;
                    alternation_regex = alternation_regex.union(&alternation_value);
                }
                Ok(alternation_regex)
            }
        }
    }

    fn to_range_unicode(class_unicode: &ClassUnicode) -> Range {
        let mut new_range = Vec::with_capacity(class_unicode.ranges().len());
        for range in class_unicode.ranges() {
            new_range.push(AnyRange::from(
                Char::new(range.start())..=Char::new(range.end()),
            ));
        }
        Range::new_from_ranges(&new_range)
    }

    fn to_range_bytes(class_bytes: &ClassBytes) -> Range {
        let mut new_range = Vec::with_capacity(class_bytes.ranges().len());
        for range in class_bytes.ranges() {
            new_range.push(AnyRange::from(
                Char::new(range.start() as char)..=Char::new(range.end() as char),
            ));
        }
        Range::new_from_ranges(&new_range)
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

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

        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("\t"));
        assert!(automaton.match_string("\n"));
        assert!(automaton.match_string("\r"));

        let regex_parsed = RegularExpression::new("(?i)a").unwrap();
        let automaton = regex_parsed.to_automaton().unwrap();

        assert!(automaton.match_string("a"));
        assert!(!automaton.match_string("A"));

        let regex_parsed = RegularExpression::new("a(?i)a(?-s).").unwrap();
        let automaton = regex_parsed.to_automaton().unwrap();

        assert!(automaton.match_string("aa\n"));
        assert!(!automaton.match_string("aAb"));

        assert!(RegularExpression::new("\\1").is_err());
        Ok(())
    }

    /*#[test]
    fn test_parse_1() -> Result<(), String> {
        let regex_parsed = RegularExpression::new("abc(?=def)").unwrap();

        println!("{:?}", regex_parsed);

        Ok(())
    }*/
}
