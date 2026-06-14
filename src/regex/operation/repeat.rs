use super::*;

impl RegularExpression {
    /// Computes the repetition of the expression between `min` and `max_opt` times; if `max_opt` is `None`, the repetition is unbounded.
    ///
    /// When `max_opt` is below `min` there is no valid repetition count and
    /// the result is the empty language, consistently with
    /// [`FastAutomaton::repeat`](crate::fast_automaton::FastAutomaton::repeat).
    #[tracing::instrument(level = "trace", skip(self), fields(min = min, max_opt = tracing::field::debug(max_opt)))]
    pub fn repeat(&self, min: u32, max_opt: Option<u32>) -> RegularExpression {
        if self.is_total() {
            return RegularExpression::new_total();
        } else if self.is_empty() {
            return RegularExpression::new_empty();
        } else if self.is_empty_string() {
            return Self::new_empty_string();
        } else if let Some(max) = max_opt {
            if max < min {
                // No valid repetition count: the language is empty. This
                // matches `FastAutomaton::repeat`, which disagreed with the
                // {""} previously returned here.
                return RegularExpression::new_empty();
            } else if max == 0 {
                return RegularExpression::new_empty_string();
            } else if min == 1 && max == 1 {
                return self.clone();
            }
        }

        match self {
            RegularExpression::Repetition(regular_expression, i_min, i_max_opt) => {
                // Only collapse (r{i_min,i_max}){min,max} into
                // r{min·i_min,max·i_max} when the bounds are gap-free AND the
                // multiplications don't overflow; the nested form is always a
                // correct fallback.
                if Self::can_simplify_nested_repetition(*i_min, *i_max_opt, min, max_opt) {
                    let new_min = min.checked_mul(*i_min);
                    let new_max = match (max_opt, i_max_opt) {
                        (Some(o_max), Some(i_max)) => o_max.checked_mul(*i_max).map(Some),
                        _ => Some(None),
                    };
                    if let (Some(new_min), Some(new_max)) = (new_min, new_max) {
                        return RegularExpression::Repetition(
                            regular_expression.clone(),
                            new_min,
                            new_max,
                        );
                    }
                }
                RegularExpression::Repetition(Box::new(self.clone()), min, max_opt)
            }
            _ => RegularExpression::Repetition(Box::new(self.clone()), min, max_opt),
        }
    }

    /// Evaluate if the repetition `(r{i_min,i_max_opt}){o_min,o_max_opt}` can be simplified to `r{i_min*o_min,i_max_opt*o_max_opt}`.
    fn can_simplify_nested_repetition(
        i_min: u32,
        i_max_opt: Option<u32>,
        o_min: u32,
        o_max_opt: Option<u32>,
    ) -> bool {
        if let Some(o_max) = o_max_opt
            && o_min == o_max
        {
            return true;
        }

        if let Some(i_max) = i_max_opt {
            // We check if there is any gap by resolving:
            // o_min * i_max >= (o_min + 1) * i_min - 1
            // <=> o_min * (i_max - i_min) >= i_min - 1
            o_min.saturating_mul(i_max.saturating_sub(i_min)) >= i_min.saturating_sub(1)
        } else if o_min > 0 {
            true
        } else {
            i_min <= 1
        }
    }
}

#[cfg(test)]
mod tests {

    use regex_charclass::char::Char;

    use crate::{CharRange, regex::RegularExpression};

    // Regression: the nested-repetition simplification used to multiply
    // bounds unchecked; huge (but valid) bounds must fall back to the nested
    // form instead of overflowing.
    #[test]
    fn repeat_bound_overflow_keeps_nested_form() {
        let a = RegularExpression::new("a").unwrap();
        let inner = a.repeat(2, Some(2)); // a{2}
        let outer = inner.repeat(u32::MAX, Some(u32::MAX)); // 2·u32::MAX overflows
        assert!(matches!(
            &outer,
            RegularExpression::Repetition(r, u32::MAX, Some(u32::MAX))
                if matches!(&**r, RegularExpression::Repetition(..))
        ));
    }

    // r{min,max} with max < min has no valid repetition count: the language
    // is empty, consistently with `FastAutomaton::repeat` (the regex side
    // used to return {""} instead).
    #[test]
    fn repeat_with_max_below_min_is_empty() {
        let a = RegularExpression::new("a").unwrap();
        assert!(a.repeat(5, Some(2)).is_empty());

        let automaton = a.to_automaton().unwrap().repeat(5, Some(2)).unwrap();
        assert!(automaton.is_empty());
    }

    #[test]
    fn test_parse_and_simplify() -> Result<(), String> {
        assert_parse_and_simplify("(xxx)*", "(x{3})*");
        assert_parse_and_simplify("(x*){3}", "x*");
        assert_parse_and_simplify("(x+)?", "x*");
        assert_parse_and_simplify("(x?)+", "x*");
        assert_parse_and_simplify("(x{0,3})+", "x*");
        assert_parse_and_simplify("(x{2,3})+", "x{2,}");
        assert_parse_and_simplify("(x{7,9})+", "(x{7,9})+");
        assert_parse_and_simplify("(x+)*", "x*");
        assert_parse_and_simplify(".*abc", ".*abc");
        assert_parse_and_simplify(".*a(b|cd)", ".*a(b|cd)");
        assert_parse_and_simplify(
            "a(bcfe|bcdg|mkv)*(abc){2,3}(abc){2}",
            "a(bc(dg|fe)|mkv)*(abc){4,5}",
        );
        assert_parse_and_simplify("((abc|fg)abc|(abc|fg)fg)", "(abc|fg){2}");
        assert_parse_and_simplify("(a{2}|a{3})", "a{2,3}");
        assert_parse_and_simplify("(a|b)", "[ab]");
        assert_parse_and_simplify("(ab|a|cd|b|ef)", "(b|ab?|cd|ef)");
        assert_parse_and_simplify("(ab|ab)", "ab");
        assert_parse_and_simplify("(ab)(ab)(ab)", "(ab){3}");
        assert_parse_and_simplify("aaaabbbbbccc", "a{4}b{5}c{3}");
        assert_parse_and_simplify("((ab))?(ab)(((ab)))((((ab)){3}))", "(ab){5,6}");
        assert_parse_and_simplify("(cd|ab)*(ab|cd)*", "(ab|cd)*");
        assert_parse_and_simplify(".*q(ab|ab|abc|ca)x", ".*q(abc?|ca)x");
        assert_parse_and_simplify(
            "((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q){1,100}",
            "(q|(a|ads|a{2}d)*abc.*def.*uif(x|ads|a{2}d)*abc.*oxs.*def(ads|ax|a{2}d)*abc.*def.*ksd){1,100}",
        );

        assert_parse_and_simplify("(a{2,4}){2,4}", "a{4,16}");
        Ok(())
    }

    fn assert_parse_and_simplify(regex: &str, regex_simplified: &str) {
        let regex_parsed = RegularExpression::new(regex).unwrap();
        assert_eq!(regex_simplified, regex_parsed.to_string());
    }

    #[test]
    fn test_repeat_simplify() -> Result<(), String> {
        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            2,
            Some(2),
            3,
            Some(3),
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            2,
            Some(2),
            2,
            Some(4),
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            3,
            Some(3),
            0,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            0,
            Some(3),
            1,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            1,
            Some(2),
            1,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            2,
            Some(3),
            1,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            3,
            Some(4),
            1,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            7,
            Some(8),
            1,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            0,
            None,
            3,
            Some(3),
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            1,
            None,
            0,
            Some(1),
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            0,
            Some(1),
            1,
            None,
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            2,
            Some(4),
            2,
            Some(4),
        );

        assert_repeat_simplify(
            &CharRange::new_from_range(Char::new('a')..=Char::new('a')),
            2,
            Some(3),
            2,
            Some(2),
        );

        Ok(())
    }

    fn assert_repeat_simplify(
        range: &CharRange,
        min1: u32,
        max1: Option<u32>,
        min2: u32,
        max2: Option<u32>,
    ) {
        let repeat = RegularExpression::Repetition(
            Box::new(RegularExpression::Repetition(
                Box::new(RegularExpression::Character(range.clone())),
                min1,
                max1,
            )),
            min2,
            max2,
        );

        let got = RegularExpression::new(&repeat.to_string()).unwrap();

        println!("{} -> {}", repeat, got);

        let repeat = repeat.to_automaton().unwrap();

        //repeat.to_dot();

        let result = got.to_automaton().unwrap();

        assert!(repeat.equivalent(&result).unwrap());
    }
}
