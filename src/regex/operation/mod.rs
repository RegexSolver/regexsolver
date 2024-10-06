use super::*;

mod concat;
mod simplify;
mod union;

impl RegularExpression {
    pub fn repeat(&self, min: u32, max_opt: Option<u32>) -> RegularExpression {
        if self.is_total() {
            return RegularExpression::new_total();
        } else if self.is_empty() {
            return RegularExpression::new_empty();
        } else if self.is_empty_string() {
            return Self::new_empty_string();
        } else if let Some(max) = max_opt {
            if max < min || max == 0 {
                return RegularExpression::new_empty_string();
            } else if min == 1 && max == 1 {
                return self.clone();
            }
        }

        match self {
            RegularExpression::Repetition(regular_expression, o_min, o_max_opt) => {
                let new_max = if let (Some(max), Some(o_max)) = (max_opt, o_max_opt) {
                    Some(max * o_max)
                } else {
                    None
                };

                let o_min = *o_min;
                if let Some(o_max) = o_max_opt {
                    let o_max = *o_max;
                    if o_min <= 1 || max_opt.is_some() && max_opt.unwrap() == min {
                        RegularExpression::Repetition(
                            regular_expression.clone(),
                            min * o_min,
                            new_max,
                        )
                    } else if o_min == o_max && o_min > 1 {
                        RegularExpression::Repetition(Box::new(self.clone()), min, max_opt)
                    } else {
                        let r = ((o_max as f64) - 1f64) / ((o_max as f64) - (o_min as f64));
                        if r > cmp::max(2, min) as f64 {
                            return RegularExpression::Repetition(
                                Box::new(self.clone()),
                                min,
                                max_opt,
                            );
                        }

                        RegularExpression::Repetition(
                            regular_expression.clone(),
                            min * o_min,
                            new_max,
                        )
                    }
                } else if o_max_opt.is_none()
                    || max_opt.is_some() && (max_opt.unwrap() == min || max_opt.unwrap() == 1)
                    || o_max_opt.is_some() && o_max_opt.unwrap() == 1
                    || max_opt.is_none() && o_min == 0
                {
                    RegularExpression::Repetition(regular_expression.clone(), min * o_min, new_max)
                } else {
                    RegularExpression::Repetition(Box::new(self.clone()), min, max_opt)
                }
            }
            _ => RegularExpression::Repetition(Box::new(self.clone()), min, max_opt),
        }
    }
}

#[cfg(test)]
mod tests {
    use regex_charclass::{char::Char, irange::RangeSet};

    use crate::regex::RegularExpression;

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
        assert_parse_and_simplify("((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q){1,100}", "(q|(a|ads|a{2}d)*abc.*def.*uif(x|ads|a{2}d)*abc.*oxs.*def(ads|ax|a{2}d)*abc.*def.*ksd){1,100}");
        Ok(())
    }

    fn assert_parse_and_simplify(regex: &str, regex_simplified: &str) {
        let regex_parsed = RegularExpression::new(regex).unwrap();
        assert_eq!(regex_simplified, regex_parsed.to_string());
    }

    #[test]
    fn test_repeat_simplify() -> Result<(), String> {
        assert_repeat_simplify(
            &RangeSet::new_from_range(Char::new('a')..=Char::new('a')),
            2,
            Some(2),
            3,
            Some(3),
        );

        assert_repeat_simplify(
            &RangeSet::new_from_range(Char::new('a')..=Char::new('a')),
            2,
            Some(2),
            2,
            Some(4),
        );

        assert_repeat_simplify(
            &RangeSet::new_from_range(Char::new('a')..=Char::new('a')),
            3,
            Some(3),
            0,
            None,
        );

        assert_repeat_simplify(
            &RangeSet::new_from_range(Char::new('a')..=Char::new('a')),
            0,
            Some(3),
            1,
            None,
        );

        assert_repeat_simplify(
            &RangeSet::new_from_range(Char::new('a')..=Char::new('a')),
            1,
            Some(2),
            1,
            None,
        );

        assert_repeat_simplify(
            &RangeSet::new_from_range(Char::new('a')..=Char::new('a')),
            2,
            Some(3),
            1,
            None,
        );

        assert_repeat_simplify(
            &RangeSet::new_from_range(Char::new('a')..=Char::new('a')),
            3,
            Some(4),
            1,
            None,
        );

        assert_repeat_simplify(
            &RangeSet::new_from_range(Char::new('a')..=Char::new('a')),
            7,
            Some(8),
            1,
            None,
        );

        assert_repeat_simplify(
            &RangeSet::new_from_range(Char::new('a')..=Char::new('a')),
            0,
            None,
            3,
            Some(3),
        );

        assert_repeat_simplify(
            &RangeSet::new_from_range(Char::new('a')..=Char::new('a')),
            1,
            None,
            0,
            Some(1),
        );

        assert_repeat_simplify(
            &RangeSet::new_from_range(Char::new('a')..=Char::new('a')),
            0,
            Some(1),
            1,
            None,
        );

        Ok(())
    }

    fn assert_repeat_simplify(
        range: &RangeSet<Char>,
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

        assert!(repeat.is_equivalent_of(&result).unwrap());
    }
}
