use std::collections::BTreeSet;

use super::*;

impl RegularExpression {
    pub fn get_common_affixes(
        &self,
        other: &RegularExpression,
    ) -> (
        Option<RegularExpression>, // common prefix
        (RegularExpression, RegularExpression),
        Option<RegularExpression>, // common suffix
    ) {
        let (mut self_regex, mut other_regex, common_prefix, common_suffix);

        (common_prefix, (self_regex, other_regex)) = self.get_common_affix(other, true);

        (common_suffix, (self_regex, other_regex)) =
            self_regex.get_common_affix(&other_regex, false);

        (common_prefix, (self_regex, other_regex), common_suffix)
    }

    pub fn get_common_affix(
        &self,
        other: &RegularExpression,
        is_prefix: bool,
    ) -> (
        Option<RegularExpression>, // common affix
        (RegularExpression, RegularExpression),
    ) {
        if self.is_empty() || other.is_empty() {
            return (None, (self.clone(), other.clone()));
        } else if self == other {
            return (
                Some(self.clone()),
                (
                    RegularExpression::new_empty_string(),
                    RegularExpression::new_empty_string(),
                ),
            );
        }

        let common_affix;
        let self_regex;
        let other_regex;

        match (self, other) {
            (RegularExpression::Concat(_), _) => {
                (common_affix, (self_regex, other_regex)) =
                    Self::opaffix_concat_and_other(self, other, is_prefix);
            }
            (_, RegularExpression::Concat(_)) => {
                (common_affix, (other_regex, self_regex)) =
                    Self::opaffix_concat_and_other(other, self, is_prefix);
            }
            (RegularExpression::Character(_), RegularExpression::Repetition(_, _, _)) => {
                (common_affix, (self_regex, other_regex)) =
                    Self::opaffix_character_and_repetition(self, other);
            }
            (RegularExpression::Repetition(_, _, _), RegularExpression::Character(_)) => {
                (common_affix, (other_regex, self_regex)) =
                    Self::opaffix_character_and_repetition(other, self);
            }
            (RegularExpression::Repetition(_, _, _), RegularExpression::Repetition(_, _, _)) => {
                (common_affix, (self_regex, other_regex)) =
                    Self::opaffix_repetition_and_repetition(self, other);
            }
            (RegularExpression::Alternation(_), RegularExpression::Alternation(_)) => {
                (common_affix, (self_regex, other_regex)) =
                    Self::opaffix_alternation_and_alternation(self, other);
            }
            (_, _) => {
                (common_affix, (self_regex, other_regex)) = (None, (self.clone(), other.clone()));
            }
        };

        (common_affix, (self_regex, other_regex))
    }

    fn opaffix_character_and_repetition(
        this_character: &RegularExpression,
        that_repetition: &RegularExpression,
    ) -> (
        Option<RegularExpression>, // common affix
        (RegularExpression, RegularExpression),
    ) {
        if let (
            RegularExpression::Character(_),
            RegularExpression::Repetition(that_regex, that_min, that_max_opt),
        ) = (this_character, that_repetition)
        {
            if this_character == &**that_regex && *that_min == 1 {
                let new_max = that_max_opt.as_ref().map(|that_max| that_max - 1);
                (
                    Some(this_character.clone()),
                    (
                        RegularExpression::new_empty_string(),
                        RegularExpression::Repetition(that_regex.clone(), 0, new_max),
                    ),
                )
            } else {
                (None, (this_character.clone(), that_repetition.clone()))
            }
        } else {
            panic!("Not character and repetition");
        }
    }

    fn opaffix_repetition_and_repetition(
        this_repetition: &RegularExpression,
        that_repetition: &RegularExpression,
    ) -> (
        Option<RegularExpression>, // common affix
        (RegularExpression, RegularExpression),
    ) {
        if let (
            RegularExpression::Repetition(this_regex, this_min, this_max_opt),
            RegularExpression::Repetition(that_regex, that_min, that_max_opt),
        ) = (this_repetition, that_repetition)
        {
            if this_regex == that_regex {
                let prefix_min = *cmp::min(this_min, that_min);
                let prefix_max_opt;
                if this_min == that_min {
                    if let Some(self_max) = this_max_opt {
                        if let Some(other_max) = that_max_opt {
                            prefix_max_opt = Some(*cmp::min(self_max, other_max));
                        } else {
                            prefix_max_opt = Some(prefix_min);
                        }
                    } else if let Some(other_max) = that_max_opt {
                        if let Some(self_max) = this_max_opt {
                            prefix_max_opt = Some(*cmp::min(self_max, other_max));
                        } else {
                            prefix_max_opt = Some(prefix_min);
                        }
                    } else {
                        prefix_max_opt = None;
                    }
                } else {
                    prefix_max_opt = Some(prefix_min);
                }
                if prefix_min != 0 {
                    if let Some(prefix_max) = prefix_max_opt {
                        let self_repeat_max =
                            this_max_opt.as_ref().map(|self_max| self_max - prefix_max);

                        let other_repeat_max = that_max_opt
                            .as_ref()
                            .map(|other_max| other_max - prefix_max);

                        let common_affix = Some(this_regex.repeat(prefix_min, prefix_max_opt));
                        let self_regex = this_regex.repeat(this_min - prefix_min, self_repeat_max);
                        let other_regex =
                            this_regex.repeat(that_min - prefix_min, other_repeat_max);

                        (common_affix, (self_regex, other_regex))
                    } else {
                        (
                            Some(*this_regex.clone()),
                            (
                                RegularExpression::new_empty_string(),
                                RegularExpression::new_empty_string(),
                            ),
                        )
                    }
                } else {
                    (None, (this_repetition.clone(), that_repetition.clone()))
                }
            } else {
                (None, (this_repetition.clone(), that_repetition.clone()))
            }
        } else {
            panic!("Not character and repetition");
        }
    }

    fn opaffix_concat_and_other(
        this_concat: &RegularExpression,
        that_other: &RegularExpression,
        is_prefix: bool,
    ) -> (
        Option<RegularExpression>, // common affix
        (RegularExpression, RegularExpression),
    ) {
        if let RegularExpression::Concat(this_elements) = this_concat {
            let mut other_temp = that_other.clone();
            let mut new_common_affix = Self::new_empty_string();
            let mut new_self_concat = Self::new_empty_string();
            let mut c = 0;

            let iter: Box<dyn Iterator<Item = _>> = if is_prefix {
                Box::new(this_elements.iter())
            } else {
                Box::new(this_elements.iter().rev())
            };
            for self_concat_element in iter {
                c += 1;
                let (common_affix_temp_opt, self_concat_element_temp);
                (
                    common_affix_temp_opt,
                    (self_concat_element_temp, other_temp),
                ) = self_concat_element.get_common_affix(&other_temp, is_prefix);

                if let Some(common_affix_temp) = common_affix_temp_opt {
                    new_common_affix = new_common_affix.concat(&common_affix_temp, is_prefix);
                }

                if !self_concat_element_temp.is_empty_string() || other_temp.is_empty_string() {
                    new_self_concat = new_self_concat.concat(&self_concat_element_temp, is_prefix);
                    break;
                }
            }
            if !new_common_affix.is_empty_string() {
                let iter: Box<dyn Iterator<Item = _>> = if is_prefix {
                    Box::new(c..this_elements.len())
                } else {
                    Box::new((0..this_elements.len() - c).rev())
                };
                for i in iter {
                    new_self_concat = new_self_concat.concat(&this_elements[i], is_prefix);
                }
                (Some(new_common_affix), (new_self_concat, other_temp))
            } else {
                (None, (this_concat.clone(), that_other.clone()))
            }
        } else {
            panic!("Not concat");
        }
    }

    fn opaffix_alternation_and_alternation(
        this_alternation: &RegularExpression,
        that_alternation: &RegularExpression,
    ) -> (
        Option<RegularExpression>, // common affix
        (RegularExpression, RegularExpression),
    ) {
        if let (
            RegularExpression::Alternation(this_elements),
            RegularExpression::Alternation(that_elements),
        ) = (this_alternation, that_alternation)
        {
            let this_elements_set: BTreeSet<_> = this_elements.iter().collect();
            let that_elements_set: BTreeSet<_> = that_elements.iter().collect();
            if this_elements_set == that_elements_set {
                (
                    Some(this_alternation.clone()),
                    (
                        RegularExpression::new_empty_string(),
                        RegularExpression::new_empty_string(),
                    ),
                )
            } else {
                (None, (this_alternation.clone(), that_alternation.clone()))
            }
        } else {
            panic!("Not character and repetition");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix() -> Result<(), String> {
        assert_regex_affix(true, ".*abc", ".*ad", ".*a", "bc", "d");

        // Character
        assert_regex_affix(true, "a", "a", "a", "", "");
        assert_regex_affix(true, "a", "a+", "a", "", "a*");
        assert_regex_affix(true, "a", "abc", "a", "", "bc");
        assert_regex_affix(true, "a", "(a|a)c", "a", "", "c");

        // Repetition
        assert_regex_affix(true, "a{1,2}", "a{1,2}", "a{1,2}", "", "");
        assert_regex_affix(true, "a{1,2}", "a", "a", "a{0,1}", "");
        assert_regex_affix(true, "a{1,2}", "a+", "a", "a?", "a*");
        assert_regex_affix(true, "a{1,2}", "abc", "a", "a?", "bc");
        assert_regex_affix(true, "a{1,2}", "(a|a)c", "a", "a?", "c");

        assert_regex_affix(true, "(ab|cd)x", "(ab|cd)y", "(ab|cd)", "x", "y");

        assert_regex_affix(true, "a+", "a+b", "a+", "", "b");

        Ok(())
    }

    #[test]
    fn test_suffix() -> Result<(), String> {
        // Character
        assert_regex_affix(false, "a", "a", "a", "", "");
        assert_regex_affix(false, "a", "a+", "a", "", "a*");
        assert_regex_affix(false, "a", "cba", "a", "", "cb");
        assert_regex_affix(false, "a", "c(a|a)", "a", "", "c");

        // Repetition
        assert_regex_affix(false, "a{1,2}", "a{1,2}", "a{1,2}", "", "");
        assert_regex_affix(false, "a{1,2}", "a", "a", "a{0,1}", "");
        assert_regex_affix(false, "a{1,2}", "a+", "a", "a?", "a*");
        assert_regex_affix(false, "a{1,2}", "cba", "a", "a?", "cb");
        assert_regex_affix(false, "a{1,2}", "c(a|a)", "a", "a?", "c");

        Ok(())
    }

    fn assert_regex_affix(
        is_prefix: bool,
        regex_1: &str,
        regex_2: &str,
        expected_affix: &str,
        expected_regex_1_t: &str,
        expected_regex_2_t: &str,
    ) {
        if is_prefix {
            println!("Prefix of {} and {}", regex_1, regex_2);
        } else {
            println!("Suffix of {} and {}", regex_1, regex_2);
        }
        let regex_1 = RegularExpression::new(regex_1).unwrap();
        let regex_2 = RegularExpression::new(regex_2).unwrap();
        let expected_prefix = RegularExpression::new(expected_affix).unwrap();
        let expected_regex_1_t = RegularExpression::new(expected_regex_1_t).unwrap();
        let expected_regex_2_t = RegularExpression::new(expected_regex_2_t).unwrap();

        let (common_prefix, (regex_1_t, regex_2_t)) = regex_1.get_common_affix(&regex_2, is_prefix);

        assert_eq!(
            Some(expected_prefix),
            common_prefix,
            "Expected common prefix mismatch"
        );
        assert_eq!(expected_regex_1_t, regex_1_t, "Expected regex1 mismatch");
        assert_eq!(expected_regex_2_t, regex_2_t, "Expected regex2 mismatch");
        assert_eq!(
            (common_prefix, (regex_2_t, regex_1_t)),
            regex_2.get_common_affix(&regex_1, is_prefix),
            "The operation is not symetrical"
        );
    }
}
