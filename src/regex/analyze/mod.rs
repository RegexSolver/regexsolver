use super::*;

mod affixes;
mod number_of_states;

impl RegularExpression {
    /// Returns the minimum and maximum length of possible matched strings.
    #[must_use]
    pub fn length(&self) -> (Option<u32>, Option<u32>) {
        match self {
            RegularExpression::Character(range) => {
                if range.is_empty() {
                    return (None, None);
                }
                (Some(1), Some(1))
            }
            RegularExpression::Repetition(regex, min, max_opt) => {
                let (min_length, max_length_opt) = regex.length();
                if let Some(min_length) = min_length {
                    let new_min_length = min.saturating_mul(min_length);
                    let new_max_length = if let Some(max_length) = max_length_opt {
                        max_opt.as_ref().map(|max| max.saturating_mul(max_length))
                    } else {
                        None
                    };
                    (Some(new_min_length), new_max_length)
                } else if min == &0 {
                    (Some(0), Some(0))
                } else {
                    (None, None)
                }
            }
            RegularExpression::Concat(concat_vec) => {
                let mut new_min_length: u32 = 0;
                let mut new_max_length: Option<u32> = Some(0);

                for concat_element in concat_vec {
                    let (min_length, max_length_opt) = concat_element.length();

                    if let Some(min_length) = min_length {
                        new_min_length = new_min_length.saturating_add(min_length);

                        if let Some(new_max) = new_max_length {
                            if let Some(max_length) = max_length_opt {
                                new_max_length = Some(new_max.saturating_add(max_length));
                            } else {
                                new_max_length = None;
                            }
                        }
                    } else {
                        return (None, None);
                    }
                }

                (Some(new_min_length), new_max_length)
            }
            RegularExpression::Alternation(alternation_vec) => {
                if alternation_vec.is_empty() {
                    return (None, None);
                }
                let mut new_min_length = u32::MAX;
                let mut new_max_length = Some(0);
                let mut any_non_empty = false;

                for alternation_element in alternation_vec {
                    let (min_length, max_length_opt) = alternation_element.length();

                    if let Some(min_length) = min_length {
                        any_non_empty = true;
                        new_min_length = cmp::min(new_min_length, min_length);

                        if let Some(new_max) = new_max_length {
                            if let Some(max_length) = max_length_opt {
                                new_max_length = Some(cmp::max(new_max, max_length));
                            } else {
                                new_max_length = None;
                            }
                        }
                    }
                }

                if !any_non_empty {
                    // Every branch was the empty language ⇒ empty language.
                    return (None, None);
                }

                (Some(new_min_length), new_max_length)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length() -> Result<(), String> {
        assert_length(".{1,1000}");
        assert_length("toto");
        assert_length(".{2,3}");
        assert_length("q(ab|ca)x");
        assert_length("q(ab|ca|ab|abc)x");
        assert_length(".*");
        assert_length(".?");
        assert_length("a*(aad|ads|a)abc.*def.*ghi");
        assert_length("(at?)");
        assert_length("(ot){3,4}");
        assert_length("(ot?d){1,4}");
        assert_length(
            "((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q){1,100}",
        );

        assert_eq!(
            FastAutomaton::new_empty().length(),
            RegularExpression::new_empty().length()
        );

        assert_eq!(
            FastAutomaton::new_total().length(),
            RegularExpression::new_total().length()
        );
        Ok(())
    }

    fn assert_length(regex: &str) {
        println!("{}", regex);
        let regex = RegularExpression::new(regex).unwrap();

        let (min, max_opt) = regex.length();

        let automaton = regex.to_automaton().unwrap();
        //automaton.to_dot();

        let (min_automaton_opt, max_automaton_opt) = automaton.length();

        assert_eq!((min_automaton_opt, max_automaton_opt), (min, max_opt));
    }
}
