use token::TokenError;

use crate::regex::RegularExpression;

use self::token::regex_token::RegexToken;

use super::*;

impl Tokenizer<'_> {
    pub fn to_regex_embedding(&self, regex: &RegularExpression) -> Vec<RegexToken> {
        let mut vec = self.to_regex_embedding_vec(regex);

        Self::append_counter_if_necessary(&mut vec);

        vec
    }

    fn append_counter_if_necessary(vec: &mut Vec<RegexToken>) {
        if let Some(last) = vec.last() {
            match last {
                RegexToken::RepetitionNone => {}
                RegexToken::Repetition(_) => {}
                RegexToken::EndGroup => {}
                RegexToken::StartGroup => {}
                RegexToken::Alternation => {}
                RegexToken::Error => todo!(),
                _ => {
                    vec.push(RegexToken::Repetition(1));
                }
            };
        }
    }

    fn to_regex_embedding_vec(&self, regex: &RegularExpression) -> Vec<RegexToken> {
        let mut vec = vec![];

        match regex {
            RegularExpression::Character(range) => {
                self.range_tokenizer
                    .range_to_embedding(range)
                    .unwrap()
                    .into_iter()
                    .for_each(|t| vec.push(RegexToken::Range(t)));
            }
            RegularExpression::Repetition(regex, min, max_opt) => {
                if matches!(
                    **regex,
                    RegularExpression::Repetition(_, _, _) | RegularExpression::Concat(_)
                ) {
                    vec.push(RegexToken::StartGroup);
                    vec.extend(self.to_regex_embedding_vec(regex));
                    vec.push(RegexToken::EndGroup);
                } else {
                    vec.extend(self.to_regex_embedding_vec(regex));
                }

                vec.push(RegexToken::Repetition(*min as u16));

                if let Some(max) = max_opt {
                    if max != min {
                        vec.push(RegexToken::Repetition(*max as u16));
                    }
                } else {
                    vec.push(RegexToken::RepetitionNone);
                }
            }
            RegularExpression::Concat(elements) => {
                for element in elements {
                    vec.extend(self.to_regex_embedding_vec(element));
                    Self::append_counter_if_necessary(&mut vec);
                }
            }
            RegularExpression::Alternation(elements) => {
                vec.push(RegexToken::StartGroup);

                for i in 0..elements.len() {
                    let element = &elements[i];
                    vec.extend(self.to_regex_embedding_vec(element));
                    Self::append_counter_if_necessary(&mut vec);
                    if i < elements.len() - 1 {
                        vec.push(RegexToken::Alternation);
                    }
                }

                vec.push(RegexToken::EndGroup);
            }
        }

        vec
    }

    pub fn from_regex_embedding(
        &self,
        vec: &[RegexToken],
    ) -> Result<RegularExpression, TokenError> {
        let mut regex_groups = vec![(RegularExpression::new_empty_string(), false)];
        let mut current_range: Option<Range> = None;
        let mut current_min = None;
        for i in 0..vec.len() {
            let token = vec[i];
            let current_group = regex_groups.len() - 1;
            match token {
                RegexToken::Range(range_token) => {
                    let range = self.range_tokenizer.token_to_range(&range_token).unwrap();
                    if let Some(curr_range) = &current_range {
                        current_range = Some(curr_range.union(range));
                    } else {
                        current_range = Some(range.clone());
                    }
                }
                RegexToken::StartGroup => {
                    regex_groups.push((RegularExpression::new_empty_string(), false));
                }
                RegexToken::EndGroup => {
                    if current_group == 0 {
                        return Err(TokenError::SyntaxError);
                    }
                    if i == vec.len() - 1 || !matches!(vec[i + 1], RegexToken::Repetition(_)) {
                        let alternation: bool = regex_groups[current_group].1;
                        Self::pop_regex_group(&mut regex_groups, &None, &None);
                        if alternation {
                            Self::pop_regex_group(&mut regex_groups, &None, &None);
                        }
                    }
                }
                RegexToken::Alternation => {
                    if regex_groups[current_group].1 {
                        Self::pop_regex_group(&mut regex_groups, &None, &None);
                    }
                    regex_groups.push((RegularExpression::new_empty_string(), true));
                }
                RegexToken::RepetitionNone => {
                    if current_min.is_some() {
                        if let Some(range) = &current_range {
                            Self::add_regex(
                                &mut regex_groups,
                                &current_min,
                                &None,
                                &RegularExpression::Character(range.clone()),
                                false,
                            );
                            current_range = None;
                        } else {
                            Self::pop_regex_group(&mut regex_groups, &current_min, &None);
                        }
                        current_min = None;
                    } else {
                        return Err(TokenError::SyntaxError);
                    }
                }
                RegexToken::Repetition(count) => {
                    if current_min.is_some()
                        || i == vec.len() - 1
                        || !matches!(vec[i + 1], RegexToken::Repetition(_))
                            && !matches!(vec[i + 1], RegexToken::RepetitionNone)
                    {
                        let min;
                        let max;
                        if current_min.is_some() {
                            min = current_min;
                            max = Some(count as u32);
                        } else {
                            min = Some(count as u32);
                            max = Some(count as u32);
                        }
                        if let Some(range) = &current_range {
                            Self::add_regex(
                                &mut regex_groups,
                                &min,
                                &max,
                                &RegularExpression::Character(range.clone()),
                                false,
                            );
                            current_range = None;
                        } else {
                            Self::pop_regex_group(&mut regex_groups, &min, &max);
                        }
                        current_min = None;
                    } else {
                        current_min = Some(count as u32);
                    }
                }
                _ => return Err(TokenError::UnknownToken),
            };
        }

        Ok(regex_groups[0].0.clone())
    }

    fn pop_regex_group(
        regex_groups: &mut Vec<(RegularExpression, bool)>,
        current_min: &Option<u32>,
        current_max: &Option<u32>,
    ) -> bool {
        if regex_groups.len() <= 1 {
            return false;
        }

        let popped_group = regex_groups.pop().unwrap();
        Self::add_regex(
            regex_groups,
            current_min,
            current_max,
            &popped_group.0,
            popped_group.1,
        );
        true
    }

    fn add_regex(
        regex_groups: &mut [(RegularExpression, bool)],
        current_min: &Option<u32>,
        current_max: &Option<u32>,
        regex: &RegularExpression,
        alternation: bool,
    ) {
        let current_group = regex_groups.len() - 1;
        let regex_to_use = if let Some(min) = current_min {
            if min == &1 && current_max.is_some() {
                if current_max.unwrap() == 1 {
                    regex.clone()
                } else {
                    RegularExpression::Repetition(Box::new(regex.clone()), *min, *current_max)
                }
            } else {
                RegularExpression::Repetition(Box::new(regex.clone()), *min, *current_max)
            }
        } else {
            regex.clone()
        };

        if alternation {
            regex_groups[current_group].0 = regex_groups[current_group].0.union(&regex_to_use);
        } else {
            regex_groups[current_group].0 =
                regex_groups[current_group].0.concat(&regex_to_use, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use embed_regex::token::Token;

    use crate::regex::RegularExpression;

    use super::*;

    #[test]
    fn test_tokenize() -> Result<(), String> {
        assert_embedding_convertion(".*");
        assert_embedding_convertion("(a|b)");
        assert_embedding_convertion("(|a)");
        assert_embedding_convertion(".*ab");
        assert_embedding_convertion("[a-e]{3}");
        assert_embedding_convertion("[a-e]{3}efg");
        assert_embedding_convertion("toto");
        assert_embedding_convertion(".{2,3}");
        assert_embedding_convertion("q(abc?|ca)x");
        assert_embedding_convertion(".*q(abc?|ca)x");
        assert_embedding_convertion("(abc){3,6}");
        assert_embedding_convertion("((|a)abd+){3}");
        /*assert_embedding_convertion(
            "((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q)",
        );*/
        Ok(())
    }

    fn assert_embedding_convertion(regex: &str) {
        let regex = RegularExpression::new(regex).unwrap();
        println!("{}", regex);

        let automaton = regex.to_automaton().unwrap().determinize().unwrap();
        //automaton.to_dot();

        let tokenizer = Tokenizer::new(&automaton);
        let embedding = tokenizer.to_regex_embedding(&regex);

        //println!("{:?}", embedding);

        // FAIR
        let embedding_u16 = RegexToken::to_fair_tokens(&embedding).unwrap();
        assert_eq!(
            embedding,
            embedding_u16
                .iter()
                .map(|&t| RegexToken::from_fair_token(t))
                .collect::<Vec<_>>()
        );

        let unembedded_regex = tokenizer.from_regex_embedding(&embedding).unwrap();
        assert_eq!(regex, unembedded_regex);

        // AI
        let embedding_u8 = RegexToken::to_ai_tokens(&embedding).unwrap();
        assert_eq!(
            embedding,
            embedding_u8
                .iter()
                .map(|&t| RegexToken::from_ai_token(t))
                .collect::<Vec<_>>()
        );

        let unembedded_regex = tokenizer.from_regex_embedding(&embedding).unwrap();
        assert_eq!(regex, unembedded_regex);
    }
}
