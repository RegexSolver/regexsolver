use token::TokenError;

use crate::{error::EngineError, fast_automaton::{condition::Condition, serializer::tokenizer::token::automaton_token::AutomatonToken}, CharRange};

use self::token::range_token::RangeToken;

use super::*;

impl Tokenizer<'_> {
    pub fn to_embedding(&self) -> Vec<AutomatonToken> {
        let mut vec = vec![];

        let mut worklist = VecDeque::new();
        let mut seen = IntSet::default();

        worklist.push_front(self.automaton.get_start_state());

        while let Some(current_state) = worklist.pop_back() {
            if !vec.is_empty() {
                // separator
                vec.push(AutomatonToken::SeparatorState)
            }
            seen.insert(current_state);

            // state
            let embedded_state =
                AutomatonToken::State(*self.state_to_token.get(&current_state).unwrap());
            vec.push(embedded_state);

            if self.automaton.is_accepted(&current_state) {
                // accept state
                vec.push(AutomatonToken::AcceptState)
            }

            for (condition, to_state) in self.automaton.transitions_from_iter(current_state) {
                if condition.is_empty() {
                    continue;
                }
                let embedded_state =
                    AutomatonToken::State(*self.state_to_token.get(to_state).unwrap());
                vec.push(embedded_state);

                if condition.is_total() {
                    vec.push(AutomatonToken::Range(RangeToken::Total));
                } else {
                    let range = condition
                        .to_range(self.automaton.get_spanning_set())
                        .expect("It should be possible to convert the condition to range.");
                    self.range_tokenizer
                        .range_to_embedding(&range)
                        .unwrap()
                        .iter()
                        .for_each(|&e| {
                            vec.push(AutomatonToken::Range(e));
                        });
                }

                if !seen.contains(to_state) {
                    worklist.push_front(*to_state);
                }
            }
        }

        vec
    }

    pub fn from_embedding(&self, vec: &Vec<AutomatonToken>) -> Result<FastAutomaton, EngineError> {
        let mut automaton = FastAutomaton::new_empty();
        automaton.apply_new_spanning_set(self.automaton.get_spanning_set())?;

        let mut from_state = None;
        let mut to_state = None;
        let mut range = CharRange::empty();
        for token in vec {
            match token {
                AutomatonToken::Range(r) => {
                    range = range.union(self.range_tokenizer.token_to_range(r).unwrap());
                }
                AutomatonToken::State(s) => {
                    while !automaton.has_state(*s) {
                        automaton.new_state();
                    }
                    if let Some(fs) = from_state {
                        if let Some(ts) = to_state {
                            Self::apply_transition(&mut automaton, fs, ts, &range)?;
                            range = CharRange::empty();
                        }
                        to_state = Some(*s);
                    } else {
                        from_state = Some(*s);
                    }
                }
                AutomatonToken::AcceptState => {
                    automaton.accept(from_state.unwrap());
                }
                AutomatonToken::SeparatorState => {
                    if let Some(to_state) = to_state {
                        Self::apply_transition(
                            &mut automaton,
                            from_state.unwrap(),
                            to_state,
                            &range,
                        )?;
                    }
                    from_state = None;
                    to_state = None;
                    range = CharRange::empty();
                }
                _ => return Err(EngineError::TokenError(TokenError::UnknownToken)),
            };
        }
        if let Some(to_state) = to_state {
            Self::apply_transition(&mut automaton, from_state.unwrap(), to_state, &range)?;
        }
        Ok(automaton)
    }

    fn apply_transition(
        automaton: &mut FastAutomaton,
        from_state: State,
        to_state: State,
        range: &CharRange,
    ) -> Result<(), EngineError> {
        let condition = Condition::from_range(range, automaton.get_spanning_set())?;
        automaton.add_transition(from_state, to_state, &condition);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    use super::*;

    #[test]
    fn test_tokenize() -> Result<(), String> {
        assert_embedding_convertion_for_fair("(a|b)");
        assert_embedding_convertion_for_fair("(|a)");
        assert_embedding_convertion_for_fair(".*ab");
        assert_embedding_convertion_for_fair("toto");
        assert_embedding_convertion_for_fair(".{2,3}");
        assert_embedding_convertion_for_fair("q(ab|ca|ab|abc)x");
        assert_embedding_convertion_for_fair(".*q(ab|ca|ab|abc)x");
        assert_embedding_convertion_for_fair(
            "((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q)",
        );
        assert_embedding_convertion_for_fair(
            "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21-\\x5a\\x53-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])+)\\])",
        );

        Ok(())
    }

    fn assert_embedding_convertion_for_fair(regex: &str) {
        assert_embedding_convertion(regex);
    }

    fn assert_embedding_convertion(regex: &str) {
        let regex = RegularExpression::new(regex).unwrap();
        println!("{}", regex);

        let automaton = regex.to_automaton().unwrap();
        let automaton = automaton.determinize().unwrap();

        let tokenizer = Tokenizer::new(&automaton);
        let embedding = tokenizer.to_embedding();

        let number_of_bases = automaton.get_spanning_set().get_number_of_spanning_ranges();
        let number_of_states = automaton.get_number_of_states();

        let embedding_usize =
            AutomatonToken::to_tokens(&embedding, number_of_bases, number_of_states).unwrap();
        let embedding: Vec<AutomatonToken> = embedding_usize
            .iter()
            .map(|&t| AutomatonToken::from_token(t, number_of_bases, number_of_states))
            .collect();

        let unembedded_automaton = tokenizer.from_embedding(&embedding).unwrap();

        assert!(
            automaton
                .difference(&unembedded_automaton)
                .unwrap()
                .is_empty()
        );
        assert!(
            unembedded_automaton
                .difference(&automaton)
                .unwrap()
                .is_empty()
        );
    }
}
