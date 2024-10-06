use token::TokenError;

use crate::error::EngineError;

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

            for (to_state, condition) in self
                .automaton
                .transitions_from_state_enumerate_iter(&current_state)
            {
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
                        .to_range(self.automaton.get_used_bases())
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
        automaton.apply_newly_used_bases(self.automaton.get_used_bases())?;

        let mut from_state = None;
        let mut to_state = None;
        let mut range = Range::empty();
        for token in vec {
            match token {
                AutomatonToken::Range(r) => {
                    range = range.union(self.range_tokenizer.token_to_range(r).unwrap());
                }
                AutomatonToken::State(s) => {
                    while !automaton.has_state((*s).into()) {
                        automaton.new_state();
                    }
                    if let Some(fs) = from_state {
                        if let Some(ts) = to_state {
                            Self::apply_transition(&mut automaton, fs, ts, &range)?;
                            range = Range::empty();
                        }
                        to_state = Some((*s).into());
                    } else {
                        from_state = Some((*s).into());
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
                    range = Range::empty();
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
        range: &Range,
    ) -> Result<(), EngineError> {
        let condition = Condition::from_range(range, automaton.get_used_bases())?;
        automaton.add_transition_to(from_state, to_state, &condition);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use embed_automaton::token::Token;

    use crate::regex::RegularExpression;

    use super::*;

    #[test]
    fn test_tokenize() -> Result<(), String> {
        assert_embedding_convertion_for_fair_and_ai("(a|b)");
        assert_embedding_convertion_for_fair_and_ai("(|a)");
        assert_embedding_convertion_for_fair_and_ai(".*ab");
        assert_embedding_convertion_for_fair_and_ai("toto");
        assert_embedding_convertion_for_fair_and_ai(".{2,3}");
        assert_embedding_convertion_for_fair_and_ai("q(ab|ca|ab|abc)x");
        assert_embedding_convertion_for_fair_and_ai(".*q(ab|ca|ab|abc)x");
        assert_embedding_convertion_for_fair(
            "((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q)",
        );
        assert_embedding_convertion_for_fair("(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21-\\x5a\\x53-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])+)\\])");

        Ok(())
    }

    fn assert_embedding_convertion_for_fair(regex: &str) {
        assert_embedding_convertion(regex, true);
    }

    fn assert_embedding_convertion_for_fair_and_ai(regex: &str) {
        assert_embedding_convertion(regex, false);
    }

    fn assert_embedding_convertion(regex: &str, ignore_ai: bool) {
        let regex = RegularExpression::new(regex).unwrap();
        println!("{}", regex);

        let automaton = regex.to_automaton().unwrap().determinize().unwrap();

        let tokenizer = Tokenizer::new(&automaton);
        let embedding = tokenizer.to_embedding();

        // FAIR
        let embedding_u16 = AutomatonToken::to_fair_tokens(&embedding).unwrap();
        let embedding: Vec<AutomatonToken> = embedding_u16
            .iter()
            .map(|&t| AutomatonToken::from_fair_token(t))
            .collect();

        let unembedded_automaton = tokenizer.from_embedding(&embedding).unwrap();

        assert!(automaton
            .subtraction(&unembedded_automaton)
            .unwrap()
            .is_empty());
        assert!(unembedded_automaton
            .subtraction(&automaton)
            .unwrap()
            .is_empty());

        if !ignore_ai {
            // AI
            let embedding_u8 = AutomatonToken::to_ai_tokens(&embedding).unwrap();
            let embedding: Vec<AutomatonToken> = embedding_u8
                .iter()
                .map(|&t| AutomatonToken::from_ai_token(t))
                .collect();

            let unembedded_automaton = tokenizer.from_embedding(&embedding).unwrap();

            assert!(automaton
                .subtraction(&unembedded_automaton)
                .unwrap()
                .is_empty());
            assert!(unembedded_automaton
                .subtraction(&automaton)
                .unwrap()
                .is_empty());
        }
    }
}
