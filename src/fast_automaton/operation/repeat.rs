use super::*;

impl FastAutomaton {
    /// Computes the repetition of the automaton between `min` and `max_opt` times; if `max_opt` is `None`, the repetition is unbounded.
    pub fn repeat(&self, min: u32, max_opt: Option<u32>) -> Result<FastAutomaton, EngineError> {
        let mut automaton = self.clone();
        if let Err(error) = automaton.repeat_mut(min, max_opt) {
            Err(error)
        } else {
            Ok(automaton)
        }
    }

    pub(crate) fn repeat_mut(&mut self, min: u32, max_opt: Option<u32>) -> Result<(), EngineError> {
        if let Some(max) = max_opt
            && min > max
        {
            self.make_empty();
            return Ok(());
        }

        let automaton_to_repeat = self.clone();

        if min == 0 && self.in_degree(self.start_state) != 0 {
            let new_state = self.new_state();
            if self.is_accepted(self.start_state) {
                self.accept(new_state);
            }

            self.add_epsilon_transition(new_state, self.start_state);
            self.start_state = new_state;

            if max_opt.is_none() {
                for accept_state in self.accept_states.clone() {
                    self.add_epsilon_transition(accept_state, self.start_state);
                }
                self.accept(self.start_state);
                return Ok(());
            }
        }

        if let Some(max) = max_opt
            && min <= 1
            && max == 1
        {
            if min == 0 {
                self.accept_states.insert(self.start_state);
            }
            return Ok(());
        }

        let iter = if min == 0 { 0..0 } else { 0..min - 1 };
        for _ in iter {
            self.concat_mut(&automaton_to_repeat)?;
        }

        if max_opt.is_none() {
            let mut automaton_to_repeat = automaton_to_repeat.clone();

            let accept_state = *automaton_to_repeat.accept_states.iter().next().unwrap();
            if automaton_to_repeat.accept_states.len() == 1
                && automaton_to_repeat.out_degree(accept_state) == 0
                && automaton_to_repeat.in_degree(automaton_to_repeat.start_state) == 0
            {
                automaton_to_repeat
                    .add_epsilon_transition(accept_state, automaton_to_repeat.start_state);
                let old_start_state = automaton_to_repeat.start_state;
                automaton_to_repeat.start_state = accept_state;
                automaton_to_repeat.remove_state(old_start_state);
            } else {
                let t = Self::transitions_from_state_set(
                    &automaton_to_repeat.transitions,
                    automaton_to_repeat.start_state,
                );
                let transitions =
                    Self::transitions_from_state_enumerate(&t, &automaton_to_repeat.removed_states);

                for state in automaton_to_repeat.accept_states.clone() {
                    for &(to_state, condition) in &transitions {
                        automaton_to_repeat.add_transition(state, *to_state, condition);
                    }
                }

                automaton_to_repeat.accept(automaton_to_repeat.get_start_state());
            }
            automaton_to_repeat.cyclic = true;

            if min == 0 {
                self.apply_model(&automaton_to_repeat);
            } else {
                self.concat_mut(&automaton_to_repeat)?;
            }

            return Ok(());
        }

        let mut end_states = self.accept_states.iter().cloned().collect::<Vec<_>>();
        for _ in cmp::max(min, 1)..max_opt.unwrap() {
            self.concat_mut(&automaton_to_repeat)?;
            end_states.extend(self.accept_states.iter());
        }
        self.accept_states.extend(end_states);
        if min == 0 {
            self.accept(self.start_state);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    #[test]
    fn test_repeat_1() -> Result<(), String> {
        let automaton = RegularExpression::parse("(a*,a*)?", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match(","));
        assert!(automaton.is_match("aaa,"));
        assert!(automaton.is_match("aaaa,aa"));
        assert!(!automaton.is_match("a"));
        assert!(!automaton.is_match("aa"));
        Ok(())
    }
}
