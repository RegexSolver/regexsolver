use condition::converter::ConditionConverter;

use crate::{error::EngineError, execution_profile::ThreadLocalParams};

use super::*;

impl FastAutomaton {
    pub fn intersection(&self, other: &FastAutomaton) -> Result<FastAutomaton, EngineError> {
        if self.is_empty() || other.is_empty() {
            return Ok(Self::new_empty());
        } else if self.is_total() {
            return Ok(other.clone());
        } else if other.is_total() {
            return Ok(self.clone());
        }
        let execution_profile = ThreadLocalParams::get_execution_profile();

        let new_spanning_set = self.spanning_set.merge(&other.spanning_set);

        let condition_converter_self_to_new =
            ConditionConverter::new(&self.spanning_set, &new_spanning_set)?;
        let condition_converter_other_to_new =
            ConditionConverter::new(&other.spanning_set, &new_spanning_set)?;

        let mut new_automaton = FastAutomaton::new_empty();
        let mut worklist =
            VecDeque::with_capacity(self.get_number_of_states() + other.get_number_of_states());
        let mut new_states: AHashMap<(usize, usize), (usize, usize, usize), _> =
            AHashMap::with_capacity(self.get_number_of_states() + other.get_number_of_states());

        let initial_pair = (
            new_automaton.start_state,
            self.start_state,
            other.start_state,
        );

        worklist.push_back(initial_pair);
        new_states.insert((self.start_state, other.start_state), initial_pair);

        while let Some(p) = worklist.pop_front() {
            execution_profile.assert_not_timed_out()?;
            if self.accept_states.contains(&p.1) && other.accept_states.contains(&p.2) {
                new_automaton.accept(p.0);
            }

            let transitions_1 =
                self.get_projected_transitions(p.1, &condition_converter_self_to_new)?;
            let transitions_2 =
                other.get_projected_transitions(p.2, &condition_converter_other_to_new)?;

            for (n1, condition_1) in transitions_1 {
                for (n2, condition_2) in &transitions_2 {
                    let intersection = condition_1.intersection(condition_2);
                    if intersection.is_empty() {
                        continue;
                    }
                    let k = (n1, *n2);
                    let r = match new_states.get(&k) {
                        Some(new_r) => *new_r,
                        None => {
                            let new_r = (new_automaton.new_state(), n1, *n2);
                            worklist.push_back(new_r);
                            new_states.insert(k, new_r);
                            new_r
                        }
                    };
                    new_automaton.add_transition_to(p.0, r.0, &intersection);
                }
            }
        }
        new_automaton.spanning_set = new_spanning_set;
        new_automaton.remove_dead_transitions();
        Ok(new_automaton)
    }

    pub fn has_intersection(&self, other: &FastAutomaton) -> Result<bool, EngineError> {
        if self.is_empty() || other.is_empty() {
            return Ok(false);
        } else if self.is_total() || other.is_total() {
            return Ok(true);
        }
        let execution_profile = ThreadLocalParams::get_execution_profile();

        let new_spanning_set = self.spanning_set.merge(&other.spanning_set);

        let condition_converter_self_to_new =
            ConditionConverter::new(&self.spanning_set, &new_spanning_set)?;
        let condition_converter_other_to_new =
            ConditionConverter::new(&other.spanning_set, &new_spanning_set)?;

        let mut new_automaton = FastAutomaton::new_empty();
        let mut worklist =
            VecDeque::with_capacity(self.get_number_of_states() + other.get_number_of_states());
        let mut new_states: AHashMap<(usize, usize), (usize, usize, usize), _> =
            AHashMap::with_capacity(self.get_number_of_states() + other.get_number_of_states());

        let initial_pair = (
            new_automaton.start_state,
            self.start_state,
            other.start_state,
        );

        worklist.push_back(initial_pair);
        new_states.insert((self.start_state, other.start_state), initial_pair);

        while let Some(p) = worklist.pop_front() {
            execution_profile.assert_not_timed_out()?;
            if self.accept_states.contains(&p.1) && other.accept_states.contains(&p.2) {
                return Ok(true);
            }

            let transitions_1 =
                self.get_projected_transitions(p.1, &condition_converter_self_to_new)?;
            let transitions_2 =
                other.get_projected_transitions(p.2, &condition_converter_other_to_new)?;

            for (n1, condition_1) in transitions_1 {
                for (n2, condition_2) in &transitions_2 {
                    let intersection = condition_1.intersection(condition_2);
                    if intersection.is_empty() {
                        continue;
                    }
                    let k = (n1, *n2);
                    let r = match new_states.get(&k) {
                        Some(new_r) => *new_r,
                        None => {
                            let new_r = (new_automaton.new_state(), n1, *n2);
                            worklist.push_back(new_r);
                            new_states.insert(k, new_r);
                            new_r
                        }
                    };
                    new_automaton.add_transition_to(p.0, r.0, &intersection);
                }
            }
        }
        Ok(false)
    }

    fn get_projected_transitions(
        &self,
        state: State,
        condition_converter: &ConditionConverter,
    ) -> Result<Vec<(State, Condition)>, EngineError> {
        let transitions_1: Result<Vec<_>, EngineError> = self
            .transitions_from_state_enumerate_iter(&state)
            .map(|(&s, c)| match condition_converter.convert(c) {
                Ok(condition) => Ok((s, condition)),
                Err(err) => Err(err),
            })
            .collect();

        transitions_1
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    #[test]
    fn test_simple_intersection_regex_1() -> Result<(), String> {
        let automaton1 = RegularExpression::new("(abc|ac|aaa)")
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::new("(abcd|ac|aba)")
            .unwrap()
            .to_automaton()
            .unwrap();
        let intersection = automaton1.intersection(&automaton2).unwrap();

        assert!(intersection.match_string("ac"));
        assert!(!intersection.match_string("abc"));
        assert!(!intersection.match_string("aaa"));
        assert!(!intersection.match_string("abcd"));
        assert!(!intersection.match_string("aba"));
        Ok(())
    }

    #[test]
    fn test_simple_intersection_regex_2() -> Result<(), String> {
        let automaton1 = RegularExpression::new("a*")
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::new("b*")
            .unwrap()
            .to_automaton()
            .unwrap();
        let intersection = automaton1.intersection(&automaton2).unwrap();

        assert!(intersection.match_string(""));
        assert!(!intersection.match_string("a"));
        assert!(!intersection.match_string("b"));
        Ok(())
    }

    #[test]
    fn test_simple_intersection_regex_3() -> Result<(), String> {
        let automaton1 = RegularExpression::new("x*")
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::new("(xxx)*")
            .unwrap()
            .to_automaton()
            .unwrap();
        let intersection = automaton1.intersection(&automaton2).unwrap();

        assert!(intersection.match_string(""));
        assert!(intersection.match_string("xxx"));
        assert!(intersection.match_string("xxxxxx"));
        assert!(!intersection.match_string("xx"));
        assert!(!intersection.match_string("xxxx"));
        Ok(())
    }

    #[test]
    fn test_complex_intersection_regex_1() -> Result<(), String> {
        let automaton1 = RegularExpression::new(".*(abc|ac|aaa)")
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::new("(abcd|ac|aba)")
            .unwrap()
            .to_automaton()
            .unwrap();
        let intersection = automaton1.intersection(&automaton2).unwrap();

        assert!(intersection.match_string("ac"));
        assert!(!intersection.match_string("aaac"));
        assert!(!intersection.match_string("abc"));
        assert!(!intersection.match_string("aaa"));
        assert!(!intersection.match_string("abcd"));
        assert!(!intersection.match_string("aba"));
        Ok(())
    }

    #[test]
    fn test_complex_intersection_regex_2() -> Result<(), String> {
        let automaton1 = RegularExpression::new("(?:[a-z0-9]+(?:\\.[a-z0-9]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21-\\x5a\\x53-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])+)\\])")
            .unwrap()
            .to_automaton().unwrap();
        let automaton2 = RegularExpression::new("avb@.*")
            .unwrap()
            .to_automaton()
            .unwrap();

        automaton1.to_dot();
        automaton2.to_dot();
        let intersection = automaton1.intersection(&automaton2).unwrap();

        assert!(!intersection.is_empty());

        assert!(intersection.match_string("avb@gmail.com"));
        Ok(())
    }
}
