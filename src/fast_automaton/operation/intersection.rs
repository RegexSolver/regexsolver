use std::borrow::Cow;

use rayon::prelude::*;

use condition::converter::ConditionConverter;

use crate::{error::EngineError, execution_profile::ExecutionProfile};

use super::*;

impl FastAutomaton {
    /// Computes the intersection between `self` and `other`.
    pub fn intersection(&self, other: &FastAutomaton) -> Result<Self, EngineError> {
        FastAutomaton::intersection_all([self, other])
    }

    /// Computes the intersection of all automata in the given iterator.
    pub fn intersection_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(
        automata: I,
    ) -> Result<Self, EngineError> {
        let mut result: Cow<'a, FastAutomaton> = Cow::Owned(FastAutomaton::new_total());

        for automaton in automata {
            result = result.intersection_internal(automaton)?;

            if result.is_empty() {
                break;
            }
        }

        Ok(result.into_owned())
    }

    /// Computes in parallel the intersection of all automata in the given iterator.
    pub fn intersection_all_par<'a, I: IntoParallelIterator<Item = &'a FastAutomaton>>(
        automata: I,
    ) -> Result<Self, EngineError> {
        let execution_profile = ExecutionProfile::get();

        let total = FastAutomaton::new_total();

        automata
            .into_par_iter()
            .try_fold(
                || total.clone(),
                |acc, next| {
                    execution_profile.apply(|| Ok(acc.intersection_internal(next)?.into_owned()))
                },
            )
            .try_reduce(
                || total.clone(),
                |acc, next| {
                    execution_profile.apply(|| Ok(acc.intersection_internal(&next)?.into_owned()))
                },
            )
    }

    fn intersection_internal<'a>(
        &self,
        other: &'a FastAutomaton,
    ) -> Result<Cow<'a, FastAutomaton>, EngineError> {
        if self.is_empty() || other.is_empty() {
            return Ok(Cow::Owned(Self::new_empty()));
        } else if self.is_total() {
            return Ok(Cow::Borrowed(other));
        } else if other.is_total() {
            return Ok(Cow::Owned(self.clone()));
        }
        let execution_profile = ExecutionProfile::get();

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

            for (condition_1, n1) in transitions_1 {
                for (condition_2, n2) in &transitions_2 {
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
                    new_automaton.add_transition(p.0, r.0, &intersection);
                }
            }
        }
        new_automaton.spanning_set = new_spanning_set;
        new_automaton.remove_dead_transitions();
        Ok(Cow::Owned(new_automaton))
    }

    /// Returns `true` if the two automata have a non-empty intersection.
    pub fn has_intersection(&self, other: &FastAutomaton) -> Result<bool, EngineError> {
        if self.is_empty() || other.is_empty() {
            return Ok(false);
        } else if self.is_total() || other.is_total() {
            return Ok(true);
        }
        let execution_profile = ExecutionProfile::get();

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

            for (condition_1, n1) in transitions_1 {
                for (condition_2, n2) in &transitions_2 {
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
                    new_automaton.add_transition(p.0, r.0, &intersection);
                }
            }
        }
        Ok(false)
    }

    fn get_projected_transitions(
        &self,
        state: State,
        condition_converter: &ConditionConverter,
    ) -> Result<Vec<(Condition, State)>, EngineError> {
        let transitions_1: Result<Vec<_>, EngineError> = self
            .transitions_from(state)
            .map(|(c, &s)| match condition_converter.convert(c) {
                Ok(condition) => Ok((condition, s)),
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
        let automaton1 = RegularExpression::parse("(abc|ac|aaa)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::parse("(abcd|ac|aba)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let intersection = automaton1.intersection(&automaton2).unwrap();

        assert!(intersection.is_match("ac"));
        assert!(!intersection.is_match("abc"));
        assert!(!intersection.is_match("aaa"));
        assert!(!intersection.is_match("abcd"));
        assert!(!intersection.is_match("aba"));
        Ok(())
    }

    #[test]
    fn test_simple_intersection_regex_2() -> Result<(), String> {
        let automaton1 = RegularExpression::parse("a*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::parse("b*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let intersection = automaton1.intersection(&automaton2).unwrap();

        assert!(intersection.is_match(""));
        assert!(!intersection.is_match("a"));
        assert!(!intersection.is_match("b"));
        Ok(())
    }

    #[test]
    fn test_simple_intersection_regex_3() -> Result<(), String> {
        let automaton1 = RegularExpression::parse("x*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::parse("(xxx)*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let intersection = automaton1.intersection(&automaton2).unwrap();

        assert!(intersection.is_match(""));
        assert!(intersection.is_match("xxx"));
        assert!(intersection.is_match("xxxxxx"));
        assert!(!intersection.is_match("xx"));
        assert!(!intersection.is_match("xxxx"));
        Ok(())
    }

    #[test]
    fn test_complex_intersection_regex_1() -> Result<(), String> {
        let automaton1 = RegularExpression::parse(".*(abc|ac|aaa)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::parse("(abcd|ac|aba)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let intersection = automaton1.intersection(&automaton2).unwrap();

        assert!(intersection.is_match("ac"));
        assert!(!intersection.is_match("aaac"));
        assert!(!intersection.is_match("abc"));
        assert!(!intersection.is_match("aaa"));
        assert!(!intersection.is_match("abcd"));
        assert!(!intersection.is_match("aba"));
        Ok(())
    }

    #[test]
    fn test_complex_intersection_regex_2() -> Result<(), String> {
        let automaton1 = RegularExpression::parse("(?:[a-z0-9]+(?:\\.[a-z0-9]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21-\\x5a\\x53-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])+)\\])", false)
            .unwrap()
            .to_automaton().unwrap();
        let automaton2 = RegularExpression::parse("avb@.*", false)
            .unwrap()
            .to_automaton()
            .unwrap();

        automaton1.print_dot();
        automaton2.print_dot();
        let intersection = automaton1.intersection(&automaton2).unwrap();

        assert!(!intersection.is_empty());

        assert!(intersection.is_match("avb@gmail.com"));
        Ok(())
    }
}
