use std::hash::BuildHasherDefault;

use condition::converter::ConditionConverter;
use rayon::prelude::*;

use crate::{error::EngineError, execution_profile::ExecutionProfile};

use super::*;

impl FastAutomaton {
    /// Computes the union between `self` and `other`.
    pub fn union(&self, other: &FastAutomaton) -> Result<Self, EngineError> {
        Self::union_all([self, other])
    }

    /// Computes the union of all automata in the given iterator.
    pub fn union_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(
        automata: I,
    ) -> Result<Self, EngineError> {
        let mut new_automaton = FastAutomaton::new_empty();
        for automaton in automata {
            new_automaton.union_mut(automaton)?;
        }
        Ok(new_automaton)
    }

    /// Computes in parallel the union of all automata in the given iterator.
    pub fn union_all_par<'a, I: IntoParallelIterator<Item = &'a FastAutomaton>>(
        automata: I,
    ) -> Result<Self, EngineError> {
        let execution_profile = ExecutionProfile::get();

        let empty = FastAutomaton::new_empty();

        automata
            .into_par_iter()
            .try_fold(
                || empty.clone(),
                |mut acc, next| {
                    execution_profile.apply(|| {
                        acc.union_mut(next)?;
                        Ok(acc)
                    })
                },
            )
            .try_reduce(
                || empty.clone(),
                |mut acc, next| {
                    execution_profile.apply(|| {
                        acc.union_mut(&next)?;
                        Ok(acc)
                    })
                },
            )
    }

    fn prepare_start_states(
        &mut self,
        other: &FastAutomaton,
        new_states: &mut IntMap<usize, usize>,
        condition_converter: &ConditionConverter,
    ) -> Result<IntSet<usize>, EngineError> {
        let mut imcomplete_states =
            IntSet::with_capacity(other.out_degree(other.start_state) + 1);
        if other.is_accepted(&other.start_state) {
            self.accept(self.start_state);
        }
        let self_start_state_in_degree = self.in_degree(self.start_state);
        let other_start_state_in_degree = other.in_degree(other.start_state);
        if self_start_state_in_degree == 0 && other_start_state_in_degree == 0 {
            // The start states can be the same state without any consequence
            new_states.insert(other.start_state, self.start_state);
            imcomplete_states.insert(self.start_state);
        } else {
            if self_start_state_in_degree != 0 {
                let new_state = self.new_state();

                self.add_epsilon_transition(new_state, self.start_state);
                self.start_state = new_state;
                new_states.insert(other.start_state, self.start_state);
                imcomplete_states.insert(self.start_state);
            }
            if other_start_state_in_degree != 0 {
                let new_state = self.new_state();
                if other.is_accepted(&other.start_state) {
                    self.accept(new_state);
                }

                new_states.insert(other.start_state, new_state);
                imcomplete_states.insert(new_state);

                for (cond, other_to_state) in other.transitions_from_vec(other.start_state) {
                    let cond = condition_converter.convert(&cond)?;
                    let to_state = match new_states.entry(other_to_state) {
                        Entry::Occupied(o) => *o.get(),
                        Entry::Vacant(v) => {
                            let new_state = self.new_state();
                            imcomplete_states.insert(new_state);
                            v.insert(new_state);
                            new_state
                        }
                    };
                    self.add_transition(self.start_state, to_state, &cond);
                }
            }
        }
        Ok(imcomplete_states)
    }

    fn prepare_accept_states(
        &mut self,
        other: &FastAutomaton,
        new_states: &mut IntMap<usize, usize>,
        imcomplete_states: &IntSet<usize>,
    ) {
        let mut self_accept_states_without_outgoing_edges = vec![];
        for &state in &self.accept_states {
            if self.out_degree(state) == 0 && !imcomplete_states.contains(&state) {
                self_accept_states_without_outgoing_edges.push(state);
            }
        }
        let accept_state_without_outgoing_edges =
            match self_accept_states_without_outgoing_edges.len() {
                1 => Some(self_accept_states_without_outgoing_edges[0]),
                n if n > 1 => {
                    let new_state = self.new_state();
                    self.accept(new_state);

                    for &accept_state in &self_accept_states_without_outgoing_edges {
                        for (from_state, condition) in self.transitions_to_vec(accept_state) {
                            self.add_transition(from_state, new_state, &condition);
                        }
                        self.remove_state(accept_state);
                    }
                    Some(new_state)
                }
                _ => None,
            };

        for &state in &other.accept_states {
            match accept_state_without_outgoing_edges {
                Some(accept_state) if other.out_degree(state) == 0 => {
                    new_states.entry(state).or_insert(accept_state);
                }
                _ => {
                    if new_states.get(&state).is_none() {
                        let new_accept_state = self.new_state();
                        self.accept(new_accept_state);
                        new_states.insert(state, new_accept_state);
                    }
                }
            }
        }
    }

    /* Important things to remember before modifying this method:
     * - the start states can't be merged if they have incoming edges
     * - the accept states can't be merged if they have outgoing edges
     */
    pub(crate) fn union_mut(&mut self, other: &FastAutomaton) -> Result<(), EngineError> {
        if other.is_empty() || self.is_total() {
            return Ok(());
        } else if other.is_total() {
            self.make_total();
            return Ok(());
        } else if self.is_empty() {
            self.apply_model(other);
            return Ok(());
        }

        let new_spanning_set = &self.spanning_set.merge(&other.spanning_set);
        self.apply_new_spanning_set(new_spanning_set)?;
        let condition_converter = ConditionConverter::new(&other.spanning_set, new_spanning_set)?;

        let mut new_states: IntMap<usize, usize> = IntMap::with_capacity_and_hasher(
            other.get_number_of_states(),
            BuildHasherDefault::default(),
        );

        let imcomplete_states =
            self.prepare_start_states(other, &mut new_states, &condition_converter)?;
        self.prepare_accept_states(other, &mut new_states, &imcomplete_states);

        for from_state in other.states() {
            let new_from_state = match new_states.entry(from_state) {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => {
                    let new_state = self.new_state();
                    v.insert(new_state);
                    new_state
                }
            };
            for (condition, to_state) in other.transitions_from(from_state) {
                let new_condition = condition_converter.convert(condition)?;
                let new_to_state = match new_states.entry(*to_state) {
                    Entry::Occupied(o) => *o.get(),
                    Entry::Vacant(v) => {
                        let new_state = self.new_state();
                        v.insert(new_state);
                        new_state
                    }
                };
                self.add_transition(new_from_state, new_to_state, &new_condition);
            }
        }
        self.cyclic = self.cyclic || other.cyclic;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    #[test]
    fn test_simple_alternation_regex_1() -> Result<(), String> {
        let automaton = RegularExpression::parse("(abc|ac|aaa)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.is_match("abc"));
        assert!(automaton.is_match("ac"));
        assert!(automaton.is_match("aaa"));
        assert!(!automaton.is_match("abcd"));
        assert!(!automaton.is_match("ab"));
        assert!(!automaton.is_match("acc"));
        assert!(!automaton.is_match("a"));
        assert!(!automaton.is_match("aaaa"));
        assert!(!automaton.is_match("aa"));
        assert!(!automaton.is_match(""));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_2() -> Result<(), String> {
        let automaton = RegularExpression::parse("(b?|b{2})", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("b"));
        assert!(automaton.is_match("bb"));
        assert!(!automaton.is_match("bbb"));
        assert!(!automaton.is_match("bbbb"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_3() -> Result<(), String> {
        let automaton = RegularExpression::parse("((a|bc)*|d)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("abcaaabcbc"));
        assert!(automaton.is_match("d"));
        assert!(!automaton.is_match("ad"));
        assert!(!automaton.is_match("abcd"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_3b() -> Result<(), String> {
        let automaton = RegularExpression::parse("(d|(a|bc)*)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("abcaaabcbc"));
        assert!(automaton.is_match("d"));
        assert!(!automaton.is_match("ad"));
        assert!(!automaton.is_match("abcd"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_3t() -> Result<(), String> {
        let automaton = RegularExpression::parse("(d*|(a|bc)*)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("abcaaabcbc"));
        assert!(automaton.is_match("d"));
        assert!(automaton.is_match("ddd"));
        assert!(!automaton.is_match("ad"));
        assert!(!automaton.is_match("abcd"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_4() -> Result<(), String> {
        let automaton = RegularExpression::parse("(a+(ba+)*|ca*c)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match("cc"));
        assert!(automaton.is_match("caaac"));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("aababa"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_5() -> Result<(), String> {
        let automaton = RegularExpression::parse("((aad|ads|a)*|q)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match("q"));
        assert!(automaton.is_match("aad"));
        assert!(automaton.is_match("ads"));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("aadadsaaa"));
        assert!(!automaton.is_match("aaaas"));
        assert!(!automaton.is_match("ad"));
        assert!(!automaton.is_match("adsq"));
        assert!(!automaton.is_match("qq"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_6() -> Result<(), String> {
        let automaton = RegularExpression::parse("(ab|)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match("ab"));
        assert!(automaton.is_match(""));
        assert!(!automaton.is_match("a"));
        assert!(!automaton.is_match("b"));
        assert!(!automaton.is_match("aab"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_7() -> Result<(), String> {
        let automaton = RegularExpression::parse("(d|a?|ab)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("d"));
        assert!(automaton.is_match("ab"));
        assert!(automaton.is_match(""));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_8() -> Result<(), String> {
        let automaton = RegularExpression::parse("((d|a?|ab)u)*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match("au"));
        assert!(automaton.is_match("du"));
        assert!(automaton.is_match("abu"));
        assert!(automaton.is_match("u"));
        assert!(automaton.is_match(""));
        Ok(())
    }
}
