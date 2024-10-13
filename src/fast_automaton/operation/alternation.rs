use std::hash::BuildHasherDefault;

use crate::error::EngineError;

use super::*;

impl FastAutomaton {
    pub fn union(&self, that: &FastAutomaton) -> Result<FastAutomaton, EngineError> {
        let mut union = self.clone();
        union.alternate(that)?;
        Ok(union)
    }

    pub fn alternation(automatons: Vec<FastAutomaton>) -> Result<FastAutomaton, EngineError> {
        if automatons.len() == 1 {
            return Ok(automatons[0].clone());
        }
        let mut new_automaton = FastAutomaton::new_empty();
        if automatons.is_empty() {
            return Ok(new_automaton);
        }
        for automaton in automatons {
            new_automaton.alternate(&automaton)?;
        }
        Ok(new_automaton)
    }

    fn prepare_start_states(
        &mut self,
        other: &FastAutomaton,
        new_states: &mut IntMap<usize, usize>,
        spanning_set: &SpanningSet,
    ) -> Result<IntSet<usize>, EngineError> {
        let mut imcomplete_states = IntSet::with_capacity(other.out_degree(other.start_state) + 1);
        let self_start_state_in_degree = self.in_degree(self.start_state);
        let other_start_state_in_degree = other.in_degree(other.start_state);
        if self_start_state_in_degree == 0 && other_start_state_in_degree == 0 {
            // The start states can be the same state without any consequence
            new_states.insert(other.start_state, self.start_state);
            imcomplete_states.insert(self.start_state);
        } else {
            if self_start_state_in_degree != 0 {
                let new_state = self.new_state();
                if self.is_accepted(&self.start_state) {
                    self.accept(new_state);
                }

                for (to_state, cond) in self.transitions_from_state_enumerate_vec(&self.start_state)
                {
                    self.add_transition_to(new_state, to_state, &cond);
                }
                self.start_state = new_state;
            }
            if other_start_state_in_degree != 0 {
                let new_state = self.new_state();
                if other.is_accepted(&other.start_state) {
                    self.accept(new_state);
                    self.accept(self.start_state);
                }

                new_states.insert(other.start_state, new_state);
                imcomplete_states.insert(new_state);

                for (other_to_state, cond) in
                    other.transitions_from_state_enumerate_vec(&other.start_state)
                {
                    let cond = cond.project_to(&other.spanning_set, spanning_set)?;
                    let to_state = match new_states.entry(other_to_state) {
                        Entry::Occupied(o) => *o.get(),
                        Entry::Vacant(v) => {
                            let new_state = self.new_state();
                            imcomplete_states.insert(new_state);
                            v.insert(new_state);
                            new_state
                        }
                    };
                    self.add_transition_to(self.start_state, to_state, &cond);
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
                1 => self_accept_states_without_outgoing_edges[0],
                n if n > 1 => {
                    let new_state = self.new_state();
                    self.accept(new_state);

                    for &accept_state in &self_accept_states_without_outgoing_edges {
                        for (from_state, condition) in self.in_transitions(accept_state) {
                            self.add_transition_to(from_state, new_state, &condition);
                        }
                        self.remove_state(accept_state);
                    }
                    new_state
                }
                _ => {
                    let new_state = self.new_state();
                    self.accept(new_state);
                    new_state
                }
            };

        for &state in &other.accept_states {
            if other.out_degree(state) == 0 {
                new_states
                    .entry(state)
                    .or_insert(accept_state_without_outgoing_edges);
            } else if new_states.get(&state).is_none() {
                let new_accept_state = self.new_state();
                self.accept(new_accept_state);
                new_states.insert(state, new_accept_state);
            }
        }
    }

    /* Important things to remember before modifying this method:
     * - the start states can't be merged if they have incoming edges
     * - the accept states can't be merged if they have outgoing edges
     */
    fn alternate(&mut self, other: &FastAutomaton) -> Result<(), EngineError> {
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

        let mut new_states: IntMap<usize, usize> = IntMap::with_capacity_and_hasher(
            other.get_number_of_states(),
            BuildHasherDefault::default(),
        );

        let imcomplete_states =
            self.prepare_start_states(other, &mut new_states, new_spanning_set)?;
        self.prepare_accept_states(other, &mut new_states, &imcomplete_states);

        for from_state in other.transitions_iter() {
            let new_from_state = match new_states.entry(from_state) {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => {
                    let new_state = self.new_state();
                    v.insert(new_state);
                    new_state
                }
            };
            for (to_state, condition) in other.transitions_from_state_enumerate_iter(&from_state) {
                let new_condition = condition.project_to(&other.spanning_set, new_spanning_set)?;
                let new_to_state = match new_states.entry(*to_state) {
                    Entry::Occupied(o) => *o.get(),
                    Entry::Vacant(v) => {
                        let new_state = self.new_state();
                        v.insert(new_state);
                        new_state
                    }
                };
                self.add_transition_to(new_from_state, new_to_state, &new_condition);
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
        let automaton = RegularExpression::new("(abc|ac|aaa)")
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.match_string("abc"));
        assert!(automaton.match_string("ac"));
        assert!(automaton.match_string("aaa"));
        assert!(!automaton.match_string("abcd"));
        assert!(!automaton.match_string("ab"));
        assert!(!automaton.match_string("acc"));
        assert!(!automaton.match_string("a"));
        assert!(!automaton.match_string("aaaa"));
        assert!(!automaton.match_string("aa"));
        assert!(!automaton.match_string(""));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_2() -> Result<(), String> {
        let automaton = RegularExpression::new("(b?|b{2})")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(automaton.match_string(""));
        assert!(automaton.match_string("b"));
        assert!(automaton.match_string("bb"));
        assert!(!automaton.match_string("bbb"));
        assert!(!automaton.match_string("bbbb"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_3() -> Result<(), String> {
        let automaton = RegularExpression::new("((a|bc)*|d)")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(automaton.match_string(""));
        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("abcaaabcbc"));
        assert!(automaton.match_string("d"));
        assert!(!automaton.match_string("ad"));
        assert!(!automaton.match_string("abcd"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_4() -> Result<(), String> {
        let automaton = RegularExpression::new("(a+(ba+)*|ca*c)")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(automaton.match_string("cc"));
        assert!(automaton.match_string("caaac"));
        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("aababa"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_5() -> Result<(), String> {
        let automaton = RegularExpression::new("((aad|ads|a)*|q)")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(automaton.match_string("q"));
        assert!(automaton.match_string("aad"));
        assert!(automaton.match_string("ads"));
        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("aadadsaaa"));
        assert!(!automaton.match_string("aaaas"));
        assert!(!automaton.match_string("ad"));
        assert!(!automaton.match_string("adsq"));
        assert!(!automaton.match_string("qq"));
        Ok(())
    }
}
