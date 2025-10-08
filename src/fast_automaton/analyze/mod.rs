use std::hash::BuildHasherDefault;

use crate::{cardinality::Cardinality, error::EngineError};

use super::*;

mod cardinality;
mod equivalence;
mod length;
mod subset;

impl FastAutomaton {
    /// Checks if the automaton matches the empty language.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.accept_states.is_empty()
    }

    /// Checks if the automaton matches all possible strings.
    #[inline]
    pub fn is_total(&self) -> bool {
        if self.accept_states.contains(&self.start_state) {
            if let Some(condition) = self.transitions[self.start_state].get(&self.start_state) {
                return condition.is_total();
            }
        }
        false
    }

    /// Checks if the automaton only matches the empty string `""`.
    #[inline]
    pub fn is_empty_string(&self) -> bool {
        self.accept_states.len() == 1
            && self.accept_states.contains(&self.start_state)
            && self.in_degree(self.start_state) == 0
    }

    /// Returns the set of all states reachable from the start state.
    pub fn get_reachable_states(&self) -> IntSet<State> {
        let mut states_map: IntMap<usize, IntSet<usize>> =
            IntMap::with_capacity_and_hasher(self.transitions.len(), BuildHasherDefault::default());
        for from_state in self.states() {
            for (condition, to_state) in self.transitions_from(from_state) {
                if condition.is_empty() {
                    continue;
                }
                match states_map.entry(*to_state) {
                    Entry::Occupied(mut o) => o.get_mut().insert(from_state),
                    Entry::Vacant(v) => {
                        let mut new_states = IntSet::default();
                        new_states.insert(from_state);
                        v.insert(new_states);
                        true
                    }
                };
            }
        }

        let mut worklist = VecDeque::from_iter(self.accept_states.iter().cloned());
        let mut live = self.accept_states.clone();
        while let Some(live_state) = worklist.pop_front() {
            if let Some(states) = states_map.get(&live_state) {
                for state in states {
                    if !live.contains(state) {
                        live.insert(*state);
                        worklist.push_back(*state);
                    }
                }
            }
        }

        live
    }

    pub(crate) fn get_spanning_bases(&self) -> Result<Vec<Condition>, EngineError> {
        self.spanning_set
            .get_spanning_ranges()
            .map(|range| Condition::from_range(range, &self.spanning_set))
            .collect()
    }
}

#[cfg(test)]
mod tests {

    use crate::fast_automaton::FastAutomaton;

    #[test]
    fn test_empty() -> Result<(), String> {
        assert!(!FastAutomaton::new_total().is_empty());
        assert!(!FastAutomaton::new_empty_string().is_empty());
        assert!(FastAutomaton::new_empty().is_empty());

        Ok(())
    }

    #[test]
    fn test_empty_string() -> Result<(), String> {
        assert!(!FastAutomaton::new_total().is_empty_string());
        assert!(FastAutomaton::new_empty_string().is_empty_string());
        assert!(!FastAutomaton::new_empty().is_empty_string());

        Ok(())
    }

    #[test]
    fn test_total() -> Result<(), String> {
        assert!(FastAutomaton::new_total().is_total());
        assert!(!FastAutomaton::new_empty_string().is_total());
        assert!(!FastAutomaton::new_empty().is_total());

        Ok(())
    }
}
