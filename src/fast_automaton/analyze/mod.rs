use std::hash::BuildHasherDefault;

use crate::{cardinality::Cardinality, error::EngineError};

use super::*;

mod cardinality;
mod equivalence;
mod length;
mod subset;

impl FastAutomaton {
    /// Checks if the automaton matches the empty language.
    ///
    /// Sound and complete: works on NFAs and non-minimal automata without
    /// requiring determinization or minimization. O(V + E) worst case, with
    /// O(1) fast paths for the common cases.
    pub fn is_empty(&self) -> bool {
        if self.accept_states.is_empty() {
            return true;
        }
        if self.accept_states.contains(&self.start_state) {
            return false;
        }
        if self.minimal {
            // A minimal automaton with at least one accept state has a
            // non-empty language (minimization prunes dead accepts).
            return false;
        }

        // Forward BFS from `start_state`; stop on first accept hit.
        let mut visited = IntSet::default();
        let mut worklist = VecDeque::new();
        visited.insert(self.start_state);
        worklist.push_back(self.start_state);

        while let Some(s) = worklist.pop_front() {
            for (cond, to) in self.transitions_from(s) {
                if cond.is_empty() {
                    continue;
                }
                if self.accept_states.contains(to) {
                    return false;
                }
                if visited.insert(*to) {
                    worklist.push_back(*to);
                }
            }
        }
        true
    }

    /// Checks if the automaton matches all possible strings.
    ///
    /// Sound and complete for **deterministic** automata: a DFA's language
    /// equals Σ\* iff every reachable state is accepting AND its outgoing
    /// conditions union to Σ. For NFAs this is sound but conservative:
    /// alternative paths may cover a character that no single reachable
    /// state covers, so callers that need an exact answer on an NFA should
    /// determinize first.
    ///
    /// O(V + E) plus one condition-union per outgoing transition.
    pub fn is_total(&self) -> bool {
        let mut visited = IntSet::default();
        let mut worklist = VecDeque::new();
        visited.insert(self.start_state);
        worklist.push_back(self.start_state);

        while let Some(s) = worklist.pop_front() {
            if !self.accept_states.contains(&s) {
                return false;
            }
            let mut covered = Condition::empty(&self.spanning_set);
            for (cond, to) in self.transitions_from(s) {
                if cond.is_empty() {
                    continue;
                }
                covered = covered.union(cond);
                if visited.insert(*to) {
                    worklist.push_back(*to);
                }
            }
            if !covered.is_total() {
                return false;
            }
        }
        true
    }

    /// Checks if the automaton only matches the empty string `""`.
    ///
    /// Sound and complete on any automaton (DFA or NFA): the language equals
    /// `{""}` iff start is accepting AND no state reachable from start by at
    /// least one non-empty transition is, or can reach, an accept state.
    /// O(V + E).
    pub fn is_empty_string(&self) -> bool {
        if !self.accept_states.contains(&self.start_state) {
            return false;
        }

        let mut visited = IntSet::default();
        let mut worklist = VecDeque::new();

        // Seed with states reachable in exactly one non-empty step from start.
        for (cond, to) in self.transitions_from(self.start_state) {
            if cond.is_empty() {
                continue;
            }
            if visited.insert(*to) {
                worklist.push_back(*to);
            }
        }

        while let Some(s) = worklist.pop_front() {
            if self.accept_states.contains(&s) {
                return false;
            }
            for (cond, to) in self.transitions_from(s) {
                if cond.is_empty() {
                    continue;
                }
                if visited.insert(*to) {
                    worklist.push_back(*to);
                }
            }
        }
        true
    }

    /// Returns the states reachable **from the start state** by following
    /// non-empty transitions (the start state is always included).
    ///
    /// This is forward reachability. Contrast with [`Self::get_live_states`],
    /// which returns the states that can **reach an accept state**
    /// (co-reachability).
    pub(crate) fn forward_reachable_states(&self) -> IntSet<State> {
        let mut visited = IntSet::default();
        let mut worklist = VecDeque::new();
        visited.insert(self.start_state);
        worklist.push_back(self.start_state);
        while let Some(s) = worklist.pop_front() {
            for (condition, to_state) in self.transitions_from(s) {
                if condition.is_empty() {
                    continue;
                }
                if visited.insert(*to_state) {
                    worklist.push_back(*to_state);
                }
            }
        }
        visited
    }

    /// Returns the "live" (co-reachable) states: those that can **reach an
    /// accept state** by following non-empty transitions. Computed by a reverse
    /// traversal from the accept states.
    ///
    /// This is co-reachability; note it is *not* the set of states reachable
    /// from the start state.
    pub fn get_live_states(&self) -> IntSet<State> {
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

    /// Returns one [`Condition`] per base of the spanning set, including the
    /// "rest" range when it is non-empty.
    ///
    /// The bases must partition the whole alphabet Σ: subset construction
    /// ([`determinize`](Self::determinize)) and Hopcroft partitioning
    /// ([`minimize`](Self::minimize)) iterate them and would otherwise silently
    /// drop transitions whose condition lies in the "rest" range. (For a
    /// spanning set with an empty rest this is exactly the spanning ranges, so
    /// well-formed automata are unaffected.)
    pub fn get_spanning_bases(&self) -> Result<Vec<Condition>, EngineError> {
        self.spanning_set
            .get_spanning_ranges_with_rest()
            .iter()
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
