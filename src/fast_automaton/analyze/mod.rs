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
    /// conditions union to Σ. For NFAs this is sound but conservative —
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

    /// Recomputes from the transition graph whether the automaton contains
    /// a cycle. Use this to refresh the [`is_cyclic`](Self::is_cyclic) cache
    /// after operations that don't maintain it.
    ///
    /// Kahn's algorithm: a directed graph has a cycle iff topological sort
    /// fails to consume all nodes. O(V + E).
    pub(crate) fn detect_cyclic(&self) -> bool {
        let total = self.get_number_of_states();
        if total == 0 {
            return false;
        }

        let mut in_deg: IntMap<State, usize> = IntMap::default();
        for s in self.states() {
            in_deg.entry(s).or_insert(0);
            for t in self.direct_states(s) {
                *in_deg.entry(t).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<State> = in_deg
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(s, _)| *s)
            .collect();

        let mut processed = 0usize;
        while let Some(s) = queue.pop_front() {
            processed += 1;
            for t in self.direct_states(s) {
                if let Some(d) = in_deg.get_mut(&t) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(t);
                    }
                }
            }
        }

        processed != total
    }

    pub fn get_spanning_bases(&self) -> Result<Vec<Condition>, EngineError> {
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
