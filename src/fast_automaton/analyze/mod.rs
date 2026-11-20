use crate::{cardinality::Cardinality, error::EngineError};

use super::*;

mod cardinality;
mod equivalence;
mod length;
#[cfg(test)]
mod reference_tests;
pub(super) mod state_sets;
mod subset;

impl FastAutomaton {
    /// Returns `true` if the automaton accepts no string.
    ///
    /// Exact whether the automaton is deterministic or not. Linear in its size.
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

    /// Returns `true` if the automaton is found to accept every string.
    ///
    /// Exact for a deterministic automaton. For a non-deterministic one, `true`
    /// is reliable, but `false` can be returned for an automaton that accepts
    /// every string: [`determinize`](Self::determinize) it first, or test
    /// `FastAutomaton::new_total().subset(&automaton)`.
    ///
    /// Linear in the size of the automaton.
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
                covered.union_with(cond);
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

    /// Returns `true` if the automaton accepts the empty string `""` and no
    /// other.
    ///
    /// Exact whether the automaton is deterministic or not. Linear in its size.
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
    /// This is forward reachability. Contrast with [`Self::live_states`],
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

    /// Returns the states from which an accept state can be reached. They are
    /// not the states that can be reached from the start state.
    ///
    /// Linear in the size of the automaton.
    pub fn live_states(&self) -> IntSet<State> {
        // Reverse BFS over the maintained `transitions_in` adjacency.
        // `transitions_in` doesn't filter empty-condition edges, so each
        // edge's condition is checked on traversal. `add_transition` adds
        // none, but `apply_new_spanning_set` can leave one, with a spanning
        // set that cannot express a condition. The lookup on `transitions`
        // also makes tombstoned predecessors fall out naturally.
        let mut worklist = VecDeque::from_iter(self.accept_states.iter().cloned());
        let mut live = self.accept_states.clone();
        while let Some(live_state) = worklist.pop_front() {
            let Some(predecessors) = self.transitions_in.get(&live_state) else {
                continue;
            };
            for &from_state in predecessors {
                if live.contains(&from_state) {
                    continue;
                }
                let takeable = self
                    .condition(from_state, live_state)
                    .is_some_and(|condition| !condition.is_empty());
                if takeable {
                    live.insert(from_state);
                    worklist.push_back(from_state);
                }
            }
        }

        live
    }

    /// Returns one [`Condition`] per base of the spanning set: one for each of
    /// its ranges, and one for its rest when it is not empty. Together, they
    /// cover every character.
    pub fn spanning_bases(&self) -> Result<Vec<Condition>, EngineError> {
        // Base `i` is by construction exactly bit `i` of a condition.
        Ok((0..self.spanning_set.spanning_ranges_with_rest_len())
            .map(|i| Condition::single_base(i, &self.spanning_set))
            .collect())
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

    // An empty-condition transition can't be taken, so a state whose only
    // path to an accept state goes through one is dead. `live_states` walks
    // `transitions_in`, which records such edges, so it must check the
    // condition instead of trusting the adjacency. `add_transition` ignores
    // an empty condition: the edge is stored directly.
    #[test]
    fn live_states_ignores_empty_condition_edges() {
        use crate::fast_automaton::condition::Condition;

        let mut a = FastAutomaton::new_empty();
        let s1 = a.new_state();
        let s2 = a.new_state();
        a.add_transition(0, s1, &Condition::total(a.spanning_set()));
        let empty = Condition::empty(a.spanning_set());
        a.transitions[s2].insert(s1, empty);
        a.transitions_in.entry(s1).or_default().insert(s2);
        a.accept(s1);
        assert!(a.has_transition(s2, s1));

        let live = a.live_states();
        assert!(live.contains(&0));
        assert!(live.contains(&s1));
        assert!(!live.contains(&s2));
    }
}
