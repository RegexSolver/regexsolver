use crate::error::EngineError;
use ahash::{AHashMap, HashSetExt};
use condition::Condition;
use regex_charclass::CharacterClass;
use spanning_set::SpanningSet;
use std::collections::VecDeque;
use std::collections::hash_map::Entry;
use std::fmt::Display;

use super::*;

pub(crate) type Transitions = IntMap<State, Condition>;

/// The identifier of a state in a [`FastAutomaton`].
pub type State = usize;

mod analyze;
mod builder;
/// Transition labels: the bitvector [`Condition`] type over an automaton's
/// spanning set of disjoint character ranges.
pub mod condition;
mod convert;
mod generate;
mod operation;
/// The [`SpanningSet`]: an automaton's partition of the alphabet into disjoint
/// character ranges, over which transition conditions are defined.
pub mod spanning_set;

pub use generate::{GenerationOptions, GenerationOrder};

/// Represents a finite-state automaton.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "non-`_mut` operations return a new automaton"]
pub struct FastAutomaton {
    transitions: Vec<Transitions>,
    transitions_in: IntMap<usize, IntSet<usize>>,
    start_state: State,
    accept_states: IntSet<State>,
    removed_states: IntSet<State>,
    spanning_set: SpanningSet,
    deterministic: bool,
    minimal: bool,
}

/// Returned by [`FastAutomaton::try_add_transition`] when adding the requested
/// condition would turn a DFA into an NFA. The automaton is left unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeterminismLost;

impl std::fmt::Display for DeterminismLost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "adding the transition would introduce overlapping conditions"
        )
    }
}

impl std::error::Error for DeterminismLost {}

impl Display for FastAutomaton {
    fn fmt(&self, sb: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(sb, "digraph Automaton {{")?;
        writeln!(sb, "\trankdir = LR;")?;
        for from_state in self.states() {
            write!(sb, "\t{from_state}")?;
            if self.accept_states.contains(&from_state) {
                writeln!(sb, "\t[shape=doublecircle,label=\"{from_state}\"];")?;
            } else {
                writeln!(sb, "\t[shape=circle,label=\"{from_state}\"];")?;
            }

            if self.start_state == from_state {
                writeln!(sb, "\tinitial [shape=plaintext,label=\"\"];")?;
                writeln!(sb, "\tinitial -> {from_state}")?;
            }
            for (cond, to_state) in self.transitions_from(from_state) {
                // The automata most worth printing are the broken ones:
                // never panic mid-format, label desynced conditions instead.
                let label = match cond.to_range(&self.spanning_set) {
                    Ok(range) => range.to_regex().replace('\\', "\\\\").replace('"', "\\\""),
                    Err(_) => String::from("<invalid condition>"),
                };
                writeln!(sb, "\t{from_state} -> {to_state} [label=\"{label}\"]")?;
            }
        }
        write!(sb, "}}")
    }
}

impl FastAutomaton {
    #[inline]
    fn assert_state_exists(&self, state: State) {
        if !self.has_state(state) {
            panic!("The state {state} does not exist");
        }
    }

    /// Returns the number of transitions to the provided state.
    #[inline]
    pub fn in_degree(&self, state: State) -> usize {
        self.transitions_in
            .get(&state)
            .unwrap_or(&IntSet::new())
            .len()
    }

    /// Returns the number of transitions from the provided state.
    /// Returns `0` if the state does not exist.
    #[inline]
    pub fn out_degree(&self, state: State) -> usize {
        if !self.has_state(state) {
            return 0;
        }
        self.transitions[state].len()
    }

    /// Returns an iterator over the automaton’s states.
    #[inline]
    pub fn states(&self) -> impl Iterator<Item = State> + '_ {
        (0..self.transitions.len()).filter(|s| !self.removed_states.contains(s))
    }

    /// Returns a vector containing the automaton’s states.
    #[inline]
    pub fn states_vec(&self) -> Vec<State> {
        self.states().collect()
    }

    /// Returns an iterator over states directly reachable from the given state in one transition.
    /// Returns an empty iterator if the state does not exist.
    #[inline]
    pub fn direct_states(&self, state: State) -> impl Iterator<Item = State> + '_ {
        self.transitions.get(state).into_iter().flat_map(move |t| {
            t.keys()
                .copied()
                .filter(|s| !self.removed_states.contains(s))
        })
    }

    /// Returns a vector of states directly reachable from the given state in one transition.
    #[inline]
    pub fn direct_states_vec(&self, state: State) -> Vec<State> {
        self.direct_states(state).collect()
    }

    /// Returns a vector of transitions to the given state.
    pub fn transitions_to_vec(&self, state: State) -> Vec<(State, Condition)> {
        // Direct `(from, state)` lookups: scanning each predecessor's whole
        // out-list made this O(predecessors × out-degree), and `minimize`
        // builds its inverse-transition table through here.
        if !self.has_state(state) {
            return vec![];
        }
        let mut in_transitions = vec![];
        for from_state in self.transitions_in.get(&state).unwrap_or(&IntSet::new()) {
            if !self.has_state(*from_state) {
                continue;
            }
            if let Some(condition) = self.condition(*from_state, state) {
                in_transitions.push((*from_state, condition.clone()));
            }
        }
        in_transitions
    }

    /// Returns a vector of transitions from the given state.
    /// Returns an empty vector if the state does not exist.
    #[inline]
    pub fn transitions_from_vec(&self, state: State) -> Vec<(Condition, State)> {
        self.transitions
            .get(state)
            .map(|t| {
                t.iter()
                    .map(|(s, c)| (c.clone(), *s))
                    .filter(|s| !self.removed_states.contains(&s.1))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns an iterator over transitions from the given state.
    /// Returns an empty iterator if the state does not exist.
    #[inline]
    pub fn transitions_from(&self, state: State) -> impl Iterator<Item = (&Condition, &State)> {
        self.transitions.get(state).into_iter().flat_map(move |t| {
            t.iter()
                .map(|(s, c)| (c, s))
                .filter(|s| !self.removed_states.contains(s.1))
        })
    }

    /// Returns `true` if there is a directed transition from `from_state` to `to_state`.
    #[inline]
    pub fn has_transition(&self, from_state: State, to_state: State) -> bool {
        if !self.has_state(from_state) || !self.has_state(to_state) {
            return false;
        }
        self.transitions[from_state].contains_key(&to_state)
    }

    fn transitions_from_state_set(transitions: &[Transitions], from_state: State) -> Transitions {
        transitions[from_state].clone()
    }

    fn transitions_from_state_enumerate<'a>(
        transitions: &'a Transitions,
        removed_states: &IntSet<State>,
    ) -> Vec<(&'a State, &'a Condition)> {
        transitions
            .iter()
            .filter(|s| !removed_states.contains(s.0))
            .collect()
    }

    /// Returns the number of states in the automaton.
    #[inline]
    pub fn number_of_states(&self) -> usize {
        self.transitions.len() - self.removed_states.len()
    }

    /// Returns a reference to the condition of the directed transition between the two states, if any.
    /// Returns `None` if either state does not exist.
    #[inline]
    pub fn condition(&self, from_state: State, to_state: State) -> Option<&Condition> {
        self.transitions
            .get(from_state)
            .and_then(|t| t.get(&to_state))
    }

    /// Returns the start state.
    #[inline]
    pub fn start_state(&self) -> State {
        self.start_state
    }

    /// Returns a reference to the set of accept (final) states.
    #[inline]
    pub fn accept_states(&self) -> &IntSet<State> {
        &self.accept_states
    }

    /// Returns a reference to the automaton's spanning set.
    #[inline]
    pub fn spanning_set(&self) -> &SpanningSet {
        &self.spanning_set
    }

    /// Returns `true` if the given state is one of the accept states.
    #[inline]
    pub fn is_accepted(&self, state: State) -> bool {
        self.accept_states.contains(&state)
    }

    /// Returns `true` if the automaton is deterministic.
    ///
    /// Note: this flag degrades monotonically. Once `add_transition` introduces
    /// an overlapping condition, the flag flips to `false` and is not
    /// re-checked by `remove_transition` or `remove_state`. The automaton may
    /// in fact be deterministic again after such removals; call
    /// [`determinize`](Self::determinize) if you need a fresh DFA.
    #[inline]
    pub fn is_deterministic(&self) -> bool {
        self.deterministic
    }

    /// Returns `true` if the automaton is minimal.
    #[inline]
    pub fn is_minimal(&self) -> bool {
        self.minimal
    }

    /// Returns `true` if the automaton contains the given state.
    #[inline]
    pub fn has_state(&self, state: State) -> bool {
        !(state >= self.transitions.len() || self.removed_states.contains(&state))
    }

    /// Returns `true` if the automaton matches the given string.
    #[tracing::instrument(level = "debug", skip(self, string), fields(states = self.number_of_states(), string_len=string.len()))]
    pub fn is_match(&self, string: &str) -> bool {
        let mut current: IntSet<State> = IntSet::default();
        current.insert(self.start_state);

        let mut next: IntSet<State> = IntSet::default();
        for c in string.chars() {
            if current.is_empty() {
                return false;
            }
            let c_u32 = c as u32;
            next.clear();
            for &state in &current {
                for (cond, to_state) in self.transitions_from(state) {
                    // A condition/spanning-set mismatch is a broken internal
                    // invariant; make it loud in debug builds instead of
                    // silently treating the transition as non-matching.
                    let matches = match cond.has_character(&c_u32, &self.spanning_set) {
                        Ok(matches) => matches,
                        Err(error) => {
                            debug_assert!(
                                false,
                                "condition desynchronized from spanning set: {error}"
                            );
                            false
                        }
                    };
                    if matches {
                        next.insert(*to_state);
                    }
                }
            }
            std::mem::swap(&mut current, &mut next);
        }

        current.iter().any(|s| self.accept_states.contains(s))
    }

    /// Returns the automaton's DOT representation.
    #[inline]
    pub fn to_dot(&self) -> String {
        format!("{self}")
    }

    /// Prints the automaton's DOT representation.
    #[inline]
    pub fn print_dot(&self) {
        println!("{self}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() -> Result<(), String> {
        let automaton = FastAutomaton::new_empty();
        assert!(automaton.is_empty());
        assert!(!automaton.is_total());
        Ok(())
    }

    #[test]
    fn test_total() -> Result<(), String> {
        let automaton = FastAutomaton::new_total();
        assert!(!automaton.is_empty());
        assert!(automaton.is_total());
        Ok(())
    }

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn test_traits() -> Result<(), String> {
        assert_send::<FastAutomaton>();
        assert_sync::<FastAutomaton>();

        Ok(())
    }

    // Read-only query methods must return gracefully (0 / None / empty
    // iterator) on out-of-range or unknown states, not index out of bounds.
    #[test]
    fn out_degree_safe_on_unknown_state() {
        let a = FastAutomaton::new_total();
        assert_eq!(a.out_degree(999), 0);
    }

    #[test]
    fn condition_safe_on_unknown_state() {
        let a = FastAutomaton::new_total();
        assert!(a.condition(999, 0).is_none());
        assert!(a.condition(0, 999).is_none());
    }

    #[test]
    fn direct_states_safe_on_unknown_state() {
        let a = FastAutomaton::new_total();
        assert_eq!(a.direct_states(999).count(), 0);
        assert_eq!(a.transitions_from(999).count(), 0);
        assert!(a.transitions_from_vec(999).is_empty());
        assert!(a.direct_states_vec(999).is_empty());
    }

    #[test]
    #[should_panic(expected = "does not exist")]
    fn remove_states_panics_clearly_on_out_of_range() {
        let mut a = FastAutomaton::new_total();
        let mut states = IntSet::default();
        states.insert(999);
        a.remove_states(&states);
    }

    #[test]
    #[should_panic(expected = "does not exist")]
    fn remove_states_panics_clearly_on_tombstoned_id() {
        let mut a = FastAutomaton::new_empty();
        let s1 = a.new_state();
        let s2 = a.new_state();
        // Remove the trailing state so it becomes a tombstone.
        let mut first = IntSet::default();
        first.insert(s2);
        a.remove_states(&first);
        // Removing it again must fail cleanly, not panic on an OOB index.
        let mut again = IntSet::default();
        again.insert(s2);
        a.remove_states(&again);
        let _ = s1;
    }

    // A valid multi-state removal (including a trailing id) still works.
    #[test]
    fn remove_states_removes_valid_ids() {
        let mut a = FastAutomaton::new_empty();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let s3 = a.new_state();
        let mut states = IntSet::default();
        states.insert(s1);
        states.insert(s3); // trailing
        a.remove_states(&states);
        assert!(a.has_state(0));
        assert!(a.has_state(s2));
        assert!(!a.has_state(s1));
        assert!(!a.has_state(s3));
    }
}
