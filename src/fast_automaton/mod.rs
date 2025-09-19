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

/// The identifier of state in an [`FastAutomaton`]
pub type State = usize;

mod analyze;
mod builder;
pub mod condition;
mod convert;
mod generate;
mod operation;
#[cfg(feature = "serializable")]
pub mod serializer;
pub mod spanning_set;

/// Represent a finite state automaton.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastAutomaton {
    transitions: Vec<Transitions>,
    transitions_in: IntMap<usize, IntSet<usize>>,
    start_state: State,
    accept_states: IntSet<State>,
    removed_states: IntSet<State>,
    spanning_set: SpanningSet,
    deterministic: bool,
    cyclic: bool,
}

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
                writeln!(
                    sb,
                    "\t{from_state} -> {to_state} [label=\"{}\"]",
                    cond.to_range(&self.spanning_set)
                        .expect("Cannot convert condition to range.")
                        .to_regex()
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                )?;
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
    #[inline]
    pub fn out_degree(&self, state: State) -> usize {
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
    #[inline]
    pub fn direct_states(&self, state: &State) -> impl Iterator<Item = State> + '_ {
        self.transitions[*state]
            .keys()
            .cloned()
            .filter(|s| !self.removed_states.contains(s))
    }

    /// Returns a vector of states directly reachable from the given state in one transition.
    #[inline]
    pub fn direct_states_vec(&self, state: &State) -> Vec<State> {
        self.direct_states(state).collect()
    }

    /// Returns a vector of transitions to the given state.
    pub fn transitions_to_vec(&self, state: State) -> Vec<(State, Condition)> {
        let mut in_transitions = vec![];
        for from_state in self.transitions_in.get(&state).unwrap_or(&IntSet::new()) {
            for (condition, to_state) in self.transitions_from_vec(*from_state) {
                if to_state == state {
                    in_transitions.push((*from_state, condition));
                    break;
                }
            }
        }
        in_transitions
    }

    /// Returns a vector of transitions from the given state.
    #[inline]
    pub fn transitions_from_vec(&self, state: State) -> Vec<(Condition, State)> {
        self.transitions[state]
            .iter()
            .map(|(s, c)| (c.clone(), *s))
            .filter(|s| !self.removed_states.contains(&s.1))
            .collect()
    }

    /// Returns an iterator over transitions from the given state.
    #[inline]
    pub fn transitions_from(
        &self,
        state: State,
    ) -> impl Iterator<Item = (&Condition, &State)> {
        self.transitions[state]
            .iter()
            .map(|(s, c)| (c, s))
            .filter(|s| !self.removed_states.contains(s.1))
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
    pub fn get_number_of_states(&self) -> usize {
        self.transitions.len() - self.removed_states.len()
    }

    /// Returns a reference to the condition of the directed transition between the two states, if any.
    #[inline]
    pub fn get_condition(&self, from_state: State, to_state: State) -> Option<&Condition> {
        self.transitions[from_state].get(&to_state)
    }

    /// Returns the start state.
    #[inline]
    pub fn get_start_state(&self) -> State {
        self.start_state
    }

    /// Returns a reference to the set of accept (final) states.
    #[inline]
    pub fn get_accept_states(&self) -> &IntSet<State> {
        &self.accept_states
    }

    /// Returns a reference to the automaton's spanning set.
    #[inline]
    pub fn get_spanning_set(&self) -> &SpanningSet {
        &self.spanning_set
    }

    /// Returns `true` if the given state is one of the accept states.
    #[inline]
    pub fn is_accepted(&self, state: &State) -> bool {
        self.accept_states.contains(state)
    }

    /// Returns `true` if the automaton is deterministic.
    #[inline]
    pub fn is_deterministic(&self) -> bool {
        self.deterministic
    }

    /// Returns `true` if the automaton contains at least one cycle.
    #[inline]
    pub fn is_cyclic(&self) -> bool {
        self.cyclic
    }

    /// Returns `true` if the automaton contains the given state.
    #[inline]
    pub fn has_state(&self, state: State) -> bool {
        !(state >= self.transitions.len() || self.removed_states.contains(&state))
    }

    /// Returns `true` if the automaton matches the given string.
    pub fn is_match(&self, string: &str) -> bool {
        let mut worklist = VecDeque::with_capacity(self.get_number_of_states());
        worklist.push_back((0, &self.start_state));

        while let Some((position, current_state)) = worklist.pop_back() {
            if string.len() == position {
                if self.accept_states.contains(current_state) {
                    return true;
                }
                continue;
            }
            let curr_char = string.chars().nth(position).unwrap() as u32;
            for (cond, to_state) in self.transitions_from(*current_state) {
                if cond.has_character(&curr_char, &self.spanning_set).unwrap() {
                    if position + 1 == string.len() {
                        if self.accept_states.contains(to_state) {
                            return true;
                        }
                    } else {
                        worklist.push_back((position + 1, to_state));
                    }
                }
            }
        }
        false
    }

    /// Returns the automaton's DOT representation.
    #[inline]
    pub fn as_dot(&self) -> String {
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
}
