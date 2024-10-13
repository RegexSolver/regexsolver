use crate::Range;
use ahash::{AHashMap, HashSetExt};
use condition::Condition;
use regex_charclass::CharacterClass;
use spanning_set::SpanningSet;
use std::collections::hash_map::Entry;
use std::collections::VecDeque;
use std::fmt::Display;

use crate::tokenizer::Tokenizer;
use crate::{IntMap, IntSet};

pub(crate) type State = usize;
pub(crate) type Transitions = IntMap<State, Condition>;

mod analyze;
mod builder;
pub mod condition;
mod convert;
mod generate;
mod operation;
mod serializer;
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
        for from_state in self.transitions_iter() {
            write!(sb, "\t{}", from_state)?;
            if self.accept_states.contains(&from_state) {
                writeln!(sb, "\t[shape=doublecircle,label=\"{}\"];", from_state)?;
            } else {
                writeln!(sb, "\t[shape=circle,label=\"{}\"];", from_state)?;
            }

            if self.start_state == from_state {
                writeln!(sb, "\tinitial [shape=plaintext,label=\"\"];")?;
                writeln!(sb, "\tinitial -> {}", from_state)?;
            }
            for (to_state, cond) in self.transitions_from_state_enumerate_iter(&from_state) {
                writeln!(
                    sb,
                    "\t{} -> {} [label=\"{}\"]",
                    from_state,
                    to_state,
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
            panic!("The state {} does not exist", state);
        }
    }

    #[inline]
    pub fn in_degree(&self, state: State) -> usize {
        self.transitions_in
            .get(&state)
            .unwrap_or(&IntSet::new())
            .len()
    }

    #[inline]
    pub fn out_degree(&self, state: State) -> usize {
        self.transitions[state].len()
    }

    pub fn in_transitions(&self, state: State) -> Vec<(usize, Condition)> {
        let mut in_transitions = vec![];
        for from_state in self.transitions_in.get(&state).unwrap_or(&IntSet::new()) {
            for (_, condition) in self.transitions_from_state_enumerate_vec(from_state) {
                in_transitions.push((*from_state, condition));
            }
        }
        in_transitions
    }

    pub fn in_states(&self, state: State) -> IntSet<State> {
        self.transitions_in
            .get(&state)
            .unwrap_or(&IntSet::new())
            .clone()
    }

    #[inline]
    pub fn transitions_iter(&self) -> impl Iterator<Item = State> + '_ {
        (0..self.transitions.len()).filter(|s| !self.removed_states.contains(s))
    }

    #[inline]
    pub fn transitions_vec(&self) -> Vec<State> {
        self.transitions_iter().collect()
    }

    #[inline]
    pub fn transitions_from_state_enumerate_iter(
        &self,
        from_state: &State,
    ) -> impl Iterator<Item = (&State, &Condition)> {
        self.transitions[*from_state]
            .iter()
            .filter(|s| !self.removed_states.contains(s.0))
    }

    #[inline]
    pub fn transitions_from_state_enumerate_iter_mut(
        &mut self,
        from_state: &State,
    ) -> impl Iterator<Item = (&usize, &mut Condition)> {
        self.transitions[*from_state]
            .iter_mut()
            .filter(|s| !self.removed_states.contains(s.0))
    }

    #[inline]
    pub fn transitions_from_state_enumerate_vec(
        &self,
        from_state: &State,
    ) -> Vec<(State, Condition)> {
        self.transitions[*from_state]
            .iter()
            .map(|(s, c)| (*s, c.clone()))
            .filter(|s| !self.removed_states.contains(&s.0))
            .collect()
    }

    #[inline]
    pub fn does_transition_exists(&self, from_state: State, to_state: State) -> bool {
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

    #[inline]
    pub fn transitions_from_state_enumerate_into_iter(
        &self,
        from_state: &State,
    ) -> impl Iterator<Item = (State, Condition)> + '_ {
        self.transitions
            .get(*from_state) // Assume transitions is a map; adjust accordingly.
            .into_iter() // Creates an iterator over Option<&V>
            .flat_map(|transitions| transitions.iter()) // Flattens into Iterator<Item = &(State, Condition)>
            .filter(move |(state, _)| !self.removed_states.contains(state)) // Filters out removed states
            .map(|(state, condition)| (*state, condition.clone())) // Creates owned data; adjust if cloning is expensive
    }

    #[inline]
    pub fn transitions_from_state_iter(
        &self,
        from_state: &State,
    ) -> impl Iterator<Item = State> + '_ {
        self.transitions[*from_state]
            .keys()
            .cloned()
            .filter(|s| !self.removed_states.contains(s))
    }

    #[inline]
    pub fn transitions_from_state(&self, from_state: &State) -> Vec<State> {
        self.transitions_from_state_iter(from_state).collect()
    }

    #[inline]
    pub fn transitions_from_state_into_iter<'a>(
        &'a self,
        from_state: &State,
    ) -> impl Iterator<Item = (State, Condition)> + 'a {
        self.transitions[*from_state]
            .clone()
            .into_iter()
            .filter(|s| !self.removed_states.contains(&s.0))
    }

    #[inline]
    pub fn get_number_of_states(&self) -> usize {
        self.transitions.len() - self.removed_states.len()
    }

    #[inline]
    pub fn get_condition(&self, from_state: &State, to_state: &State) -> Option<&Condition> {
        self.transitions[*from_state].get(to_state)
    }

    #[inline]
    pub fn get_start_state(&self) -> State {
        self.start_state
    }

    #[inline]
    pub fn get_removed_states(&self) -> &IntSet<State> {
        &self.removed_states
    }

    #[inline]
    pub fn get_accept_states(&self) -> &IntSet<State> {
        &self.accept_states
    }

    #[inline]
    pub fn get_spanning_set(&self) -> &SpanningSet {
        &self.spanning_set
    }

    #[inline]
    pub fn is_accepted(&self, state: &State) -> bool {
        self.accept_states.contains(state)
    }

    #[inline]
    pub fn is_determinitic(&self) -> bool {
        self.deterministic
    }

    #[inline]
    pub fn is_cyclic(&self) -> bool {
        self.cyclic
    }

    #[inline]
    pub fn has_state(&self, state: State) -> bool {
        !(state >= self.transitions.len() || self.removed_states.contains(&state))
    }

    pub fn match_string(&self, input: &str) -> bool {
        let mut worklist = VecDeque::with_capacity(self.get_number_of_states());
        worklist.push_back((0, &self.start_state));

        while let Some((position, current_state)) = worklist.pop_back() {
            if input.len() == position {
                if self.accept_states.contains(current_state) {
                    return true;
                }
                continue;
            }
            let curr_char = input.chars().nth(position).unwrap() as u32;
            for (to_state, cond) in self.transitions_from_state_enumerate_iter(current_state) {
                if cond.has_character(&curr_char, &self.spanning_set).unwrap() {
                    if position + 1 == input.len() {
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

    #[inline]
    pub fn to_dot(&self) {
        println!("{}", self);
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
}
