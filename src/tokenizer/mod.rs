use std::{cmp::Ordering, collections::VecDeque, vec};

use ahash::HashMapExt;
use crate::fast_automaton::spanning_set::SpanningSet;
use crate::Range;

use crate::{
    fast_automaton::{FastAutomaton, State},
    IntMap, IntSet,
};

use self::{range_tokenizer::RangeTokenizer, token::automaton_token::AutomatonToken};

mod embed_automaton;
mod embed_regex;
mod embed_regex_operations;
pub mod range_tokenizer;
pub mod token;

#[derive(Debug)]
pub struct Tokenizer<'a> {
    range_tokenizer: RangeTokenizer<'a>,
    automaton: &'a FastAutomaton,
    state_to_token: IntMap<usize, u16>,
}

impl Tokenizer<'_> {
    pub fn new(automaton: &FastAutomaton) -> Tokenizer<'_> {
        let mut worklist = VecDeque::with_capacity(automaton.get_number_of_states());
        let mut seen = IntSet::default();

        worklist.push_front(automaton.get_start_state());

        let mut state_counter: u16 = 0;
        let mut state_to_token = IntMap::with_capacity(automaton.get_number_of_states());

        while let Some(current_state) = worklist.pop_back() {
            if !seen.insert(current_state) {
                continue;
            }

            state_to_token.insert(current_state, state_counter);
            state_counter += 1;

            automaton
                .transitions_from_state_enumerate_iter(&current_state)
                .filter(|(_, c)| !c.is_empty())
                .for_each(|(to_state, _)| {
                    if !seen.contains(to_state) {
                        worklist.push_front(*to_state);
                    }
                });
        }

        Tokenizer {
            range_tokenizer: RangeTokenizer::new(automaton.get_spanning_set()),
            automaton,
            state_to_token,
        }
    }

    pub fn get_number_of_spanning_ranges(&self) -> usize {
        self.range_tokenizer.get_number_of_spanning_ranges()
    }

    pub fn get_spanning_set(&self) -> &SpanningSet {
        self.range_tokenizer.get_spanning_set()
    }
}

#[cfg(test)]
mod tests {}
