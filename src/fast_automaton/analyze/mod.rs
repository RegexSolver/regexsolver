use std::hash::BuildHasherDefault;

use crate::{cardinality::Cardinality, error::EngineError};

use super::*;

mod cardinality;
mod equivalence;
mod length;
mod subset;

impl FastAutomaton {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.accept_states.is_empty()
    }

    #[inline]
    pub fn is_total(&self) -> bool {
        if self.accept_states.contains(&self.start_state) {
            if let Some(condition) = self.transitions[self.start_state].get(&self.start_state) {
                return condition.is_total();
            }
        }
        false
    }

    pub fn get_reacheable_states(&self) -> IntSet<State> {
        let mut states_map: IntMap<usize, IntSet<usize>> =
            IntMap::with_capacity_and_hasher(self.transitions.len(), BuildHasherDefault::default());
        for from_state in self.transitions_iter() {
            for (to_state, transition) in self.transitions_from_state_enumerate_iter(&from_state) {
                if transition.is_empty() {
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

    pub fn get_bases(&self) -> Result<Vec<Condition>, EngineError> {
        self.used_bases.get_bases().map(|range| {
            Condition::from_range(range, &self.used_bases)
        }).collect()
    }
}
