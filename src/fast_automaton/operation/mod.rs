use std::{cmp, hash::Hasher};

use ahash::AHasher;

use super::*;

mod union;
mod concat;
mod determinize;
mod intersection;
mod difference;
mod repeat;

impl FastAutomaton {
    pub(crate) fn remove_dead_transitions(&mut self) {
        if !self.is_empty() {
            let reacheable_states = self.get_reachable_states();

            let mut dead_states = IntSet::default();
            for from_state in self.states() {
                if !reacheable_states.contains(&from_state) {
                    dead_states.insert(from_state);
                }
            }
            self.remove_states(&dead_states);
        } else {
            self.make_empty();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    #[test]
    fn test_remove_dead_states() -> Result<(), String> {
        let automaton1 = RegularExpression::parse("(abc|ac|aaa)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::parse("(abcd|ac|aba)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let intersection = automaton1.intersection(&automaton2).unwrap();
        assert_eq!(3, intersection.get_number_of_states());
        assert_eq!(3, intersection.get_reachable_states().len());
        Ok(())
    }
}
