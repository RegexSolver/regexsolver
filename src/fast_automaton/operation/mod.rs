use std::cmp;

use super::*;

mod concat;
mod determinize;
mod difference;
mod intersection;
mod minimize;
mod repeat;
mod union;

impl FastAutomaton {
    /// Removes "dead" states (those that cannot reach any accept state), since
    /// they never contribute to the language. If the language is empty the whole
    /// automaton collapses to the canonical empty automaton.
    pub fn remove_dead_states(&mut self) {
        if !self.is_empty() {
            let live_states = self.live_states();

            let mut dead_states = IntSet::default();
            for from_state in self.states() {
                if !live_states.contains(&from_state) {
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
        assert_eq!(3, intersection.number_of_states());
        assert_eq!(3, intersection.live_states().len());
        Ok(())
    }
}
