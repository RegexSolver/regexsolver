use std::hash::BuildHasherDefault;

use crate::EngineError;

use super::*;

impl FastAutomaton {
    /// Totalize the automaton. Precondition: `self.deterministic` is true
    /// (the only caller, `complement`, determinizes first).
    fn totalize(&mut self) -> Result<(), EngineError> {
        debug_assert!(self.deterministic, "totalize requires a DFA");

        let crash_state = self.new_state();
        let mut transitions_to_crash_state: IntMap<State, Condition> =
            IntMap::with_capacity_and_hasher(
                self.get_number_of_states(),
                BuildHasherDefault::default(),
            );

        let mut ranges = Vec::with_capacity(self.get_number_of_states());
        for from_state in self.states() {
            let mut new_condition = Condition::empty(&self.spanning_set);
            for (condition, _) in self.transitions_from(from_state) {
                new_condition = new_condition.union(condition);
                ranges.push(condition.to_range(self.get_spanning_set())?);
            }

            new_condition = new_condition.complement();

            transitions_to_crash_state.insert(from_state, new_condition);
        }

        for (from_state, condition) in &transitions_to_crash_state {
            self.add_transition(*from_state, crash_state, condition);
            ranges.push(condition.to_range(self.get_spanning_set())?);
        }

        let new_spanning_set = SpanningSet::compute_spanning_set(&ranges);
        self.apply_new_spanning_set(&new_spanning_set)?;

        if self.in_degree(crash_state) == 1 {
            // Only the self-loop points to crash; nothing else needs it.
            self.remove_state(crash_state);
        }
        Ok(())
    }

    /// Complements the automaton.
    ///
    /// If `self` is non-deterministic, it is determinized in place first.
    pub fn complement(&mut self) -> Result<(), EngineError> {
        if !self.deterministic {
            *self = self.determinize()?.into_owned();
        }
        self.totalize()?;

        let mut new_accept_states = IntSet::default();
        for state in self.states() {
            if self.accept_states.contains(&state) {
                continue;
            }
            new_accept_states.insert(state);
        }

        self.accept_states = new_accept_states;
        Ok(())
    }

    /// Computes the difference between `self` and `other`.
    ///
    /// If `other` is non-deterministic, it is determinized first.
    pub fn difference(&self, other: &FastAutomaton) -> Result<FastAutomaton, EngineError> {
        let mut complement = other.determinize()?.into_owned();
        complement.complement()?;
        self.intersection(&complement)
    }
}

#[cfg(test)]
mod tests {
    use crate::fast_automaton::FastAutomaton;
    use crate::regex::RegularExpression;

    // `totalize` adds a `crash_state` with a total self-loop, so the complement
    // of a finite language is infinite (it matches arbitrarily long strings via
    // the crash-state loop).
    #[test]
    fn complement_of_finite_is_infinite() {
        let mut a = RegularExpression::parse("abc", false)
            .unwrap()
            .to_automaton()
            .unwrap();

        a.complement().unwrap();

        assert!(!a.is_match("abc"), "complement must not match 'abc'");
        assert!(a.is_match("x"));
        assert!(a.is_match("xx"));
        assert!(a.is_match("xxxxxxxxxx"));
    }

    // empty.complement() = Σ*.
    #[test]
    fn complement_of_empty_is_total() {
        let mut a = FastAutomaton::new_empty();
        a.complement().unwrap();
        assert!(a.is_total(), "complement of ∅ must be Σ*");
    }
}
