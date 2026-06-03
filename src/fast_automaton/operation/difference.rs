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
        } else {
            // crash_state has incoming edges from real states and a total
            // self-loop, so the automaton now contains a cycle.
            self.cyclic = true;
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

    // Regression: `totalize` adds a `crash_state` with a total self-loop
    // whenever the input isn't already total. That self-loop makes the
    // automaton cyclic. The flag is now updated when the crash state
    // survives.
    #[test]
    fn complement_updates_cyclic_flag() {
        let mut a = RegularExpression::parse("abc", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(!a.is_cyclic(), "precondition: 'abc' is acyclic");

        a.complement().unwrap();

        assert!(a.is_match("x"));
        assert!(a.is_match("xx"));
        assert!(a.is_match("xxxxxxxxxx"));

        assert!(
            a.is_cyclic(),
            "complement of finite acyclic must be cyclic (crash self-loop)"
        );
    }

    // Regression: empty.complement() = Σ* which is cyclic. Same root cause
    // as above plus `new_total` flag fix.
    #[test]
    fn complement_of_empty_is_cyclic() {
        let mut a = FastAutomaton::new_empty();
        a.complement().unwrap();
        assert!(a.is_cyclic(), "Σ* must report cyclic");
    }
}
