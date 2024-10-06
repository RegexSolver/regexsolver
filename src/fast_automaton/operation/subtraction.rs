use std::hash::BuildHasherDefault;

use crate::EngineError;

use super::*;

impl FastAutomaton {
    fn totalize(&mut self) -> Result<(), EngineError> {
        if !self.is_determinitic() {
            return Err(EngineError::AutomatonShouldBeDeterministic);
        }
        let crash_state = self.new_state();
        let mut transitions_to_crash_state: IntMap<State, Condition> =
            IntMap::with_capacity_and_hasher(
                self.get_number_of_states(),
                BuildHasherDefault::default(),
            );

        let mut ranges = Vec::with_capacity(self.get_number_of_states());
        for from_state in self.transitions_iter() {
            let mut new_condition = Condition::empty(&self.used_bases);
            for (_, condition) in self.transitions_from_state_enumerate_iter(&from_state) {
                new_condition = new_condition.union(condition);
                ranges.push(condition.to_range(self.get_used_bases())?);
            }

            new_condition = new_condition.complement();

            transitions_to_crash_state.insert(from_state, new_condition);
        }

        for (from_state, condition) in &transitions_to_crash_state {
            self.add_transition_to(*from_state, crash_state, condition);
            ranges.push(condition.to_range(self.get_used_bases())?);
        }

        let new_bases = UsedBases::compute_used_bases(&ranges);
        self.project_to_bases(&new_bases)?;

        if self.in_degree(crash_state) == 1 {
            self.remove_state(crash_state);
        }
        Ok(())
    }

    fn project_to_bases(&mut self, new_bases: &UsedBases) -> Result<(), EngineError> {
        let previous_bases = self.used_bases.clone();

        for from_state in self.transitions_vec() {
            for (_, condition) in self.transitions_from_state_enumerate_iter_mut(&from_state) {
                *condition = condition.project_to(&previous_bases, new_bases)?;
            }
        }

        self.used_bases = new_bases.clone();
        Ok(())
    }

    pub fn complement(&mut self) -> Result<(), EngineError> {
        self.totalize()?;

        let mut new_accept_states = IntSet::default();
        for state in self.transitions_iter() {
            if self.accept_states.contains(&state) {
                continue;
            }
            new_accept_states.insert(state);
        }

        self.accept_states = new_accept_states;
        Ok(())
    }

    pub fn subtraction(&self, other: &FastAutomaton) -> Result<FastAutomaton, EngineError> {
        let mut complement = other.clone();
        match complement.complement() {
            Ok(()) => self.intersection(&complement),
            Err(err) => Err(err),
        }
    }
}

#[cfg(test)]
mod tests {}
