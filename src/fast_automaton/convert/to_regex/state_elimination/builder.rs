use ahash::HashMapExt;

use super::*;

impl Gnfa {
    pub(super) fn from_automaton(automaton: &FastAutomaton) -> Result<Gnfa, EngineError> {
        let mut state_elimination_automaton = Gnfa {
            start_state: 0,  // start_state is not set yet
            accept_state: 0, // accept_state is not set yet
            transitions: Vec::with_capacity(automaton.number_of_states()),
            transitions_in: IntMap::with_capacity(automaton.number_of_states()),
            removed_states: IntSet::with_capacity(automaton.number_of_states()),
            empty: false,
        };

        if automaton.is_empty() {
            state_elimination_automaton.empty = true;
            return Ok(state_elimination_automaton);
        }

        let mut states_map = IntMap::with_capacity(automaton.number_of_states());

        for from_state in automaton.states() {
            let new_from_state = *states_map
                .entry(from_state)
                .or_insert_with(|| state_elimination_automaton.new_state());
            for (condition, to_state) in automaton.transitions_from(from_state) {
                let new_to_state = *states_map
                    .entry(*to_state)
                    .or_insert_with(|| state_elimination_automaton.new_state());

                let range = condition.to_range(automaton.spanning_set())?;
                state_elimination_automaton.add_transition(
                    new_from_state,
                    new_to_state,
                    RegularExpression::Character(range),
                );
            }
        }

        if automaton.in_degree(automaton.start_state()) == 0 {
            // If the start state does not have any incoming state we just set it
            state_elimination_automaton.start_state =
                *states_map.get(&automaton.start_state()).unwrap();
        } else {
            // If not we create a new state that will be the new start state
            state_elimination_automaton.start_state = state_elimination_automaton.new_state();

            let previous_start_state = *states_map.get(&automaton.start_state()).unwrap();
            // We add an empty string transition to the new start state
            state_elimination_automaton.add_transition(
                state_elimination_automaton.start_state,
                previous_start_state,
                RegularExpression::new_empty_string(),
            );
        }

        let accept_state = *automaton.accept_states().iter().next().unwrap();
        if automaton.accept_states().len() == 1 && automaton.out_degree(accept_state) == 0 {
            // If there is only one accept state we just set it
            state_elimination_automaton.accept_state = *states_map
                .get(automaton.accept_states().iter().next().unwrap())
                .unwrap();
        } else {
            // If not we create a new state that will be the new accept state
            state_elimination_automaton.accept_state = state_elimination_automaton.new_state();
            for accept_state in automaton.accept_states() {
                let accept_state = *states_map.get(accept_state).unwrap();
                // We add an empty string transition to the new accept state
                state_elimination_automaton.add_transition(
                    accept_state,
                    state_elimination_automaton.accept_state,
                    RegularExpression::new_empty_string(),
                );
            }
        }

        Ok(state_elimination_automaton)
    }

    fn new_state(&mut self) -> usize {
        if let Some(new_state) = self.removed_states.clone().iter().next() {
            self.removed_states.remove(new_state);
            self.transitions_in.insert(*new_state, IntSet::new());
            *new_state
        } else {
            self.transitions.push(IntMap::default());
            self.transitions_in
                .insert(self.transitions.len() - 1, IntSet::new());
            self.transitions.len() - 1
        }
    }

    #[inline]
    pub(super) fn has_state(&self, state: State) -> bool {
        !(state >= self.transitions.len() || self.removed_states.contains(&state))
    }

    #[inline]
    fn assert_state_exists(&self, state: State) {
        if !self.has_state(state) {
            panic!("The state {state} does not exist");
        }
    }

    pub(crate) fn add_transition(
        &mut self,
        from_state: State,
        to_state: State,
        transition: RegularExpression,
    ) {
        self.assert_state_exists(from_state);
        if from_state != to_state {
            self.assert_state_exists(to_state);
        }

        self.transitions_in
            .entry(to_state)
            .or_default()
            .insert(from_state);
        match self.transitions[from_state].entry(to_state) {
            Entry::Occupied(mut o) => {
                let merged = transition.union(o.get());
                *o.get_mut() = merged;
            }
            Entry::Vacant(v) => {
                v.insert(transition);
            }
        };
    }

    pub(super) fn remove_state(&mut self, state: State) {
        self.assert_state_exists(state);
        if self.start_state == state || self.accept_state == state {
            panic!(
                "Can not remove the state {state}, it is still used as start state or accept state."
            );
        }
        self.transitions_in.remove(&state);
        if self.transitions.len() - 1 == state {
            self.transitions.remove(state);

            // Compact tombstones that are now trailing; see
            // `FastAutomaton::remove_state`.
            let mut s = state;
            while s > 0 && self.removed_states.contains(&(s - 1)) {
                s -= 1;
                self.transitions.remove(s);
                self.removed_states.remove(&s);
            }
        } else {
            self.transitions[state].clear();
            self.removed_states.insert(state);
        }

        for transitions in self.transitions.iter_mut() {
            transitions.remove(&state);
        }
        for transitions in self.transitions_in.values_mut() {
            transitions.remove(&state);
        }
    }
}
