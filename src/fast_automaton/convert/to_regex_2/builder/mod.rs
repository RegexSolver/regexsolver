use super::*;

mod scc;

impl StateEliminationAutomaton<Range> {
    pub fn new(automaton: &FastAutomaton) -> Result<Option<Self>, EngineError> {
        if automaton.is_empty() {
            return Ok(None);
        }

        let mut state_elimination_automaton = StateEliminationAutomaton {
            start_state: 0,  // start_state is not set yet
            accept_state: 0, // accept_state is not set yet
            transitions: Vec::with_capacity(automaton.get_number_of_states()),
            transitions_in: IntMap::with_capacity(automaton.get_number_of_states()),
            removed_states: IntSet::new(),
            cyclic: false,
        };

        let mut states_map = IntMap::with_capacity(automaton.get_number_of_states());

        for from_state in automaton.transitions_iter() {
            let new_from_state = *states_map
                .entry(from_state)
                .or_insert_with(|| state_elimination_automaton.new_state());
            for (to_state, condition) in
                automaton.transitions_from_state_enumerate_into_iter(&from_state)
            {
                let new_to_state = *states_map
                    .entry(to_state)
                    .or_insert_with(|| state_elimination_automaton.new_state());

                state_elimination_automaton.add_transition_to(
                    new_from_state,
                    new_to_state,
                    GraphTransition::Weight(condition.to_range(automaton.get_spanning_set())?),
                );
            }
        }

        state_elimination_automaton.start_state =
            *states_map.get(&automaton.get_start_state()).unwrap(); // We finally set start_state

        if automaton.get_accept_states().len() == 1 {
            // If there is only one accept state with just set it
            state_elimination_automaton.accept_state = *states_map
                .get(automaton.get_accept_states().iter().next().unwrap())
                .unwrap();
        } else {
            // If not we create a new state that will be the new accept state
            state_elimination_automaton.accept_state = state_elimination_automaton.new_state();
            for accept_state in automaton.get_accept_states() {
                let accept_state = *states_map.get(accept_state).unwrap();
                // We add an empty string transition to the new accept state
                state_elimination_automaton.add_transition_to(
                    accept_state,
                    state_elimination_automaton.accept_state,
                    GraphTransition::Epsilon,
                );
            }
        }
        state_elimination_automaton.identify_and_apply_components()?;
        Ok(Some(state_elimination_automaton))
    }

    pub fn new_state(&mut self) -> usize {
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
    pub fn has_state(&self, state: State) -> bool {
        !(state >= self.transitions.len() || self.removed_states.contains(&state))
    }

    #[inline]
    fn assert_state_exists(&self, state: State) {
        if !self.has_state(state) {
            panic!("The state {} does not exist", state);
        }
    }

    pub fn add_transition_to(
        &mut self,
        from_state: State,
        to_state: State,
        transition: GraphTransition<Range>,
    ) {
        self.assert_state_exists(from_state);
        if from_state != to_state {
            self.assert_state_exists(to_state);
        }

        self.transitions_in
            .entry(to_state)
            .or_insert(IntSet::new())
            .insert(from_state);
        match self.transitions[from_state].entry(to_state) {
            Entry::Occupied(mut o) => {
                if let (GraphTransition::Weight(current_regex), GraphTransition::Weight(regex)) =
                    (o.get(), transition)
                {
                    o.insert(GraphTransition::Weight(current_regex.union(&regex)));
                } else {
                    panic!("Cannot add transition");
                }
            }
            Entry::Vacant(v) => {
                v.insert(transition);
            }
        };
    }

    pub fn remove_state(&mut self, state: State) {
        self.assert_state_exists(state);
        if self.start_state == state || self.accept_state == state {
            panic!(
                "Can not remove the state {}, it is still used as start state or accept state.",
                state
            );
        }
        self.transitions_in.remove(&state);
        if self.transitions.len() - 1 == state {
            self.transitions.remove(state);

            let mut s = state;
            while self.removed_states.contains(&s) {
                self.transitions.remove(s);
                self.removed_states.remove(&s);
                s -= 1;
            }
        } else {
            self.transitions[state].clear();
            self.removed_states.insert(state);
        }

        for transitions in self.transitions.iter_mut() {
            transitions.remove(&state);
        }
        for (_, transitions) in self.transitions_in.iter_mut() {
            transitions.remove(&state);
        }
    }

    pub fn remove_transition(&mut self, from_state: State, to_state: State) {
        self.assert_state_exists(from_state);
        if from_state != to_state {
            self.assert_state_exists(to_state);
        }

        if let Some(from_states) = self.transitions_in.get_mut(&to_state) {
            from_states.remove(&from_state);
        }

        self.transitions[from_state].remove(&to_state);
    }

    pub fn get_transition(&self, from_state: State, to_state: State) -> Option<&GraphTransition<Range>> {
        self.transitions.get(from_state)?.get(&to_state)
    }
}
