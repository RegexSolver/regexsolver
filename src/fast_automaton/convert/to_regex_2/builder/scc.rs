use super::*;

impl StateEliminationAutomaton<Range> {
    pub fn identify_and_apply_components(&mut self) -> Result<(), EngineError> {
        let mut index = 0;
        let mut stack = Vec::new();
        let mut indices = vec![-1; self.transitions.len()];
        let mut lowlink = vec![-1; self.transitions.len()];
        let mut on_stack = vec![false; self.transitions.len()];
        let mut scc = Vec::new();

        for state in self.states_iter() {
            if self.removed_states.contains(&state) {
                continue;
            }
            if indices[state] == -1 {
                self.strongconnect(
                    state,
                    &mut index,
                    &mut stack,
                    &mut indices,
                    &mut lowlink,
                    &mut on_stack,
                    &mut scc,
                );
            }
        }

        let scc = scc
            .into_iter()
            .filter(|states| {
                let first_state = states.iter().next().unwrap();
                let self_loop = if let Some(transitions_in) = self.transitions_in.get(first_state) {
                    transitions_in.contains(first_state)
                } else {
                    false
                };
                if states.len() == 1 && !self_loop {
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();

        for component in scc {
            self.build_component(&component)?;
        }

        self.cyclic = false;

        Ok(())
    }

    fn strongconnect(
        &self,
        v: usize,
        index: &mut usize,
        stack: &mut Vec<usize>,
        indices: &mut Vec<i32>,
        lowlink: &mut Vec<i32>,
        on_stack: &mut Vec<bool>,
        scc: &mut Vec<Vec<usize>>,
    ) {
        indices[v] = *index as i32;
        lowlink[v] = *index as i32;
        *index += 1;
        stack.push(v);
        on_stack[v] = true;

        if let Some(neighbors) = self.transitions.get(v) {
            for &w in neighbors.keys() {
                if indices[w] == -1 {
                    self.strongconnect(w, index, stack, indices, lowlink, on_stack, scc);
                    lowlink[v] = lowlink[v].min(lowlink[w]);
                } else if on_stack[w] {
                    lowlink[v] = lowlink[v].min(indices[w]);
                }
            }
        }

        if lowlink[v] == indices[v] {
            let mut component = Vec::new();
            while let Some(w) = stack.pop() {
                on_stack[w] = false;
                component.push(w);
                if w == v {
                    break;
                }
            }
            scc.push(component);
        }
    }

    fn build_component(&mut self, states: &[usize]) -> Result<(), EngineError> {
        let state_set = states.iter().copied().collect::<IntSet<usize>>();
        let mut start_states = IntMap::new();
        let mut accept_states = IntMap::new();

        let mut state_elimination_automaton = StateEliminationAutomaton {
            start_state: 0,  // start_state is not set yet
            accept_state: 0, // accept_state is not set yet
            transitions: Vec::with_capacity(states.len()),
            transitions_in: IntMap::with_capacity(states.len()),
            removed_states: IntSet::new(),
            cyclic: true,
        };

        let mut states_map = IntMap::with_capacity(states.len());
        for from_state in states {
            if *from_state == self.accept_state {
                self.accept_state = self.new_state();
                self.add_transition_to(*from_state, self.accept_state, GraphTransition::Epsilon);
            }
            if *from_state == self.start_state {
                self.start_state = self.new_state();
                self.add_transition_to(self.start_state, *from_state, GraphTransition::Epsilon);
            }
            let from_state_new = *states_map
                .entry(*from_state)
                .or_insert_with(|| state_elimination_automaton.new_state());
            for (to_state, transition) in self.transitions_from_state_enumerate_iter(from_state) {
                if !state_set.contains(to_state) {
                    accept_states
                        .entry(*to_state)
                        .or_insert_with(|| Vec::new())
                        .push((from_state_new, transition.clone()));
                    continue;
                }

                let to_state_new = *states_map
                    .entry(*to_state)
                    .or_insert_with(|| state_elimination_automaton.new_state());

                state_elimination_automaton.add_transition_to(
                    from_state_new,
                    to_state_new,
                    transition.clone(),
                );
            }

            for (parent_state, transition) in self.in_transitions_vec(*from_state) {
                if !state_set.contains(&parent_state) {
                    start_states
                        .entry(from_state_new)
                        .or_insert_with(|| Vec::new())
                        .push((parent_state, transition.clone()));
                }
            }
        }

        for state in states {
            self.remove_state(*state);
        }

        for (start_state, parent_states) in &start_states {
            for (parent_state, transition) in parent_states {
                let new_parent_state = if !transition.is_empty_string() {
                    let new_parent_state = self.new_state();

                    self.add_transition_to(*parent_state, new_parent_state, transition.clone());
                    new_parent_state
                } else {
                    *parent_state
                };
                for (target_state, accept_states_transition) in &accept_states {
                    let mut new_automaton = state_elimination_automaton.clone();

                    let target_state = if accept_states_transition.len() > 1 {
                        new_automaton.accept_state = new_automaton.new_state();
                        for (accept_state, transition) in accept_states_transition {
                            new_automaton.add_transition_to(
                                *accept_state,
                                new_automaton.accept_state,
                                transition.clone(),
                            );
                        }
                        *target_state
                    } else {
                        let (accept_state, transition) =
                            accept_states_transition.iter().next().unwrap();

                        new_automaton.accept_state = *accept_state;
                        if !transition.is_empty_string() {
                            let new_target_state = self.new_state();
                            self.add_transition_to(
                                new_target_state,
                                *target_state,
                                transition.clone(),
                            );
                            new_target_state
                        } else {
                            *target_state
                        }
                    };

                    new_automaton.start_state = *start_state;

                    self.add_transition_to(
                        new_parent_state,
                        target_state,
                        GraphTransition::Graph(new_automaton),
                    );
                }
            }
        }

        Ok(())
    }
}
