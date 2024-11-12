use std::hash::BuildHasherDefault;

use crate::execution_profile::ExecutionProfile;

use super::*;

impl StateEliminationAutomaton<Range> {
    pub fn convert_to_regex(
        &self,
        execution_profile: &ExecutionProfile,
    ) -> Result<Option<RegularExpression>, EngineError> {
        if self.cyclic {
            return self.convert_graph_to_regex(execution_profile);
        }
        execution_profile.assert_not_timed_out()?;

        let mut regex_map: IntMap<usize, RegularExpression> = IntMap::with_capacity_and_hasher(
            self.get_number_of_states(),
            BuildHasherDefault::default(),
        );
        regex_map.insert(self.start_state, RegularExpression::new_empty_string());
        for from_state in self.states_topo_vec() {
            let current_regex = if let Some(current_regex) = regex_map.get(&from_state) {
                current_regex.clone()
            } else {
                RegularExpression::new_empty_string()
            };
            if let Some(transitions) = self.transitions.get(from_state) {
                for (to_state, transition) in transitions {
                    let transition_regex = match transition {
                        GraphTransition::Graph(graph) => {
                            if let Some(regex) = graph.convert_graph_to_regex(execution_profile)? {
                                regex
                            } else {
                                return Ok(None);
                            }
                        }
                        GraphTransition::Weight(range) => {
                            RegularExpression::Character(range.clone())
                        }
                        GraphTransition::Epsilon => RegularExpression::new_empty_string(),
                    };
                    let new_regex = current_regex.concat(&transition_regex, true);
                    match regex_map.entry(*to_state) {
                        Entry::Occupied(mut o) => {
                            o.insert(new_regex.union(o.get()).simplify());
                        }
                        Entry::Vacant(v) => {
                            v.insert(new_regex);
                        }
                    };
                }
            }
        }

        Ok(regex_map.get(&self.accept_state).cloned())
    }

    fn convert_graph_to_regex(
        &self,
        execution_profile: &ExecutionProfile,
    ) -> Result<Option<RegularExpression>, EngineError> {
        execution_profile.assert_not_timed_out()?;
        if let Some(regex) = self.convert_shape_dot_star(execution_profile)? {
            return Ok(Some(regex));
        } else if let Some(regex) = self.convert_shape_self_loop(execution_profile)? {
            return Ok(Some(regex));
        }
        Ok(None)
    }

    /// We try to idenfify the regex following the shape:
    /// A*B
    fn convert_shape_dot_star(
        &self,
        execution_profile: &ExecutionProfile,
    ) -> Result<Option<RegularExpression>, EngineError> {
        if self.get_number_of_states() < 2 {
            return Ok(None);
        }
        //self.to_dot();
        let mut dot_value =
            if let Some(dot_value) = self.get_transition(self.start_state, self.start_state) {
                if let Some(dot_value) = dot_value.get_weight() {
                    dot_value.clone()
                } else {
                    return Ok(None);
                }
            } else {
                return Ok(None);
            };

        for state in self.states_iter() {
            if state == self.start_state {
                continue;
            }
            let weight = if let Some(weight) = self.get_transition(state, self.start_state) {
                if let Some(weight) = weight.get_weight() {
                    weight
                } else {
                    return Ok(None);
                }
            } else if state == self.accept_state {
                continue;
            } else {
                return Ok(None);
            };

            if !dot_value.contains_all(weight) {
                return Ok(None);
            }
        }

        let mut graph = self.clone();

        for (from_state, transition) in graph.in_transitions_vec(graph.start_state) {
            let weight = if let Some(weight) = transition.get_weight() {
                weight
            } else {
                return Ok(None);
            };
            dot_value = dot_value.union(weight);
            graph.remove_transition(from_state, graph.start_state);
        }

        let mut worklist = VecDeque::new();
        let mut seen = IntSet::with_capacity(graph.get_number_of_states());

        worklist.push_back(graph.start_state);
        seen.insert(self.start_state);

        while let Some(from_state) = worklist.pop_front() {
            for to_state in graph.transitions_from_state_vec(&from_state) {
                let transition =
                    if let Some(transition) = graph.get_transition(from_state, to_state) {
                        transition
                    } else {
                        return Ok(None);
                    };
                let weight = if let Some(weight) = transition.get_weight() {
                    weight
                } else {
                    continue;
                };
                dot_value = dot_value.union(weight);
                if seen.contains(&to_state) {
                    if graph.accept_state != to_state || to_state == from_state {
                        graph.remove_transition(from_state, to_state);
                    }
                } else {
                    seen.insert(to_state);
                    worklist.push_back(to_state);
                }
            }
        }

        graph.add_transition_to(
            self.start_state,
            self.start_state,
            GraphTransition::Weight(dot_value),
        );

        graph.identify_and_apply_components()?;
        graph.convert_to_regex(execution_profile)
    }

    /// We try to identify the regex following the shape:
    /// A*B
    fn convert_shape_self_loop(
        &self,
        execution_profile: &ExecutionProfile,
    ) -> Result<Option<RegularExpression>, EngineError> {
        let mut graph = self.clone();

        graph.accept_state = graph.new_state();

        for (from_state, transition) in graph.in_transitions_vec(self.start_state) {
            graph.remove_transition(from_state, self.start_state);

            graph.add_transition_to(from_state, graph.accept_state, transition);
        }

        graph.identify_and_apply_components()?;

        let a_part = if let Some(a_part) = graph.convert_to_regex(execution_profile)? {
            a_part
        } else {
            return Ok(None);
        };

        let mut graph = self.clone();

        for (from_state, _) in graph.in_transitions_vec(self.start_state) {
            graph.remove_transition(from_state, self.start_state);
        }

        graph.identify_and_apply_components()?;
        let b_part = if let Some(b_part) = graph.convert_to_regex(execution_profile)? {
            b_part
        } else {
            return Ok(None);
        };

        let regex = a_part.repeat(0, None).concat(&b_part, true);

        Ok(Some(regex))
    }
}
