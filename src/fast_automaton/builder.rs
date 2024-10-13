use crate::error::EngineError;

use super::*;

impl FastAutomaton {
    #[inline]
    pub fn new_empty() -> Self {
        Self {
            transitions: vec![Transitions::default()],
            transitions_in: IntMap::default(),
            start_state: 0,
            accept_states: IntSet::default(),
            removed_states: IntSet::default(),
            spanning_set: SpanningSet::new_empty(),
            deterministic: true,
            cyclic: false,
        }
    }

    #[inline]
    pub fn new_empty_string() -> Self {
        let mut automaton = Self::new_empty();
        automaton.accept(automaton.start_state);
        automaton
    }

    #[inline]
    pub fn new_total() -> Self {
        let mut automaton: FastAutomaton = Self::new_empty();
        automaton.spanning_set = SpanningSet::new_total();
        automaton.accept(automaton.start_state);
        automaton.add_transition_to(
            0,
            0,
            &Condition::total(&automaton.spanning_set),
        );
        automaton
    }

    #[inline]
    pub fn make_empty(&mut self) {
        self.apply_model(&Self::new_empty())
    }

    #[inline]
    pub fn make_total(&mut self) {
        self.apply_model(&Self::new_total())
    }

    pub fn make_from_range(range: &Range) -> Result<Self, EngineError> {
        let mut automaton = Self::new_empty();
        if range.is_empty() {
            return Ok(automaton);
        }
        let new_state = automaton.new_state();

        let spanning_set = SpanningSet::compute_spanning_set(&[range.clone()]);
        let condition = Condition::from_range(range, &spanning_set)?;
        automaton.spanning_set = spanning_set;
        automaton.add_transition_to(0, new_state, &condition);
        automaton.accept(new_state);
        Ok(automaton)
    }

    pub fn apply_new_spanning_set(
        &mut self,
        new_spanning_set: &SpanningSet,
    ) -> Result<(), EngineError> {
        if new_spanning_set == &self.spanning_set {
            return Ok(());
        }
        for from_state in &self.transitions_vec() {
            for to_state in self.transitions_from_state(from_state) {
                match self.transitions[*from_state].entry(to_state) {
                    Entry::Occupied(mut o) => {
                        o.insert(
                            o.get()
                                .project_to(&self.spanning_set, new_spanning_set)?,
                        );
                    }
                    Entry::Vacant(_) => {}
                };
            }
        }
        self.spanning_set = new_spanning_set.clone();
        Ok(())
    }

    #[inline]
    pub fn apply_model(&mut self, model: &FastAutomaton) {
        self.transitions = model.transitions.clone();
        self.start_state = model.start_state;
        self.accept_states = model.accept_states.clone();
        self.removed_states = model.removed_states.clone();
        self.spanning_set = model.spanning_set.clone();
        self.deterministic = model.deterministic;
        self.cyclic = model.cyclic;
    }

    #[inline]
    pub fn new_state(&mut self) -> State {
        if let Some(new_state) = self.removed_states.clone().iter().next() {
            self.removed_states.remove(new_state);
            *new_state
        } else {
            self.transitions.push(Transitions::default());
            self.transitions.len() - 1
        }
    }

    #[inline]
    pub fn accept(&mut self, state: State) {
        self.assert_state_exists(state);
        self.accept_states.insert(state);
    }

    pub fn add_transition_to(&mut self, from_state: State, to_state: State, new_cond: &Condition) {
        self.assert_state_exists(from_state);
        if from_state != to_state {
            self.assert_state_exists(to_state);
        }
        if new_cond.is_empty() {
            return;
        }

        if self.deterministic {
            let mut deterministic = true;
            for (state, condition) in self.transitions_from_state_enumerate_iter(&from_state) {
                if state == &to_state {
                    continue;
                }
                if condition.has_intersection(new_cond) {
                    deterministic = false;
                    break;
                }
            }
            self.deterministic = deterministic;
        }

        self.transitions_in
            .entry(to_state)
            .or_default()
            .insert(from_state);
        match self.transitions[from_state].entry(to_state) {
            Entry::Occupied(mut o) => {
                o.insert(o.get().union(new_cond));
            }
            Entry::Vacant(v) => {
                v.insert(new_cond.clone());
            }
        };
    }

    pub fn add_epsilon(&mut self, from_state: State, to_state: State) {
        if from_state == to_state {
            return;
        }
        self.assert_state_exists(from_state);
        self.assert_state_exists(to_state);
        if self.accept_states.contains(&to_state) {
            self.accept_states.insert(from_state);
        }

        let transitions_to: Vec<_> = self.transitions_from_state_into_iter(&to_state).collect();

        for (state, cond) in transitions_to {
            if self.deterministic {
                let mut deterministic = true;
                for (s, c) in self.transitions_from_state_enumerate_iter(&from_state) {
                    if state == *s {
                        continue;
                    }
                    if c.has_intersection(&cond) {
                        deterministic = false;
                        break;
                    }
                }
                self.deterministic = deterministic;
            }
            self.transitions_in
                .entry(state)
                .or_default()
                .insert(from_state);
            match self.transitions[from_state].entry(state) {
                Entry::Occupied(mut o) => {
                    o.insert(o.get().union(&cond));
                }
                Entry::Vacant(v) => {
                    v.insert(cond);
                }
            };
        }
    }

    pub fn remove_state(&mut self, state: State) {
        self.assert_state_exists(state);
        if self.start_state == state {
            panic!(
                "Can not remove the state {}, it is still used as start state.",
                state
            );
        }
        self.accept_states.remove(&state);
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

    pub fn remove_states(&mut self, states: &IntSet<State>) {
        self.accept_states.retain(|e| !states.contains(e));

        let mut states_to_remove = Vec::with_capacity(states.len());

        for &state in states {
            if self.start_state == state {
                panic!(
                    "Can not remove the state {}, it is still used as start state.",
                    state
                );
            }
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
            states_to_remove.push(state);
        }
        if states_to_remove.is_empty() {
            return;
        }

        for transitions in self.transitions.iter_mut() {
            for state in &states_to_remove {
                if transitions.is_empty() {
                    break;
                }

                transitions.remove(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    #[test]
    fn test_regex_build_deterministic_automaton() -> Result<(), String> {
        assert_regex_build_deterministic_automaton("...", true);
        assert_regex_build_deterministic_automaton(".*", true);
        assert_regex_build_deterministic_automaton(".*abc", false);
        assert_regex_build_deterministic_automaton(".{12}abc", true);
        assert_regex_build_deterministic_automaton(".{12,13}abc", false);
        Ok(())
    }

    fn assert_regex_build_deterministic_automaton(regex: &str, deterministic: bool) {
        let automaton = RegularExpression::new(regex)
            .unwrap()
            .to_automaton()
            .unwrap();
        assert_eq!(deterministic, automaton.is_determinitic());
    }
}
