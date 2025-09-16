use condition::converter::ConditionConverter;

use crate::error::EngineError;

use super::*;

impl FastAutomaton {
    /// Creates an automaton that matches the empty language.
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

    /// Creates an automaton that only matches the empty string `""`.
    #[inline]
    pub fn new_empty_string() -> Self {
        let mut automaton = Self::new_empty();
        automaton.accept(automaton.start_state);
        automaton
    }

    /// Creates an automaton that matches all possible strings.
    #[inline]
    pub fn new_total() -> Self {
        let mut automaton: FastAutomaton = Self::new_empty();
        automaton.spanning_set = SpanningSet::new_total();
        automaton.accept(automaton.start_state);
        automaton.add_transition(0, 0, &Condition::total(&automaton.spanning_set));
        automaton
    }

    /// Creates an automaton that matches one of the characters in the given [`CharRange`].
    pub fn new_from_range(range: &CharRange) -> Self {
        let mut automaton = Self::new_empty();
        if range.is_empty() {
            return automaton;
        }
        let new_state = automaton.new_state();

        let spanning_set = SpanningSet::compute_spanning_set(&[range.clone()]);
        let condition = Condition::from_range(range, &spanning_set).expect("The spanning set should be valid");
        automaton.spanning_set = spanning_set;
        automaton.add_transition(0, new_state, &condition);
        automaton.accept(new_state);
        automaton
    }

    /// Creates a new state and returns its identifier.
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

    /// Marks the provided state as an accepting (final) state.
    #[inline]
    pub fn accept(&mut self, state: State) {
        self.assert_state_exists(state);
        self.accept_states.insert(state);
    }

    /// Creates a new transition with the given condition; the condition must follow the automaton’s current spanning set.
    ///
    /// This method accepts a [`Condition`] rather than a raw character set. To build a [`Condition`], call:
    /// ```rust
    /// # use regexsolver::CharRange;
    /// # use regexsolver::fast_automaton::{condition::Condition, spanning_set::SpanningSet};
    /// # let range = CharRange::total();
    /// # let spanning_set = SpanningSet::new_total();
    /// Condition::from_range(&range, &spanning_set);
    /// ```
    /// where `spanning_set` is the automaton's current [`SpanningSet`]. The [`CharRange`] you pass must be fully covered by that spanning set. If it isn't, you have two options:
    ///
    /// 1. Merge an existing spanning set with another:
    /// ```rust
    /// # use regexsolver::fast_automaton::spanning_set::SpanningSet;
    /// # let old_set = SpanningSet::new_total();
    /// # let other_set = SpanningSet::new_total();
    /// let new_set = SpanningSet::merge(&old_set, &other_set);
    /// ```
    ///
    /// 2. Recompute from a list of ranges:
    /// ```rust
    /// # use regexsolver::CharRange;
    /// # use regexsolver::fast_automaton::spanning_set::SpanningSet;
    /// # let range_set1 = CharRange::total();
    /// # let range_set2 = CharRange::total();
    /// let new_set = SpanningSet::compute_spanning_set(&[range_set1, range_set2]);
    /// ```
    ///
    /// After constructing `new_set`, apply it to the automaton:
    /// ```rust
    /// # use regexsolver::fast_automaton::{FastAutomaton, spanning_set::SpanningSet};
    /// # let mut fast_automaton = FastAutomaton::new_total();
    /// # let new_set = SpanningSet::new_total();
    /// fast_automaton.apply_new_spanning_set(&new_set);
    /// ```
    ///
    /// This design allows us to perform unions, intersections, and complements of transition conditions in O(1) time, but it does add some complexity to automaton construction. For more details, you can check [this article](https://alexvbrdn.me/post/optimizing-transition-conditions-automaton-representation).
    pub fn add_transition(&mut self, from_state: State, to_state: State, new_cond: &Condition) {
        self.assert_state_exists(from_state);
        if from_state != to_state {
            self.assert_state_exists(to_state);
        }
        if new_cond.is_empty() {
            return;
        }

        if self.deterministic {
            let mut deterministic = true;
            for (condition, state) in self.transitions_from(from_state) {
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

    /// Creates a new epsilon transition between the two states.
    pub fn add_epsilon_transition(&mut self, from_state: State, to_state: State) {
        if from_state == to_state {
            return;
        }
        self.assert_state_exists(from_state);
        self.assert_state_exists(to_state);
        if self.accept_states.contains(&to_state) {
            self.accept_states.insert(from_state);
        }

        let transitions_to: Vec<_> = self
            .transitions_from(to_state)
            .map(|(cond, to_state)| (cond.clone(), *to_state))
            .collect();

        for (cond, state) in transitions_to {
            if self.deterministic {
                let mut deterministic = true;
                for (c, s) in self.transitions_from(from_state) {
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

    pub fn remove_transition(&mut self, from_state: State, to_state: State) {
        self.assert_state_exists(from_state);
        if from_state != to_state {
            self.assert_state_exists(to_state);
        }

        self.transitions_in
            .entry(to_state)
            .or_default()
            .remove(&from_state);
        self.transitions[from_state].remove(&to_state);
    }

    /// Removes the state and all its connected transitions; panics if it's a start state.
    pub fn remove_state(&mut self, state: State) {
        self.assert_state_exists(state);
        if self.start_state == state {
            panic!("Can not remove the state {state}, it is still used as start state.");
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

    /// Remove the provided states from the automaton. Remove all the transitions they are connected to. Panic if one of the state is used as a start state.
    pub fn remove_states(&mut self, states: &IntSet<State>) {
        self.accept_states.retain(|e| !states.contains(e));

        let mut states_to_remove = Vec::with_capacity(states.len());

        for &state in states {
            if self.start_state == state {
                panic!("Can not remove the state {state}, it is still used as start state.");
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

    /// Applies the provided spanning set and projects all existing conditions onto it.
    pub fn apply_new_spanning_set(
        &mut self,
        new_spanning_set: &SpanningSet,
    ) -> Result<(), EngineError> {
        if new_spanning_set == &self.spanning_set {
            return Ok(());
        }
        let condition_converter = ConditionConverter::new(&self.spanning_set, new_spanning_set)?;
        for from_state in &self.states_vec() {
            for to_state in self.direct_states_vec(from_state) {
                match self.transitions[*from_state].entry(to_state) {
                    Entry::Occupied(mut o) => {
                        o.insert(condition_converter.convert(o.get())?);
                    }
                    Entry::Vacant(_) => {}
                };
            }
        }
        self.spanning_set = new_spanning_set.clone();
        Ok(())
    }

    #[inline]
    pub(crate) fn make_empty(&mut self) {
        self.apply_model(&Self::new_empty())
    }

    #[inline]
    pub(crate) fn make_total(&mut self) {
        self.apply_model(&Self::new_total())
    }

    #[inline]
    pub(crate) fn apply_model(&mut self, model: &FastAutomaton) {
        self.transitions = model.transitions.clone();
        self.transitions_in = model.transitions_in.clone();
        self.start_state = model.start_state;
        self.accept_states = model.accept_states.clone();
        self.removed_states = model.removed_states.clone();
        self.spanning_set = model.spanning_set.clone();
        self.deterministic = model.deterministic;
        self.cyclic = model.cyclic;
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
        let automaton = RegularExpression::parse(regex, false)
            .unwrap()
            .to_automaton()
            .unwrap();
        assert_eq!(deterministic, automaton.is_determinitic());
    }
}
