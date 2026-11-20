use condition::converter::ConditionConverter;

use crate::error::EngineError;

use super::*;

impl FastAutomaton {
    /// Creates an automaton that accepts no string.
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
            minimal: true,
        }
    }

    /// Creates an automaton that accepts the empty string `""` and no other.
    #[inline]
    pub fn new_empty_string() -> Self {
        let mut automaton = Self::new_empty();
        automaton.accept(automaton.start_state);
        automaton.minimal = true;
        automaton
    }

    /// Creates an automaton that accepts every string.
    #[inline]
    pub fn new_total() -> Self {
        let mut automaton: FastAutomaton = Self::new_empty();
        automaton.spanning_set = SpanningSet::new_total();
        automaton.accept(automaton.start_state);
        automaton.add_transition(0, 0, &Condition::total(&automaton.spanning_set));
        automaton.minimal = true;
        automaton
    }

    /// Creates an automaton that accepts the strings made of one character of
    /// `range`.
    pub fn new_from_range(range: &CharRange) -> Self {
        let mut automaton = Self::new_empty();
        if range.is_empty() {
            return automaton;
        }
        let new_state = automaton.new_state();

        let spanning_set = SpanningSet::compute_spanning_set(std::slice::from_ref(range));
        let condition =
            Condition::from_range(range, &spanning_set).expect("The spanning set should be valid");
        automaton.spanning_set = spanning_set;
        automaton.add_transition(0, new_state, &condition);
        automaton.accept(new_state);
        automaton.minimal = true;
        automaton
    }

    /// Adds a state and returns its identifier.
    #[inline]
    pub fn new_state(&mut self) -> State {
        self.minimal = false;
        if let Some(new_state) = self.removed_states.iter().next().copied() {
            self.removed_states.remove(&new_state);
            new_state
        } else {
            self.transitions.push(Transitions::default());
            self.transitions.len() - 1
        }
    }

    /// Makes `state` an accept state.
    ///
    /// # Panics
    ///
    /// Panics if `state` does not exist.
    #[inline]
    pub fn accept(&mut self, state: State) {
        self.assert_state_exists(state);
        self.minimal = false;
        self.accept_states.insert(state);
    }

    /// Makes `state` a state that does not accept.
    ///
    /// # Panics
    ///
    /// Panics if `state` does not exist.
    #[inline]
    pub fn unaccept(&mut self, state: State) {
        self.assert_state_exists(state);
        self.minimal = false;
        self.accept_states.remove(&state);
    }

    /// Adds a transition from `from_state` to `to_state` for the characters of
    /// `new_cond`, which must be expressed over the automaton's current
    /// [`SpanningSet`].
    ///
    /// If the two states already have a transition, the characters are added to
    /// it. An empty condition adds nothing.
    ///
    /// [`add_transition_from_range`](Self::add_transition_from_range) takes a
    /// [`CharRange`] and maintains the spanning set: use it unless you manage
    /// the spanning set yourself.
    ///
    /// # Building the condition
    ///
    /// [`Condition::from_range`] converts a [`CharRange`] that the spanning set
    /// can express:
    ///
    /// ```rust
    /// # use regexsolver::CharRange;
    /// # use regexsolver::fast_automaton::{condition::Condition, spanning_set::SpanningSet};
    /// # let range = CharRange::total();
    /// # let spanning_set = SpanningSet::new_total();
    /// Condition::from_range(&range, &spanning_set);
    /// ```
    ///
    /// When it cannot, compute a spanning set that can, with
    /// [`SpanningSet::merge`] or [`SpanningSet::compute_spanning_set`], and
    /// apply it to the automaton first:
    ///
    /// ```rust
    /// # use regexsolver::CharRange;
    /// # use regexsolver::fast_automaton::{FastAutomaton, spanning_set::SpanningSet};
    /// # let mut automaton = FastAutomaton::new_total();
    /// # let range = CharRange::total();
    /// let spanning_set = automaton
    ///     .spanning_set()
    ///     .merge(&SpanningSet::compute_spanning_set(&[range]));
    /// automaton.apply_new_spanning_set(&spanning_set).unwrap();
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if a state does not exist.
    pub fn add_transition(&mut self, from_state: State, to_state: State, new_cond: &Condition) {
        self.add_transition_cow(from_state, to_state, Cow::Borrowed(new_cond));
    }

    /// [`add_transition`](Self::add_transition) of a condition the caller
    /// has no further use for, which is stored instead of being cloned.
    pub(crate) fn add_owned_transition(
        &mut self,
        from_state: State,
        to_state: State,
        new_cond: Condition,
    ) {
        self.add_transition_cow(from_state, to_state, Cow::Owned(new_cond));
    }

    fn add_transition_cow(
        &mut self,
        from_state: State,
        to_state: State,
        new_cond: Cow<'_, Condition>,
    ) {
        self.assert_state_exists(from_state);
        if from_state != to_state {
            self.assert_state_exists(to_state);
        }
        if new_cond.is_empty() {
            return;
        }

        self.minimal = false;
        if self.deterministic {
            let mut deterministic = true;
            for (condition, state) in self.transitions_from(from_state) {
                if state == &to_state {
                    continue;
                }
                if condition.has_intersection(&new_cond) {
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
                o.get_mut().union_with(&new_cond);
            }
            Entry::Vacant(v) => {
                v.insert(new_cond.into_owned());
            }
        };
    }

    /// Adds a transition from `from_state` to `to_state` for the characters of
    /// `range`.
    ///
    /// If the two states already have a transition, the characters are added to
    /// it. An empty range adds nothing.
    ///
    /// When the automaton's spanning set cannot express `range`, it is
    /// extended, and the conditions of the existing transitions are converted
    /// to it: that takes time proportional to their number.
    ///
    /// # Panics
    ///
    /// Panics if a state does not exist and `range` is not empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use regexsolver::CharRange;
    /// use regexsolver::fast_automaton::FastAutomaton;
    /// use regex_charclass::char::Char;
    ///
    /// let mut automaton = FastAutomaton::new_empty();
    /// let s1 = automaton.new_state();
    /// automaton.accept(s1);
    ///
    /// let a_to_c = CharRange::new_from_range(Char::new('a')..=Char::new('c'));
    /// automaton.add_transition_from_range(0, s1, &a_to_c).unwrap();
    ///
    /// assert!(automaton.is_match("b"));
    /// assert!(!automaton.is_match("d"));
    /// ```
    pub fn add_transition_from_range(
        &mut self,
        from_state: State,
        to_state: State,
        range: &CharRange,
    ) -> Result<(), EngineError> {
        if range.is_empty() {
            return Ok(());
        }

        // Fast path: the range is exactly expressible in the current
        // spanning set. `Condition::from_range` alone cannot tell us that
        // (it silently drops partially-covered bases), so round-trip the
        // condition to check exactness.
        if let Ok(condition) = Condition::from_range(range, &self.spanning_set)
            && condition.to_range(&self.spanning_set)? == *range
        {
            self.add_transition(from_state, to_state, &condition);
            return Ok(());
        }

        // The range is not (fully) covered: extend the spanning set,
        // re-project the existing conditions, then add.
        let new_spanning_set =
            self.spanning_set
                .merge(&SpanningSet::compute_spanning_set(std::slice::from_ref(
                    range,
                )));
        self.apply_new_spanning_set(&new_spanning_set)?;

        let condition = Condition::from_range(range, &self.spanning_set)?;
        self.add_transition(from_state, to_state, &condition);
        Ok(())
    }

    /// Adds a transition as [`add_transition`](Self::add_transition) does,
    /// unless it would make a deterministic automaton non-deterministic.
    ///
    /// # Errors
    ///
    /// Returns [`DeterminismLost`] in that case, and leaves the automaton
    /// unchanged.
    ///
    /// # Panics
    ///
    /// Panics if a state does not exist.
    pub fn try_add_transition(
        &mut self,
        from_state: State,
        to_state: State,
        new_cond: &Condition,
    ) -> Result<(), super::DeterminismLost> {
        self.assert_state_exists(from_state);
        if from_state != to_state {
            self.assert_state_exists(to_state);
        }
        if new_cond.is_empty() {
            return Ok(());
        }
        if self.deterministic {
            for (condition, state) in self.transitions_from(from_state) {
                if *state == to_state {
                    continue;
                }
                if condition.has_intersection(new_cond) {
                    return Err(super::DeterminismLost);
                }
            }
        }
        self.add_transition(from_state, to_state, new_cond);
        Ok(())
    }

    /// Adds an epsilon transition from `from_state` to `to_state`: `from_state`
    /// gets a copy of the transitions of `to_state`, and becomes an accept
    /// state if `to_state` is one.
    ///
    /// The copy is made at the time of the call: transitions added to
    /// `to_state` afterwards are not copied. When you build an automaton step
    /// by step, add its epsilon transitions last.
    ///
    /// # Panics
    ///
    /// Panics if a state does not exist.
    pub fn add_epsilon_transition(&mut self, from_state: State, to_state: State) {
        if from_state == to_state {
            return;
        }
        self.assert_state_exists(from_state);
        self.assert_state_exists(to_state);

        self.minimal = false;

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
                    o.get_mut().union_with(&cond);
                }
                Entry::Vacant(v) => {
                    v.insert(cond);
                }
            };
        }
    }

    /// Removes the transition from `from_state` to `to_state`, if there is one.
    ///
    /// # Panics
    ///
    /// Panics if a state does not exist.
    pub fn remove_transition(&mut self, from_state: State, to_state: State) {
        self.assert_state_exists(from_state);
        if from_state != to_state {
            self.assert_state_exists(to_state);
        }

        self.minimal = false;

        self.transitions_in
            .entry(to_state)
            .or_default()
            .remove(&from_state);
        self.transitions[from_state].remove(&to_state);
    }

    /// Removes `state` and its transitions.
    ///
    /// # Panics
    ///
    /// Panics if `state` does not exist or is the start state.
    pub fn remove_state(&mut self, state: State) {
        self.assert_state_exists(state);
        if self.start_state == state {
            panic!("Can not remove the state {state}, it is still used as start state.");
        }
        self.minimal = false;
        self.accept_states.remove(&state);
        self.unlink_state(state);
        if self.transitions.len() - 1 == state {
            self.transitions.remove(state);

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
    }

    /// Removes every transition from or to `state`, in time proportional to
    /// their number.
    fn unlink_state(&mut self, state: State) {
        if let Some(predecessors) = self.transitions_in.remove(&state) {
            for predecessor in predecessors {
                self.transitions[predecessor].remove(&state);
            }
        }
        for (successor, _) in std::mem::take(&mut self.transitions[state]) {
            if let Some(predecessors) = self.transitions_in.get_mut(&successor) {
                predecessors.remove(&state);
            }
        }
    }

    /// Removes `states` and their transitions.
    ///
    /// # Panics
    ///
    /// Panics if one of them does not exist or is the start state.
    pub fn remove_states(&mut self, states: &IntSet<State>) {
        for &state in states {
            self.assert_state_exists(state);
            if self.start_state == state {
                panic!("Can not remove the state {state}, it is still used as start state.");
            }
        }

        if states.is_empty() {
            return;
        }

        self.accept_states.retain(|e| !states.contains(e));

        self.minimal = false;

        for &state in states {
            self.unlink_state(state);
        }
        for &state in states {
            if self.transitions.len() - 1 == state {
                self.transitions.remove(state);

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
        }
    }

    /// Replaces the spanning set with the smallest one that expresses the
    /// conditions of the transitions.
    pub fn recompute_minimal_spanning_set(&mut self) -> Result<(), EngineError> {
        let mut ranges = Vec::with_capacity(self.number_of_states());

        for state in self.states() {
            for (condition, _) in self.transitions_from(state) {
                ranges.push(condition.to_range(&self.spanning_set)?);
            }
        }

        let new_spanning_set = SpanningSet::compute_spanning_set(&ranges);

        self.apply_new_spanning_set(&new_spanning_set)
    }

    /// Replaces the spanning set with `new_spanning_set`, which has to express
    /// the conditions of all the transitions, and converts them to it.
    pub fn apply_new_spanning_set(
        &mut self,
        new_spanning_set: &SpanningSet,
    ) -> Result<(), EngineError> {
        if new_spanning_set == &self.spanning_set {
            return Ok(());
        }
        let condition_converter = ConditionConverter::new(&self.spanning_set, new_spanning_set)?;
        // Removed states keep a cleared transition map (see `remove_state`),
        // so every stored condition can be converted in place directly.
        for transitions in self.transitions.iter_mut() {
            for condition in transitions.values_mut() {
                *condition = condition_converter.convert(condition)?;
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
    pub(crate) fn make_empty_string(&mut self) {
        self.apply_model(&Self::new_empty_string())
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
        self.minimal = model.minimal;
    }
}

#[cfg(test)]
mod tests {
    use crate::IntSet;
    use crate::fast_automaton::FastAutomaton;
    use crate::fast_automaton::condition::Condition;
    use crate::regex::RegularExpression;

    fn rng(a: char, b: char) -> crate::CharRange {
        use regex_charclass::char::Char;
        crate::CharRange::new_from_range(Char::new(a)..=Char::new(b))
    }

    #[test]
    fn small_mutators_and_queries() {
        let mut a = FastAutomaton::new_empty();
        let s1 = a.new_state();
        let s2 = a.new_state();
        a.add_transition_from_range(0, s1, &rng('a', 'a')).unwrap();
        a.accept(s1);

        assert!(a.is_accepted(s1));
        assert!(a.has_transition(0, s1));
        assert!(a.condition(0, s1).is_some());
        assert_eq!(a.in_degree(s1), 1);
        assert_eq!(a.out_degree(0), 1);
        assert!(a.is_match("a"));

        // try_add_transition: refuses determinism-breaking additions and
        // leaves the automaton untouched on Err.
        let condition_a = Condition::from_range(&rng('a', 'a'), a.spanning_set()).unwrap();
        assert!(a.is_deterministic());
        assert!(a.try_add_transition(0, s2, &condition_a).is_err());
        assert!(a.is_deterministic());
        assert!(!a.has_transition(0, s2));
        // ...but accepts disjoint conditions.
        let condition_not_a = condition_a.complement();
        a.try_add_transition(0, s2, &condition_not_a).unwrap();
        assert!(a.is_deterministic());
        assert!(a.has_transition(0, s2));

        // unaccept flips membership and the language.
        a.unaccept(s1);
        assert!(!a.is_accepted(s1));
        assert!(!a.is_match("a"));
        a.accept(s1);
        assert!(a.is_match("a"));

        // remove_transition removes the edge and updates queries.
        a.remove_transition(0, s1);
        assert!(!a.has_transition(0, s1));
        assert!(a.condition(0, s1).is_none());
        assert_eq!(a.in_degree(s1), 0);
        assert!(!a.is_match("a"));
    }

    #[test]
    fn add_transition_from_range_extends_the_spanning_set() {
        let mut automaton = FastAutomaton::new_empty();
        let s1 = automaton.new_state();
        let s2 = automaton.new_state();
        automaton.accept(s2);

        // Both ranges extend the (initially empty) spanning set.
        automaton
            .add_transition_from_range(0, s1, &rng('a', 'c'))
            .unwrap();
        automaton
            .add_transition_from_range(s1, s2, &rng('x', 'z'))
            .unwrap();

        assert!(automaton.is_match("ax"));
        assert!(automaton.is_match("cz"));
        assert!(!automaton.is_match("aa"));
        assert!(!automaton.is_match("x"));

        // An exactly-covered range takes the fast path: same spanning set.
        let before = automaton.spanning_set().clone();
        automaton
            .add_transition_from_range(0, s1, &rng('x', 'z'))
            .unwrap();
        assert_eq!(&before, automaton.spanning_set());
        assert!(automaton.is_match("zx"));

        // An empty range adds nothing.
        automaton
            .add_transition_from_range(0, s2, &crate::CharRange::empty())
            .unwrap();
        assert!(!automaton.is_match("a"));
    }

    // `Condition::from_range` only sets bits for fully-covered bases, so
    // adding a range that partially covers the "rest" (here [a-e] over an
    // existing [a-c] base) must refine the spanning set to stay exact rather
    // than truncate [a-e] to [a-c].
    #[test]
    fn add_transition_from_range_is_exact_on_partial_coverage() {
        let mut automaton = FastAutomaton::new_empty();
        let s1 = automaton.new_state();
        automaton.accept(s1);

        automaton
            .add_transition_from_range(0, s1, &rng('a', 'c'))
            .unwrap();
        // Contains the whole [a-c] base but only part of the rest.
        automaton
            .add_transition_from_range(0, s1, &rng('a', 'e'))
            .unwrap();

        for accepted in ["a", "b", "c", "d", "e"] {
            assert!(automaton.is_match(accepted), "{accepted:?} must match");
        }
        assert!(!automaton.is_match("f"));
    }

    // Removing the trailing state must also physically drop tombstones that
    // become trailing, so `transitions` does not stay at its peak length
    // forever.
    #[test]
    fn remove_state_compacts_trailing_tombstones() {
        let mut a = FastAutomaton::new_empty(); // state 0
        let s1 = a.new_state();
        let s2 = a.new_state();
        let s3 = a.new_state();

        a.remove_state(s2); // tombstoned (not trailing)
        assert_eq!(4, a.transitions.len());
        a.remove_state(s3); // trailing: pops s3 AND compacts the s2 tombstone
        assert_eq!(2, a.transitions.len());
        assert!(a.removed_states.is_empty());
        assert!(a.has_state(s1));

        // Same through `remove_states`, in one call.
        let mut a = FastAutomaton::new_empty();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let s3 = a.new_state();
        let mut to_remove = IntSet::default();
        to_remove.insert(s2);
        to_remove.insert(s3);
        a.remove_states(&to_remove);
        assert_eq!(2, a.transitions.len());
        assert!(a.removed_states.is_empty());
        assert!(a.has_state(s1));
    }

    // `remove_states` must perform the same `transitions_in` cleanup as the
    // single-state `remove_state` (drop entries keyed by removed states and
    // purge them from surviving predecessor sets); otherwise `in_degree` of a
    // removed state stays stale for callers like repeat/concat/union.
    #[test]
    fn remove_states_cleans_transitions_in() {
        let mut a = FastAutomaton::new_empty();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let cond = Condition::total(a.spanning_set());
        a.add_transition(0, s1, &cond);
        a.add_transition(0, s2, &cond);
        a.accept(s1);
        a.accept(s2);

        assert_eq!(a.in_degree(s1), 1);
        assert_eq!(a.in_degree(s2), 1);

        let mut to_remove = IntSet::default();
        to_remove.insert(s1);
        a.remove_states(&to_remove);

        // After removing s1, its in_degree should report 0 (or, equivalently,
        // queries on a removed state should be a clean no-op). Currently it
        // still reports the pre-removal count.
        assert_eq!(a.in_degree(s1), 0, "in_degree of removed state should be 0");
        assert_eq!(a.in_degree(s2), 1);
    }

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
        assert_eq!(deterministic, automaton.is_deterministic());
    }
}
