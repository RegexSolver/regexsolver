use std::hash::BuildHasherDefault;

use condition::converter::SharedSpanningSet;

use crate::fast_automaton::analyze::state_sets::{Id, Operand};
use crate::{EngineError, execution_profile::ExecutionProfile};

use super::*;

impl FastAutomaton {
    /// Totalize the automaton. Precondition: `self.deterministic` is true
    /// (the only caller, `complement`, determinizes first).
    fn totalize(&mut self) -> Result<(), EngineError> {
        debug_assert!(self.deterministic, "totalize requires a DFA");

        let crash_state = self.new_state();
        let mut transitions_to_crash_state: IntMap<State, Condition> =
            IntMap::with_capacity_and_hasher(
                self.number_of_states(),
                BuildHasherDefault::default(),
            );

        let mut ranges = Vec::with_capacity(self.number_of_states());
        for from_state in self.states() {
            let mut new_condition = Condition::empty(&self.spanning_set);
            for (condition, _) in self.transitions_from(from_state) {
                new_condition = new_condition.union(condition);
                ranges.push(condition.to_range(self.spanning_set())?);
            }

            new_condition = new_condition.complement();

            transitions_to_crash_state.insert(from_state, new_condition);
        }

        for (from_state, condition) in &transitions_to_crash_state {
            self.add_transition(*from_state, crash_state, condition);
            ranges.push(condition.to_range(self.spanning_set())?);
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
    /// If `self` is non-deterministic, it is determinized in place first,
    /// unless the execution profile disables implicit determinization, in
    /// which case [`EngineError::DeterministicAutomatonRequired`] is
    /// returned.
    #[tracing::instrument(level = "debug", skip_all, fields(states = self.number_of_states(), deterministic = self.is_deterministic()))]
    pub fn complement(&mut self) -> Result<(), EngineError> {
        if !self.deterministic {
            *self = self.determinize_implicit()?.into_owned();
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
    /// `other` is not determinized or complemented first: a state of the
    /// result pairs a state of `self` with the set of states of `other` that
    /// the same word reaches, and only the pairs a word reaches are built.
    /// The result is still exponential in `other` in the worst case; the
    /// execution profile's timeout and `max_number_of_states` apply to it.
    #[tracing::instrument(level = "debug", skip_all, fields(self_states = self.number_of_states(), self_deterministic = self.is_deterministic(), other_states = other.number_of_states(), other_deterministic = other.is_deterministic()))]
    pub fn difference(&self, other: &FastAutomaton) -> Result<FastAutomaton, EngineError> {
        if self.is_empty() || other.is_total() || self == other {
            return Ok(Self::new_empty());
        } else if other.is_empty() {
            return Ok(self.clone());
        }
        let execution_profile = ExecutionProfile::get();

        let shared = SharedSpanningSet::new(&self.spanning_set, &other.spanning_set);
        let (self_converter, other_converter) = shared.converters()?;
        let left = Operand::new(self, 0, self_converter.as_ref())?;
        let right = Operand::new(other, 0, other_converter.as_ref())?;
        let spanning_set = shared.spanning_set();

        let Some(start) = left.start else {
            return Ok(Self::new_empty());
        };

        let mut difference = FastAutomaton::new_empty();
        // The sets of the pairs, back to back.
        let mut sets: Vec<Id> = right.start.into_iter().collect();
        if left.accepts(start) && !right.is_accepting(&sets) {
            difference.accept(difference.start_state);
        }
        // The state of each pair, keyed by the state of `self` followed by
        // the set of states of `other`.
        let mut states: AHashMap<Vec<Id>, State> = AHashMap::new();
        let mut key = vec![start];
        key.extend_from_slice(&sets);
        states.insert(key.clone(), difference.start_state);
        let mut worklist = VecDeque::new();
        worklist.push_back((difference.start_state, start, 0..sets.len()));

        let mut successors = vec![Vec::new(); spanning_set.spanning_ranges_with_rest_len()];
        // The pairs one transition of `self` leads to: the state of the
        // pair, a base that leads to it, and all the bases that do.
        let mut targets: Vec<(State, usize, Condition)> = Vec::new();

        while let Some((state, from, set)) = worklist.pop_front() {
            execution_profile.assert_not_timed_out()?;
            execution_profile.assert_max_number_of_states(difference.number_of_states())?;

            right.successors(&sets[set], &mut successors);

            for (condition, to) in left.transitions_from(from) {
                targets.clear();
                for base in condition.iter_set_bits() {
                    let set = &successors[base];
                    if let Some((_, _, bases)) = targets
                        .iter_mut()
                        .find(|(_, other_base, _)| successors[*other_base] == *set)
                    {
                        bases.insert_base(base);
                        continue;
                    }
                    key.clear();
                    key.push(*to);
                    key.extend_from_slice(set);
                    let target = match states.get(key.as_slice()) {
                        Some(target) => *target,
                        None => {
                            let target = difference.new_state();
                            if left.accepts(*to) && !right.is_accepting(set) {
                                difference.accept(target);
                            }
                            states.insert(key.clone(), target);
                            worklist.push_back((target, *to, sets.len()..sets.len() + set.len()));
                            sets.extend_from_slice(set);
                            target
                        }
                    };
                    targets.push((target, base, Condition::single_base(base, spanning_set)));
                }
                for (target, _, bases) in &targets {
                    difference.add_transition(state, *target, bases);
                }
            }
        }
        difference.spanning_set = shared.into_spanning_set();
        difference.remove_dead_states();
        Ok(difference)
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
