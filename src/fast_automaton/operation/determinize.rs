use bit_set::BitSet;

use crate::{EngineError, execution_profile::ExecutionProfile};

use super::*;

impl FastAutomaton {
    /// [`determinize`](Self::determinize) on behalf of an operation that
    /// requires a deterministic automaton: when the execution profile
    /// disables implicit determinization, a non-deterministic input is
    /// rejected with [`EngineError::DeterministicAutomatonRequired`] instead
    /// of being converted. Already-deterministic automata always pass.
    pub(crate) fn determinize_implicit(&self) -> Result<Cow<'_, Self>, EngineError> {
        if !self.deterministic {
            ExecutionProfile::get().assert_implicit_determinization_allowed()?;
        }
        self.determinize()
    }

    /// Determinizes the automaton and returns the result.
    #[tracing::instrument(level = "debug", skip_all, fields(states = self.number_of_states(), deterministic = self.is_deterministic()))]
    pub fn determinize(&self) -> Result<Cow<'_, Self>, EngineError> {
        if self.deterministic {
            return Ok(Cow::Borrowed(self));
        }
        let execution_profile = ExecutionProfile::get();

        let bases = self.spanning_bases()?;

        let mut worklist = VecDeque::with_capacity(self.number_of_states());

        let map_capacity = (self.number_of_states() as f64 / 0.75).ceil() as usize;
        let mut new_states = AHashMap::with_capacity(map_capacity);

        let mut accept_states = BitSet::new();
        for &state in &self.accept_states {
            accept_states.insert(state);
        }

        let mut new_automaton = FastAutomaton::new_empty();
        new_automaton.spanning_set = self.spanning_set.clone();

        let mut initial_state = BitSet::new();
        initial_state.insert(self.start_state);

        worklist.push_back((initial_state.clone(), new_automaton.start_state));
        new_states.insert(initial_state, new_automaton.start_state);

        let mut new_states_to_add = BitSet::new();
        while let Some((states, r)) = worklist.pop_front() {
            execution_profile.assert_not_timed_out()?;
            execution_profile.assert_max_number_of_states(new_states.len())?;

            if !states.is_disjoint(&accept_states) {
                new_automaton.accept(r);
            }

            for base in &bases {
                for from_state in &states {
                    for (cond, to_state) in self.transitions_from(from_state) {
                        if cond.has_intersection(base) {
                            new_states_to_add.insert(*to_state);
                        }
                    }
                }
                if !new_states_to_add.is_empty() {
                    match new_states.entry(new_states_to_add.clone()) {
                        Entry::Occupied(o) => {
                            let q = *o.get();

                            new_states_to_add.clear();

                            new_automaton.add_transition(r, q, base);
                        }
                        Entry::Vacant(v) => {
                            let new_q = new_automaton.new_state();
                            v.insert(new_q);

                            let new_states = std::mem::take(&mut new_states_to_add);
                            worklist.push_back((new_states, new_q));

                            new_automaton.add_transition(r, new_q, base);
                        }
                    };
                }
            }
        }

        Ok(Cow::Owned(new_automaton))
    }
}

#[cfg(test)]
mod tests {
    use crate::CharRange;
    use crate::fast_automaton::FastAutomaton;
    use crate::fast_automaton::condition::Condition;
    use crate::fast_automaton::spanning_set::SpanningSet;
    use crate::regex::RegularExpression;
    use regex_charclass::char::Char;

    // Subset construction iterates `spanning_bases`, which must include the
    // spanning set's "rest" range: otherwise a transition whose condition
    // lies in the rest range would be dropped, giving a DFA with a smaller
    // language than the input NFA.
    #[test]
    fn determinize_keeps_rest_range_transitions() {
        let rng = |c: char| {
            let c = Char::new(c);
            CharRange::new_from_range(c..=c)
        };
        let ss = SpanningSet::compute_spanning_set(&[rng('a'), rng('b')]);
        let rest = ss.rest().clone();

        let mut a = FastAutomaton::new_empty();
        a.apply_new_spanning_set(&ss).unwrap();
        a.new_state();
        a.add_transition(0, 1, &Condition::from_range(&rest, &ss).unwrap()); // 0 -[^ab]-> 1
        a.add_transition(1, 0, &Condition::from_range(&rng('a'), &ss).unwrap());
        a.add_transition(1, 1, &Condition::from_range(&rng('a'), &ss).unwrap()); // nondeterministic
        a.accept(1);

        assert!(!a.is_deterministic());
        assert!(a.is_match("\u{0}"), "a should accept a [^ab] character");

        let d = a.determinize().unwrap();
        assert!(d.is_deterministic());
        assert!(
            d.is_match("\u{0}"),
            "determinize dropped the [^ab] transition"
        );
        assert!(a.equivalent(&d).unwrap());
    }

    #[test]
    fn test_determinize_regex() -> Result<(), String> {
        assert_determinization("(aad|ads|a)");
        assert_determinization(".*ab.*de");
        assert_determinization(".*de");
        assert_determinization("abc.*def");
        assert_determinization("a(bcfe|bcdg|mkv)*");
        assert_determinization("(aad|ads|a)*abc.*def.*ghi");
        assert_determinization(
            "((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q)",
        );

        Ok(())
    }

    fn assert_determinization(regex: &str) {
        println!(":{}", regex);
        let automaton = RegularExpression::parse(regex, false)
            .unwrap()
            .to_automaton()
            .unwrap();
        println!("States Before: {}", automaton.number_of_states());
        let deterministic_automaton = automaton.determinize().unwrap();
        println!(
            "States After: {}",
            deterministic_automaton.number_of_states()
        );
        assert!(deterministic_automaton.is_deterministic());
        //deterministic_automaton.print_dot();
        assert!(
            automaton
                .difference(&deterministic_automaton)
                .unwrap()
                .is_empty()
        );
    }
}
