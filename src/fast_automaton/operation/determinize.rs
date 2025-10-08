use bit_set::BitSet;

use crate::{EngineError, execution_profile::ExecutionProfile};

use super::*;

impl FastAutomaton {
    /// Determinizes the automaton and returns the result.
    pub fn determinize(&self) -> Result<Cow<Self>, EngineError> {
        if self.deterministic {
            return Ok(Cow::Borrowed(self));
        }
        let execution_profile = ExecutionProfile::get();

        let bases = self.get_spanning_bases()?;

        let mut worklist = VecDeque::with_capacity(self.get_number_of_states());

        let map_capacity = (self.get_number_of_states() as f64 / 0.75).ceil() as usize;
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

            if !states.is_disjoint(&accept_states) {
                new_automaton.accept_states.insert(r);
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
    use crate::regex::RegularExpression;

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
        println!("States Before: {}", automaton.get_number_of_states());
        let deterministic_automaton = automaton.determinize().unwrap();
        println!(
            "States After: {}",
            deterministic_automaton.get_number_of_states()
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
