use ahash::HashMapExt;

use crate::{execution_profile::ThreadLocalParams, EngineError};

use super::*;

impl FastAutomaton {
    pub fn determinize(&self) -> Result<Self, EngineError> {
        if self.deterministic {
            return Ok(self.clone());
        }
        let execution_profile = ThreadLocalParams::get_execution_profile();

        let bases = self.get_bases()?;

        let initial_vec = VecDeque::from(vec![self.start_state]);

        let mut worklist = VecDeque::with_capacity(self.get_number_of_states());

        let map_capacity = (self.get_number_of_states() as f64 / 0.75).ceil() as usize;
        let mut new_states = IntMap::with_capacity(map_capacity);

        let mut new_automaton = FastAutomaton::new_empty();
        new_automaton.used_bases = self.used_bases.clone();

        worklist.push_back((vec![self.start_state], new_automaton.start_state));
        new_states.insert(Self::simple_hash(&initial_vec), new_automaton.start_state);

        let mut new_states_to_add = VecDeque::with_capacity(self.get_number_of_states());
        while let Some((states, r)) = worklist.pop_front() {
            execution_profile.is_timed_out()?;

            for state in &states {
                if self.accept_states.contains(state) {
                    new_automaton.accept_states.insert(r);
                    break;
                }
            }

            for base in &bases {
                for from_state in &states {
                    for (to_state, cond) in self.transitions_from_state_enumerate_iter(from_state) {
                        if cond.has_intersection(base) {
                            match new_states_to_add.binary_search(to_state) {
                                Ok(_) => {} // element already in vector @ `pos`
                                Err(pos) => new_states_to_add.insert(pos, *to_state),
                            };
                        }
                    }
                }
                if !new_states_to_add.is_empty() {
                    let q = match new_states.entry(Self::simple_hash(&new_states_to_add)) {
                        Entry::Occupied(o) => *o.get(),
                        Entry::Vacant(v) => {
                            let new_q = new_automaton.new_state();
                            worklist
                                .push_back((new_states_to_add.iter().cloned().collect(), new_q));
                            v.insert(new_q);
                            new_q
                        }
                    };

                    new_automaton.add_transition_to(r, q, base);
                }
                new_states_to_add.clear();
            }
        }
        Ok(new_automaton)
    }

    fn simple_hash(list: &VecDeque<usize>) -> u64 {
        let mut hasher = AHasher::default();
        for &item in list {
            hasher.write_usize(item);
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    #[test]
    fn test_determinize_1() -> Result<(), String> {
        let automaton = RegularExpression::new(".*ab")
            .unwrap()
            .to_automaton()
            .unwrap();

        let deterministic_automaton = automaton.determinize().unwrap();

        assert!(deterministic_automaton.is_determinitic());

        Ok(())
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
        let automaton = RegularExpression::new(regex)
            .unwrap()
            .to_automaton()
            .unwrap();
        //automaton.compute_determinization_cost();
        //println!("Determinization Cost: {:?}", automaton.determinisation_cost);
        println!("States Before: {}", automaton.get_number_of_states());
        let deterministic_automaton = automaton.determinize().unwrap();
        println!(
            "States After: {}",
            deterministic_automaton.get_number_of_states()
        );
        assert!(deterministic_automaton.is_determinitic());
        assert!(automaton
            .subtraction(&deterministic_automaton)
            .unwrap()
            .is_empty());
    }
}
