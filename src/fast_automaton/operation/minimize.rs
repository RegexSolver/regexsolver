use crate::execution_profile::ExecutionProfile;

use super::*;

impl FastAutomaton {
    /// Minimizes the automaton using Hopcroft's Algorithm.
    ///
    /// If `self` is non-deterministic, it is determinized in place first,
    /// unless the [`ExecutionProfile`] disables implicit determinization, in
    /// which case [`EngineError::DeterministicAutomatonRequired`] is
    /// returned.
    #[tracing::instrument(level = "debug", skip_all, fields(states = self.number_of_states(), deterministic = self.is_deterministic(), minimal = self.is_minimal()))]
    pub fn minimize(&mut self) -> Result<(), EngineError> {
        // The `minimal` flag is conservatively cleared on every mutation, so
        // it can be trusted here; this also keeps the
        // `minimize_after_determinization` profile from paying a second
        // Hopcroft pass when callers minimize an already-minimized result.
        if self.minimal {
            return Ok(());
        }
        if !self.deterministic {
            *self = self.determinize_implicit()?.into_owned();
        }
        let execution_profile = ExecutionProfile::get();

        // Drop states unreachable from the start. A minimal automaton has none,
        // and downstream invariants rely on it; in particular `is_empty`'s
        // fast path treats any minimal automaton with an accept state as
        // non-empty, which only holds if every accept state is reachable.
        let reachable = self.forward_reachable_states();
        let unreachable: IntSet<State> = self.states().filter(|s| !reachable.contains(s)).collect();
        if !unreachable.is_empty() {
            self.remove_states(&unreachable);
        }

        self.remove_dead_states();

        let max_states = self.transitions.len();

        let all_states: IntSet<usize> = self.states().collect();
        let accept_states: IntSet<usize> = self.accept_states().iter().cloned().collect();
        let non_accept_states: IntSet<usize> =
            all_states.difference(&accept_states).cloned().collect();

        let mut partitions: Vec<IntSet<usize>> = vec![accept_states, non_accept_states];
        partitions.retain(|p| !p.is_empty());

        let mut state_to_partition = vec![0; max_states];
        for (i, partition) in partitions.iter().enumerate() {
            for &state in partition {
                state_to_partition[state] = i;
            }
        }

        let mut worklist: Vec<usize> = (0..partitions.len()).collect();
        let mut in_worklist: Vec<bool> = vec![true; max_states];

        let bases = self.spanning_bases()?;

        let mut inverse_transitions: Vec<Vec<(usize, Condition)>> = vec![Vec::new(); max_states];
        for to_state in self.states() {
            for (from_state, condition) in self.transitions_to_vec(to_state) {
                inverse_transitions[to_state].push((from_state, condition));
            }
        }

        let mut x = IntSet::with_capacity(self.number_of_states());

        let mut intersection_states: Vec<Vec<usize>> = vec![Vec::new(); max_states];
        let mut touched_partitions: Vec<usize> = Vec::with_capacity(max_states);

        while let Some(a_idx) = worklist.pop() {
            execution_profile.assert_not_timed_out()?;
            in_worklist[a_idx] = false;

            let a = partitions[a_idx].clone();

            for base in &bases {
                x.clear();

                // Find states that transition into partition 'A' on 'base'
                for &to_state in &a {
                    for (from_state, condition) in &inverse_transitions[to_state] {
                        if base.has_intersection(condition) {
                            x.insert(*from_state);
                        }
                    }
                }

                if x.is_empty() {
                    continue;
                }

                // TARGETED SPLITTING: Only evaluate partitions we know overlap with 'x'
                for &state in &x {
                    let p_idx = state_to_partition[state];
                    if intersection_states[p_idx].is_empty() {
                        touched_partitions.push(p_idx);
                    }
                    intersection_states[p_idx].push(state);
                }

                // Process only the affected partitions
                for &p_idx in &touched_partitions {
                    let int_states = &mut intersection_states[p_idx];
                    let y_len = partitions[p_idx].len();

                    // If the partition is fully contained in 'x', no split happens.
                    if int_states.len() == y_len {
                        int_states.clear();
                        continue;
                    }

                    // A split happens! 'int_states' becomes the new partition.
                    let new_idx = partitions.len();
                    let mut new_part = IntSet::with_capacity(int_states.len());

                    for &state in int_states.iter() {
                        partitions[p_idx].remove(&state); // Remove from original (forming the difference)
                        new_part.insert(state); // Add to new partition (forming the intersection)
                        state_to_partition[state] = new_idx; // Update the lookup array
                    }

                    let diff_len = partitions[p_idx].len();
                    let int_len = new_part.len();

                    partitions.push(new_part);
                    in_worklist.push(false);

                    // Worklist update
                    if in_worklist[p_idx] || int_len <= diff_len {
                        worklist.push(new_idx);
                        in_worklist[new_idx] = true;
                    } else {
                        worklist.push(p_idx);
                        in_worklist[p_idx] = true;
                    }

                    int_states.clear();
                }
                touched_partitions.clear();
            }
        }

        if partitions.len() == all_states.len() {
            self.minimal = true;
            return Ok(());
        }

        self.rebuild_automaton_from_partition(&partitions)?;

        self.minimal = true;
        Ok(())
    }

    fn rebuild_automaton_from_partition(
        &mut self,
        partitions: &[IntSet<usize>],
    ) -> Result<(), EngineError> {
        let mut state_to_rep = vec![0; self.transitions.len()];
        let mut representatives = Vec::with_capacity(partitions.len());

        for partition in partitions {
            let representative = if partition.contains(&self.start_state()) {
                self.start_state()
            } else {
                *partition
                    .iter()
                    .next()
                    .expect("A partition cannot be empty")
            };

            representatives.push(representative);

            for &state in partition {
                state_to_rep[state] = representative;
            }
        }

        let mut transitions_to_update = Vec::new();
        for &rep in &representatives {
            for (condition, old_target) in self.transitions_from_vec(rep) {
                let new_target = state_to_rep[old_target];
                transitions_to_update.push((rep, condition, new_target));
            }
        }

        for partition in partitions {
            for &state in partition {
                let rep = state_to_rep[state];
                if state != rep {
                    self.remove_state(state);
                }
            }
        }

        for (from, condition, to) in transitions_to_update {
            self.add_transition(from, to, &condition);
        }

        self.recompute_minimal_spanning_set()
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    #[test]
    fn test_minimize_various_regexes() -> Result<(), String> {
        let test_cases = [
            "a",
            "a|b",
            "ab",
            "a|a",
            "a(b|c)d|a(b|c)d",
            "(ab|ab|ab)",
            "a*|b*",
            "(a|b)*a(a|b)*",
            "(abc|de)",
            "a(b|c)*d",
            "((a|b)c|(a|b)d)",
            "a+b?",
            "(a+b)*",
        ];

        for regex in test_cases {
            assert_minimize(regex)?;
        }

        Ok(())
    }

    fn assert_minimize(regex: &str) -> Result<(), String> {
        println!("{regex}");
        let automaton = RegularExpression::parse(regex, false)
            .unwrap()
            .to_automaton()
            .unwrap();

        let automaton = automaton.determinize().unwrap().into_owned();
        let mut minimized_automaton = automaton.clone();
        minimized_automaton.minimize().unwrap();

        assert!(automaton.equivalent(&minimized_automaton).unwrap());

        assert!(minimized_automaton.is_deterministic());
        assert!(minimized_automaton.is_minimal());
        Ok(())
    }

    #[test]
    fn test_minimize_union_complement_total() -> Result<(), String> {
        let automaton = RegularExpression::parse("(abc|de)", false)
            .unwrap()
            .to_automaton()
            .unwrap();

        let automaton = automaton.determinize().unwrap();
        let mut complement = automaton.clone().into_owned();
        complement.complement().unwrap();

        let union = automaton.union(&complement).unwrap();
        let mut union = union.determinize().unwrap().into_owned();

        assert!(union.is_deterministic());
        assert!(!union.is_minimal());

        union.minimize().unwrap();
        assert!(union.is_total());

        assert!(union.is_deterministic());
        assert!(union.is_minimal());
        Ok(())
    }
}
