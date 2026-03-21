use super::*;

impl FastAutomaton {
    /// Computes the repetition of the automaton between `min` and `max_opt` times; if `max_opt` is `None`, the repetition is unbounded.
    pub fn repeat(&self, min: u32, max_opt: Option<u32>) -> Result<FastAutomaton, EngineError> {
        let mut automaton = self.clone();
        if let Err(error) = automaton.repeat_mut(min, max_opt) {
            Err(error)
        } else {
            Ok(automaton)
        }
    }

    pub(crate) fn repeat_mut(&mut self, min: u32, max_opt: Option<u32>) -> Result<(), EngineError> {
        ExecutionProfile::get()
            .assert_max_number_of_states(self.repeat_state_count_heuristic(min, max_opt))?;

        if let Some(max) = max_opt
            && min > max
        {
            self.make_empty();
            return Ok(());
        }

        let automaton_to_repeat = self.clone();

        if min == 0 && self.in_degree(self.start_state) != 0 {
            let new_state = self.new_state();
            if self.is_accepted(self.start_state) {
                self.accept(new_state);
            }

            self.add_epsilon_transition(new_state, self.start_state);
            self.start_state = new_state;

            if max_opt.is_none() {
                for accept_state in self.accept_states.clone() {
                    self.add_epsilon_transition(accept_state, self.start_state);
                }
                self.accept(self.start_state);
                return Ok(());
            }
        }

        if let Some(max) = max_opt
            && min <= 1
            && max == 1
        {
            if min == 0 {
                self.accept_states.insert(self.start_state);
            }
            return Ok(());
        }

        let iter = if min == 0 { 0..0 } else { 0..min - 1 };
        for _ in iter {
            self.concat_mut(&automaton_to_repeat)?;
        }

        if max_opt.is_none() {
            let mut automaton_to_repeat = automaton_to_repeat.clone();

            let accept_state = *automaton_to_repeat.accept_states.iter().next().unwrap();
            if automaton_to_repeat.accept_states.len() == 1
                && automaton_to_repeat.out_degree(accept_state) == 0
                && automaton_to_repeat.in_degree(automaton_to_repeat.start_state) == 0
            {
                automaton_to_repeat
                    .add_epsilon_transition(accept_state, automaton_to_repeat.start_state);
                let old_start_state = automaton_to_repeat.start_state;
                automaton_to_repeat.start_state = accept_state;
                automaton_to_repeat.remove_state(old_start_state);
            } else {
                let t = Self::transitions_from_state_set(
                    &automaton_to_repeat.transitions,
                    automaton_to_repeat.start_state,
                );
                let transitions =
                    Self::transitions_from_state_enumerate(&t, &automaton_to_repeat.removed_states);

                for state in automaton_to_repeat.accept_states.clone() {
                    for &(to_state, condition) in &transitions {
                        automaton_to_repeat.add_transition(state, *to_state, condition);
                    }
                }

                automaton_to_repeat.accept(automaton_to_repeat.get_start_state());
            }
            automaton_to_repeat.cyclic = true;

            if min == 0 {
                self.apply_model(&automaton_to_repeat);
            } else {
                self.concat_mut(&automaton_to_repeat)?;
            }

            return Ok(());
        }

        let mut end_states = self.accept_states.iter().cloned().collect::<Vec<_>>();
        for _ in cmp::max(min, 1)..max_opt.unwrap() {
            self.concat_mut(&automaton_to_repeat)?;
            end_states.extend(self.accept_states.iter());
        }
        self.accept_states.extend(end_states);
        if min == 0 {
            self.accept(self.start_state);
        }
        Ok(())
    }

    /// Computes the expected number of states after calling `repeat_mut`,
    /// reusing the concatenation heuristic to determine loop costs.
    fn repeat_state_count_heuristic(&self, min: u32, max_opt: Option<u32>) -> usize {
        // 1. Invalid range clears the automaton
        if let Some(max) = max_opt
            && min > max
        {
            return 0;
        }

        let v_original = self.get_number_of_states();
        if v_original == 0 {
            return 0;
        }

        let mut current_states = v_original;
        let in_deg_start = self.in_degree(self.start_state) > 0;

        // --- REUSE CONCAT HEURISTIC HERE ---
        // Calculate the state delta for a single concatenation.
        let concat_cost = self.concat_state_count_heuristic(self) - v_original;

        // 2. Early state allocation for 0-minimum repeats with incoming start edges
        if min == 0 && in_deg_start {
            current_states += 1;
            if max_opt.is_none() {
                return current_states;
            }
        }

        // 3. Simple cases: 0..=1 or 1..=1 repetitions
        if let Some(max) = max_opt
            && min <= 1
            && max == 1
        {
            return current_states;
        }

        // 4. Minimum repetitions loop
        let min_iters = if min == 0 { 0 } else { min - 1 };
        current_states += min_iters as usize * concat_cost;

        // 5. Infinite repetition (max_opt is None)
        if max_opt.is_none() {
            let mut v_modified = v_original;
            let mut mod_start_in_deg_gt_0 = in_deg_start;
            let acc_out_gt_0 = self.accept_states.iter().any(|&s| self.out_degree(s) > 0);

            // Check if it triggers the start-state removal optimization block
            if self.accept_states.len() == 1 {
                let accept_state = *self.accept_states.iter().next().unwrap();
                if self.out_degree(accept_state) == 0 && !in_deg_start {
                    // The old start state is removed in the cloned automaton
                    v_modified -= 1;
                    mod_start_in_deg_gt_0 = self.in_degree(accept_state) > 0;
                }
            }

            if min == 0 {
                return v_modified;
            } else {
                // Calculate the final virtual concatenation cost manually since
                // we can't pass a "virtually modified" automaton to concat_state_count_heuristic
                let final_concat_cost = if mod_start_in_deg_gt_0 && acc_out_gt_0 {
                    v_modified
                } else {
                    v_modified.saturating_sub(1)
                };
                return current_states + final_concat_cost;
            }
        }

        // 6. Finite maximum repetition loop
        let max = max_opt.unwrap();
        let loop_start = if min > 1 { min } else { 1 };
        let max_iters = max.saturating_sub(loop_start);

        current_states += max_iters as usize * concat_cost;

        current_states
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    #[test]
    fn test_repeat_1() -> Result<(), String> {
        let automaton = RegularExpression::parse("(a*,a*)?", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match(","));
        assert!(automaton.is_match("aaa,"));
        assert!(automaton.is_match("aaaa,aa"));
        assert!(!automaton.is_match("a"));
        assert!(!automaton.is_match("aa"));
        Ok(())
    }

    #[test]
    fn test_heuristic() -> Result<(), String> {
        assert_heuristic(".{900}");
        assert_heuristic("[a-z]+");
        assert_heuristic("[a-z]+@");

        assert_heuristic("[0-9]+[A-Z]*");
        assert_heuristic("a+(ba+)*");
        assert_heuristic("((a|bc)*|d)");
        assert_heuristic(".*");
        assert_heuristic("(ac|ads|a)*");
        assert_heuristic("((aad|ads|a)*|q)");

        assert_heuristic(
            r"john[!#-'\*\+\-/-9=\?\^-\u{007e}]*(\.[!#-'\*\+\-/-9=\?\^-\u{007e}](\.?[!#-'\*\+\-/-9=\?\^-\u{007e}])*)?\.?doe@example\.com",
        );

        assert_heuristic("(?:A+(?:\\.[AB]+)*|\"(?:C|\\\\D)*\")@");
        assert_heuristic(
            "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@",
        );
        assert_heuristic("((aad|ads|a)*abc.*uif(aad|ads|x)*|q)");
        Ok(())
    }

    fn assert_heuristic(regex: &str) {
        println!("Testing regex: {regex}");

        let automaton = RegularExpression::parse(regex, false)
            .unwrap()
            .to_automaton()
            .unwrap();

        // A matrix of test cases covering all edge cases in the repeat logic
        let test_cases = vec![
            (0, Some(0)),  // Zero-repeat
            (0, Some(1)),  // Optional once
            (1, Some(1)),  // Exactly once
            (5, Some(10)), // Standard finite range
            (0, None),     // Zero or more (Kleene star)
            (1, None),     // One or more (Kleene plus)
            (3, None),     // Finite minimum, infinite maximum
        ];

        for (min, max_opt) in test_cases {
            // Clone the original automaton to avoid mutating it across iterations
            let mut actual_automaton = automaton.clone();

            // Execute the actual mutation (assuming repeat_mut is the core method)
            actual_automaton.repeat_mut(min, max_opt).unwrap();

            let actual_states = actual_automaton.get_number_of_states();
            let heuristic_states = automaton.repeat_state_count_heuristic(min, max_opt);

            assert_eq!(
                actual_states, heuristic_states,
                "Mismatch for regex '{}' with min={}, max={:?}.\nExpected (heuristic): {}\nActual (computed): {}",
                regex, min, max_opt, heuristic_states, actual_states
            );
        }
    }
}
