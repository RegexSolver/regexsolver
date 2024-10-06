use std::hash::BuildHasherDefault;

use crate::error::EngineError;

use super::*;

impl FastAutomaton {
    pub fn concatenate(automatons: Vec<FastAutomaton>) -> Result<FastAutomaton, EngineError> {
        if automatons.len() == 1 {
            return Ok(automatons[0].clone());
        }
        let mut new_automaton = FastAutomaton::new_empty_string();
        if automatons.is_empty() {
            return Ok(new_automaton);
        }
        for automaton in automatons {
            new_automaton.concat(&automaton)?;
        }

        Ok(new_automaton)
    }

    pub fn repeat(&mut self, min: u32, max_opt: Option<u32>) -> Result<(), EngineError> {
        if let Some(max) = max_opt {
            if min > max {
                self.make_empty();
                return Ok(());
            }
        }

        let automaton_to_repeat = self.clone();

        if min == 0 && self.in_degree(self.start_state) != 0 {
            let new_state = self.new_state();
            if self.is_accepted(&self.start_state) {
                self.accept(new_state);
            }

            for to_state in self.transitions_from_state(&self.start_state) {
                self.add_epsilon(new_state, to_state);
            }
            self.start_state = new_state;

            if max_opt.is_none() {
                for accept_state in self.accept_states.clone() {
                    self.add_epsilon(accept_state, self.start_state);
                }
                self.accept(self.start_state);
                return Ok(());
            }
        }

        if let Some(max) = max_opt {
            if min <= 1 && max == 1 {
                if min == 0 {
                    self.accept_states.insert(self.start_state);
                }
                return Ok(());
            }
        }

        let iter = if min == 0 { 0..0 } else { 0..min - 1 };
        for _ in iter {
            self.concat(&automaton_to_repeat)?;
        }

        if max_opt.is_none() {
            let mut automaton_to_repeat = automaton_to_repeat.clone();

            let accept_state = *automaton_to_repeat.accept_states.iter().next().unwrap();
            if automaton_to_repeat.accept_states.len() == 1
                && automaton_to_repeat.out_degree(accept_state) == 0
                && automaton_to_repeat.in_degree(automaton_to_repeat.start_state) == 0
            {
                automaton_to_repeat.add_epsilon(accept_state, automaton_to_repeat.start_state);
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
                        automaton_to_repeat.add_transition_to(state, *to_state, condition);
                    }
                }

                automaton_to_repeat.accept(automaton_to_repeat.get_start_state());
            }
            automaton_to_repeat.cyclic = true;

            if min == 0 {
                self.apply_model(&automaton_to_repeat);
            } else {
                self.concat(&automaton_to_repeat)?;
            }

            return Ok(());
        }

        let mut end_states = self.accept_states.iter().cloned().collect::<Vec<_>>();
        for _ in cmp::max(min, 1)..max_opt.unwrap() {
            self.concat(&automaton_to_repeat)?;
            end_states.extend(self.accept_states.iter());
        }
        self.accept_states.extend(end_states);
        if min == 0 {
            self.accept(self.start_state);
        }
        Ok(())
    }

    fn concat(&mut self, other: &FastAutomaton) -> Result<(), EngineError> {
        if other.is_empty() {
            return Ok(());
        }
        if self.is_empty() {
            self.apply_model(other);
            return Ok(());
        }

        let newly_used_bases = &self.used_bases.merge(&other.used_bases);
        self.apply_newly_used_bases(newly_used_bases)?;

        let mut new_states: IntMap<usize, usize> = IntMap::with_capacity_and_hasher(
            other.get_number_of_states(),
            BuildHasherDefault::default(),
        );

        let start_state_and_accept_states_not_mergeable = other.in_degree(other.start_state) > 0
            && self
                .accept_states
                .iter()
                .cloned()
                .any(|s| self.out_degree(s) > 0);

        let accept_states = self.accept_states.iter().cloned().collect::<Vec<usize>>();

        self.accept_states.clear();

        if other.accept_states.contains(&other.start_state) {
            for &accept_state in accept_states.iter() {
                self.accept(accept_state);
            }
        }

        if start_state_and_accept_states_not_mergeable {
            let new_start_state = new_states
                .entry(other.start_state)
                .or_insert(self.new_state());
            if other.accept_states.contains(&other.start_state) {
                self.accept(*new_start_state);
            }
        }

        for from_state in other.transitions_iter() {
            let new_from_states = match new_states.entry(from_state) {
                Entry::Occupied(o) => {
                    vec![*o.get()]
                }
                Entry::Vacant(v) => {
                    if from_state == other.start_state {
                        accept_states.clone()
                    } else {
                        let new_state = self.new_state();
                        if other.accept_states.contains(&from_state) {
                            self.accept(new_state);
                        }
                        v.insert(new_state);
                        vec![new_state]
                    }
                }
            };

            for (to_state, condition) in other.transitions_from_state_enumerate_iter(&from_state) {
                let new_to_states = match new_states.entry(*to_state) {
                    Entry::Occupied(o) => {
                        vec![*o.get()]
                    }
                    Entry::Vacant(v) => {
                        if *to_state == other.start_state {
                            accept_states.clone()
                        } else {
                            let new_state = self.new_state();
                            if other.accept_states.contains(to_state) {
                                self.accept(new_state);
                            }
                            v.insert(new_state);
                            vec![new_state]
                        }
                    }
                };
                let projected_condition =
                    condition.project_to(&other.used_bases, newly_used_bases)?;
                for new_from_state in new_from_states.iter() {
                    for new_to_state in new_to_states.iter() {
                        self.add_transition_to(
                            *new_from_state,
                            *new_to_state,
                            &projected_condition,
                        );
                    }
                }
            }
        }

        if start_state_and_accept_states_not_mergeable {
            if let Some(&other_start_state) = new_states.get(&other.start_state) {
                for accept_state in &accept_states {
                    self.add_epsilon(*accept_state, other_start_state);
                }
            }
        }
        self.cyclic = self.cyclic || other.cyclic;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::regex::RegularExpression;

    #[test]
    fn test_simple_concatenation_regex() -> Result<(), String> {
        let automaton = RegularExpression::new("abc")
            .unwrap()
            .to_automaton()
            .unwrap();

        automaton.to_dot();
        assert!(automaton.match_string("abc"));
        assert!(!automaton.match_string("abcd"));
        assert!(!automaton.match_string("ab"));
        assert!(!automaton.match_string(""));
        Ok(())
    }

    #[test]
    fn test_simple_concat_alternation_regex() -> Result<(), String> {
        let automaton = RegularExpression::new("0101(abc|ac|aaa)")
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.match_string("0101abc"));
        assert!(automaton.match_string("0101ac"));
        assert!(automaton.match_string("0101aaa"));
        assert!(!automaton.match_string("abc"));
        assert!(!automaton.match_string("0101abcd"));
        assert!(!automaton.match_string("ab"));
        assert!(!automaton.match_string("acc"));
        assert!(!automaton.match_string("a"));
        assert!(!automaton.match_string("aaaa"));
        assert!(!automaton.match_string("aa"));
        assert!(!automaton.match_string(""));
        Ok(())
    }

    #[test]
    fn test_simple_concat_repeat_regex() -> Result<(), String> {
        let automaton = RegularExpression::new("A+B*")
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.match_string("AAABBB"));
        assert!(automaton.match_string("AA"));
        assert!(automaton.match_string("AB"));
        assert!(!automaton.match_string("B"));
        assert!(!automaton.match_string("ABA"));
        assert!(!automaton.match_string(""));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_01() -> Result<(), String> {
        let automaton = RegularExpression::new("a+")
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("aa"));
        assert!(automaton.match_string("aaaaaaa"));
        assert!(!automaton.match_string("ab"));
        assert!(!automaton.match_string(""));

        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_02() -> Result<(), String> {
        let automaton = RegularExpression::new("a*c")
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.match_string("c"));
        assert!(automaton.match_string("ac"));
        assert!(automaton.match_string("aac"));
        assert!(automaton.match_string("aaaaaaac"));
        assert!(!automaton.match_string("abc"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_03() -> Result<(), String> {
        let automaton = RegularExpression::new("(ab){3,4}")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(automaton.match_string("ababab"));
        assert!(automaton.match_string("abababab"));
        assert!(!automaton.match_string("ab"));
        assert!(!automaton.match_string("abab"));
        assert!(!automaton.match_string("ababababab"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_04() -> Result<(), String> {
        let automaton = RegularExpression::new("a{3,}")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(automaton.match_string("aaa"));
        assert!(automaton.match_string("aaaaa"));
        assert!(!automaton.match_string("a"));
        assert!(!automaton.match_string("aa"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_05() -> Result<(), String> {
        let automaton = RegularExpression::new("a?")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(automaton.match_string(""));
        assert!(automaton.match_string("a"));
        assert!(!automaton.match_string("aa"));
        assert!(!automaton.match_string("aaa"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_06() -> Result<(), String> {
        let automaton = RegularExpression::new("a{0,2}")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(automaton.match_string(""));
        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("aa"));
        assert!(!automaton.match_string("aaa"));
        assert!(!automaton.match_string("aaaa"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_07() -> Result<(), String> {
        let automaton = RegularExpression::new("a{1,3}")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(!automaton.match_string(""));
        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("aa"));
        assert!(automaton.match_string("aaa"));
        assert!(!automaton.match_string("aaaa"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_08() -> Result<(), String> {
        let automaton = RegularExpression::new("a+(ba+)*")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(!automaton.match_string(""));
        assert!(!automaton.match_string("aab"));
        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("aaa"));
        assert!(automaton.match_string("aba"));
        assert!(automaton.match_string("aaba"));
        assert!(automaton.match_string("aabaaa"));
        assert!(automaton.match_string("aaabaaabaaba"));
        assert!(!automaton.match_string("aaabbaa"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_09() -> Result<(), String> {
        let automaton = RegularExpression::new("(ac|ads|a)*")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(automaton.match_string(""));
        assert!(automaton.match_string("ac"));
        assert!(automaton.match_string("ads"));
        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("acaadsac"));
        assert!(automaton.match_string("adsaaaaaaaacaa"));
        assert!(!automaton.match_string("as"));
        assert!(!automaton.match_string("ad"));
        assert!(!automaton.match_string("c"));
        assert!(!automaton.match_string("ds"));
        assert!(!automaton.match_string("d"));
        assert!(!automaton.match_string("s"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_10() -> Result<(), String> {
        let automaton = RegularExpression::new("(ef|ads|a)+")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(!automaton.match_string(""));
        assert!(automaton.match_string("ef"));
        assert!(automaton.match_string("ads"));
        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("efadsa"));
        assert!(automaton.match_string("aaadsefef"));
        assert!(!automaton.match_string("as"));
        assert!(!automaton.match_string("ad"));
        assert!(!automaton.match_string("e"));
        assert!(!automaton.match_string("ds"));
        assert!(!automaton.match_string("d"));
        assert!(!automaton.match_string("s"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_11() -> Result<(), String> {
        let automaton = RegularExpression::new("(a|bc)*")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(automaton.match_string(""));
        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("bc"));
        assert!(automaton.match_string("abcbca"));
        assert!(automaton.match_string("bcabcbcaaaa"));
        assert!(!automaton.match_string("b"));
        assert!(!automaton.match_string("c"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_12() -> Result<(), String> {
        let automaton = RegularExpression::new("([ab]*a)?")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(automaton.match_string(""));
        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("aa"));
        assert!(automaton.match_string("ba"));
        assert!(automaton.match_string("aba"));
        assert!(automaton.match_string("abbaabbaba"));
        assert!(!automaton.match_string("b"));
        assert!(!automaton.match_string("abab"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_13() -> Result<(), String> {
        let automaton = RegularExpression::new("([ab]*a)*")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert!(automaton.match_string(""));
        assert!(automaton.match_string("a"));
        assert!(automaton.match_string("aa"));
        assert!(automaton.match_string("ba"));
        assert!(automaton.match_string("aba"));
        assert!(automaton.match_string("abbaabbaba"));
        assert!(!automaton.match_string("b"));
        assert!(!automaton.match_string("abab"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_right_number_of_states_1() -> Result<(), String> {
        let automaton = RegularExpression::new("a*")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert_eq!(1, automaton.get_number_of_states());
        Ok(())
    }

    #[test]
    fn test_simple_concat_right_number_of_states_2() -> Result<(), String> {
        let automaton = RegularExpression::new("(a*bc)")
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.to_dot();
        assert_eq!(3, automaton.get_number_of_states());
        Ok(())
    }
}
//(a|bc)*
