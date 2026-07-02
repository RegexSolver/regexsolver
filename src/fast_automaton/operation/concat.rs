use std::hash::BuildHasherDefault;

use condition::converter::ConditionConverter;

use crate::error::EngineError;

use super::*;

impl FastAutomaton {
    /// Computes the concatenation between `self` and `other`.
    pub fn concat(&self, other: &FastAutomaton) -> Result<Self, EngineError> {
        Self::concat_all([self, other])
    }

    /// Computes the concatenation of all automata in the given iterator.
    #[tracing::instrument(level = "debug", skip_all)]
    pub fn concat_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(
        automata: I,
    ) -> Result<Self, EngineError> {
        let mut new_automaton = FastAutomaton::new_empty_string();
        for automaton in automata {
            new_automaton.concat_mut(automaton)?;
        }

        Ok(new_automaton)
    }

    pub(crate) fn concat_mut(&mut self, other: &FastAutomaton) -> Result<(), EngineError> {
        self.concat_mut_with(other, false)
    }

    /// Concatenation where `force_no_merge` prevents merging `other`'s start
    /// state into `self`'s accept states, always introducing a fresh start
    /// state for `other` reached by epsilon transitions. Used by `repeat` to
    /// keep accept states "clean" when they must remain accepting (so they do
    /// not inherit the next copy's transitions).
    pub(crate) fn concat_mut_with(
        &mut self,
        other: &FastAutomaton,
        force_no_merge: bool,
    ) -> Result<(), EngineError> {
        let execution_profile = ExecutionProfile::get();
        execution_profile.assert_not_timed_out()?;
        execution_profile.assert_max_number_of_states(self.concat_state_count_heuristic(other))?;

        if other.is_empty() {
            self.make_empty();
            return Ok(());
        } else if other.is_empty_string() {
            return Ok(());
        }

        if self.is_empty() {
            self.make_empty();
            return Ok(());
        } else if self.is_empty_string() {
            self.apply_model(other);
            return Ok(());
        }

        let new_spanning_set = &self.spanning_set.merge(&other.spanning_set);
        self.apply_new_spanning_set(new_spanning_set)?;
        let condition_converter = ConditionConverter::new(&other.spanning_set, new_spanning_set)?;

        let mut new_states: IntMap<usize, usize> = IntMap::with_capacity_and_hasher(
            other.number_of_states(),
            BuildHasherDefault::default(),
        );

        let start_state_and_accept_states_not_mergeable = force_no_merge
            || (other.in_degree(other.start_state) > 0
                && self
                    .accept_states
                    .iter()
                    .cloned()
                    .any(|s| self.out_degree(s) > 0));

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

        for from_state in other.states() {
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

            for (condition, to_state) in other.transitions_from(from_state) {
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
                let projected_condition = condition_converter.convert(condition)?;
                for new_from_state in new_from_states.iter() {
                    for new_to_state in new_to_states.iter() {
                        self.add_transition(*new_from_state, *new_to_state, &projected_condition);
                    }
                }
            }
        }

        if start_state_and_accept_states_not_mergeable
            && let Some(&other_start_state) = new_states.get(&other.start_state)
        {
            for accept_state in &accept_states {
                self.add_epsilon_transition(*accept_state, other_start_state);
            }
        }

        self.minimal = false;
        Ok(())
    }

    pub(crate) fn concat_state_count_heuristic(&self, other: &FastAutomaton) -> usize {
        if other.is_empty() {
            return 1;
        } else if other.is_empty_string() {
            return self.number_of_states();
        }

        if self.is_empty() {
            return 1;
        } else if self.is_empty_string() {
            return other.number_of_states();
        }

        // Determine if we are forced to create a new state to avoid unintended loops
        let start_state_and_accept_states_not_mergeable = other.in_degree(other.start_state) > 0
            && self
                .accept_states
                .iter()
                .cloned()
                .any(|s| self.out_degree(s) > 0);

        let v1 = self.number_of_states();
        let v2 = other.number_of_states();

        // Apply the heuristic
        if start_state_and_accept_states_not_mergeable {
            v1 + v2
        } else {
            v1 + v2 - 1
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{fast_automaton::FastAutomaton, regex::RegularExpression};

    #[test]
    fn bug_concat_empty_left() {
        let e = FastAutomaton::new_empty();
        let t = FastAutomaton::new_total();
        let r = e.concat(&t).unwrap();
        assert!(r.is_empty(), "∅ · Σ* must be ∅, got something non-empty");
    }

    #[test]
    fn bug_concat_empty_right() {
        let e = FastAutomaton::new_empty();
        let t = FastAutomaton::new_total();
        let r = t.concat(&e).unwrap();
        assert!(r.is_empty(), "Σ* · ∅ must be ∅, got something non-empty");
    }

    #[test]
    fn bug_term_concat_with_empty() {
        use crate::Term;
        let a = Term::from_automaton(
            RegularExpression::parse("abc", false)
                .unwrap()
                .to_automaton()
                .unwrap(),
        );
        let e = Term::from_automaton(FastAutomaton::new_empty());
        let r = a.concat(&[e]).unwrap();
        assert!(r.is_empty().unwrap(), "'abc' · ∅ must be ∅");
    }

    #[test]
    fn test_simple_concatenation_regex() -> Result<(), String> {
        let automaton = RegularExpression::parse("abc", false)
            .unwrap()
            .to_automaton()
            .unwrap();

        automaton.print_dot();
        assert!(automaton.is_match("abc"));
        assert!(!automaton.is_match("abcd"));
        assert!(!automaton.is_match("ab"));
        assert!(!automaton.is_match(""));
        Ok(())
    }

    #[test]
    fn test_simple_concat_alternation_regex() -> Result<(), String> {
        let automaton = RegularExpression::parse("0101(abc|ac|aaa)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.is_match("0101abc"));
        assert!(automaton.is_match("0101ac"));
        assert!(automaton.is_match("0101aaa"));
        assert!(!automaton.is_match("abc"));
        assert!(!automaton.is_match("0101abcd"));
        assert!(!automaton.is_match("ab"));
        assert!(!automaton.is_match("acc"));
        assert!(!automaton.is_match("a"));
        assert!(!automaton.is_match("aaaa"));
        assert!(!automaton.is_match("aa"));
        assert!(!automaton.is_match(""));
        Ok(())
    }

    #[test]
    fn test_simple_concat_repeat_regex() -> Result<(), String> {
        let automaton = RegularExpression::parse("A+B*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.is_match("AAABBB"));
        assert!(automaton.is_match("AA"));
        assert!(automaton.is_match("AB"));
        assert!(!automaton.is_match("B"));
        assert!(!automaton.is_match("ABA"));
        assert!(!automaton.is_match(""));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_01() -> Result<(), String> {
        let automaton = RegularExpression::parse("a+", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("aa"));
        assert!(automaton.is_match("aaaaaaa"));
        assert!(!automaton.is_match("ab"));
        assert!(!automaton.is_match(""));

        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_02() -> Result<(), String> {
        let automaton = RegularExpression::parse("a*c", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.is_match("c"));
        assert!(automaton.is_match("ac"));
        assert!(automaton.is_match("aac"));
        assert!(automaton.is_match("aaaaaaac"));
        assert!(!automaton.is_match("abc"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_03() -> Result<(), String> {
        let automaton = RegularExpression::parse("(ab){3,4}", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match("ababab"));
        assert!(automaton.is_match("abababab"));
        assert!(!automaton.is_match("ab"));
        assert!(!automaton.is_match("abab"));
        assert!(!automaton.is_match("ababababab"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_04() -> Result<(), String> {
        let automaton = RegularExpression::parse("a{3,}", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match("aaa"));
        assert!(automaton.is_match("aaaaa"));
        assert!(!automaton.is_match("a"));
        assert!(!automaton.is_match("aa"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_05() -> Result<(), String> {
        let automaton = RegularExpression::parse("a?", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("a"));
        assert!(!automaton.is_match("aa"));
        assert!(!automaton.is_match("aaa"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_06() -> Result<(), String> {
        let automaton = RegularExpression::parse("a{0,2}", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("aa"));
        assert!(!automaton.is_match("aaa"));
        assert!(!automaton.is_match("aaaa"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_07() -> Result<(), String> {
        let automaton = RegularExpression::parse("a{1,3}", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(!automaton.is_match(""));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("aa"));
        assert!(automaton.is_match("aaa"));
        assert!(!automaton.is_match("aaaa"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_08() -> Result<(), String> {
        let automaton = RegularExpression::parse("a+(ba+)*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(!automaton.is_match(""));
        assert!(!automaton.is_match("aab"));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("aaa"));
        assert!(automaton.is_match("aba"));
        assert!(automaton.is_match("aaba"));
        assert!(automaton.is_match("aabaaa"));
        assert!(automaton.is_match("aaabaaabaaba"));
        assert!(!automaton.is_match("aaabbaa"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_09() -> Result<(), String> {
        let automaton = RegularExpression::parse("(ac|ads|a)*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("ac"));
        assert!(automaton.is_match("ads"));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("acaadsac"));
        assert!(automaton.is_match("adsaaaaaaaacaa"));
        assert!(!automaton.is_match("as"));
        assert!(!automaton.is_match("ad"));
        assert!(!automaton.is_match("c"));
        assert!(!automaton.is_match("ds"));
        assert!(!automaton.is_match("d"));
        assert!(!automaton.is_match("s"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_10() -> Result<(), String> {
        let automaton = RegularExpression::parse("(ef|ads|a)+", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(!automaton.is_match(""));
        assert!(automaton.is_match("ef"));
        assert!(automaton.is_match("ads"));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("efadsa"));
        assert!(automaton.is_match("aaadsefef"));
        assert!(!automaton.is_match("as"));
        assert!(!automaton.is_match("ad"));
        assert!(!automaton.is_match("e"));
        assert!(!automaton.is_match("ds"));
        assert!(!automaton.is_match("d"));
        assert!(!automaton.is_match("s"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_11() -> Result<(), String> {
        let automaton = RegularExpression::parse("(a|bc)*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("bc"));
        assert!(automaton.is_match("abcbca"));
        assert!(automaton.is_match("bcabcbcaaaa"));
        assert!(!automaton.is_match("b"));
        assert!(!automaton.is_match("c"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_12() -> Result<(), String> {
        let automaton = RegularExpression::parse("([ab]*a)?", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("aa"));
        assert!(automaton.is_match("ba"));
        assert!(automaton.is_match("aba"));
        assert!(automaton.is_match("abbaabbaba"));
        assert!(!automaton.is_match("b"));
        assert!(!automaton.is_match("abab"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_regex_13() -> Result<(), String> {
        let automaton = RegularExpression::parse("([ab]*a)*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("aa"));
        assert!(automaton.is_match("ba"));
        assert!(automaton.is_match("aba"));
        assert!(automaton.is_match("abbaabbaba"));
        assert!(!automaton.is_match("b"));
        assert!(!automaton.is_match("abab"));
        Ok(())
    }

    #[test]
    fn test_simple_repeat_right_number_of_states_1() -> Result<(), String> {
        let automaton = RegularExpression::parse("a*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert_eq!(1, automaton.number_of_states());
        Ok(())
    }

    #[test]
    fn test_simple_concat_right_number_of_states_2() -> Result<(), String> {
        let automaton = RegularExpression::parse("(a*bc)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert_eq!(3, automaton.number_of_states());
        Ok(())
    }

    #[test]
    fn test_heuristic() -> Result<(), String> {
        assert_heuristic(".{900}", "[a-z]+");

        assert_heuristic("[a-z]+@", "[0-9]+[A-Z]*");

        assert_heuristic("a+(ba+)*", "((a|bc)*|d)");

        assert_heuristic(".*", "(ac|ads|a)*");

        assert_heuristic(
            "((aad|ads|a)*|q)",
            r"john[!#-'\*\+\-/-9=\?\^-\u{007e}]*(\.[!#-'\*\+\-/-9=\?\^-\u{007e}](\.?[!#-'\*\+\-/-9=\?\^-\u{007e}])*)?\.?doe@example\.com",
        );

        assert_heuristic(
            "(?:A+(?:\\.[AB]+)*|\"(?:C|\\\\D)*\")@",
            "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@",
        );

        assert_heuristic("((aad|ads|a)*abc.*uif(aad|ads|x)*|q)", ".*");

        assert_heuristic(
            ".{900}",
            r"john[!#-'\*\+\-/-9=\?\^-\u{007e}]*(\.[!#-'\*\+\-/-9=\?\^-\u{007e}](\.?[!#-'\*\+\-/-9=\?\^-\u{007e}])*)?\.?doe@example\.com",
        );

        Ok(())
    }

    fn assert_heuristic(regex1: &str, regex2: &str) {
        println!(
            "Testing concat heuristic for: '{}' and '{}'",
            regex1, regex2
        );

        let automaton1 = RegularExpression::parse(regex1, false)
            .unwrap()
            .to_automaton()
            .unwrap();

        let automaton2 = RegularExpression::parse(regex2, false)
            .unwrap()
            .to_automaton()
            .unwrap();

        // Helper closure to run the test and assert
        let test_pair = |a1: &FastAutomaton, a2: &FastAutomaton, desc: &str| {
            let mut actual_concat = a1.clone();

            // Execute the actual mutation
            actual_concat.concat_mut(a2).unwrap();

            let actual_states = actual_concat.number_of_states();
            let heuristic_states = a1.concat_state_count_heuristic(a2);

            assert_eq!(
                actual_states, heuristic_states,
                "Mismatch for {}.\nExpected (heuristic): {}\nActual (computed): {}",
                desc, heuristic_states, actual_states
            );
        };

        // Test 1: regex1 + regex2
        test_pair(
            &automaton1,
            &automaton2,
            &format!("'{}' + '{}'", regex1, regex2),
        );

        // Test 2: regex2 + regex1 (Reverse order)
        test_pair(
            &automaton2,
            &automaton1,
            &format!("'{}' + '{}'", regex2, regex1),
        );

        // Test 3: regex1 + regex1 (Self-concatenation, crucial for your repeat logic)
        test_pair(
            &automaton1,
            &automaton1,
            &format!("'{}' + '{}' (Self)", regex1, regex1),
        );

        // Test 4 & 5: Empty automaton edge cases
        let empty_automaton = FastAutomaton::new_empty();

        test_pair(
            &empty_automaton,
            &automaton2,
            &format!("Empty + '{}'", regex2),
        );
        test_pair(
            &automaton1,
            &empty_automaton,
            &format!("'{}' + Empty", regex1),
        );
    }
}
//(a|bc)*
