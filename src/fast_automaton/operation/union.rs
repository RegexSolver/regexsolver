use std::hash::BuildHasherDefault;

use condition::converter::ConditionConverter;
use rayon::prelude::*;

use crate::{error::EngineError, execution_profile::ExecutionProfile};

use super::*;

impl FastAutomaton {
    /// Computes the union between `self` and `other`.
    pub fn union(&self, other: &FastAutomaton) -> Result<Self, EngineError> {
        Self::union_all([self, other])
    }

    /// Computes the union of all automata in the given iterator.
    pub fn union_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(
        automata: I,
    ) -> Result<Self, EngineError> {
        let mut new_automaton = FastAutomaton::new_empty();
        for automaton in automata {
            new_automaton.union_mut(automaton)?;
        }
        Ok(new_automaton)
    }

    /// Computes in parallel the union of all automata in the given iterator.
    pub fn union_all_par<'a, I: IntoParallelIterator<Item = &'a FastAutomaton>>(
        automata: I,
    ) -> Result<Self, EngineError> {
        let execution_profile = ExecutionProfile::get();

        let empty = FastAutomaton::new_empty();

        automata
            .into_par_iter()
            .try_fold(
                || empty.clone(),
                |mut acc, next| {
                    execution_profile.apply(|| {
                        acc.union_mut(next)?;
                        Ok(acc)
                    })
                },
            )
            .try_reduce(
                || empty.clone(),
                |mut acc, next| {
                    execution_profile.apply(|| {
                        acc.union_mut(&next)?;
                        Ok(acc)
                    })
                },
            )
    }

    fn prepare_start_states(
        &mut self,
        other: &FastAutomaton,
        new_states: &mut IntMap<usize, usize>,
        condition_converter: &ConditionConverter,
    ) -> Result<IntSet<usize>, EngineError> {
        let mut imcomplete_states = IntSet::with_capacity(other.out_degree(other.start_state) + 1);
        // If `other` accepts the empty string we must make the union's *entry*
        // state accepting — but only after the start state is finalized below.
        // Marking the current start eagerly is wrong when it has incoming edges
        // (e.g. a self-loop) and is about to be demoted behind a fresh start:
        // the demoted state would then wrongly accept the strings on its loop.
        let self_start_state_in_degree = self.in_degree(self.start_state);
        let other_start_state_in_degree = other.in_degree(other.start_state);
        if self_start_state_in_degree == 0 && other_start_state_in_degree == 0 {
            // The start states can be the same state without any consequence
            new_states.insert(other.start_state, self.start_state);
            imcomplete_states.insert(self.start_state);
        } else {
            if self_start_state_in_degree != 0 {
                let new_state = self.new_state();

                self.add_epsilon_transition(new_state, self.start_state);
                self.start_state = new_state;
                new_states.insert(other.start_state, self.start_state);
                imcomplete_states.insert(self.start_state);
            }
            if other_start_state_in_degree != 0 {
                let new_state = self.new_state();
                if other.is_accepted(other.start_state) {
                    self.accept(new_state);
                }

                new_states.insert(other.start_state, new_state);
                imcomplete_states.insert(new_state);

                for (cond, other_to_state) in other.transitions_from_vec(other.start_state) {
                    let cond = condition_converter.convert(&cond)?;
                    let to_state = match new_states.entry(other_to_state) {
                        Entry::Occupied(o) => *o.get(),
                        Entry::Vacant(v) => {
                            let new_state = self.new_state();
                            imcomplete_states.insert(new_state);
                            v.insert(new_state);
                            new_state
                        }
                    };
                    self.add_transition(self.start_state, to_state, &cond);
                }
            }
        }
        // Now that `self.start_state` is the final entry state, record `other`'s
        // empty-string acceptance there. `self`'s own empty-string acceptance is
        // preserved by the start handling above (a freshly created start
        // inherits it through the epsilon transition).
        if other.is_accepted(other.start_state) {
            self.accept(self.start_state);
        }
        Ok(imcomplete_states)
    }

    fn prepare_accept_states(
        &mut self,
        other: &FastAutomaton,
        new_states: &mut IntMap<usize, usize>,
        imcomplete_states: &IntSet<usize>,
    ) {
        let mut self_accept_states_without_outgoing_edges = vec![];
        for &state in &self.accept_states {
            // The start state must never be a merge candidate: the n > 1
            // branch below removes the merged states, and removing the start
            // state panics (e.g. an accepting start with no outgoing edges,
            // unioned with an operand whose start has incoming edges).
            if self.out_degree(state) == 0
                && !imcomplete_states.contains(&state)
                && state != self.start_state
            {
                self_accept_states_without_outgoing_edges.push(state);
            }
        }
        let accept_state_without_outgoing_edges =
            match self_accept_states_without_outgoing_edges.len() {
                1 => Some(self_accept_states_without_outgoing_edges[0]),
                n if n > 1 => {
                    let new_state = self.new_state();
                    self.accept(new_state);

                    for &accept_state in &self_accept_states_without_outgoing_edges {
                        for (from_state, condition) in self.transitions_to_vec(accept_state) {
                            self.add_transition(from_state, new_state, &condition);
                        }
                        self.remove_state(accept_state);
                    }
                    Some(new_state)
                }
                _ => None,
            };

        for &state in &other.accept_states {
            // Resolve the self-state that represents `state`, allocating one if
            // it is not mapped yet, then mark it accepting. The accept flag must
            // be applied even when `state` was already mapped during
            // `prepare_start_states` (e.g. a start state with incoming edges
            // whose outgoing edges reach this accept state); otherwise the
            // union would silently drop `other`'s acceptance.
            let mapped = match accept_state_without_outgoing_edges {
                Some(accept_state) if other.out_degree(state) == 0 => {
                    *new_states.entry(state).or_insert(accept_state)
                }
                _ => match new_states.get(&state) {
                    Some(&mapped) => mapped,
                    None => {
                        let new_accept_state = self.new_state();
                        new_states.insert(state, new_accept_state);
                        new_accept_state
                    }
                },
            };
            self.accept(mapped);
        }
    }

    /* Important things to remember before modifying this method:
     * - the start states can't be merged if they have incoming edges
     * - the accept states can't be merged if they have outgoing edges
     */
    pub(crate) fn union_mut(&mut self, other: &FastAutomaton) -> Result<(), EngineError> {
        ExecutionProfile::get()
            .assert_max_number_of_states(self.union_state_count_heuristic(other))?;

        if other.is_empty() || self.is_total() {
            return Ok(());
        } else if other.is_total() {
            self.make_total();
            return Ok(());
        } else if self.is_empty() {
            self.apply_model(other);
            return Ok(());
        }

        let new_spanning_set = &self.spanning_set.merge(&other.spanning_set);
        self.apply_new_spanning_set(new_spanning_set)?;
        let condition_converter = ConditionConverter::new(&other.spanning_set, new_spanning_set)?;

        let mut new_states: IntMap<usize, usize> = IntMap::with_capacity_and_hasher(
            other.get_number_of_states(),
            BuildHasherDefault::default(),
        );

        let imcomplete_states =
            self.prepare_start_states(other, &mut new_states, &condition_converter)?;
        self.prepare_accept_states(other, &mut new_states, &imcomplete_states);

        for from_state in other.states() {
            let new_from_state = match new_states.entry(from_state) {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => {
                    let new_state = self.new_state();
                    v.insert(new_state);
                    new_state
                }
            };
            for (condition, to_state) in other.transitions_from(from_state) {
                let new_condition = condition_converter.convert(condition)?;
                let new_to_state = match new_states.entry(*to_state) {
                    Entry::Occupied(o) => *o.get(),
                    Entry::Vacant(v) => {
                        let new_state = self.new_state();
                        v.insert(new_state);
                        new_state
                    }
                };
                self.add_transition(new_from_state, new_to_state, &new_condition);
            }
        }
        self.minimal = false;
        Ok(())
    }

    /// Computes the expected number of states after calling `union_mut`.
    fn union_state_count_heuristic(&self, other: &FastAutomaton) -> usize {
        // Edge cases
        if other.is_empty() || self.is_total() {
            return self.get_number_of_states();
        } else if other.is_total() || self.is_empty() {
            return other.get_number_of_states();
        }

        let v1 = self.get_number_of_states();
        let v2 = other.get_number_of_states();

        let self_in = self.in_degree(self.start_state);
        let other_in = other.in_degree(other.start_state);

        let mut total_delta: i32 = 0;

        // --- 1. Start States Math ---
        if self_in == 0 && other_in == 0 {
            total_delta -= 1;
        } else if self_in != 0 && other_in != 0 {
            total_delta += 1;
        }

        // Track which 'other' states are already mapped in the start phase
        // so we don't double-count them when calculating accept state savings.
        let mut mapped_other_states = std::collections::HashSet::new();
        mapped_other_states.insert(other.start_state);

        if other_in != 0 {
            for (_, to_state) in other.transitions_from(other.start_state) {
                mapped_other_states.insert(*to_state);
            }
        }

        // --- 2. Accept States Math ---
        // Gather self's accept states. If other.start_state is accepted,
        // it virtually triggers self.accept(self.start_state) early.
        let mut self_accepts: std::collections::HashSet<usize> =
            self.accept_states.iter().cloned().collect();

        if other.is_accepted(other.start_state) {
            self_accepts.insert(self.start_state);
        }

        let mut n = 0;

        for &state in &self_accepts {
            // Mirror `prepare_accept_states`: a state that is (still) the
            // start after the start-state phase is never a merge candidate.
            // When `self_in != 0` the original start gets demoted behind a
            // fresh start, so it *does* participate.
            let is_excluded = self_in == 0 && state == self.start_state;
            if self.out_degree(state) == 0 && !is_excluded {
                n += 1;
            }
        }

        let has_acc_target = n >= 1;

        // If n > 1, we replace `n` states with exactly 1 unified state.
        if n > 1 {
            total_delta += 1 - n;
        }

        // Calculate mappings for other's accept states
        if has_acc_target {
            for &state in &other.accept_states {
                if other.out_degree(state) == 0 && !mapped_other_states.contains(&state) {
                    total_delta -= 1;
                }
            }
        }

        (v1 as i32 + v2 as i32 + total_delta) as usize
    }
}

#[cfg(test)]
mod tests {
    use crate::{Term, fast_automaton::FastAutomaton, regex::RegularExpression};

    // Regression: unioning with the empty-string language used to drop the
    // other operand's acceptance. When `other`'s start state has incoming edges
    // its outgoing edges (and the accept states they reach) are mapped during
    // `prepare_start_states`; `prepare_accept_states` then failed to mark those
    // already-mapped images accepting, so `union({""}, "a+")` matched only ""
    // instead of "" and "a", "aa", ...
    #[test]
    fn union_with_empty_string_keeps_other_accepts() {
        let empty_string = RegularExpression::parse("", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let a_plus = RegularExpression::parse("a+", false)
            .unwrap()
            .to_automaton()
            .unwrap();

        let u = empty_string.union(&a_plus).unwrap();
        assert!(u.is_match(""), "union must keep \"\"");
        assert!(u.is_match("a"), "union dropped the other operand's language");
        assert!(u.is_match("aaa"));

        // It must be equivalent regardless of operand order.
        let u2 = a_plus.union(&empty_string).unwrap();
        assert!(Term::from_automaton(u).equivalent(&Term::from_automaton(u2)).unwrap());
    }

    // Regression: `prepare_accept_states` merges accept states without
    // outgoing edges and removes the originals. When `self`'s accepting start
    // (no outgoing edges) met an operand whose start has incoming edges, the
    // start landed in the merge list and `remove_state(start)` panicked.
    #[test]
    fn union_does_not_remove_accepting_start() {
        use crate::CharRange;
        use crate::fast_automaton::condition::Condition;
        use crate::fast_automaton::spanning_set::SpanningSet;
        use regex_charclass::char::Char;

        let rng = |c: char| {
            let c = Char::new(c);
            CharRange::new_from_range(c..=c)
        };
        let ss = SpanningSet::compute_spanning_set(&[rng('a'), rng('b')]);

        // a: two accepting states without outgoing edges, one being the start.
        let mut a = FastAutomaton::new_empty();
        a.apply_new_spanning_set(&ss).unwrap();
        a.new_state();
        a.accept(0);
        a.accept(1);

        // b: start has an incoming edge (1 -a-> 0) but no outgoing edges.
        let mut b = FastAutomaton::new_empty();
        b.apply_new_spanning_set(&ss).unwrap();
        b.new_state();
        b.add_transition(1, 0, &Condition::from_range(&rng('a'), &ss).unwrap());
        b.accept(0);

        let u = a.union(&b).unwrap(); // used to panic
        assert!(u.is_match(""), "union must keep the empty string");
    }

    // Regression: unioning a language whose start state has a self-loop with the
    // empty string used to mark that looping start accepting, so `a*b | ""`
    // wrongly matched "a", "aa", ... The empty-string acceptance must land on
    // the union's entry state, not on a demoted looping state.
    #[test]
    fn union_with_empty_string_does_not_over_accept() {
        let a_star_b = RegularExpression::parse("a*b", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let empty_string = RegularExpression::parse("", false)
            .unwrap()
            .to_automaton()
            .unwrap();

        let u = a_star_b.union(&empty_string).unwrap();
        assert!(u.is_match(""), "(a*b)? must match \"\"");
        assert!(u.is_match("b"));
        assert!(u.is_match("ab"));
        assert!(u.is_match("aab"));
        assert!(
            !u.is_match("a"),
            "union wrongly accepted 'a' (looping start marked accepting)"
        );
        assert!(!u.is_match("aa"));
    }

    #[test]
    fn test_simple_alternation_regex_1() -> Result<(), String> {
        let automaton = RegularExpression::parse("(abc|ac|aaa)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        assert!(automaton.is_match("abc"));
        assert!(automaton.is_match("ac"));
        assert!(automaton.is_match("aaa"));
        assert!(!automaton.is_match("abcd"));
        assert!(!automaton.is_match("ab"));
        assert!(!automaton.is_match("acc"));
        assert!(!automaton.is_match("a"));
        assert!(!automaton.is_match("aaaa"));
        assert!(!automaton.is_match("aa"));
        assert!(!automaton.is_match(""));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_2() -> Result<(), String> {
        let automaton = RegularExpression::parse("(b?|b{2})", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("b"));
        assert!(automaton.is_match("bb"));
        assert!(!automaton.is_match("bbb"));
        assert!(!automaton.is_match("bbbb"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_3() -> Result<(), String> {
        let automaton = RegularExpression::parse("((a|bc)*|d)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("abcaaabcbc"));
        assert!(automaton.is_match("d"));
        assert!(!automaton.is_match("ad"));
        assert!(!automaton.is_match("abcd"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_3b() -> Result<(), String> {
        let automaton = RegularExpression::parse("(d|(a|bc)*)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("abcaaabcbc"));
        assert!(automaton.is_match("d"));
        assert!(!automaton.is_match("ad"));
        assert!(!automaton.is_match("abcd"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_3t() -> Result<(), String> {
        let automaton = RegularExpression::parse("(d*|(a|bc)*)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match(""));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("abcaaabcbc"));
        assert!(automaton.is_match("d"));
        assert!(automaton.is_match("ddd"));
        assert!(!automaton.is_match("ad"));
        assert!(!automaton.is_match("abcd"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_4() -> Result<(), String> {
        let automaton = RegularExpression::parse("(a+(ba+)*|ca*c)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match("cc"));
        assert!(automaton.is_match("caaac"));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("aababa"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_5() -> Result<(), String> {
        let automaton = RegularExpression::parse("((aad|ads|a)*|q)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match("q"));
        assert!(automaton.is_match("aad"));
        assert!(automaton.is_match("ads"));
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("aadadsaaa"));
        assert!(!automaton.is_match("aaaas"));
        assert!(!automaton.is_match("ad"));
        assert!(!automaton.is_match("adsq"));
        assert!(!automaton.is_match("qq"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_6() -> Result<(), String> {
        let automaton = RegularExpression::parse("(ab|)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match("ab"));
        assert!(automaton.is_match(""));
        assert!(!automaton.is_match("a"));
        assert!(!automaton.is_match("b"));
        assert!(!automaton.is_match("aab"));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_7() -> Result<(), String> {
        let automaton = RegularExpression::parse("(d|a?|ab)", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match("a"));
        assert!(automaton.is_match("d"));
        assert!(automaton.is_match("ab"));
        assert!(automaton.is_match(""));
        Ok(())
    }

    #[test]
    fn test_simple_alternation_regex_8() -> Result<(), String> {
        let automaton = RegularExpression::parse("((d|a?|ab)u)*", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        automaton.print_dot();
        assert!(automaton.is_match("au"));
        assert!(automaton.is_match("du"));
        assert!(automaton.is_match("abu"));
        assert!(automaton.is_match("u"));
        assert!(automaton.is_match(""));
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
        println!("Testing union heuristic for: '{}' | '{}'", regex1, regex2);

        let automaton1 = RegularExpression::parse(regex1, false)
            .unwrap()
            .to_automaton()
            .unwrap();

        let automaton2 = RegularExpression::parse(regex2, false)
            .unwrap()
            .to_automaton()
            .unwrap();

        let test_pair = |a1: &FastAutomaton, a2: &FastAutomaton, desc: &str| {
            let mut actual_union = a1.clone();
            actual_union.union_mut(a2).unwrap();

            let actual_states = actual_union.get_number_of_states();
            let heuristic_states = a1.union_state_count_heuristic(a2);

            assert_eq!(
                actual_states, heuristic_states,
                "Mismatch for {}.\nExpected (heuristic): {}\nActual (computed): {}",
                desc, heuristic_states, actual_states
            );
        };

        // Test standard union: A | B
        test_pair(
            &automaton1,
            &automaton2,
            &format!("'{}' | '{}'", regex1, regex2),
        );

        // Test reverse union: B | A
        test_pair(
            &automaton2,
            &automaton1,
            &format!("'{}' | '{}'", regex2, regex1),
        );

        // Test self-union: A | A
        test_pair(
            &automaton1,
            &automaton1,
            &format!("'{}' | '{}' (Self)", regex1, regex1),
        );

        // Test Empty states
        let empty_automaton = FastAutomaton::new_empty();

        test_pair(
            &empty_automaton,
            &automaton2,
            &format!("Empty | '{}'", regex2),
        );
        test_pair(
            &automaton1,
            &empty_automaton,
            &format!("'{}' | Empty", regex1),
        );
    }
}
