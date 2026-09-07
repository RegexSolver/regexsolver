use super::*;

impl FastAutomaton {
    /// Computes the repetition of the automaton between `min` and `max_opt` times; if `max_opt` is `None`, the repetition is unbounded.
    #[tracing::instrument(level = "debug", skip(self), fields(states = self.number_of_states(), deterministic = self.is_deterministic(), min = min, max_opt = tracing::field::debug(max_opt)))]
    pub fn repeat(&self, min: u32, max_opt: Option<u32>) -> Result<FastAutomaton, EngineError> {
        let mut automaton = self.clone();
        automaton.repeat_mut(min, max_opt)?;
        Ok(automaton)
    }

    pub(crate) fn repeat_mut(&mut self, min: u32, max_opt: Option<u32>) -> Result<(), EngineError> {
        let execution_profile = ExecutionProfile::get();
        execution_profile.assert_not_timed_out()?;
        execution_profile
            .assert_max_number_of_states(self.repeat_state_count_heuristic(min, max_opt))?;

        if let Some(max) = max_opt
            && min > max
        {
            self.make_empty();
            return Ok(());
        }

        // r⁰ = {""} for any language (max == 0 implies min == 0 here, since
        // min > max already returned above). Without this, the general path
        // below would leave the original language reachable and return
        // L ∪ {""} instead of just {""}.
        if max_opt == Some(0) {
            self.make_empty_string();
            return Ok(());
        }

        // The empty-string language is a fixpoint of repetition: {""}{m,n} = {""}
        // for any valid m ≤ n. Returning early also avoids the unbounded
        // construction below, whose single-state "tight loop" branch would
        // otherwise try to remove the start state and panic.
        if self.is_empty_string() {
            return Ok(());
        }

        // Empty language: ∅⁰ = {""}, ∅ⁿ = ∅ for n ≥ 1. The general algorithm
        // below assumes a non-empty language; bail out before it can panic.
        // This must be the semantic `is_empty()` check, not just
        // `accept_states.is_empty()`: an automaton whose accept states are
        // all unreachable is the empty language too, and the construction
        // below breaks on it (concatenation prunes the dead accepts, leaving
        // stale state ids in the accept frontier).
        if self.is_empty() {
            if min == 0 {
                // ∅⁰ is exactly {""}: replace the whole automaton instead
                // of marking the start accepting: a dead automaton can still
                // have reachable transitions (e.g. a self-loop on a
                // non-accepting start), and an accepting start would wrongly
                // revive them into (label)*.
                self.make_empty_string();
            }
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
                // Through `accept()`, not a direct insert: the language
                // changes (it gains ""), so the `minimal` flag must clear.
                self.accept(self.start_state);
            }
            return Ok(());
        }

        // From here on `self` and `automaton_to_repeat` are known to be
        // neither ∅ nor {""} (checked above), and concatenating two such
        // languages preserves that: the loops call the concatenation core
        // directly, since re-checking the growing chain on every iteration is
        // quadratic.
        let iter = if min == 0 { 0..0 } else { 0..min - 1 };
        for _ in iter {
            self.concat_mut_nondegenerate(&automaton_to_repeat, false)?;
        }

        if max_opt.is_none() {
            if min == 0 {
                // r* with a start state that has no incoming edges (the
                // in_degree > 0 case already returned above): loop the single
                // copy in place by letting each accept state re-enter the
                // start, and make the start accepting.
                let mut star = automaton_to_repeat.clone();

                let accept_state = *star.accept_states.iter().next().unwrap();
                if star.accept_states.len() == 1
                    && star.out_degree(accept_state) == 0
                    && star.in_degree(star.start_state) == 0
                {
                    star.add_epsilon_transition(accept_state, star.start_state);
                    let old_start_state = star.start_state;
                    star.start_state = accept_state;
                    star.remove_state(old_start_state);
                } else {
                    let t = Self::transitions_from_state_set(&star.transitions, star.start_state);
                    let transitions =
                        Self::transitions_from_state_enumerate(&t, &star.removed_states);

                    for state in star.accept_states.clone() {
                        for &(to_state, condition) in &transitions {
                            star.add_transition(state, *to_state, condition);
                        }
                    }

                    star.accept(star.start_state());
                }

                self.apply_model(&star);
            } else {
                // r{min,} = rᵐⁱⁿ · r*. Build the star part via recursion rather
                // than looping `automaton_to_repeat` in place: when the start
                // state has incoming edges, `repeat(0, None)` introduces a
                // clean accepting start instead of marking the looping start
                // accepting, which would otherwise accept partial copies
                // (e.g. `(a*b)+` matching "aaba").
                // The star of a non-degenerate language is non-degenerate: it
                // keeps every string of `r` and gains "".
                let star = automaton_to_repeat.repeat(0, None)?;
                self.concat_mut_nondegenerate(&star, false)?;
            }

            return Ok(());
        }

        // Finite maximum: append the optional copies one at a time, keeping
        // `self` with a single accept frontier so the chain stays linear, and
        // collect each copy boundary in `end_states` to mark accepting at the
        // end (stopping after any copy in `min..=max` is valid).
        //
        // When the copy's start state has incoming edges, merging it into the
        // previous copy's accept state would let that (re-marked accepting)
        // junction inherit the copy's own transitions and accept partial
        // copies (e.g. `(a*b){1,3}` matching "ba"). In that case we force a
        // non-merging concatenation so each boundary is a clean accept state
        // reached by an epsilon transition.
        let force_no_merge = automaton_to_repeat.in_degree(automaton_to_repeat.start_state) > 0;
        let mut end_states = self.accept_states.iter().cloned().collect::<Vec<_>>();
        for _ in cmp::max(min, 1)..max_opt.unwrap() {
            self.concat_mut_nondegenerate(&automaton_to_repeat, force_no_merge)?;
            end_states.extend(self.accept_states.iter());
        }
        for end_state in end_states {
            self.accept(end_state);
        }
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

        // 1b. r⁰ = {""} (a single state); see `repeat_mut`.
        if max_opt == Some(0) {
            return 1;
        }

        let v_original = self.number_of_states();
        if v_original == 0 {
            return 0;
        }

        let mut current_states = v_original;
        let in_deg_start = self.in_degree(self.start_state) > 0;

        // The state delta for a single concatenation, reusing the concat
        // heuristic. It short-circuits to a *smaller* value than `v_original`
        // for degenerate languages (∅ → 1, {""} → the operand size), so the
        // delta must saturate: `repeat_mut` early-returns for those inputs
        // right after this estimate anyway.
        let concat_cost = self
            .concat_state_count_heuristic(self, false)
            .saturating_sub(v_original);

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
            if min == 0 {
                // In-place looped r*: a single accept state with no outgoing
                // edges and an incoming-edge-free start drops the old start
                // state (`v_original - 1`); otherwise the state count is
                // unchanged (the `min == 0 && in_deg_start` case already
                // returned in step 2).
                let mut v_modified = v_original;
                if self.accept_states.len() == 1 {
                    let accept_state = *self.accept_states.iter().next().unwrap();
                    if self.out_degree(accept_state) == 0 && !in_deg_start {
                        v_modified -= 1;
                    }
                }
                return v_modified;
            } else {
                // r{min,} = rᵐⁱⁿ · r*. `current_states` already accounts for
                // the rᵐⁱⁿ part. The star's size and whether its start has
                // incoming edges follow directly from the `repeat_mut(0, None)`
                // construction, so derive them here rather than building the
                // star automaton just to measure it:
                // - start with incoming edges → fresh accepting start
                //   (v + 1 states); epsilon "transitions" copy outgoing edges,
                //   so nothing ever points into the fresh start;
                // - single dead-end accept with an incoming-edge-free start →
                //   in-place loop rooted at the old accept (v - 1 states),
                //   which keeps incoming edges (its surviving predecessors, or
                //   the self-loop copied from a direct start→accept edge);
                // - otherwise → in-place loop (v states) keeping the
                //   incoming-edge-free start.
                let acc_out_gt_0 = self.accept_states.iter().any(|&s| self.out_degree(s) > 0);
                let (star_states, star_start_has_in_edges) = if in_deg_start {
                    (v_original + 1, false)
                } else if self.accept_states.len() == 1
                    && self
                        .accept_states
                        .iter()
                        .next()
                        .is_some_and(|&s| self.out_degree(s) == 0)
                {
                    (v_original - 1, true)
                } else {
                    (v_original, false)
                };
                let not_mergeable = star_start_has_in_edges && acc_out_gt_0;
                let final_concat_cost = if not_mergeable {
                    star_states
                } else {
                    star_states.saturating_sub(1)
                };
                return current_states + final_concat_cost;
            }
        }

        // 6. Finite maximum repetition loop
        //
        // The mandatory copies (handled above) merge as plain `r`. Each
        // optional tail copy merges as well (`v - 1` new states), except when
        // the start state has an incoming edge: the non-merging concatenation
        // then introduces a fresh start state, costing `v` per copy.
        let max = max_opt.unwrap();
        let loop_start = if min > 1 { min } else { 1 };
        let max_iters = max.saturating_sub(loop_start);

        let optional_states = v_original + if in_deg_start { 1 } else { 0 };
        current_states += max_iters as usize * (optional_states - 1);

        current_states
    }
}

#[cfg(test)]
mod tests {
    // Building a large bounded repetition must stay linear in the bound: the
    // per-copy concatenations run against a growing chain, and re-checking
    // that chain's emptiness on every copy made this quadratic (~10 s in
    // debug builds at this size, milliseconds when linear).
    #[test]
    fn repeat_large_bounded_stays_linear() {
        let automaton = crate::regex::RegularExpression::parse("[ab]{5000}", false)
            .unwrap()
            .to_automaton()
            .unwrap();

        assert_eq!(5001, automaton.number_of_states());
        assert!(automaton.is_match(&"ab".repeat(2500)));
        assert!(!automaton.is_match(&"ab".repeat(2499)));
    }

    // Repeating an empty-language automaton must respect ∅* = {""} and
    // ∅ⁿ = ∅ even when the emptiness comes from unreachable accept states or
    // dead-but-reachable transitions (rather than an absent accept set): the
    // repeat must not revive those dead transitions.
    #[test]
    fn repeat_of_unreachable_accept_empty_language() {
        let mut a = crate::fast_automaton::FastAutomaton::new_empty();
        let s1 = a.new_state();
        a.accept(s1); // unreachable accept: the language is ∅
        assert!(a.is_empty());

        let star = a.repeat(0, None).unwrap(); // ∅* = {""}
        assert!(star.is_match(""));
        assert!(!star.is_match("a"));

        assert!(a.repeat(1, Some(2)).unwrap().is_empty()); // ∅{1,2} = ∅
        assert!(a.repeat(2, None).unwrap().is_empty()); // ∅{2,} = ∅

        // A dead automaton with REACHABLE transitions: ∅* must still be
        // exactly {""}, without reviving the dead self-loop into b*.
        let range_b = crate::CharRange::new_from_range(
            regex_charclass::char::Char::new('b')..=regex_charclass::char::Char::new('b'),
        );
        let mut dead_loop = crate::fast_automaton::FastAutomaton::new_empty();
        dead_loop.add_transition_from_range(0, 0, &range_b).unwrap();
        assert!(dead_loop.is_empty());

        let star = dead_loop.repeat(0, None).unwrap();
        assert!(star.is_match(""));
        assert!(!star.is_match("b"), "∅* must not contain \"b\"");
        assert!(dead_loop.repeat(1, None).unwrap().is_empty());
    }

    // Repeating a multi-state empty-language automaton must not underflow the
    // state-count heuristic (the concat heuristic short-circuits ∅ to 1) in
    // the public `repeat` before the empty-language early-return runs.
    #[test]
    fn repeat_of_multi_state_empty_language_does_not_underflow() {
        let mut a = crate::fast_automaton::FastAutomaton::new_empty();
        a.new_state(); // ≥ 2 states, no accept states: the empty language

        let star = a.repeat(0, None).unwrap(); // ∅* = {""}
        assert!(star.is_match(""));
        assert!(!star.is_match("a"));

        let plus = a.repeat(1, None).unwrap(); // ∅⁺ = ∅
        assert!(plus.is_empty());

        let bounded = a.repeat(2, Some(3)).unwrap(); // ∅{2,3} = ∅
        assert!(bounded.is_empty());
    }

    // The r{0,1} fast path changes the language (it gains ""), so it must go
    // through `accept()` and clear the `minimal` flag; otherwise `minimize()`
    // (which trusts the flag) would refuse to minimize the mutated automaton.
    #[test]
    fn repeat_zero_or_one_clears_the_minimal_flag() {
        let mut a = crate::regex::RegularExpression::new("ab")
            .unwrap()
            .to_automaton()
            .unwrap();
        a.minimize().unwrap();
        assert!(a.is_minimal());
        assert!(!a.is_match(""));

        a.repeat_mut(0, Some(1)).unwrap();
        assert!(a.is_match(""));
        assert!(a.is_match("ab"));
        assert!(!a.is_minimal(), "the language changed: the flag must clear");
    }

    use crate::fast_automaton::FastAutomaton;
    use crate::regex::RegularExpression;

    // r⁰ must be exactly {""} for a non-empty language, not L ∪ {""}.
    #[test]
    fn bug_repeat_zero_zero_on_non_empty() {
        let a = RegularExpression::parse("abc", false)
            .unwrap()
            .to_automaton()
            .unwrap();
        let r = a.repeat(0, Some(0)).unwrap();
        assert!(r.is_match(""), "L^0 must contain \"\"");
        assert!(
            !r.is_match("abc"),
            "L^0 must NOT contain L (got 'abc' match)"
        );
    }

    // {""} is a fixpoint of repetition: every bound must return {""} without
    // panicking (in particular the unbounded "tight loop" branch must not try
    // to remove the single state while it is still the start state).
    #[test]
    fn repeat_of_empty_string_is_fixpoint() {
        let empty_string = FastAutomaton::new_empty_string();
        for (min, max) in [
            (0, None),
            (1, None),
            (3, None),
            (0, Some(1)),
            (2, Some(5)),
            (0, Some(0)),
        ] {
            let r = empty_string.repeat(min, max).unwrap();
            assert!(r.is_match(""), "{{\"\"}}{{{min},{max:?}}} must match \"\"");
            assert!(
                !r.is_match("a"),
                "{{\"\"}}{{{min},{max:?}}} must match only \"\""
            );
        }
    }

    // Unbounded repetition of the empty language must not panic (the branch
    // must not assume an accept state exists): ∅* = {""} and ∅⁺ = ∅.
    #[test]
    fn empty_repeat_unbounded_does_not_panic() {
        let empty = FastAutomaton::new_empty();
        // Expected: ∅* = {""}.
        let r = empty.repeat(0, None).expect("should not error");
        assert!(r.is_match(""));
        assert!(!r.is_match("a"));

        // Expected: ∅⁺ = ∅.
        let r = empty.repeat(1, None).expect("should not error");
        assert!(!r.is_match(""));
        assert!(!r.is_match("a"));
    }

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
        assert_heuristic("b*a");
        assert_heuristic("a*b");
        assert_heuristic("ba*");
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

            let actual_states = actual_automaton.number_of_states();
            let heuristic_states = automaton.repeat_state_count_heuristic(min, max_opt);

            assert_eq!(
                actual_states, heuristic_states,
                "Mismatch for regex '{}' with min={}, max={:?}.\nExpected (heuristic): {}\nActual (computed): {}",
                regex, min, max_opt, heuristic_states, actual_states
            );
        }
    }
}
