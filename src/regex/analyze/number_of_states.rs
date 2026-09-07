use std::cmp;

use crate::regex::RegularExpression;

#[derive(Clone, Debug)]
struct AbstractStateMetadata {
    has_incoming_edges: bool,
    has_outgoing_edges: bool,
}

impl AbstractStateMetadata {
    pub(crate) fn new(has_incoming_edges: bool, has_outgoing_edges: bool) -> Self {
        AbstractStateMetadata {
            has_incoming_edges,
            has_outgoing_edges,
        }
    }
}

#[derive(Clone, Debug)]
struct AbstractNFAMetadata {
    start: AbstractStateMetadata,
    accepted: Vec<AbstractStateMetadata>,
    number_of_states: usize,
    /// Whether the language contains the empty string, i.e. the start state
    /// is accepting. Exact for the constructions modeled here.
    accepts_empty_string: bool,
    /// Whether some accept state may sit one transition away from the start
    /// state. Over-approximate (may be `true` when none does), never
    /// under-approximate: `alternate` withholds a merge discount on it, so
    /// erring towards `true` keeps the estimate an upper bound.
    accept_adjacent_to_start: bool,
}

impl AbstractNFAMetadata {
    pub(crate) fn new() -> Self {
        AbstractNFAMetadata {
            start: AbstractStateMetadata::new(false, true),
            accepted: vec![AbstractStateMetadata::new(true, false)],
            number_of_states: 2,
            accepts_empty_string: false,
            accept_adjacent_to_start: true,
        }
    }

    pub(crate) fn new_empty_string() -> Self {
        AbstractNFAMetadata {
            start: AbstractStateMetadata::new(false, false),
            accepted: vec![AbstractStateMetadata::new(false, false)],
            number_of_states: 1,
            accepts_empty_string: true,
            accept_adjacent_to_start: false,
        }
    }

    pub(crate) fn new_empty() -> Self {
        AbstractNFAMetadata {
            start: AbstractStateMetadata::new(false, false),
            accepted: vec![],
            number_of_states: 1,
            accepts_empty_string: false,
            accept_adjacent_to_start: false,
        }
    }

    /// [`accept_adjacent_to_start`](Self::accept_adjacent_to_start) of the
    /// concatenation `self · nfa`: the boundary attaches `nfa`'s structure to
    /// `self`'s accept states, so an accept can only end up next to the start
    /// through an accepting start on one side of the boundary.
    fn concat_accept_adjacency(&self, nfa: &AbstractNFAMetadata) -> bool {
        (nfa.accepts_empty_string && self.accept_adjacent_to_start)
            || (self.accepts_empty_string && nfa.accept_adjacent_to_start)
    }

    pub(crate) fn concat(&self, nfa: &AbstractNFAMetadata) -> Self {
        let is_empty_string = |m: &AbstractNFAMetadata| {
            m.number_of_states == 1 && !m.accepted.is_empty() && !m.start.has_outgoing_edges
        };
        if is_empty_string(nfa) {
            return self.clone();
        }
        if is_empty_string(self) {
            return nfa.clone();
        }

        let start_state_and_accept_states_not_mergeable =
            nfa.start.has_incoming_edges && self.accepted.iter().any(|s| s.has_outgoing_edges);

        if start_state_and_accept_states_not_mergeable {
            AbstractNFAMetadata {
                start: self.start.clone(),
                accepted: nfa.accepted.clone(),
                number_of_states: self.number_of_states.saturating_add(nfa.number_of_states),
                accepts_empty_string: self.accepts_empty_string && nfa.accepts_empty_string,
                accept_adjacent_to_start: self.concat_accept_adjacency(nfa),
            }
        } else {
            AbstractNFAMetadata {
                start: self.start.clone(),
                accepted: nfa.accepted.clone(),
                number_of_states: self.number_of_states.saturating_add(nfa.number_of_states) - 1,
                accepts_empty_string: self.accepts_empty_string && nfa.accepts_empty_string,
                accept_adjacent_to_start: self.concat_accept_adjacency(nfa),
            }
        }
    }

    pub(crate) fn repeat(&self, min: u32, max_opt: &Option<u32>) -> Self {
        // r⁰ = {""} (the empty-string automaton, a single state).
        if max_opt == &Some(0) {
            return Self::new_empty_string();
        }

        if self.accepted.is_empty() {
            return if min == 0 {
                Self::new_empty_string()
            } else {
                self.clone()
            };
        }
        if max_opt.is_none() && min >= 1 {
            let appended_copy_cost = if self.start.has_incoming_edges
                && self.accepted.iter().any(|s| s.has_outgoing_edges)
            {
                self.number_of_states
            } else {
                self.number_of_states - 1
            };
            let mandatory = AbstractNFAMetadata {
                start: self.start.clone(),
                accepted: self.accepted.clone(),
                number_of_states: self
                    .number_of_states
                    .saturating_add((min as usize - 1).saturating_mul(appended_copy_cost)),
                accepts_empty_string: self.accepts_empty_string,
                // An accept of rᵐⁱⁿ can neighbour the start only when it is
                // one copy deep, or when copies collapse over "" ∈ r.
                accept_adjacent_to_start: self.accept_adjacent_to_start
                    && (min == 1 || self.accepts_empty_string),
            };
            return mandatory.concat(&self.repeat(0, &None));
        }

        let start_state_not_mergeable = self.start.has_incoming_edges;
        let accepted_not_mergeable = self.accepted.iter().any(|s| s.has_outgoing_edges);
        let start_state_or_accept_states_not_mergeable =
            start_state_not_mergeable || accepted_not_mergeable;

        let mut return_start = self.start.clone();
        let mut return_accepted = self.accepted.clone();

        if min == 0 && start_state_not_mergeable {
            return_start.has_incoming_edges = false;
        }

        if max_opt.is_none() {
            for accepted in return_accepted.iter_mut() {
                accepted.has_outgoing_edges = true;
            }
        }

        if min == 0
            && !start_state_or_accept_states_not_mergeable
            && max_opt.is_none()
            && self.accepted.len() == 1
        {
            return_start.has_incoming_edges = true;
            return_accepted.push(return_start.clone());

            return AbstractNFAMetadata {
                start: return_start,
                accepted: return_accepted,
                number_of_states: (self.number_of_states - 1).max(1),
                accepts_empty_string: true,
                accept_adjacent_to_start: self.accept_adjacent_to_start,
            };
        }

        if min == 0 {
            return_accepted.push(return_start.clone());
        }

        if let Some(max) = max_opt
            && *max > cmp::max(min, 1)
        {
            return_accepted.push(AbstractStateMetadata::new(true, true));
        }

        let return_number_of_states = if let Some(max) = max_opt {
            // Mirror `repeat_mut`: rᵐⁱⁿ mandatory copies built by merging
            // concatenation, then `max - max(min,1)` optional tail copies. A
            // tail copy whose start has incoming edges is concatenated without
            // merging (a fresh start state, so +`number_of_states`); otherwise
            // it merges (+`number_of_states - 1`).
            let max = *max as usize;
            let merge_cost = if start_state_not_mergeable && accepted_not_mergeable {
                self.number_of_states
            } else {
                self.number_of_states - 1
            };
            let tail_cost = if start_state_not_mergeable {
                self.number_of_states
            } else {
                self.number_of_states - 1
            };

            if min == 0 {
                let base = self
                    .number_of_states
                    .saturating_add(if start_state_not_mergeable { 1 } else { 0 });
                base.saturating_add(max.saturating_sub(1).saturating_mul(tail_cost))
            } else {
                let mandatory = self
                    .number_of_states
                    .saturating_add((min as usize - 1).saturating_mul(merge_cost));
                mandatory.saturating_add(max.saturating_sub(min as usize).saturating_mul(tail_cost))
            }
        } else {
            let mult = if start_state_not_mergeable {
                self.number_of_states
            } else {
                self.number_of_states - 1
            };

            (cmp::max(min, 1) as usize)
                .saturating_mul(mult)
                .saturating_add(1)
        };

        AbstractNFAMetadata {
            start: return_start,
            accepted: return_accepted,
            number_of_states: return_number_of_states,
            accepts_empty_string: min == 0 || self.accepts_empty_string,
            // An accept can neighbour the start only when it is one copy deep
            // (min <= 1), or when copies collapse over "" ∈ r.
            accept_adjacent_to_start: self.accept_adjacent_to_start
                && (min <= 1 || self.accepts_empty_string),
        }
    }

    pub(crate) fn alternate(&mut self, nfa: &AbstractNFAMetadata) -> Self {
        let self_start_state_not_mergeable = self.start.has_incoming_edges;
        let self_accepted_not_mergeable = self.accepted.iter().any(|s| s.has_outgoing_edges);

        let nfa_start_state_not_mergeable = nfa.start.has_incoming_edges;
        let nfa_accepted_not_mergeable = nfa.accepted.iter().any(|s| s.has_outgoing_edges);

        let return_start = AbstractStateMetadata::new(false, true);
        let mut return_accepted = vec![];

        let mut return_number_of_states =
            self.number_of_states.saturating_add(nfa.number_of_states);

        if !self_start_state_not_mergeable && !nfa_start_state_not_mergeable {
            return_number_of_states -= 1;
        } else if self_start_state_not_mergeable && nfa_start_state_not_mergeable {
            return_number_of_states = return_number_of_states.saturating_add(1);
        }

        // A looping start (incoming edges) makes the union materialize
        // the start's direct successors before accept states are merged,
        // and an accept among those successors never merges (e.g. `a*a`,
        // whose accept hangs directly off the looping start). Withhold the
        // saving when an accept may sit there: an upper bound may
        // overshoot, but never undershoot.
        let nfa_accept_beside_looping_start =
            nfa_start_state_not_mergeable && nfa.accept_adjacent_to_start;

        if !self_accepted_not_mergeable
            && !nfa_accepted_not_mergeable
            && !nfa_accept_beside_looping_start
            && !self.accepted.is_empty()
            && !nfa.accepted.is_empty()
            && self.number_of_states > 1
            && nfa.number_of_states > 1
        {
            return_number_of_states -= 1;
            return_accepted.push(AbstractStateMetadata::new(true, false));
        } else {
            let mut extend = |operand: &AbstractNFAMetadata| {
                if operand.number_of_states == 1
                    && !operand.accepted.is_empty()
                    && !operand.start.has_incoming_edges
                {
                    return_accepted.push(AbstractStateMetadata::new(false, true));
                } else {
                    return_accepted.extend(operand.accepted.clone());
                }
            };
            extend(self);
            extend(nfa);
        }

        AbstractNFAMetadata {
            start: return_start,
            accepted: return_accepted,
            accepts_empty_string: self.accepts_empty_string || nfa.accepts_empty_string,
            // The union's entry state carries both operands' start edges.
            accept_adjacent_to_start: self.accept_adjacent_to_start || nfa.accept_adjacent_to_start,
            // Both merge discounts can apply to two single-state {""}
            // operands (e.g. `a{0,0}|b{0,0}`); clamp so the count never
            // reaches zero (see `repeat`).
            number_of_states: return_number_of_states.max(1),
        }
    }
}

impl RegularExpression {
    pub(crate) fn get_number_of_states_in_nfa(&self) -> usize {
        self.evaluate_number_of_states_in_nfa().number_of_states
    }

    fn evaluate_number_of_states_in_nfa(&self) -> AbstractNFAMetadata {
        match self {
            RegularExpression::Character(range) => {
                if range.is_empty() {
                    AbstractNFAMetadata::new_empty()
                } else {
                    AbstractNFAMetadata::new()
                }
            }
            RegularExpression::Repetition(regex, min, max_opt) => regex
                .evaluate_number_of_states_in_nfa()
                .repeat(*min, max_opt),
            RegularExpression::Concat(concat_vec) => {
                if concat_vec.is_empty() {
                    return AbstractNFAMetadata::new_empty_string();
                }
                let mut nfa_metadata = concat_vec[0].evaluate_number_of_states_in_nfa();
                for concat in concat_vec.iter().skip(1) {
                    nfa_metadata = nfa_metadata.concat(&concat.evaluate_number_of_states_in_nfa());
                }
                nfa_metadata
            }
            RegularExpression::Alternation(alternation_vec) => {
                if alternation_vec.is_empty() {
                    return AbstractNFAMetadata::new_empty();
                }
                let mut nfa_metadata = alternation_vec[0].evaluate_number_of_states_in_nfa();
                for alternation in alternation_vec.iter().skip(1) {
                    nfa_metadata =
                        nfa_metadata.alternate(&alternation.evaluate_number_of_states_in_nfa());
                }
                nfa_metadata
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_of_states_in_nfa() -> Result<(), String> {
        assert_number_of_states_in_nfa("(ab|c)+");
        assert_number_of_states_in_nfa("A+");
        assert_number_of_states_in_nfa("B*");
        assert_number_of_states_in_nfa("([ab]*a)");

        assert_number_of_states_in_nfa("ba*");
        assert_number_of_states_in_nfa("b*a");
        assert_number_of_states_in_nfa("(b*a)*");
        assert_number_of_states_in_nfa("(ba*)*");

        assert_number_of_states_in_nfa("(b*a)?");
        assert_number_of_states_in_nfa("(ba*)?");

        assert_number_of_states_in_nfa("(b*a){1,2}");
        assert_number_of_states_in_nfa("(ba*){1,2}");
        assert_number_of_states_in_nfa("(b*a){5,26}");
        assert_number_of_states_in_nfa("(ba*){5,26}");

        // Unbounded with min >= 1 over a self-looping start (r{min,} = rᵐⁱⁿ·r*).
        assert_number_of_states_in_nfa("(b*a){1,}");
        assert_number_of_states_in_nfa("(b*a){2,}");
        assert_number_of_states_in_nfa("(b*a){5,}");
        assert_number_of_states_in_nfa("(a*b){1,}");
        assert_number_of_states_in_nfa("(a*b){3,}");

        // Operands whose start has incoming edges AND whose accept states have
        // outgoing edges: the `min == 0` construction allocates a fresh start
        // with no incoming edges, and the metadata must say so; keeping the
        // flag set overcounted the following concatenation by one state.
        assert_number_of_states_in_nfa("(a*ba*){2,}");
        assert_number_of_states_in_nfa("(a*ba*){5,}");
        assert_number_of_states_in_nfa("(a?b?){2,}");
        assert_number_of_states_in_nfa("a+(b*a)?");
        assert_number_of_states_in_nfa("a+(b*a){0,2}");
        assert_number_of_states_in_nfa("(b*a){0,3}");

        assert_number_of_states_in_nfa("");
        assert_number_of_states_in_nfa("toto");
        assert_number_of_states_in_nfa("A+B*");

        assert_number_of_states_in_nfa("a+");

        assert_number_of_states_in_nfa("ba+");
        assert_number_of_states_in_nfa("ba");
        assert_number_of_states_in_nfa("(ba)*");
        assert_number_of_states_in_nfa("(ba+)*");
        assert_number_of_states_in_nfa("(ab)*");
        assert_number_of_states_in_nfa("(ab){0,3}");
        assert_number_of_states_in_nfa("a*b*");
        assert_number_of_states_in_nfa("(a*b*){0,3}");
        assert_number_of_states_in_nfa(".{1,1000}");
        assert_number_of_states_in_nfa(".{2,3}");

        assert_number_of_states_in_nfa("a+(ba)*");
        assert_number_of_states_in_nfa("a+(ba+)*");
        assert_number_of_states_in_nfa("ca*c");

        assert_number_of_states_in_nfa(".*");
        assert_number_of_states_in_nfa(".?");

        assert_number_of_states_in_nfa("(at?)");
        assert_number_of_states_in_nfa("(ot){3,4}");
        assert_number_of_states_in_nfa("(ot?d){1,4}");

        assert_number_of_states_in_nfa("(ab|ca)");
        assert_number_of_states_in_nfa("q(ab|ca)x");

        assert_number_of_states_in_nfa("(sr)*");
        assert_number_of_states_in_nfa("((sr)*|q)");

        assert_number_of_states_in_nfa("(b*a|ba*|ba)");
        assert_number_of_states_in_nfa("(a+(ba+)*|ca*c)");

        assert_number_of_states_in_nfa("q(ab|ca|ab|abc)x");
        assert_number_of_states_in_nfa("a*(aad|ads|a)abc.*def.*ghi");
        assert_number_of_states_in_nfa(
            "((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q){1,100}",
        );
        Ok(())
    }

    // The unbounded `r{min,}` estimate is a closed form (not a loop over
    // `min`), so a short pattern with a huge minimum like `a{4294967295,}`
    // saturates the estimate and the state budget rejects it immediately,
    // rather than burning CPU inside the estimator.
    #[test]
    fn huge_unbounded_repetition_is_rejected_quickly() {
        use crate::error::EngineError;
        use crate::execution_profile::ExecutionProfileBuilder;

        ExecutionProfileBuilder::new()
            .max_number_of_states(100)
            .build()
            .run(|| {
                let regex = RegularExpression::new("a{4294967295,}").unwrap();
                assert_eq!(
                    EngineError::AutomatonHasTooManyStates,
                    regex.to_automaton().unwrap_err()
                );
            });
    }

    // The estimator's arithmetic saturates, so nested huge (but parseable)
    // bounds do not overflow; the budget then rejects the pattern.
    #[test]
    fn nested_huge_bounds_saturate_instead_of_overflowing() {
        use crate::error::EngineError;
        use crate::execution_profile::ExecutionProfileBuilder;

        ExecutionProfileBuilder::new()
            .max_number_of_states(100)
            .build()
            .run(|| {
                let regex =
                    RegularExpression::new("((a{4294967295}){4294967295}){4294967295}").unwrap();
                assert_eq!(
                    EngineError::AutomatonHasTooManyStates,
                    regex.to_automaton().unwrap_err()
                );

                // A `?`-wrapped saturated repetition whose operand start has
                // incoming edges takes the `min == 0` finite-max arm, whose
                // `+ 1` for the fresh start state must saturate too (it used
                // to be an unchecked add that overflowed in debug builds).
                let regex =
                    RegularExpression::new("((((a*b){4294967295}){4294967295}){4294967295})?")
                        .unwrap();
                assert_eq!(
                    EngineError::AutomatonHasTooManyStates,
                    regex.to_automaton().unwrap_err()
                );
            });
    }

    // Directly-constructed (unsimplified) repetitions over {""}
    // sub-expressions, shapes the string parser simplifies away but any user
    // of the public enum can build, must not drive the abstract state count to
    // zero, which would underflow the merge discounts.
    #[test]
    fn degenerate_repetitions_do_not_underflow() {
        use std::collections::VecDeque;

        let atom = RegularExpression::new("a").unwrap();
        // a{0,0} denotes {""} without being the canonical empty-string form.
        let empty_string = RegularExpression::Repetition(Box::new(atom), 0, Some(0));
        let star_of_alternation = RegularExpression::Repetition(
            Box::new(RegularExpression::Alternation(vec![
                empty_string.clone(),
                empty_string.clone(),
            ])),
            0,
            None,
        );
        let star_of_concat = RegularExpression::Repetition(
            Box::new(RegularExpression::Concat(VecDeque::from([
                empty_string.clone(),
                RegularExpression::Repetition(Box::new(empty_string), 0, None),
            ]))),
            0,
            None,
        );

        for regex in [star_of_alternation, star_of_concat] {
            let estimate = regex.get_number_of_states_in_nfa();
            assert!(estimate >= 1, "state estimate of {regex} must be >= 1");
            let automaton = regex.to_automaton().unwrap();
            assert!(automaton.number_of_states() >= 1);
        }
    }

    fn assert_number_of_states_in_nfa(regex: &str) {
        println!("{}", regex);
        let regex = RegularExpression::new(regex).unwrap();

        let number_of_states_in_nfa = regex.get_number_of_states_in_nfa();

        let automaton = regex.to_automaton().unwrap();

        assert_eq!(automaton.number_of_states(), number_of_states_in_nfa);
    }

    mod prop {
        use super::*;
        use crate::CharRange;
        use crate::execution_profile::ExecutionProfileBuilder;
        use proptest::prelude::*;
        use regex_charclass::char::Char;

        fn letter(c: char) -> CharRange {
            let c = Char::new(c);
            CharRange::new_from_range(c..=c)
        }

        /// Regular-expression trees built directly over the enum, covering
        /// shapes the string parser simplifies away, with occasional huge
        /// repetition bounds to exercise the saturating arithmetic.
        fn arb_regex_tree() -> impl Strategy<Value = RegularExpression> {
            let leaf = prop_oneof![
                4 => Just(letter('a')),
                4 => Just(letter('b')),
                1 => Just(CharRange::empty()),
                1 => Just(CharRange::total()),
            ]
            .prop_map(RegularExpression::Character);
            leaf.prop_recursive(4, 32, 3, |inner| {
                let bound = prop_oneof![
                    8 => 0u32..=3,
                    1 => u32::MAX - 2..=u32::MAX,
                ];
                prop_oneof![
                    3 => (inner.clone(), bound.clone(), bound, any::<bool>()).prop_map(
                        |(r, min, extra, has_max)| {
                            // max is min + extra, so the bounds are always valid
                            let max = if has_max {
                                Some(min.saturating_add(extra))
                            } else {
                                None
                            };
                            RegularExpression::Repetition(Box::new(r), min, max)
                        }
                    ),
                    2 => proptest::collection::vec(inner.clone(), 1..=3)
                        .prop_map(|v| RegularExpression::Concat(v.into())),
                    2 => proptest::collection::vec(inner, 1..=3)
                        .prop_map(RegularExpression::Alternation),
                ]
            })
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// The estimate must never *under*-estimate: the state budget
            /// rejects a pattern when the estimate exceeds it, so an
            /// under-estimate would let an oversized construction through
            /// (the denial-of-service direction). It must also never panic:
            /// huge but parseable bounds have to saturate, not overflow.
            ///
            /// Exactness (`==`) intentionally is not asserted here: the
            /// estimate is exact for the deterministic corpus above but only
            /// an upper bound in general (e.g. `(ab|c)+`).
            #[test]
            fn estimate_is_a_sound_upper_bound(regex in arb_regex_tree()) {
                let estimate = regex.get_number_of_states_in_nfa();
                prop_assert!(estimate >= 1, "state estimate of {} must be >= 1", regex);

                let automaton = ExecutionProfileBuilder::new()
                    .max_number_of_states(4096)
                    .execution_timeout(2000)
                    .build()
                    .run(|| regex.to_automaton());
                if let Ok(automaton) = automaton {
                    prop_assert!(
                        automaton.number_of_states() <= estimate,
                        "the estimate under-estimated {}: estimate {} < actual {}",
                        regex,
                        estimate,
                        automaton.number_of_states()
                    );
                }
            }
        }
    }
}
