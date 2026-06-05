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
}

impl AbstractNFAMetadata {
    pub(crate) fn new() -> Self {
        AbstractNFAMetadata {
            start: AbstractStateMetadata::new(false, true),
            accepted: vec![AbstractStateMetadata::new(true, false)],
            number_of_states: 2,
        }
    }

    pub(crate) fn new_empty_string() -> Self {
        AbstractNFAMetadata {
            start: AbstractStateMetadata::new(false, false),
            accepted: vec![AbstractStateMetadata::new(false, false)],
            number_of_states: 1,
        }
    }

    pub(crate) fn new_empty() -> Self {
        AbstractNFAMetadata {
            start: AbstractStateMetadata::new(false, false),
            accepted: vec![],
            number_of_states: 1,
        }
    }

    pub(crate) fn concat(&self, nfa: &AbstractNFAMetadata) -> Self {
        let start_state_and_accept_states_not_mergeable =
            nfa.start.has_incoming_edges && self.accepted.iter().any(|s| s.has_outgoing_edges);

        if start_state_and_accept_states_not_mergeable {
            AbstractNFAMetadata {
                start: self.start.clone(),
                accepted: nfa.accepted.clone(),
                number_of_states: self.number_of_states + nfa.number_of_states,
            }
        } else {
            AbstractNFAMetadata {
                start: self.start.clone(),
                accepted: nfa.accepted.clone(),
                number_of_states: self.number_of_states + nfa.number_of_states - 1,
            }
        }
    }

    pub(crate) fn repeat(&self, min: u32, max_opt: &Option<u32>) -> Self {
        // r⁰ = {""} (the empty-string automaton, a single state).
        if max_opt == &Some(0) {
            return Self::new_empty_string();
        }

        // Unbounded with min >= 1: `repeat_mut` builds r{min,} = rᵐⁱⁿ · r*.
        // Mirror that here (mandatory copies via merging concatenation, then a
        // recursively-built star) so the predicted count stays consistent with
        // the construction even when the start state has incoming edges.
        if max_opt.is_none() && min >= 1 {
            let mut acc = self.clone();
            for _ in 1..min {
                acc = acc.concat(self);
            }
            return acc.concat(&self.repeat(0, &None));
        }

        let start_state_not_mergeable = self.start.has_incoming_edges;
        let accepted_not_mergeable = self.accepted.iter().any(|s| s.has_outgoing_edges);
        let start_state_or_accept_states_not_mergeable =
            start_state_not_mergeable || accepted_not_mergeable;

        let mut return_start = self.start.clone();
        let mut return_accepted = self.accepted.clone();

        if max_opt.is_none() {
            for accepted in return_accepted.iter_mut() {
                accepted.has_outgoing_edges = true;
            }
        }

        if min == 0 && !start_state_or_accept_states_not_mergeable {
            return_start.has_incoming_edges = true;
            return_accepted.push(return_start.clone());
            if max_opt.is_none() {
                let return_number_of_states = if !start_state_or_accept_states_not_mergeable {
                    // An automaton always has at least one state. Degenerate
                    // sub-expressions denoting {""} (e.g. an unsimplified
                    // `(a{0,0})*`) reach this point with a single state, and
                    // the merge discount must not drive the count to zero —
                    // every later `- 1` in this module relies on counts
                    // staying >= 1.
                    (self.number_of_states - 1).max(1)
                } else {
                    self.number_of_states
                };
                return AbstractNFAMetadata {
                    start: return_start,
                    accepted: return_accepted,
                    number_of_states: return_number_of_states,
                };
            }
        }

        if min == 0 {
            return_accepted.push(return_start.clone());
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
                let base = self.number_of_states + if start_state_not_mergeable { 1 } else { 0 };
                base + max.saturating_sub(1) * tail_cost
            } else {
                let mandatory = self.number_of_states + (min as usize - 1) * merge_cost;
                mandatory + max.saturating_sub(min as usize) * tail_cost
            }
        } else {
            let mult = if start_state_not_mergeable {
                self.number_of_states
            } else {
                self.number_of_states - 1
            };

            cmp::max(min, 1) as usize * mult + 1
        };

        AbstractNFAMetadata {
            start: return_start,
            accepted: return_accepted,
            number_of_states: return_number_of_states,
        }
    }

    pub(crate) fn alternate(&mut self, nfa: &AbstractNFAMetadata) -> Self {
        let self_start_state_not_mergeable = self.start.has_incoming_edges;
        let self_accepted_not_mergeable = self.accepted.iter().any(|s| s.has_outgoing_edges);

        let nfa_start_state_not_mergeable = nfa.start.has_incoming_edges;
        let nfa_accepted_not_mergeable = nfa.accepted.iter().any(|s| s.has_outgoing_edges);

        let return_start = AbstractStateMetadata::new(false, true);
        let mut return_accepted = vec![];

        let mut return_number_of_states = self.number_of_states + nfa.number_of_states;

        if !self_start_state_not_mergeable && !nfa_start_state_not_mergeable {
            return_number_of_states -= 1;
        }

        if !self_accepted_not_mergeable && !nfa_accepted_not_mergeable {
            return_number_of_states -= 1;
            return_accepted.push(AbstractStateMetadata::new(true, false));
        } else {
            return_accepted.extend(self.accepted.clone());
            return_accepted.extend(nfa.accepted.clone());
        }

        AbstractNFAMetadata {
            start: return_start,
            accepted: return_accepted,
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
            RegularExpression::Character(..) => AbstractNFAMetadata::new(),
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
        //TODO:
        //assert_number_of_states_in_nfa("(ab|c)+");
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

    // Regression: directly-constructed (unsimplified) repetitions over {""}
    // sub-expressions — shapes the string parser simplifies away but any user
    // of the public enum can build — used to drive the abstract state count
    // to zero, after which the merge discounts underflowed and panicked.
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
            assert!(automaton.get_number_of_states() >= 1);
        }
    }

    fn assert_number_of_states_in_nfa(regex: &str) {
        println!("{}", regex);
        let regex = RegularExpression::new(regex).unwrap();

        //regex.to_automaton().unwrap().to_dot();

        let number_of_states_in_nfa = regex.get_number_of_states_in_nfa();

        let automaton = regex.to_automaton().unwrap();

        assert_eq!(automaton.get_number_of_states(), number_of_states_in_nfa);
    }
}
