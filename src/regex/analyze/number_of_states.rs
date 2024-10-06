use std::cmp;

use crate::regex::RegularExpression;

#[derive(Clone, Debug)]
struct AbstractStateMetadata {
    has_incoming_edges: bool,
    has_outgoing_edges: bool,
}

impl AbstractStateMetadata {
    pub fn new(has_incoming_edges: bool, has_outgoing_edges: bool) -> Self {
        AbstractStateMetadata {
            has_incoming_edges,
            has_outgoing_edges,
        }
    }
}

#[derive(Debug)]
struct AbstractNFAMetadata {
    start: AbstractStateMetadata,
    accepted: Vec<AbstractStateMetadata>,
    number_of_states: usize,
}

impl AbstractNFAMetadata {
    pub fn new() -> Self {
        AbstractNFAMetadata {
            start: AbstractStateMetadata::new(false, true),
            accepted: vec![AbstractStateMetadata::new(true, false)],
            number_of_states: 2,
        }
    }

    pub fn new_empty_string() -> Self {
        AbstractNFAMetadata {
            start: AbstractStateMetadata::new(false, false),
            accepted: vec![AbstractStateMetadata::new(false, false)],
            number_of_states: 1,
        }
    }

    pub fn new_empty() -> Self {
        AbstractNFAMetadata {
            start: AbstractStateMetadata::new(false, false),
            accepted: vec![],
            number_of_states: 1,
        }
    }

    pub fn concat(&self, nfa: &AbstractNFAMetadata) -> Self {
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

    pub fn repeat(&self, min: u32, max_opt: &Option<u32>) -> Self {
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
                    self.number_of_states - 1
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
            let mult = if start_state_not_mergeable && (accepted_not_mergeable || min == 0) {
                self.number_of_states
            } else {
                self.number_of_states - 1
            };

            *max as usize * mult + 1
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

    pub fn alternate(&mut self, nfa: &AbstractNFAMetadata) -> Self {
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
            number_of_states: return_number_of_states,
        }
    }
}

impl RegularExpression {
    pub fn get_number_of_states_in_nfa(&self) -> usize {
        self.evaluate_number_of_states_in_nfa().number_of_states
    }

    fn evaluate_number_of_states_in_nfa(&self) -> AbstractNFAMetadata {
        match self {
            RegularExpression::Character(_) => AbstractNFAMetadata::new(),
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
        assert_number_of_states_in_nfa("((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q){1,100}");
        Ok(())
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
