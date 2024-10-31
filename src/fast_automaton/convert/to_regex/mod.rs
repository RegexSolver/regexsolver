use std::{
    collections::{hash_map::Entry, VecDeque},
    fmt::Display,
};

use ahash::{HashMapExt, HashSetExt};
use log::error;
use nohash_hasher::IntMap;

use crate::{error::EngineError, execution_profile::ThreadLocalParams, regex::RegularExpression};

use super::{FastAutomaton, IntSet, Range, State};

mod builder;
mod transform;

#[derive(Clone, Debug)]
enum GraphTransition<T> {
    Graph(StateEliminationAutomaton<T>),
    Weight(T),
    Epsilon,
}

impl<T> GraphTransition<T> {
    pub fn is_empty_string(&self) -> bool {
        matches!(self, GraphTransition::Epsilon)
    }

    pub fn get_weight(&self) -> Option<&T> {
        if let GraphTransition::Weight(weight) = self {
            Some(weight)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct StateEliminationAutomaton<T> {
    start_state: usize,
    accept_state: usize,
    transitions: Vec<IntMap<State, GraphTransition<T>>>,
    transitions_in: IntMap<usize, IntSet<usize>>,
    removed_states: IntSet<State>,
    cyclic: bool,
}

impl Display for StateEliminationAutomaton<Range> {
    fn fmt(&self, sb: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_graph_dot(sb, None)
    }
}

impl StateEliminationAutomaton<Range> {
    #[cfg(test)]
    #[inline]
    pub fn to_dot(&self) {
        println!("{}", self);
    }

    #[inline]
    fn to_graph_dot(
        &self,
        sb: &mut std::fmt::Formatter<'_>,
        prefix: Option<&str>,
    ) -> std::fmt::Result {
        let is_subgraph;
        let indent;
        let prefix = if let Some(prefix) = prefix {
            writeln!(sb, "\tsubgraph cluster_{} {{", prefix)?;
            writeln!(sb, "\t\tlabel = \"{} - cyclic={}\";", prefix, self.cyclic)?;
            indent = "\t";
            is_subgraph = true;
            prefix
        } else {
            writeln!(sb, "digraph Automaton {{")?;
            writeln!(sb, "\trankdir = LR;")?;
            writeln!(sb, "\tlabel = \"cyclic={}\";", self.cyclic)?;
            indent = "";
            is_subgraph = false;
            ""
        };

        for from_state in self.states_iter() {
            let from_state_with_prefix = if is_subgraph {
                format!("S{prefix}_{from_state}")
            } else {
                format!("S{from_state}")
            };

            write!(sb, "{indent}\t{}", from_state_with_prefix)?;
            if !is_subgraph && self.accept_state == from_state {
                writeln!(sb, "\t[shape=doublecircle,label=\"{}\"];", from_state)?;
            } else {
                writeln!(sb, "{indent}\t[shape=circle,label=\"{}\"];", from_state)?;
            }

            if !is_subgraph && self.start_state == from_state {
                writeln!(sb, "\tinitial [shape=plaintext,label=\"\"];")?;
                writeln!(sb, "\tinitial -> {}", from_state_with_prefix)?;
            }
            for (to_state, weight) in self.transitions_from_state_enumerate_iter(&from_state) {
                let to_state_with_prefix = if is_subgraph {
                    format!("S{prefix}_{to_state}")
                } else {
                    format!("S{to_state}")
                };

                match weight {
                    GraphTransition::Graph(state_elimination_automaton) => {
                        let subgraph_prefix = if is_subgraph {
                            format!("{prefix}_{from_state}_{to_state}")
                        } else {
                            format!("{from_state}_{to_state}")
                        };
                        state_elimination_automaton.to_graph_dot(sb, Some(&subgraph_prefix))?;
                        writeln!(sb)?;
                        let subgraph_start_state = format!(
                            "S{}_{}",
                            subgraph_prefix, state_elimination_automaton.start_state
                        );
                        writeln!(
                            sb,
                            "{indent}\t{} -> {} [label=\"ε\"]",
                            from_state_with_prefix, subgraph_start_state
                        )?;

                        let subgraph_accept_state = format!(
                            "S{}_{}",
                            subgraph_prefix, state_elimination_automaton.accept_state
                        );
                        writeln!(
                            sb,
                            "{indent}\t{} -> {} [label=\"ε\"]",
                            subgraph_accept_state, to_state_with_prefix
                        )
                    }
                    GraphTransition::Weight(range) => {
                        writeln!(
                            sb,
                            "{indent}\t{} -> {} [label=\"{}\"]",
                            from_state_with_prefix,
                            to_state_with_prefix,
                            RegularExpression::Character(range.clone())
                                .to_string()
                                .replace('\\', "\\\\")
                                .replace('"', "\\\"")
                        )
                    }
                    GraphTransition::Epsilon => writeln!(
                        sb,
                        "{indent}\t{} -> {} [label=\"ε\"]",
                        from_state_with_prefix, to_state_with_prefix
                    ),
                }?;
            }
        }
        write!(sb, "{indent}}}")
    }

    #[inline]
    pub fn states_iter(&self) -> impl Iterator<Item = State> + '_ {
        (0..self.transitions.len()).filter(|s| !self.removed_states.contains(s))
    }

    #[inline]
    pub fn transitions_from_state_enumerate_iter(
        &self,
        from_state: &State,
    ) -> impl Iterator<Item = (&State, &GraphTransition<Range>)> {
        self.transitions[*from_state]
            .iter()
            .filter(|s| !self.removed_states.contains(s.0))
    }

    #[inline]
    pub fn transitions_from_state_vec(&self, from_state: &State) -> Vec<State> {
        self.transitions[*from_state]
            .keys()
            .filter(|s| !self.removed_states.contains(s))
            .copied()
            .collect()
    }

    pub fn in_transitions_vec(&self, to_state: State) -> Vec<(State, GraphTransition<Range>)> {
        let mut in_transitions = vec![];
        for from_state in self.transitions_in.get(&to_state).unwrap_or(&IntSet::new()) {
            for (state, transition) in self.transitions_from_state_enumerate_iter(from_state) {
                if to_state == *state {
                    in_transitions.push((*from_state, transition.clone()));
                }
            }
        }
        in_transitions
    }

    pub fn states_topo_vec(&self) -> Vec<State> {
        if self.cyclic {
            panic!("The graph has a cycle");
        }

        let mut in_degree: IntMap<State, usize> = self
            .transitions_in
            .iter()
            .map(|(state, parents)| (*state, parents.len()))
            .collect();

        let mut worklist: VecDeque<State> = VecDeque::new();
        for (&state, &degree) in &in_degree {
            if degree == 0 {
                worklist.push_back(state);
            }
        }

        let mut sorted_order = Vec::with_capacity(self.get_number_of_states());
        while let Some(state) = worklist.pop_front() {
            sorted_order.push(state);

            if let Some(neighbors) = self.transitions.get(state) {
                let neighbors = neighbors.keys();
                for &neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(&neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            worklist.push_back(neighbor);
                        }
                    }
                }
            }
        }

        if sorted_order.len() == self.get_number_of_states() {
            sorted_order
        } else {
            panic!("The graph has a cycle");
        }
    }

    #[inline]
    pub fn get_number_of_states(&self) -> usize {
        self.transitions.len() - self.removed_states.len()
    }
}

impl FastAutomaton {
    /// Try to convert the current FastAutomaton to a RegularExpression.
    /// If it cannot find an equivalent regex it returns None.
    /// This method is still a work in progress.
    pub fn to_regex(&self) -> Option<RegularExpression> {
        if self.is_empty() {
            return Some(RegularExpression::new_empty());
        }
        let execution_profile = ThreadLocalParams::get_execution_profile();
        if let Ok(graph) = StateEliminationAutomaton::new(self) {
            if let Ok(regex) = graph?.convert_to_regex(&execution_profile) {
                let regex = regex?;
                match regex.to_automaton() {
                    Ok(automaton) => match self.is_equivalent_of(&automaton) {
                        Ok(result) => {
                            if !result {
                                println!("Not equivalent with:");
                                //automaton.to_dot();
                                println!(
                                "The automaton is not equivalent to the generated regex; automaton={} regex={}",
                                serde_json::to_string(self).unwrap(),
                                regex
                            );
                                None
                            } else {
                                Some(regex)
                            }
                        }
                        Err(err) => {
                            println!("{err}");
                            None
                        }
                    },
                    Err(err) => {
                        if let crate::error::EngineError::RegexSyntaxError(_) = err {
                            error!(
                            "The generated regex can not be converted to automaton to be checked for equivalence (Syntax Error); automaton={} regex={}",
                            serde_json::to_string(self).unwrap(),
                            regex
                        );
                        }
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_() -> Result<(), String> {
        let automaton = RegularExpression::new("a+(ba+)*")
            .unwrap()
            .to_automaton()
            .unwrap()
            .determinize()
            .unwrap();

        let automaton = StateEliminationAutomaton::new(&automaton).unwrap().unwrap();

        automaton.to_dot();

        Ok(())
    }

    #[test]
    fn test_convert() -> Result<(), String> {
        assert_convert(".*sf");
        assert_convert(".*sf.*uif(ab|de)");

        assert_convert(".*ab");

        assert_convert("(abc|fg){2}");
        assert_convert("a{2,3}");
        assert_convert("a(bcfe|bcdg|mkv){1,2}");
        assert_convert("a(bcfe|bcdg|mkv){5,6}");
        assert_convert("a(bcfe|bcdg|mkv){0,8}");
        assert_convert("a(bcfe|bcdg|mkv){3,20}");

        assert_convert("a+");

        assert_convert("a*bc*");
        assert_convert("(abc)*(a|ab)");
        assert_convert("a(bcfe|bcdg)*");
        assert_convert(".*abc");
        assert_convert(".*abc.*def");
        assert_convert("a(bcfe|bcdg|mkv)*");

        assert_convert("(bc|a)*");

        assert_convert(".*a(bc|d)");
        assert_convert("abc.*def.*uif(ab|de)");

        assert_convert("(b+a+)*");
        assert_convert("a+(ba+)*");
        Ok(())
    }

    fn assert_convert(regex: &str) {
        let input_regex = RegularExpression::new(regex).unwrap();
        println!("IN                     : {}", input_regex);
        let input_automaton = input_regex.to_automaton().unwrap();

        //input_automaton.to_dot();

        let output_regex = input_automaton.to_regex().unwrap();
        println!("OUT (non deterministic): {}", output_regex);
        let output_automaton = output_regex.to_automaton().unwrap();

        assert!(input_automaton.is_equivalent_of(&output_automaton).unwrap());

        let input_automaton = input_automaton.determinize().unwrap();

        //input_automaton.to_dot();

        let output_regex = input_automaton.to_regex().unwrap();
        println!("OUT (deterministic)    : {}", output_regex);
        let output_automaton = output_regex.to_automaton().unwrap();

        assert!(input_automaton.is_equivalent_of(&output_automaton).unwrap());
    }

    #[test]
    fn test_convert_after_operation_1() -> Result<(), String> {
        let automaton1 = RegularExpression::new("(ab|cd)")
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::new("ab")
            .unwrap()
            .to_automaton()
            .unwrap()
            .determinize()
            .unwrap();

        let result = automaton1.subtraction(&automaton2).unwrap();

        result.to_dot();

        let output_regex = result.to_regex().unwrap();
        assert_eq!("cd", output_regex.to_string());

        Ok(())
    }

    #[test]
    fn test_convert_after_operation_2() -> Result<(), String> {
        let automaton1 = RegularExpression::new("a*")
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::new("b*")
            .unwrap()
            .to_automaton()
            .unwrap();

        let result = automaton1.intersection(&automaton2).unwrap();

        result.to_dot();

        let output_regex = result.to_regex().unwrap();
        assert_eq!("", output_regex.to_string());

        Ok(())
    }

    #[test]
    fn test_convert_after_operation_3() -> Result<(), String> {
        let automaton1 = RegularExpression::new("x*")
            .unwrap()
            .to_automaton()
            .unwrap();
        let automaton2 = RegularExpression::new("(xxx)*")
            .unwrap()
            .to_automaton()
            .unwrap()
            .determinize()
            .unwrap();

        let result = automaton1.subtraction(&automaton2).unwrap();
        result.to_dot();

        let result = result.to_regex().unwrap();

        assert_eq!("(x{3})*x{1,2}", result.to_string());

        Ok(())
    }
}
