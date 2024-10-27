use std::{collections::hash_map::Entry, fmt::Display};

use ahash::{HashMapExt, HashSetExt};
use nohash_hasher::IntMap;

use crate::{error::EngineError, regex::RegularExpression};

use super::{FastAutomaton, IntSet, State};

mod builder;

#[derive(Clone, Debug)]
enum GraphTransition<T> {
    Graph(StateEliminationAutomaton<T>),
    Weight(T),
}

impl GraphTransition<RegularExpression> {
    pub fn is_empty_string(&self) -> bool {
        match self {
            GraphTransition::Graph(_) => false,
            GraphTransition::Weight(regex) => regex.is_empty_string(),
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
}

impl Display for StateEliminationAutomaton<RegularExpression> {
    fn fmt(&self, sb: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_graph_dot(sb, None)
    }
}

impl StateEliminationAutomaton<RegularExpression> {
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
            writeln!(sb, "\t\tlabel = \"{}\";", prefix)?;
            indent = "\t";
            is_subgraph = true;
            prefix
        } else {
            writeln!(sb, "digraph Automaton {{")?;
            writeln!(sb, "\trankdir = LR;")?;
            indent = "";
            is_subgraph = false;
            ""
        };

        for from_state in self.transitions_iter() {
            let from_state_with_prefix = if is_subgraph {
                format!("S{prefix}_{from_state}")
            } else {
                format!("S{from_state}")
            };

            write!(sb, "{indent}\t{}", from_state_with_prefix)?;
            if !is_subgraph && self.accept_state == from_state {
                writeln!(
                    sb,
                    "\t[shape=doublecircle,label=\"{}\"];",
                    from_state
                )?;
            } else {
                writeln!(
                    sb,
                    "{indent}\t[shape=circle,label=\"{}\"];",
                    from_state
                )?;
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
                            "{indent}\t{} -> {} [label=\"\"]",
                            from_state_with_prefix, subgraph_start_state
                        )?;

                        let subgraph_accept_state = format!(
                            "S{}_{}",
                            subgraph_prefix, state_elimination_automaton.accept_state
                        );
                        writeln!(
                            sb,
                            "{indent}\t{} -> {} [label=\"\"]",
                            subgraph_accept_state, to_state_with_prefix
                        )
                    }
                    GraphTransition::Weight(regex) => {
                        writeln!(
                            sb,
                            "{indent}\t{} -> {} [label=\"{}\"]",
                            from_state_with_prefix,
                            to_state_with_prefix,
                            regex.to_string().replace('\\', "\\\\").replace('"', "\\\"")
                        )
                    }
                }?;
            }
        }
        write!(sb, "{indent}}}")
    }

    #[inline]
    pub fn transitions_iter(&self) -> impl Iterator<Item = State> + '_ {
        (0..self.transitions.len()).filter(|s| !self.removed_states.contains(s))
    }

    #[inline]
    pub fn transitions_from_state_enumerate_iter(
        &self,
        from_state: &State,
    ) -> impl Iterator<Item = (&State, &GraphTransition<RegularExpression>)> {
        self.transitions[*from_state]
            .iter()
            .filter(|s| !self.removed_states.contains(s.0))
    }

    pub fn in_transitions_vec(
        &self,
        to_state: State,
    ) -> Vec<(State, GraphTransition<RegularExpression>)> {
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
}

#[cfg(test)]
mod tests {
    use crate::EngineError;

    use super::*;

    #[test]
    fn test_convert() -> Result<(), String> {
        let automaton = RegularExpression::new("(ab?)*")
            .unwrap()
            .to_automaton()
            .unwrap()
            .determinize()
            .unwrap();

        let automaton = StateEliminationAutomaton::new(&automaton).unwrap().unwrap();

        automaton.to_dot();

        Ok(())
    }
}
