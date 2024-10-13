use crate::{
    error::EngineError,
    execution_profile::{ExecutionProfile, ThreadLocalParams},
};

use super::*;
use ahash::AHashSet;
use log::error;
use petgraph::{
    algo::tarjan_scc,
    graph::NodeIndex,
    stable_graph::StableGraph,
    visit::{EdgeRef, Topo},
    Direction,
};
use std::hash::BuildHasherDefault;

mod shapes;

#[derive(Clone, Debug)]
struct GraphWrapper {
    graph: StableGraph<u32, RegularExpression>,
    start_state: NodeIndex,
    accept_state: NodeIndex,
    spanning_set: SpanningSet,
    node_index_count: u32,
}

impl GraphWrapper {
    pub fn new(automaton: &FastAutomaton) -> GraphWrapper {
        let mut graph = StableGraph::<u32, RegularExpression>::new();
        let mut added_nodes = IntMap::default();
        let mut count = 0;

        for from_state in automaton.transitions_iter() {
            let from_node = match added_nodes.entry(from_state) {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => {
                    let i = count;
                    count += 1;
                    *v.insert(graph.add_node(i))
                }
            };

            for (&to_state, condition) in
                automaton.transitions_from_state_enumerate_iter(&from_state)
            {
                let to_node = match added_nodes.entry(to_state) {
                    Entry::Occupied(o) => *o.get(),
                    Entry::Vacant(v) => {
                        let i = count;
                        count += 1;
                        *v.insert(graph.add_node(i))
                    }
                };
                graph.add_edge(
                    from_node,
                    to_node,
                    RegularExpression::Character(
                        condition.to_range(&automaton.spanning_set).expect(
                            "The condition should have been able to be converted to range.",
                        ),
                    ),
                );
            }
        }
        let i = count;
        count += 1;
        let new_accept_state = graph.add_node(i);
        for accept_state in &automaton.accept_states {
            graph.add_edge(
                *added_nodes.get(accept_state).unwrap(),
                new_accept_state,
                RegularExpression::new_empty_string(),
            );
        }

        GraphWrapper {
            graph,
            start_state: *added_nodes.get(&automaton.start_state).unwrap(),
            accept_state: new_accept_state,
            spanning_set: automaton.spanning_set.clone(),
            node_index_count: count,
        }
    }

    pub fn add_node(&mut self) -> NodeIndex<u32> {
        let i = self.node_index_count;
        self.node_index_count += 1;

        self.graph.add_node(i)
    }

    fn get_strongly_connected_components(
        graph: &StableGraph<u32, RegularExpression>,
    ) -> Vec<Vec<NodeIndex>> {
        tarjan_scc(&graph)
            .into_iter()
            .filter(|group| group.len() != 1 || graph.contains_edge(group[0], group[0]))
            .collect()
    }

    fn subgraph_from_group(
        &self,
        states: &AHashSet<NodeIndex>,
    ) -> (
        StableGraph<u32, RegularExpression>,
        IntMap<usize, NodeIndex>,
    ) {
        let mut new_graph = StableGraph::new();
        let mut index_map = IntMap::default();

        for &node in states {
            let new_index = new_graph.add_node(*self.graph.node_weight(node).unwrap());
            index_map.insert(node.index(), new_index);
        }

        for &node in states {
            for edge in self.graph.edges(node) {
                let (source, target) = (edge.source(), edge.target());
                if states.contains(&source) && states.contains(&target) {
                    let new_source = *index_map.get(&source.index()).unwrap();
                    let new_target = *index_map.get(&target.index()).unwrap();
                    new_graph.add_edge(new_source, new_target, edge.weight().clone());
                }
            }
        }

        (new_graph, index_map)
    }

    pub fn convert_to_regex(
        &mut self,
        execution_profile: &ExecutionProfile,
    ) -> Result<Option<RegularExpression>, EngineError> {
        let groups = Self::get_strongly_connected_components(&self.graph);

        for group in groups {
            execution_profile.assert_not_timed_out()?;

            let mut group_start_states = AHashSet::new();
            let mut group_accept_states = AHashSet::new();
            for state in &group {
                if state == &self.start_state {
                    group_start_states.insert(*state);
                }
                for edge in self.graph.edges_directed(*state, Direction::Incoming) {
                    if !group.contains(&edge.source()) {
                        group_start_states.insert(*state);
                        break;
                    }
                }

                if state == &self.accept_state {
                    group_accept_states.insert(*state);
                }
                for edge in self.graph.edges_directed(*state, Direction::Outgoing) {
                    if !group.contains(&edge.target()) {
                        group_accept_states.insert(*state);
                        break;
                    }
                }
            }

            let group: AHashSet<_> = group.iter().cloned().collect();

            let (subgraph, group_map) = self.subgraph_from_group(&group);

            for state in &group {
                let mut edges_to_remove = vec![];
                let mut remove_node = true;
                for edge in self.graph.edges(*state) {
                    if group.contains(&edge.source()) && group.contains(&edge.target()) {
                        edges_to_remove.push(edge.id());
                    } else {
                        remove_node = false;
                    }
                }
                edges_to_remove.iter().for_each(|edge_index| {
                    self.graph.remove_edge(*edge_index);
                });
                if remove_node
                    && !group_start_states.contains(state)
                    && !group_accept_states.contains(state)
                {
                    self.graph.remove_node(*state);
                }
            }

            for &group_start_state in &group_start_states {
                let subgraph_start_state = *group_map.get(&group_start_state.index()).unwrap();
                for &group_accept_state in &group_accept_states {
                    let subgraph_accept_state =
                        *group_map.get(&group_accept_state.index()).unwrap();
                    let mut wrapped_graph = GraphWrapper {
                        graph: subgraph.clone(),
                        start_state: subgraph_start_state,
                        accept_state: subgraph_accept_state,
                        spanning_set: self.spanning_set.clone(),
                        node_index_count: self.node_index_count,
                    };

                    let regex;
                    if let Some(v) = wrapped_graph.convert_group(execution_profile) {
                        regex = v;
                    } else {
                        return Ok(None);
                    }

                    if group_start_state == group_accept_state {
                        let new_accept_state = self.add_node();
                        if group_accept_state == self.accept_state {
                            self.accept_state = new_accept_state;
                        }
                        let mut edge_to_add = vec![];
                        for edge in self
                            .graph
                            .edges_directed(group_accept_state, Direction::Outgoing)
                        {
                            edge_to_add.push((edge.target(), edge.weight().clone(), edge.id()));
                        }

                        for edge in edge_to_add {
                            self.graph.add_edge(new_accept_state, edge.0, edge.1);
                            self.graph.remove_edge(edge.2);
                        }
                        self.graph
                            .add_edge(group_start_state, new_accept_state, regex);
                    } else {
                        self.graph
                            .add_edge(group_start_state, group_accept_state, regex);
                    }
                }
            }
        }

        let mut regex_map: IntMap<usize, RegularExpression> = IntMap::with_capacity_and_hasher(
            self.graph.node_count(),
            BuildHasherDefault::default(),
        );
        let mut topo = Topo::new(&self.graph);
        while let Some(node) = topo.next(&self.graph) {
            let current_regex = if let Some(current_regex) = regex_map.get(&node.index()) {
                current_regex.clone()
            } else {
                RegularExpression::new_empty_string()
            };
            for edge in self.graph.edges_directed(node, Direction::Outgoing) {
                let new_regex =
                    current_regex.concat(self.graph.edge_weight(edge.id()).unwrap(), true);
                match regex_map.entry(edge.target().index()) {
                    Entry::Occupied(mut o) => {
                        o.insert(new_regex.union(o.get()).simplify());
                    }
                    Entry::Vacant(v) => {
                        v.insert(new_regex);
                    }
                };
            }
        }

        let result = regex_map.get(&self.accept_state.index()).cloned();

        Ok(result)
    }

    fn convert_group(&mut self, execution_profile: &ExecutionProfile) -> Option<RegularExpression> {
        if self.start_state == self.accept_state {
            let new_accept_state = self.add_node();

            let edges: Vec<_> = self
                .graph
                .edges_directed(self.start_state, Direction::Incoming)
                .map(|e| (e.id(), e.source(), e.weight().clone()))
                .collect();

            for edge in edges {
                self.graph.remove_edge(edge.0);
                self.graph.add_edge(edge.1, new_accept_state, edge.2);
            }
            self.accept_state = new_accept_state;

            if let Ok(regex) = self.convert_to_regex(execution_profile) {
                return Some(regex?.repeat(0, None));
            } else {
                return None;
            }
        }

        self.identify_and_convert_shape(execution_profile)
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
        if let Ok(regex) = GraphWrapper::new(self).convert_to_regex(&execution_profile) {
            let regex = regex?;
            // Checking if correct
            match regex.to_automaton() {
                Ok(automaton) => match self.is_equivalent_of(&automaton) {
                    Ok(result) => {
                        if !result {
                            println!("Not equivalent with:");
                            automaton.to_dot();
                            error!(
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
    }
}

#[cfg(test)]
mod tests {
    use crate::EngineError;

    use super::*;

    #[test]
    fn test_convert() -> Result<(), String> {
        assert_convert("a+");
        assert_convert("a{2,3}");
        assert_convert("(abc|fg){2}");
        assert_convert("a{2,3}");
        assert_convert("a*bc*");
        assert_convert("a(bcfe|bcdg)*");
        assert_convert(".*abc");
        assert_convert(".*abc.*def");
        assert_convert("a(bcfe|bcdg|mkv)*");
        assert_convert("a(bcfe|bcdg|mkv){1,2}");
        assert_convert("a(bcfe|bcdg|mkv){5,6}");
        assert_convert("a(bcfe|bcdg|mkv){0,8}");
        assert_convert("a(bcfe|bcdg|mkv){3,20}");
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
    fn test_dot() -> Result<(), EngineError> {
        let automaton = RegularExpression::new("(ba+)*")?
            .to_automaton()?
            .determinize()
            .unwrap();

        automaton.to_dot();
        Ok(())
    }
}
