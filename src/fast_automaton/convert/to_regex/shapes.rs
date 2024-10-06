use std::cmp;

use ahash::AHashSet;
use petgraph::visit::{Bfs, IntoNodeIdentifiers, NodeRef};

use super::*;

impl GraphWrapper {
    pub fn identify_and_convert_shape(
        &mut self,
        execution_profile: &ExecutionProfile,
    ) -> Option<RegularExpression> {
        let result = self.identify_and_convert_dot_star_sequence(execution_profile);
        if result.is_some() {
            return result;
        }

        let result = self.identify_and_convert_self_loop(execution_profile);
        if result.is_some() {
            return result;
        }

        None
    }

    fn identify_and_convert_self_loop(
        &mut self,
        execution_profile: &ExecutionProfile,
    ) -> Option<RegularExpression> {
        if self.graph.node_count() <= 1 {
            return None;
        }

        let edge_index = self.graph.find_edge(self.accept_state, self.start_state)?;

        let weigth = self.graph.remove_edge(edge_index)?;

        let regex;

        if let Ok(v) = self.convert_to_regex(execution_profile) {
            regex = v?;
        } else {
            return None;
        }

        let result;
        if let RegularExpression::Repetition(repeated_regex, min, max_opt) = &regex {
            let regex1 = weigth
                .concat(&repeated_regex.repeat(cmp::max(1, *min), *max_opt), true)
                .repeat(0, None);

            let regex2 = weigth.concat(&regex, true).repeat(0, Some(1));

            result = regex.concat(&regex1, true).concat(&regex2, true);
        } else {
            let regex1 = weigth.concat(&regex, true).repeat(0, None);

            result = regex.concat(&regex1, true);
        }

        Some(result)
    }

    fn identify_and_convert_dot_star_sequence(
        &mut self,
        execution_profile: &ExecutionProfile,
    ) -> Option<RegularExpression> {
        let mut dot_value = Self::weight_to_range(
            self.graph
                .edges_connecting(self.start_state, self.start_state)
                .next()?
                .weight(),
        )?;

        for node in self.graph.node_identifiers() {
            if node == self.start_state {
                continue;
            }
            let weight = Self::weight_to_range(
                self.graph
                    .edges_connecting(node, self.start_state)
                    .next()?
                    .weight(),
            )?;
            if !dot_value.contains_all(&weight) {
                return None;
            }
        }

        let graph = self.graph.clone();
        let edges: Vec<_> = graph
            .edges_directed(self.start_state, Direction::Incoming)
            .collect();

        for edge in edges {
            println!("Here:");
            let v = Self::weight_to_range(edge.weight())?;
            println!("{dot_value} {}/{:?}", v, v);
            dot_value = dot_value.union(&Self::weight_to_range(edge.weight())?);
            self.graph.remove_edge(edge.id());
        }

        let graph = self.graph.clone();
        let mut bfs = Bfs::new(&graph, self.start_state);
        let mut visited = AHashSet::new();
        while let Some(node) = bfs.next(&graph) {
            visited.insert(node.id());

            for edge in graph.edges_directed(node, Direction::Outgoing) {
                dot_value = dot_value.union(&Self::weight_to_range(edge.weight())?);
                if visited.contains(&edge.target())
                    && (self.accept_state != edge.target() || edge.target() == node.id())
                {
                    self.graph.remove_edge(edge.id());
                }
            }
        }

        self.graph.add_edge(
            self.start_state,
            self.start_state,
            RegularExpression::Character(dot_value),
        );

        self.convert_to_regex(execution_profile).unwrap_or_default()
    }

    fn weight_to_range(weight: &RegularExpression) -> Option<Range> {
        if let RegularExpression::Character(range) = weight {
            Some(range.clone())
        } else {
            None
        }
    }
}
