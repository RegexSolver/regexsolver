use std::hash::BuildHasherDefault;

use super::*;

impl FastAutomaton {
    pub fn get_cardinality(&self) -> Option<Cardinality<u32>> {
        if self.is_empty() {
            return Some(Cardinality::Integer(0));
        } else if self.cyclic || self.is_total() {
            return Some(Cardinality::Infinite);
        } else if !self.deterministic {
            return None;
        }

        let topologically_sorted_states = self.topological_sorted_states();
        if topologically_sorted_states.is_none() {
            return Some(Cardinality::Infinite);
        }
        let topologically_sorted_states = topologically_sorted_states.unwrap();

        let len = self.transitions.len();
        let mut distances: IntMap<usize, u32> =
            IntMap::with_capacity_and_hasher(len, BuildHasherDefault::default());

        distances.insert(self.start_state, 1);
        for state in topologically_sorted_states {
            let current_distance = *distances.entry(state).or_insert(0);
            if let Some(to_states) = self.transitions.get(state) {
                for (to_state, condition) in to_states {
                    if let Some(distance) = current_distance.checked_mul(
                        condition
                            .get_cardinality(&self.spanning_set)
                            .expect("It should be possible to get the cardinality of a condition."),
                    ) {
                        if let Some(new_distance) =
                            distances.get(to_state).unwrap_or(&0).checked_add(distance)
                        {
                            distances.insert(*to_state, new_distance);
                            continue;
                        }
                    }

                    return Some(Cardinality::BigInteger);
                }
            }
        }

        let mut temp_cardinality: u32 = 0;
        for accept_state in &self.accept_states {
            if let Some(distance) = distances.get(accept_state) {
                if let Some(add) = temp_cardinality.checked_add(*distance) {
                    temp_cardinality = add;
                    continue;
                }
                return Some(Cardinality::BigInteger);
            }
        }
        Some(Cardinality::Integer(temp_cardinality))
    }

    fn topological_sorted_states(&self) -> Option<Vec<usize>> {
        let len = self.get_number_of_states();
        let mut in_degree: IntMap<usize, i32> =
            IntMap::with_capacity_and_hasher(len, BuildHasherDefault::default());
        let mut queue = VecDeque::with_capacity(len);
        let mut order = Vec::with_capacity(len);

        for from_state in &self.transitions_vec() {
            in_degree.entry(*from_state).or_insert(0);
            for to_state in self.transitions_from_state_iter(from_state) {
                *in_degree.entry(to_state).or_insert(0) += 1;
            }
        }

        for (state, degree) in &in_degree {
            if degree == &0 {
                queue.push_back(*state);
            }
        }

        while let Some(from_state) = queue.pop_front() {
            order.push(from_state);
            for to_state in self.transitions_from_state_iter(&from_state) {
                *in_degree.entry(to_state).or_default() -= 1;

                if in_degree[&to_state] == 0 {
                    queue.push_back(to_state);
                }
            }
        }

        if order.len() != len {
            None
        } else {
            Some(order)
        }
    }
}
