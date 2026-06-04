use std::hash::BuildHasherDefault;

use super::*;

impl FastAutomaton {
    /// Returns the cardinality of the automaton (i.e., the number of possible matched strings).
    pub fn get_cardinality(&self) -> Cardinality<u32> {
        if self.is_empty() {
            return Cardinality::Integer(0);
        } else if self.is_total() {
            return Cardinality::Infinite;
        }

        // A cycle means infinitely many strings. `topological_sorted_states`
        // returns `None` exactly when the transition graph is cyclic and needs
        // no determinism, so this also covers cyclic non-deterministic inputs.
        let topologically_sorted_states = match self.topological_sorted_states() {
            None => return Cardinality::Infinite,
            Some(states) => states,
        };

        // The finite count below assumes deterministic (single-path) transitions.
        assert!(
            self.is_deterministic(),
            "The automaton should be deterministic."
        );

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
                    ) && let Some(new_distance) =
                        distances.get(to_state).unwrap_or(&0).checked_add(distance)
                    {
                        distances.insert(*to_state, new_distance);
                        continue;
                    }

                    return Cardinality::BigInteger;
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
                return Cardinality::BigInteger;
            }
        }
        Cardinality::Integer(temp_cardinality)
    }

    fn topological_sorted_states(&self) -> Option<Vec<usize>> {
        let len = self.get_number_of_states();
        let mut in_degree: IntMap<usize, i32> =
            IntMap::with_capacity_and_hasher(len, BuildHasherDefault::default());
        let mut queue = VecDeque::with_capacity(len);
        let mut order = Vec::with_capacity(len);

        for &from_state in &self.states_vec() {
            in_degree.entry(from_state).or_insert(0);
            for to_state in self.direct_states(from_state) {
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
            for to_state in self.direct_states(from_state) {
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
