use std::hash::BuildHasherDefault;

use super::*;

impl FastAutomaton {
    /// Returns the cardinality of the automaton (i.e., the number of possible matched strings).
    ///
    /// Works on non-deterministic automata too: acyclic NFAs are determinized
    /// internally (the only fallible step, subject to the
    /// [`crate::execution_profile::ExecutionProfile`] budget, and rejected
    /// with [`EngineError::DeterministicAutomatonRequired`] when the profile
    /// disables implicit determinization).
    ///
    /// As in [`length`](Self::length), only cycles **on accepting
    /// paths** make the count infinite: cycles among dead or unreachable
    /// states don't add a single matched string.
    #[tracing::instrument(level = "debug", skip_all, fields(states = self.number_of_states(), deterministic = self.is_deterministic()))]
    pub fn cardinality(&self) -> Result<Cardinality<u32>, EngineError> {
        if self.is_empty() {
            return Ok(Cardinality::Integer(0));
        } else if self.is_total() {
            return Ok(Cardinality::Infinite);
        }

        // Only states on an accepting path (reachable from the start AND
        // able to reach an accept) contribute strings; everything else is
        // excluded from both the cycle check and the count.
        let live = self.live_states();
        let relevant: IntSet<State> = self
            .forward_reachable_states()
            .intersection(&live)
            .copied()
            .collect();

        // A cycle among relevant states means infinitely many strings.
        // `topological_sorted_states` returns `None` exactly when that
        // subgraph is cyclic and needs no determinism, so this also covers
        // cyclic non-deterministic inputs.
        let topologically_sorted_states = match self.topological_sorted_states(&relevant) {
            None => return Ok(Cardinality::Infinite),
            Some(states) => states,
        };

        // The finite count below assumes deterministic (single-path)
        // transitions. Determinizing an automaton with a finite language
        // yields one whose relevant subgraph is acyclic too, so the
        // recursion takes the deterministic path on the second call.
        if !self.is_deterministic() {
            return self.determinize_implicit()?.cardinality();
        }

        let len = self.transitions.len();
        let mut distances: IntMap<usize, u32> =
            IntMap::with_capacity_and_hasher(len, BuildHasherDefault::default());

        distances.insert(self.start_state, 1);
        for state in topologically_sorted_states {
            let current_distance = *distances.entry(state).or_insert(0);
            if let Some(to_states) = self.transitions.get(state) {
                for (to_state, condition) in to_states {
                    if !relevant.contains(to_state) {
                        continue;
                    }
                    let condition_cardinality = condition.cardinality(&self.spanning_set)?;
                    if let Some(distance) = current_distance.checked_mul(condition_cardinality)
                        && let Some(new_distance) =
                            distances.get(to_state).unwrap_or(&0).checked_add(distance)
                    {
                        distances.insert(*to_state, new_distance);
                        continue;
                    }

                    return Ok(Cardinality::BigInteger);
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
                return Ok(Cardinality::BigInteger);
            }
        }
        Ok(Cardinality::Integer(temp_cardinality))
    }

    /// Kahn's algorithm restricted to the `relevant` subgraph (transitions
    /// with empty conditions can't be taken and are ignored). Returns `None`
    /// when that subgraph contains a cycle.
    fn topological_sorted_states(&self, relevant: &IntSet<State>) -> Option<Vec<usize>> {
        let len = relevant.len();
        let mut in_degree: IntMap<usize, i32> =
            IntMap::with_capacity_and_hasher(len, BuildHasherDefault::default());
        let mut queue = VecDeque::with_capacity(len);
        let mut order = Vec::with_capacity(len);

        let successors = |from_state: State| {
            self.transitions_from(from_state)
                .filter(|(condition, to_state)| {
                    !condition.is_empty() && relevant.contains(to_state)
                })
                .map(|(_, to_state)| *to_state)
        };

        for &from_state in relevant {
            in_degree.entry(from_state).or_insert(0);
            for to_state in successors(from_state) {
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
            for to_state in successors(from_state) {
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

#[cfg(test)]
mod tests {
    use crate::cardinality::Cardinality;
    use crate::fast_automaton::FastAutomaton;
    use crate::fast_automaton::condition::Condition;

    // Regression (found by the brute-force enumeration proptest): the cycle
    // check used to run over ALL states, so a cycle among dead states made
    // the cardinality of a finite language Infinite. Only cycles on
    // accepting paths count.
    #[test]
    fn get_cardinality_ignores_dead_cycles() {
        let mut a = FastAutomaton::new_empty();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let cond = Condition::total(a.spanning_set());
        a.accept(0);
        a.add_transition(0, s1, &cond);
        a.add_transition(s1, s2, &cond);
        a.add_transition(s2, s1, &cond);
        // s1, s2 can't reach an accept → language is {""} only.

        assert_eq!(a.cardinality().unwrap(), Cardinality::Integer(1));
    }

    // Regression: `cardinality` used to `assert!` determinism and panic
    // on acyclic NFAs (the only nondeterministic inputs that reach the finite
    // count; cyclic ones return Infinite earlier). It now determinizes
    // internally.
    #[test]
    fn get_cardinality_determinizes_acyclic_nfas() {
        let mut a = FastAutomaton::new_empty();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let cond = Condition::total(a.spanning_set());
        // Two overlapping transitions from the start: nondeterministic, but
        // both lead to accepting states after exactly one character.
        a.add_transition(0, s1, &cond);
        a.add_transition(0, s2, &cond);
        a.accept(s1);
        a.accept(s2);
        assert!(!a.is_deterministic());

        let cardinality = a.cardinality().unwrap();
        let expected = a.determinize().unwrap().cardinality().unwrap();
        assert_eq!(cardinality, expected);
        assert!(matches!(cardinality, Cardinality::Integer(n) if n > 0));
    }
}
