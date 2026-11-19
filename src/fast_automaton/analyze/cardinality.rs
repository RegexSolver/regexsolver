use std::hash::BuildHasherDefault;

use crate::execution_profile::ExecutionProfile;

use super::state_sets::{Id, Operand};
use super::*;

impl FastAutomaton {
    /// Returns the cardinality of the automaton (i.e., the number of possible matched strings).
    ///
    /// Non-deterministic automata are not determinized: when the language is
    /// finite, its strings are counted over the sets of states a word can
    /// reach. That count can visit exponentially many sets; the
    /// [`crate::execution_profile::ExecutionProfile`]'s timeout applies, and
    /// its `max_number_of_states` caps the number of sets visited.
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

        // The count below is a count of paths, which is a count of strings
        // only when a string has a single path.
        if !self.is_deterministic() {
            return self.cardinality_over_state_sets();
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
                    if let Some(distance) = current_distance.checked_mul(condition_cardinality) {
                        let slot = distances.entry(*to_state).or_insert(0);
                        if let Some(new_distance) = slot.checked_add(distance) {
                            *slot = new_distance;
                            continue;
                        }
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

    /// Counts the strings of a non-deterministic automaton whose language is
    /// finite.
    ///
    /// Several paths can spell the same string, so strings are counted from
    /// the *set* of states a word reaches, which is the same for all of its
    /// paths: one if the set holds an accept state, plus, for each character,
    /// the count of the set's successor under that character. Each set is
    /// counted once, as determinization would create it once.
    ///
    /// The count stops at the first set that overflows it: some word reaches
    /// that set, so the language holds at least as many strings.
    fn cardinality_over_state_sets(&self) -> Result<Cardinality<u32>, EngineError> {
        let execution_profile = ExecutionProfile::get();

        let operand = Operand::new(self, 0, None)?;
        let Some(start) = operand.start else {
            return Ok(Cardinality::Integer(0));
        };
        // The number of characters of each base.
        let widths = self
            .spanning_bases()?
            .iter()
            .map(|base| base.cardinality(&self.spanning_set))
            .collect::<Result<Vec<u32>, EngineError>>()?;

        // The number of strings accepted from each set counted so far.
        let mut counts: AHashMap<Vec<Id>, u32> = AHashMap::new();
        let mut successors = vec![Vec::new(); widths.len()];

        // The non-empty successors of a set, each with the width of its base.
        let mut successors_of = |set: &[Id]| -> Vec<(Vec<Id>, u32)> {
            operand.successors(set, &mut successors);
            successors
                .iter()
                .zip(&widths)
                .filter(|(successor, _)| !successor.is_empty())
                .map(|(successor, width)| (successor.clone(), *width))
                .collect()
        };
        // The sets being counted, each waiting for the one above it: a set,
        // its successors, and how many of them are counted already. The
        // language is finite, so a set never waits for itself.
        let mut pending = vec![(vec![start], successors_of(&[start]), 0)];

        loop {
            execution_profile.assert_not_timed_out()?;
            execution_profile.assert_max_number_of_states(counts.len() + pending.len())?;
            let Some((set, next, counted)) = pending.last_mut() else {
                break;
            };

            // Count the first successor that is not counted yet, and come
            // back to this set afterwards.
            if let Some(uncounted) = next[*counted..]
                .iter()
                .position(|(successor, _)| !counts.contains_key(successor))
            {
                *counted += uncounted;
                let successor = next[*counted].0.clone();
                let next = successors_of(&successor);
                pending.push((successor, next, 0));
                continue;
            }

            let mut count = u32::from(operand.is_accepting(set));
            for (successor, width) in next.iter() {
                let Some(sum) = counts[successor]
                    .checked_mul(*width)
                    .and_then(|strings| count.checked_add(strings))
                else {
                    return Ok(Cardinality::BigInteger);
                };
                count = sum;
            }
            let (set, _, _) = pending.pop().expect("`pending` has a last element");
            counts.insert(set, count);
        }

        Ok(Cardinality::Integer(counts[&vec![start]]))
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

    // Only cycles on accepting paths make a language infinite: a cycle among
    // dead states (that cannot reach an accept) must not turn a finite
    // language's cardinality into Infinite.
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

    // `cardinality` returns a finite count for an acyclic NFA (the only
    // nondeterministic input that reaches the finite count; cyclic ones
    // return Infinite earlier) rather than requiring a DFA, and counts each
    // string once however many paths spell it.
    #[test]
    fn get_cardinality_counts_strings_of_acyclic_nfas() {
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
