use super::*;

impl FastAutomaton {
    /// Returns the minimum and maximum length of matched strings.
    ///
    /// Cycles are only treated as "language-extending" if they sit on an
    /// accepting path. Cycles among dead states (states that can't reach any
    /// accept) don't extend the language and therefore don't make the max
    /// infinite.
    ///
    /// Runs in O(V + E): the minimum is a BFS distance; the maximum is a
    /// longest path over the subgraph of states lying on accepting paths,
    /// which is unbounded exactly when that subgraph has a cycle (any such
    /// cycle can be pumped).
    #[must_use]
    pub fn get_length(&self) -> (Option<u32>, Option<u32>) {
        // States that can reach an accept state. If the start state can't,
        // the language is empty.
        let live = self.get_live_states();
        if !live.contains(&self.start_state) {
            return (None, None);
        }

        // BFS from the start over live states only — every state on an
        // accepting path is live, so this loses no accepting path. BFS visits
        // in non-decreasing depth, hence the first accept hit is the minimum.
        // The visited set (reachable ∩ live) is exactly the subgraph relevant
        // for the maximum.
        let mut min = None;
        let mut visited = IntSet::default();
        let mut worklist = VecDeque::with_capacity(self.get_number_of_states());
        visited.insert(self.start_state);
        worklist.push_back((self.start_state, 0u32));
        while let Some((state, length)) = worklist.pop_front() {
            if min.is_none() && self.accept_states.contains(&state) {
                min = Some(length);
            }
            for (condition, to_state) in self.transitions_from(state) {
                if condition.is_empty() || !live.contains(to_state) {
                    continue;
                }
                if visited.insert(*to_state) {
                    worklist.push_back((*to_state, length + 1));
                }
            }
        }

        // Longest path via Kahn's algorithm on the visited subgraph. In the
        // acyclic case the topological order covers all visited states and
        // every state's longest distance is final when it is dequeued.
        let mut in_degree: IntMap<State, u32> = IntMap::default();
        for &from in &visited {
            in_degree.entry(from).or_insert(0);
            for (condition, to_state) in self.transitions_from(from) {
                if condition.is_empty() || !visited.contains(to_state) {
                    continue;
                }
                *in_degree.entry(*to_state).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<State> = in_degree
            .iter()
            .filter(|&(_, &degree)| degree == 0)
            .map(|(&state, _)| state)
            .collect();

        let mut longest: IntMap<State, u32> = IntMap::default();
        longest.insert(self.start_state, 0);
        let mut max = None;
        let mut processed = 0usize;
        while let Some(from) = queue.pop_front() {
            processed += 1;
            let length = *longest.get(&from).unwrap_or(&0);
            if self.accept_states.contains(&from) {
                max = Some(max.map_or(length, |m: u32| m.max(length)));
            }
            for (condition, to_state) in self.transitions_from(from) {
                if condition.is_empty() || !visited.contains(to_state) {
                    continue;
                }
                longest
                    .entry(*to_state)
                    .and_modify(|l| *l = (*l).max(length + 1))
                    .or_insert(length + 1);
                let degree = in_degree
                    .get_mut(to_state)
                    .expect("every visited target was counted above");
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(*to_state);
                }
            }
        }

        if processed != visited.len() {
            // A cycle on an accepting path: matched strings can be pumped
            // arbitrarily, the maximum is unbounded.
            return (min, None);
        }

        (min, max)
    }
}

#[cfg(test)]
mod tests {
    use crate::fast_automaton::FastAutomaton;
    use crate::fast_automaton::condition::Condition;

    // Regression: `get_length` used to set `max = None` on any cycle
    // reachable from start, even dead cycles among non-accepting states that
    // cannot reach an accept. Such cycles don't extend the language; the
    // max must remain finite. Now fixed by filtering branches to the live
    // (co-reachable-from-accept) subgraph.
    #[test]
    fn get_length_handles_dead_cycle() {
        let mut a = FastAutomaton::new_empty();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let cond = Condition::total(a.get_spanning_set());
        a.accept(0);
        a.add_transition(0, s1, &cond);
        a.add_transition(s1, s2, &cond);
        a.add_transition(s2, s1, &cond);
        // s1, s2 not accepting → language is {""} only.

        let (min, max) = a.get_length();
        assert_eq!(min, Some(0), "min length of {{\"\"}} is 0");
        assert_eq!(
            max,
            Some(0),
            "max length of {{\"\"}} is 0; got {max:?} (cycle is dead, shouldn't extend the language)"
        );
    }

    #[test]
    fn get_length_finite_and_infinite() {
        // Chain 0 -> 1 -> 2, accepts {0, 2}: min 0, max 2.
        let mut a = FastAutomaton::new_empty();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let cond = Condition::total(a.get_spanning_set());
        a.add_transition(0, s1, &cond);
        a.add_transition(s1, s2, &cond);
        a.accept(0);
        a.accept(s2);
        assert_eq!(a.get_length(), (Some(0), Some(2)));

        // Live cycle 0 <-> 1, accept {1}: min 1, max unbounded.
        let mut a = FastAutomaton::new_empty();
        let s1 = a.new_state();
        let cond = Condition::total(a.get_spanning_set());
        a.add_transition(0, s1, &cond);
        a.add_transition(s1, 0, &cond);
        a.accept(s1);
        assert_eq!(a.get_length(), (Some(1), None));
    }

    // Regression: `get_length` used to enumerate paths with a cloned `seen`
    // set per branch — exponential time and memory on branching DAGs. A chain
    // of diamonds has 2^k paths; the linear algorithm must handle it
    // instantly.
    #[test]
    fn get_length_linear_on_branching_dag() {
        const DIAMONDS: usize = 24;

        let mut a = FastAutomaton::new_empty();
        let cond = Condition::total(a.get_spanning_set());
        let mut current = 0;
        for _ in 0..DIAMONDS {
            let upper = a.new_state();
            let lower = a.new_state();
            let next = a.new_state();
            a.add_transition(current, upper, &cond);
            a.add_transition(current, lower, &cond);
            a.add_transition(upper, next, &cond);
            a.add_transition(lower, next, &cond);
            current = next;
        }
        a.accept(current);

        let expected = 2 * DIAMONDS as u32;
        assert_eq!(a.get_length(), (Some(expected), Some(expected)));
    }
}
