use super::*;

impl FastAutomaton {
    /// Returns the minimum and maximum length of matched strings.
    ///
    /// Cycles are only treated as "language-extending" if they sit on an
    /// accepting path. Cycles among dead states (states that can't reach any
    /// accept) don't extend the language and therefore don't make the max
    /// infinite.
    pub fn get_length(&self) -> (Option<u32>, Option<u32>) {
        if self.is_empty() {
            return (None, None);
        } else if self.is_total() {
            return (Some(0), None);
        }

        // States that lie on some accepting path. Walking only these prunes
        // dead branches whose cycles cannot extend the language.
        let live = self.get_reachable_states();
        if !live.contains(&self.start_state) {
            return (None, None);
        }

        let mut min = None;
        let mut is_infinite = false;

        let mut worklist = VecDeque::with_capacity(self.get_number_of_states());
        worklist.push_back((self.start_state, 0, IntSet::default()));

        while let Some(element) = worklist.pop_front() {
            let state = element.0;
            let length = element.1;
            let mut seen = element.2;
            if min.is_some() && length > min.unwrap() {
                continue;
            }
            if self.accept_states.contains(&state) && (min.is_none() || length < min.unwrap()) {
                min = Some(length);
            }
            seen.insert(state);

            for to_state in self.direct_states(state) {
                if !live.contains(&to_state) {
                    continue;
                }
                if to_state == state || seen.contains(&to_state) {
                    is_infinite = true;
                    continue;
                }
                worklist.push_back((to_state, length + 1, seen.clone()));
            }
        }

        if is_infinite || min.is_none() {
            return (min, None);
        }

        let mut max = None;

        worklist.clear();
        worklist.push_back((self.start_state, 0, IntSet::default()));

        while let Some(element) = worklist.pop_back() {
            let state = element.0;
            let length = element.1;
            let mut seen = element.2;
            if self.accept_states.contains(&state) && (max.is_none() || length > max.unwrap()) {
                max = Some(length);
            }
            seen.insert(state);

            for to_state in self.direct_states(state) {
                if !live.contains(&to_state) {
                    continue;
                }
                if to_state == state || seen.contains(&to_state) {
                    max = None;
                    break;
                }
                worklist.push_back((to_state, length + 1, seen.clone()));
            }
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
}