use super::*;

impl FastAutomaton {
    pub fn get_length(&self) -> (Option<u32>, Option<u32>) {
        if self.is_empty() {
            return (None, None);
        } else if self.is_total(){
            return (Some(0), None);
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

            for to_state in self.transitions_from_state_iter(&state) {
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

            for to_state in self.transitions_from_state_iter(&state) {
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