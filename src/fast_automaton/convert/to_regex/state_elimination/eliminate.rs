use super::*;

impl Gnfa {
    pub(super) fn convert(&mut self) -> RegularExpression {
        if self.empty {
            return RegularExpression::new_empty();
        }

        while let Some(state) = self.get_next_state_to_eliminate() {
            self.eliminate_state(state);
        }

        self.get_transition(self.start_state, self.accept_state)
            .cloned()
            .unwrap_or(RegularExpression::new_empty_string())
    }

    fn get_next_state_to_eliminate(&self) -> Option<usize> {
        let states: Vec<usize> = self
            .all_states_iter()
            .filter(|&s| s != self.start_state && s != self.accept_state)
            .collect();

        states
            .into_par_iter()
            .filter_map(|state| {
                let preds = self.transitions_to_vec(state);
                let succs = self.transitions_from_vec(state);

                let in_deg = preds.len() as u128;
                let out_deg = succs.len() as u128;

                if in_deg == 0 || out_deg == 0 {
                    let score = (state as u128) & 0xFF;
                    return Some((score, state));
                }

                let mut score: u128 = in_deg * out_deg;

                if self.has_self_loop(state) {
                    score = score + (score >> 1);
                }

                let mut label_cost: u128 = 0;

                for (_, regex) in &preds {
                    label_cost += regex.evaluate_complexity() as u128;
                }
                for (regex, _) in &succs {
                    label_cost += regex.evaluate_complexity() as u128;
                }
                if let Some(re) = self.get_transition(state, state) {
                    label_cost += (re.evaluate_complexity() as u128) * 2;
                }

                score = score.saturating_add(label_cost);

                let tie = (state as u128) & 0xFFFF;
                Some((score.saturating_add(tie), state))
            })
            .reduce_with(|a, b| if a.0 < b.0 { a } else { b })
            .map(|(_, state)| state)
    }

    fn eliminate_state(&mut self, k: usize) {
        if self.removed_states.contains(&k) {
            return;
        }

        let in_states = self
            .transitions_in
            .get(&k)
            .unwrap()
            .iter()
            .cloned()
            .filter(|&s| s != k)
            .collect::<Vec<_>>();
        let out_states = self.transitions[k]
            .keys()
            .cloned()
            .filter(|&s| s != k)
            .collect::<Vec<_>>();

        for p in in_states {
            for &q in &out_states {
                self.bridge(p, k, q);
            }
        }

        self.remove_state(k);
    }

    fn bridge(&mut self, p: usize, k: usize, q: usize) {
        let rpk = self.get_transition(p, k);
        let rkk = self.get_transition(k, k);
        let rkq = self.get_transition(k, q);

        if let (Some(rpk), Some(rkq)) = (rpk, rkq) {
            let mut regex = rpk.clone();
            if let Some(rkk) = rkk {
                //regex = RegularExpression::Concat(VecDeque::from_iter(vec![regex, RegularExpression::Repetition(Box::new(rkk.clone()), 0, None)]));
                regex = regex.concat(&rkk.repeat(0, None), true);
            }
            //regex = RegularExpression::Concat(VecDeque::from_iter(vec![regex, rkq.clone()]));
            regex = regex.concat(rkq, true);
            self.add_transition(p, q, regex);
        }
    }
}
