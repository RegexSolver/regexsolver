use super::*;

impl Gnfa {
    pub(super) fn convert(&mut self) -> Result<RegularExpression, EngineError> {
        if self.empty {
            return Ok(RegularExpression::new_empty());
        }

        let execution_profile = crate::execution_profile::ExecutionProfile::get();

        // Cached elimination scores, indexed by state. Ids are stable during
        // elimination (no state is created; tombstone compaction only pops
        // trailing, already-removed entries), and `None` marks
        // non-candidates: start, accept, and eliminated states.
        //
        // A state's score depends only on its own in/out edges, and
        // eliminating `k` only touches edges incident to k's predecessors
        // and successors — so exactly those need re-scoring each round.
        let mut scores: Vec<Option<u128>> = vec![None; self.transitions.len()];
        for state in self.all_states_iter() {
            if state != self.start_state && state != self.accept_state {
                scores[state] = Some(self.score_state(state));
            }
        }

        loop {
            execution_profile.assert_not_timed_out()?;

            // Lowest score wins; `<=` keeps the last minimal state on ties
            // (the order `convert_reference` pins).
            let mut best: Option<(u128, usize)> = None;
            for (state, &score) in scores.iter().enumerate() {
                if let Some(score) = score
                    && best.is_none_or(|(best_score, _)| score <= best_score)
                {
                    best = Some((score, state));
                }
            }
            let Some((_, k)) = best else {
                break;
            };

            scores[k] = None;
            let (predecessors, successors) = self.eliminate_state(k);
            for state in predecessors.into_iter().chain(successors) {
                if scores[state].is_some() {
                    scores[state] = Some(self.score_state(state));
                }
            }
        }

        // Moved out, not cloned: the `Gnfa` is discarded right after.
        Ok(self
            .transitions
            .get_mut(self.start_state)
            .and_then(|transitions| transitions.remove(&self.accept_state))
            .unwrap_or_else(RegularExpression::new_empty_string))
    }

    /// The elimination score. It reads only `state`'s own degrees and label
    /// complexities, which is what makes the cached, neighbors-only
    /// re-scoring in [`convert`](Self::convert) sound.
    fn score_state(&self, state: usize) -> u128 {
        let mut in_deg: u128 = 0;
        let mut label_cost: u128 = 0;
        for (_, regex) in self.transitions_to_iter(state) {
            in_deg += 1;
            label_cost += regex.evaluate_complexity() as u128;
        }
        let mut out_deg: u128 = 0;
        for (regex, _) in self.transitions_from_iter(state) {
            out_deg += 1;
            label_cost += regex.evaluate_complexity() as u128;
        }

        if in_deg == 0 || out_deg == 0 {
            return (state as u128) & 0xFF;
        }

        let mut score: u128 = in_deg * out_deg;

        if self.has_self_loop(state) {
            score = score + (score >> 1);
        }

        if let Some(re) = self.get_transition(state, state) {
            label_cost += (re.evaluate_complexity() as u128) * 2;
        }

        score = score.saturating_add(label_cost);

        let tie = (state as u128) & 0xFFFF;
        score.saturating_add(tie)
    }

    /// Bridges every predecessor to every successor and removes `k`,
    /// returning those predecessors and successors (`k` excluded): the only
    /// states whose elimination scores the operation changed.
    fn eliminate_state(&mut self, k: usize) -> (Vec<usize>, Vec<usize>) {
        if self.removed_states.contains(&k) {
            return (vec![], vec![]);
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

        // The k→k self-loop star is the same for every (p, q) pair, and
        // bridging never touches the (k, k) edge, so build it once.
        let star = self
            .get_transition(k, k)
            .map(|self_loop| self_loop.repeat(0, None));

        for &p in &in_states {
            for &q in &out_states {
                self.bridge(p, k, q, star.as_ref());
            }
        }

        self.remove_state(k);

        (in_states, out_states)
    }

    fn bridge(&mut self, p: usize, k: usize, q: usize, star: Option<&RegularExpression>) {
        let rpk = self.get_transition(p, k);
        let rkq = self.get_transition(k, q);

        if let (Some(rpk), Some(rkq)) = (rpk, rkq) {
            let mut regex = rpk.clone();
            if let Some(star) = star {
                regex = regex.concat(star, true);
            }
            regex = regex.concat(rkq, true);
            self.add_transition(p, q, regex);
        }
    }
}

#[cfg(test)]
impl Gnfa {
    /// [`convert`](Self::convert) with the score cache disabled: every
    /// candidate is re-scored from scratch each round, with the serial fold
    /// (last minimal state wins on ties) the cached version replaced. The
    /// oracle proving the cache never yields a stale score — i.e. the
    /// produced pattern is identical to the pre-cache implementation's.
    pub(super) fn convert_reference(&mut self) -> Result<RegularExpression, EngineError> {
        if self.empty {
            return Ok(RegularExpression::new_empty());
        }

        loop {
            let best = self
                .all_states_iter()
                .filter(|&s| s != self.start_state && s != self.accept_state)
                .map(|state| (self.score_state(state), state))
                .reduce(|a, b| if a.0 < b.0 { a } else { b });
            let Some((_, state)) = best else {
                break;
            };
            self.eliminate_state(state);
        }

        Ok(self
            .transitions
            .get_mut(self.start_state)
            .and_then(|transitions| transitions.remove(&self.accept_state))
            .unwrap_or_else(RegularExpression::new_empty_string))
    }
}
