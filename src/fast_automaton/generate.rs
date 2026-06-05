use crate::{EngineError, execution_profile::ExecutionProfile};
use ahash::{AHashSet, RandomState};
use indexmap::IndexSet;

use super::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Eq, PartialEq)]
struct QueueItem {
    score: usize,
    depth: usize,
    state: usize,
    ranges: Vec<CharRange>,
    hash: u64,
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .cmp(&self.score)
            .then_with(|| self.depth.cmp(&other.depth))
            .then_with(|| self.state.cmp(&other.state))
            .then_with(|| self.hash.cmp(&other.hash))
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl FastAutomaton {
    /// Generates up to `limit` distinct strings matched by the automaton, skipping the first `offset` strings.
    ///
    /// Strings are only guaranteed to be distinct **within a single call**:
    /// the offset fast-skips by counting paths, and in a non-deterministic
    /// automaton the same string can be reached through several paths, so
    /// calls with different offsets may repeat strings (or skip some).
    /// [`determinize`](Self::determinize) (and ideally
    /// [`minimize`](Self::minimize)) first to make pages disjoint.
    pub fn generate_strings(
        &self,
        limit: usize,
        mut offset: usize,
    ) -> Result<Vec<String>, EngineError> {
        if self.is_empty() || limit == 0 {
            return Ok(vec![]);
        }

        let (_, max) = self.get_length();
        let max_len = max.unwrap_or(u32::MAX) as usize;

        let execution_profile = ExecutionProfile::get();
        let num_states = self.transitions.len();

        // -----------------------------------------------------------------
        // 1. REVERSE BFS: Precalculate exact distances to Accept State
        // -----------------------------------------------------------------
        let mut incoming = vec![vec![]; num_states];
        let mut dist_q = std::collections::VecDeque::new();
        let mut dist = vec![usize::MAX; num_states];

        for state in self.states() {
            if self.is_accepted(state as _) {
                dist[state] = 0;
                dist_q.push_back(state);
            }
            for (_cond, &to_state) in self.transitions_from(state as _) {
                incoming[to_state].push(state);
            }
        }

        while let Some(state) = dist_q.pop_front() {
            let d = dist[state];
            for &prev in &incoming[state] {
                if dist[prev] == usize::MAX {
                    dist[prev] = d + 1;
                    dist_q.push_back(prev);
                }
            }
        }

        // -----------------------------------------------------------------
        // 2. A* SEARCH: Find matching strings instantly
        // -----------------------------------------------------------------
        let mut ranges_cache = AHashMap::with_capacity(num_states);
        let mut strings = IndexSet::with_capacity_and_hasher(limit, RandomState::default());
        let mut visited = AHashSet::with_capacity(num_states);

        let mut q = BinaryHeap::new();
        let start_state = self.get_start_state();

        // If the start state can't reach an accept state, exit immediately
        if dist[start_state] != usize::MAX {
            q.push(QueueItem {
                score: dist[start_state],
                depth: 0,
                state: start_state,
                ranges: vec![],
                hash: 0u64,
            });
        }

        while let Some(QueueItem {
            score: _,
            depth: current_depth,
            state,
            mut ranges,
            hash: h,
        }) = q.pop()
        {
            execution_profile.assert_not_timed_out()?;

            if self.is_accepted(state) {
                if current_depth == 0 {
                    if offset > 0 {
                        offset -= 1;
                    } else {
                        strings.insert(String::new());
                    }
                } else {
                    Self::ranges_to_strings(
                        &mut strings,
                        &ranges,
                        limit,
                        &mut offset,
                        &execution_profile,
                    )?;
                }

                if strings.len() >= limit {
                    break;
                }
            }

            if current_depth >= max_len {
                continue;
            }

            let next_depth = current_depth + 1;
            let mut valid_transitions = Vec::new();

            for (cond, &to_state) in self.transitions_from(state) {
                let to_state_usize = to_state;

                // DEAD-END PRUNING: Instantly kill paths that cannot accept
                if dist[to_state_usize] == usize::MAX {
                    continue;
                }

                let hash =
                    Self::path_mix(h, Self::mix64(state as u64 ^ Self::mix64(to_state as u64)));

                if visited.insert((to_state, next_depth, hash)) {
                    let range = ranges_cache
                        .entry(cond)
                        .or_insert_with(|| cond.to_range(&self.spanning_set).unwrap())
                        .clone();

                    valid_transitions.push((to_state_usize, range, hash));
                }
            }

            // Vector Reuse Optimization
            if let Some((last_state, last_range, last_hash)) = valid_transitions.pop() {
                for (to_state, range, hash) in valid_transitions {
                    let mut new_ranges = ranges.clone();
                    new_ranges.push(range);
                    q.push(QueueItem {
                        score: next_depth + dist[to_state], // A* Score Formula
                        depth: next_depth,
                        state: to_state,
                        ranges: new_ranges,
                        hash,
                    });
                }

                ranges.push(last_range);
                q.push(QueueItem {
                    score: next_depth + dist[last_state], // A* Score Formula
                    depth: next_depth,
                    state: last_state,
                    ranges,
                    hash: last_hash,
                });
            }
        }

        Ok(strings.into_iter().collect())
    }

    fn ranges_to_strings(
        strings: &mut IndexSet<String, RandomState>,
        ranges: &Vec<CharRange>,
        count: usize,
        offset: &mut usize,
        execution_profile: &ExecutionProfile,
    ) -> Result<(), EngineError> {
        if strings.len() >= count {
            return Ok(());
        }

        let range_lengths: Vec<usize> = ranges
            .iter()
            .map(|r| r.get_cardinality() as usize)
            .collect();

        let mut total_combinations = 1usize;
        for &len in &range_lengths {
            total_combinations = total_combinations.saturating_mul(len);
        }

        if *offset >= total_combinations {
            *offset -= total_combinations;
            return Ok(());
        }

        let mut current_str = String::with_capacity(ranges.len());
        Self::generate_combinations(
            ranges,
            &range_lengths,
            0,
            &mut current_str,
            strings,
            count,
            offset,
            execution_profile,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_combinations(
        ranges: &Vec<CharRange>,
        range_lengths: &[usize],
        depth: usize,
        current_str: &mut String,
        strings: &mut IndexSet<String, RandomState>,
        count: usize,
        offset: &mut usize,
        execution_profile: &ExecutionProfile,
    ) -> Result<(), EngineError> {
        if strings.len() >= count {
            return Ok(());
        }

        if depth == ranges.len() {
            if *offset > 0 {
                *offset -= 1;
            } else {
                strings.insert(current_str.clone());
            }
            return Ok(());
        }

        // Calculate combinations for the remaining suffix of ranges
        let mut sub_combinations = 1usize;
        for &len in &range_lengths[depth + 1..] {
            sub_combinations = sub_combinations.saturating_mul(len);
        }

        for ch in ranges[depth].clone().iter() {
            execution_profile.assert_not_timed_out()?;

            // If skipping this character's subtree fits within the remaining offset
            if *offset >= sub_combinations {
                *offset -= sub_combinations;
                continue;
            }

            current_str.push(ch.to_char());
            Self::generate_combinations(
                ranges,
                range_lengths,
                depth + 1,
                current_str,
                strings,
                count,
                offset,
                execution_profile,
            )?;
            current_str.pop();

            if strings.len() >= count {
                break;
            }
        }

        Ok(())
    }

    #[inline]
    fn mix64(mut x: u64) -> u64 {
        // splitmix64
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn path_mix(h: u64, x: u64) -> u64 {
        h.wrapping_mul(0x9E3779B97F4A7C15).rotate_left(7) ^ x
    }
}

#[cfg(test)]
mod tests {
    use crate::{cardinality::Cardinality, regex::RegularExpression};
    use regex::Regex;

    #[test]
    fn test_generate_strings_1() -> Result<(), String> {
        let automaton =
            RegularExpression::parse(".*ab.*c(de|fg).*dab.*c(de|fg).*ab.*c(de|fg).*dab.*c", true)
                .unwrap()
                .to_automaton()
                .unwrap();

        let automaton = automaton.determinize().unwrap();
        automaton.generate_strings(30, 0).unwrap();

        Ok(())
    }

    #[test]
    fn test_generate_strings_2() -> Result<(), String> {
        let automaton = RegularExpression::parse("(abc|de){2}", true)
            .unwrap()
            .to_automaton()
            .unwrap();

        let automaton = automaton.determinize().unwrap();
        let strings = automaton.generate_strings(2, 0).unwrap();
        assert_eq!(2, strings.len());

        let strings = automaton.generate_strings(2, 2).unwrap();
        assert_eq!(2, strings.len());

        Ok(())
    }

    #[test]
    fn test_generate_strings_3() -> Result<(), String> {
        assert_generate_strings(r"<([A-Za-z][A-Za-z0-9]*)[^>]*?/>", 500);
        assert_generate_strings("a{100}[a-z]", 100);
        assert_generate_strings("(ab|cd)e", 100);
        assert_generate_strings("[a-z]+", 100);
        assert_generate_strings("[a-z]+@", 100);
        assert_generate_strings("ù", 1000);

        assert_generate_strings("[0-9]+[A-Z]*", 500);
        assert_generate_strings("a+(ba+)*", 200);
        assert_generate_strings("((a|bc)*|d)", 200);
        assert_generate_strings(".*", 50);
        assert_generate_strings("(ac|ads|a)*", 200);
        assert_generate_strings("((aad|ads|a)*|q)", 200);

        assert_generate_strings(
            r"john[!#-'\*\+\-/-9=\?\^-\u{007e}]*(\.[!#-'\*\+\-/-9=\?\^-\u{007e}](\.?[!#-'\*\+\-/-9=\?\^-\u{007e}])*)?\.?doe@example\.com",
            1000,
        );

        assert_generate_strings("(?:A+(?:\\.[AB]+)*|\"(?:C|\\\\D)*\")@", 500);
        assert_generate_strings(
            "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@",
            500,
        );
        assert_generate_strings("((aad|ads|a)*abc.*uif(aad|ads|x)*|q)", 1000);

        Ok(())
    }

    #[test]
    fn test_generate_strings_offset() -> Result<(), String> {
        assert_generate_strings_offset(".{900}");
        assert_generate_strings_offset("[a-z]+");
        assert_generate_strings_offset("[a-z]+@");

        assert_generate_strings_offset("[0-9]+[A-Z]*");
        assert_generate_strings_offset("a+(ba+)*");
        assert_generate_strings_offset("((a|bc)*|d)");
        assert_generate_strings_offset(".*");
        assert_generate_strings_offset("(ac|ads|a)*");
        assert_generate_strings_offset("((aad|ads|a)*|q)");

        assert_generate_strings_offset(
            r"john[!#-'\*\+\-/-9=\?\^-\u{007e}]*(\.[!#-'\*\+\-/-9=\?\^-\u{007e}](\.?[!#-'\*\+\-/-9=\?\^-\u{007e}])*)?\.?doe@example\.com",
        );

        assert_generate_strings_offset("(?:A+(?:\\.[AB]+)*|\"(?:C|\\\\D)*\")@");
        assert_generate_strings_offset(
            "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@",
        );
        assert_generate_strings_offset("((aad|ads|a)*abc.*uif(aad|ads|x)*|q)");

        Ok(())
    }

    fn assert_generate_strings_offset(regex: &str) {
        println!("regex: {regex}");
        let automaton = RegularExpression::parse(regex, false)
            .unwrap()
            .to_automaton()
            .unwrap();

        // Generate 30 strings at once
        let all_strings = automaton.generate_strings(30, 0).unwrap();

        //println!("all_strings {:?}", all_strings);

        // Generate the same 30 strings in chunks of 10
        let chunk1 = automaton.generate_strings(10, 0).unwrap();
        let chunk2 = automaton.generate_strings(10, 10).unwrap();
        let chunk3 = automaton.generate_strings(10, 20).unwrap();

        /*
        println!("chunk1 {:?}", chunk1);
        println!("chunk2 {:?}", chunk2);
        println!("chunk3 {:?}", chunk3);
        */

        assert_eq!(all_strings.len(), 30, "Should generate exactly 30 strings");
        assert_eq!(chunk1.len(), 10);
        assert_eq!(chunk2.len(), 10);
        assert_eq!(chunk3.len(), 10);

        // Combine the chunks
        let mut combined = chunk1;
        combined.extend(chunk2);
        combined.extend(chunk3);

        // Prove that generating in chunks perfectly matches the bulk generation
        assert_eq!(
            all_strings, combined,
            "Chunked generation did not match bulk generation"
        );

        let cardinality = automaton.get_cardinality().unwrap();

        if let Cardinality::Integer(count) = cardinality {
            let empty_chunk = automaton.generate_strings(10, count as usize).unwrap();
            assert!(empty_chunk.is_empty(), "Chunk past limits should be empty");
        }
    }

    fn assert_generate_strings(regex: &str, number: usize) {
        println!(":{}", regex);
        let automaton = RegularExpression::parse(regex, false)
            .unwrap()
            .to_automaton()
            .unwrap();

        let re = Regex::new(&format!("(?s)^{}$", regex)).unwrap();

        // Modified to include an offset of 0
        let strings = automaton.generate_strings(number, 0).unwrap();
        println!("nb of strings: {}/{}", strings.len(), number);
        assert!(number >= strings.len());
        for string in strings {
            if !re.is_match(&string) {
                for byte in string.as_bytes() {
                    print!("{:02x} ", byte);
                }
                panic!("'{string}'")
            }
            assert!(re.is_match(&string), "'{string}'");
        }
    }
}
