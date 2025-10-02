use crate::{EngineError, execution_profile::ExecutionProfile};
use ahash::AHashSet;

use super::*;

impl FastAutomaton {
    /// Generates `count` strings matched by the automaton.
    pub fn generate_strings(&self, count: usize) -> Result<Vec<String>, EngineError> {
        if self.is_empty() {
            return Ok(vec![]);
        }

        let (min, max) = self.get_length();
        let max_len = if let Some(max) = max {
            max
        } else {
            let min = min.expect("A non empty automaton should have a minimum length");
            min.saturating_add(100)
        } as usize;

        let execution_profile = ExecutionProfile::get();

        let mut ranges_cache = AHashMap::with_capacity(self.get_number_of_states());
        let mut strings = AHashSet::with_capacity(count);
        let mut visited = AHashSet::with_capacity(self.get_number_of_states());
        let mut q = VecDeque::with_capacity(self.get_number_of_states());
        q.push_back((self.get_start_state(), vec![], 0u64));
        while let Some((state, ranges, h)) = q.pop_front() {
            execution_profile.assert_not_timed_out()?;

            if ranges.len() > max_len {
                continue;
            }

            if self.is_accepted(state) {
                if ranges.is_empty() {
                    strings.insert(String::new());
                } else {
                    Self::ranges_to_strings(&mut strings, &ranges, count, &execution_profile)?;
                }

                if strings.len() >= count {
                    break;
                }
            }

            for (cond, &to_state) in self.transitions_from(state) {
                let hash =
                    Self::path_mix(h, Self::mix64(state as u64 ^ Self::mix64(to_state as u64)));

                if visited.insert((to_state, ranges.len() + 1, hash)) {
                    let mut new_ranges = ranges.clone();
                    new_ranges.push(
                        ranges_cache
                            .entry(cond)
                            .or_insert_with(|| cond.to_range(&self.spanning_set).unwrap())
                            .clone(),
                    );

                    q.push_back((to_state, new_ranges, hash));
                }
            }
        }
        let mut strings: Vec<String> = strings.into_iter().collect();
        strings.sort_unstable_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        Ok(strings)
    }

    pub fn ranges_to_strings(
        strings: &mut AHashSet<String>,
        ranges: &Vec<CharRange>,
        count: usize,
        execution_profile: &ExecutionProfile,
    ) -> Result<(), EngineError> {
        let n = count - strings.len();
        if n == 0 {
            return Ok(());
        }

        let mut end = false;
        let mut out: Vec<String> = Vec::with_capacity(n);
        out.push(String::with_capacity(ranges.len()));
        for r in ranges {
            let mut next = Vec::with_capacity(n);
            for prefix in out.into_iter() {
                execution_profile.assert_not_timed_out()?;
                for ch in r.clone().iter() {
                    let mut s = prefix.clone();
                    s.push(ch.to_char());
                    next.push(s);
                    if next.len() == n {
                        end = true;
                        break;
                    }
                }
                if end {
                    end = false;
                    break;
                }
            }
            out = next;
            if out.is_empty() {
                break;
            }
        }
        strings.extend(out);
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
    use regex::Regex;

    use crate::regex::RegularExpression;

    #[test]
    fn test_generate_strings() -> Result<(), String> {
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
        assert_generate_strings("[0-9]+[A-Z]*", 500);
        assert_generate_strings("a+(ba+)*", 200);
        assert_generate_strings("((a|bc)*|d)", 200);
        assert_generate_strings(".*", 50);
        assert_generate_strings("(ac|ads|a)*", 200);
        assert_generate_strings("((aad|ads|a)*|q)", 200);
        assert_generate_strings("((aad|ads|a)*abc.*uif(aad|ads|x)*|q)", 1000);
        //((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q){1,5}

        Ok(())
    }

    fn assert_generate_strings(regex: &str, number: usize) {
        println!(":{}", regex);
        let automaton = RegularExpression::parse(regex, false)
            .unwrap()
            .to_automaton()
            .unwrap();
        //println!("{}", automaton.get_number_of_states());
        //automaton.to_dot();
        let re = Regex::new(&format!("(?s)^{}$", regex)).unwrap();

        let strings = automaton.generate_strings(number).unwrap();
        println!("nb of strings: {}/{}", strings.len(), number);
        assert!(number >= strings.len());
        for string in strings {
            // println!("{string}");
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
