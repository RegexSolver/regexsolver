use std::cmp;

use crate::{execution_profile::ThreadLocalParams, EngineError};
use ahash::AHashSet;

use super::*;

impl FastAutomaton {
    /*pub fn generate_strings(&self, number: usize) -> Result<AHashSet<String>, EngineError> {
        if self.is_empty() {
            return Ok(AHashSet::new());
        }

        let bases = self
            .get_bases()?
            .into_iter()
            .map(|c| {
                let range = c.to_range(self.get_used_bases()).unwrap();
                (c, range)
            })
            .collect::<Vec<_>>();

        let capacity = cmp::min(number, 1000);
        let mut strings = AHashSet::with_capacity(capacity);
        let mut pq = BinaryHeap::with_capacity(capacity);
        let mut visited = AHashSet::with_capacity(capacity);

        pq.push(Reverse((String::new(), vec![self.start_state])));
        if self.is_accepted(&self.start_state) {
            strings.insert("".to_string());
            if strings.len() >= number {
                return Ok(strings);
            }
        }

        while let Some(Reverse((string, states))) = pq.pop() {
            if !visited.insert((states.clone(), string.clone())) {
                continue;
            }

            for (base, range) in &bases {
                for from_state in &states {
                    let mut accepted = false;
                    let mut next_states = Vec::with_capacity(self.get_number_of_states());
                    for (to_state, condition) in
                        self.transitions_from_state_enumerate_iter(&from_state)
                    {
                        if condition.has_intersection(base) {
                            next_states.push(*to_state);
                            if self.is_accepted(to_state) {
                                accepted = true;
                            }
                        }
                    }
                    if !next_states.is_empty() {
                        next_states.shrink_to_fit();
                        next_states.sort_unstable();

                        let mut c = 0;
                        for char in range.iter() {
                            let mut new_string = String::with_capacity(string.capacity() + 1);
                            new_string.push_str(&string);
                            new_string.push(char.to_char());
                            //println!("char {char}, new_string {new_string}");
                            if accepted {
                                strings.insert(new_string.clone());
                                if strings.len() >= number {
                                    return Ok(strings);
                                }
                            }
                            pq.push(Reverse((new_string, next_states.clone())));
                            c += 1;
                            if c >= number {
                                break;
                            }
                        }
                    }
                }
            }
        }

        Ok(strings)
    }*/

    pub fn generate_strings(&self, number: usize) -> Result<AHashSet<String>, EngineError> {
        if self.is_empty() {
            return Ok(AHashSet::new());
        }

        let mut strings = AHashSet::with_capacity(cmp::min(number, 1000));

        let execution_profile = ThreadLocalParams::get_execution_profile();

        let mut ranges_cache: AHashMap<&Condition, Range> =
            AHashMap::with_capacity(self.get_number_of_states());

        let mut worklist: VecDeque<(Vec<Range>, usize)> =
            VecDeque::with_capacity(cmp::min(number, 1000));
        let mut visited = AHashSet::with_capacity(cmp::min(number, 1000));

        worklist.push_back((vec![], self.start_state));
        while let Some((ranges, state)) = worklist.pop_front() {
            execution_profile.is_timed_out()?;
            if self.accept_states.contains(&state) {
                if ranges.is_empty() {
                    strings.insert(String::new());
                } else {
                    let mut end = false;
                    let mut ranges_iter: Vec<_> = ranges.iter().map(|range| range.iter()).collect();
                    while strings.len() < number {
                        let mut string = vec![];
                        for i in 0..ranges.len() {
                            if let Some(character) = ranges_iter[i].next() {
                                string.push(character);
                            } else {
                                ranges_iter[i] = ranges[i].iter();
                                if i + 1 < ranges.len() {
                                    string.push(ranges_iter[i].next().unwrap());
                                } else {
                                    end = true;
                                    break;
                                }
                            }
                        }
                        if end {
                            break;
                        }
                        strings.insert(string.into_iter().map(|c| c.to_char()).collect());
                    }
                }

                if strings.len() == number {
                    break;
                }
            }
            for (to_state, cond) in self.transitions_from_state_enumerate_iter(&state) {
                let range = match ranges_cache.entry(cond) {
                    Entry::Occupied(o) => o.get().clone(),
                    Entry::Vacant(v) => {
                        let range = cond.to_range(&self.used_bases)?;
                        v.insert(range.clone());
                        range
                    }
                };
                if range.is_empty() {
                    continue;
                }
                let mut new_ranges = ranges.clone();
                new_ranges.push(range);
                let element = (new_ranges, *to_state);

                if !visited.contains(&element) {
                    visited.insert(element.clone());
                    worklist.push_back(element);
                }
            }
        }

        Ok(strings)
    }
}

#[cfg(test)]
mod tests {
    use regex::Regex;

    use crate::regex::RegularExpression;

    #[test]
    fn test_generate_strings() -> Result<(), String> {
        assert_generate_strings("ù", 1000);

        assert_generate_strings("(?:A+(?:\\.[AB]+)*|\"(?:C|\\\\D)*\")@", 500);
        assert_generate_strings(
            "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@",
            500
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
        let automaton = RegularExpression::new(regex)
            .unwrap()
            .to_automaton()
            .unwrap();
        println!("{}", automaton.get_number_of_states());
        //automaton.to_dot();
        let re = Regex::new(&format!("(?s)^{}$", regex)).unwrap();

        let strings = automaton.generate_strings(number).unwrap();
        let mut strings: Vec<_> = strings.iter().collect();
        strings.sort_unstable();
        println!("nb of strings: {}/{}", strings.len(), number);
        assert!(number >= strings.len());
        for string in strings {
            if !re.is_match(string) {
                for byte in string.as_bytes() {
                    print!("{:02x} ", byte);
                }
                panic!("'{string}'")
            }
            assert!(re.is_match(string), "'{string}'");
        }
    }
}
