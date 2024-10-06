use self::cardinality::Cardinality;

use super::*;

mod affixes;
mod number_of_states;

impl RegularExpression {
    pub fn get_length(&self) -> (Option<u32>, Option<u32>) {
        match self {
            RegularExpression::Character(range) => {
                if range.is_empty() {
                    return (None, None);
                }
                (Some(1), Some(1))
            }
            RegularExpression::Repetition(regex, min, max_opt) => {
                let (min_length, max_length_opt) = regex.get_length();
                if let Some(min_length) = min_length {
                    let new_min_length = min * min_length;
                    let new_max_length = if let Some(max_length) = max_length_opt {
                        max_opt.as_ref().map(|max| max * max_length)
                    } else {
                        None
                    };
                    (Some(new_min_length), new_max_length)
                } else if min == &0 {
                    (Some(0), Some(0))
                } else {
                    (None, None)
                }
            }
            RegularExpression::Concat(concat_vec) => {
                let mut new_min_length = 0;
                let mut new_max_length = Some(0);

                for concat_element in concat_vec {
                    let (min_length, max_length_opt) = concat_element.get_length();

                    if let Some(min_length) = min_length {
                        new_min_length += min_length;

                        if let Some(new_max) = new_max_length {
                            if let Some(max_length) = max_length_opt {
                                new_max_length = Some(new_max + max_length);
                            } else {
                                new_max_length = None;
                            }
                        }
                    } else {
                        return (None, None);
                    }
                }

                (Some(new_min_length), new_max_length)
            }
            RegularExpression::Alternation(alternation_vec) => {
                if alternation_vec.is_empty() {
                    return (None, None);
                }
                let mut new_min_length = u32::MAX;
                let mut new_max_length = Some(0);

                for alternation_element in alternation_vec {
                    let (min_length, max_length_opt) = alternation_element.get_length();

                    if let Some(min_length) = min_length {
                        new_min_length = cmp::min(new_min_length, min_length);

                        if let Some(new_max) = new_max_length {
                            if let Some(max_length) = max_length_opt {
                                new_max_length = Some(cmp::max(new_max, max_length));
                            } else {
                                new_max_length = None;
                            }
                        }
                    } else {
                        return (None, None);
                    }
                }

                (Some(new_min_length), new_max_length)
            }
        }
    }

    pub fn get_cardinality(&self) -> Cardinality<u32> {
        if self.is_empty() {
            return Cardinality::Integer(0);
        } else if self.is_total() {
            return Cardinality::Infinite;
        }
        match self {
            RegularExpression::Character(range) => Cardinality::Integer(range.get_cardinality()),
            RegularExpression::Repetition(regular_expression, min, max_opt) => {
                if let Some(max) = max_opt {
                    let regex_cardinality = regular_expression.get_cardinality();
                    if let Cardinality::Integer(cardinality) = regex_cardinality {
                        let mut cardinality_temp: u32 = 0;
                        for i in *min..*max + 1 {
                            if let Some(pow) = cardinality.checked_pow(i) {
                                if let Some(add) = cardinality_temp.checked_add(pow) {
                                    cardinality_temp = add;
                                } else {
                                    return Cardinality::BigInteger;
                                }
                            } else {
                                return Cardinality::BigInteger;
                            }
                        }
                        Cardinality::Integer(cardinality_temp)
                    } else {
                        regex_cardinality
                    }
                } else {
                    Cardinality::Infinite
                }
            }
            RegularExpression::Concat(concat) => {
                let mut cardinality: u32 = 1;
                for concat_element in concat {
                    let element_cardinality = concat_element.get_cardinality();
                    if let Cardinality::Integer(element_cardinality) = element_cardinality {
                        if let Some(mult) = cardinality.checked_mul(element_cardinality) {
                            cardinality = mult;
                        } else {
                            return Cardinality::BigInteger;
                        }
                    } else {
                        return element_cardinality;
                    }
                }
                Cardinality::Integer(cardinality)
            }
            RegularExpression::Alternation(alternation) => {
                let mut cardinality: u32 = 0;
                for alternation_element in alternation {
                    let element_cardinality = alternation_element.get_cardinality();
                    if let Cardinality::Integer(element_cardinality) = element_cardinality {
                        if let Some(add) = cardinality.checked_add(element_cardinality) {
                            cardinality = add;
                        } else {
                            return Cardinality::BigInteger;
                        }
                    } else {
                        return element_cardinality;
                    }
                }
                Cardinality::Integer(cardinality)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length() -> Result<(), String> {
        assert_length(".{1,1000}");
        assert_length("toto");
        assert_length(".{2,3}");
        assert_length("q(ab|ca)x");
        assert_length("q(ab|ca|ab|abc)x");
        assert_length(".*");
        assert_length(".?");
        assert_length("a*(aad|ads|a)abc.*def.*ghi");
        assert_length("(at?)");
        assert_length("(ot){3,4}");
        assert_length("(ot?d){1,4}");
        assert_length("((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q){1,100}");

        assert_eq!(
            FastAutomaton::new_empty().get_length(),
            RegularExpression::new_empty().get_length()
        );

        assert_eq!(
            FastAutomaton::new_total().get_length(),
            RegularExpression::new_total().get_length()
        );
        Ok(())
    }

    fn assert_length(regex: &str) {
        println!("{}", regex);
        let regex = RegularExpression::new(regex).unwrap();

        let (min, max_opt) = regex.get_length();

        let automaton = regex.to_automaton().unwrap();
        //automaton.to_dot();

        let (min_automaton_opt, max_automaton_opt) = automaton.get_length();

        assert_eq!((min_automaton_opt, max_automaton_opt), (min, max_opt));
    }

    #[test]
    fn test_cardinality() -> Result<(), String> {
        assert_cardinality(".{1,1000}");
        assert_cardinality("toto");
        assert_cardinality(".");
        assert_cardinality(".{2,3}");
        assert_cardinality("q(ab|ca|abc)x");
        assert_cardinality("q(ab|ca|ab|abc)x");
        assert_cardinality(".*");
        assert_cardinality(".?");
        assert_cardinality("a*(aad|ads|a)abc.*def.*ghi");
        assert_cardinality("(at?)");
        assert_cardinality("(ot){3,4}");
        assert_cardinality("(t){1,3}");
        assert_cardinality("(ot?d){1,4}");
        assert_cardinality("((aad|ads|a)*abc.*def.*uif(aad|ads|x)*abc.*oxs.*def(aad|ads|ax)*abc.*def.*ksd|q){1,100}");
        Ok(())
    }

    fn assert_cardinality(regex: &str) {
        println!("{}", regex);
        let regex = RegularExpression::new(regex).unwrap();

        let cardinality = regex.get_cardinality();

        let mut automaton = regex.to_automaton().unwrap();

        if !automaton.is_cyclic() {
            automaton = automaton.determinize().unwrap();
        }

        //automaton.to_dot();

        let expected = automaton.get_cardinality().unwrap();

        assert_eq!(expected, cardinality);
    }
}
