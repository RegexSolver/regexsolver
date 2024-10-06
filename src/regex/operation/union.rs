use std::collections::BTreeSet;

use super::*;

impl RegularExpression {
    pub fn union(&self, other: &RegularExpression) -> RegularExpression {
        if self.is_total() || other.is_total() {
            return RegularExpression::new_total();
        } else if self.is_empty() {
            return other.clone();
        } else if other.is_empty() || self == other {
            return self.clone();
        } else if other.is_empty_string() {
            return self.clone().repeat(0, Some(1));
        } else if self.is_empty_string() {
            return other.clone().repeat(0, Some(1));
        }
        match (self, other) {
            (
                RegularExpression::Character(self_range),
                RegularExpression::Character(other_range),
            ) => RegularExpression::Character(self_range.union(other_range)),
            (RegularExpression::Character(_), RegularExpression::Repetition(_, _, _)) => {
                Self::opunion_character_and_repetition(self, other)
            }
            (RegularExpression::Character(_), RegularExpression::Concat(_)) => {
                Self::opunion_character_and_concat(self, other)
            }
            (RegularExpression::Character(_), RegularExpression::Alternation(_)) => {
                Self::opunion_character_and_alternation(self, other)
            }
            (RegularExpression::Repetition(_, _, _), RegularExpression::Character(_)) => {
                Self::opunion_character_and_repetition(other, self)
            }
            (RegularExpression::Repetition(_, _, _), RegularExpression::Repetition(_, _, _)) => {
                Self::opunion_repetition_and_repetition(self, other)
            }
            (RegularExpression::Repetition(_, _, _), RegularExpression::Concat(_)) => {
                Self::opunion_concat_and_repetition(other, self)
            }
            (RegularExpression::Repetition(_, _, _), RegularExpression::Alternation(_)) => {
                Self::opunion_repetition_and_alternation(self, other)
            }
            (RegularExpression::Concat(_), RegularExpression::Character(_)) => {
                Self::opunion_character_and_concat(other, self)
            }
            (RegularExpression::Concat(_), RegularExpression::Repetition(_, _, _)) => {
                Self::opunion_concat_and_repetition(self, other)
            }
            (RegularExpression::Concat(_), RegularExpression::Concat(_)) => {
                Self::opunion_common_affixes(self, other)
            }
            (RegularExpression::Concat(_), RegularExpression::Alternation(_)) => {
                Self::opunion_concat_and_alternation(self, other)
            }
            (RegularExpression::Alternation(_), RegularExpression::Character(_)) => {
                Self::opunion_character_and_alternation(other, self)
            }
            (RegularExpression::Alternation(_), RegularExpression::Repetition(_, _, _)) => {
                Self::opunion_repetition_and_alternation(other, self)
            }
            (RegularExpression::Alternation(_), RegularExpression::Concat(_)) => {
                Self::opunion_concat_and_alternation(other, self)
            }
            (RegularExpression::Alternation(self_elements), RegularExpression::Alternation(_)) => {
                let mut new_alternation = other.clone();
                for self_element in self_elements {
                    new_alternation = new_alternation.union(self_element);
                }

                new_alternation
            }
        }
    }

    fn opunion_character_and_repetition(
        this_character: &RegularExpression,
        that_repetition: &RegularExpression,
    ) -> RegularExpression {
        if let (
            RegularExpression::Character(_),
            RegularExpression::Repetition(that_regex, that_min, that_max_opt),
        ) = (this_character, that_repetition)
        {
            if this_character == &**that_regex && *that_min <= 2 {
                RegularExpression::Repetition(
                    that_regex.clone(),
                    cmp::min(1, *that_min),
                    *that_max_opt,
                )
            } else {
                let mut alternate = vec![this_character.clone(), that_repetition.clone()];
                alternate.sort_unstable();
                RegularExpression::Alternation(alternate)
            }
        } else {
            panic!(
                "Not character and repetition {:?} {:?}",
                this_character, that_repetition
            )
        }
    }

    fn opunion_common_affixes(
        this: &RegularExpression,
        that: &RegularExpression,
    ) -> RegularExpression {
        let (prefix, (self_regex, other_regex), suffix) = this.get_common_affixes(that);
        let mut regex = RegularExpression::new_empty_string();
        if let Some(prefix) = &prefix {
            regex = regex.concat(prefix, true);
        }

        let regex_from_alternate = if !self_regex.is_empty_string() {
            if !other_regex.is_empty_string() {
                if prefix.is_none() && suffix.is_none() {
                    let mut alternate_elements = vec![self_regex, other_regex];
                    alternate_elements.sort_unstable();
                    RegularExpression::Alternation(alternate_elements)
                } else {
                    self_regex.union(&other_regex)
                }
            } else {
                RegularExpression::Repetition(Box::new(self_regex), 0, Some(1))
            }
        } else if !other_regex.is_empty_string() {
            RegularExpression::Repetition(Box::new(other_regex), 0, Some(1))
        } else {
            RegularExpression::new_empty_string()
        };

        regex = regex.concat(&regex_from_alternate, true);

        if let Some(suffix) = suffix {
            regex = regex.concat(&suffix, true);
        }
        regex
    }

    fn opunion_character_and_alternation(
        this_character: &RegularExpression,
        that_alternation: &RegularExpression,
    ) -> RegularExpression {
        if let (
            RegularExpression::Character(this_range),
            RegularExpression::Alternation(that_elements),
        ) = (this_character, that_alternation)
        {
            let mut set = BTreeSet::new();

            let mut had_character_union = false;
            for element in that_elements {
                if let RegularExpression::Character(range) = element {
                    set.insert(RegularExpression::Character(this_range.union(range)));
                    had_character_union = true;
                } else if matches!(element, RegularExpression::Repetition(_, _, _)) {
                    let repetition =
                        Self::opunion_character_and_repetition(this_character, element);
                    if matches!(repetition, RegularExpression::Repetition(_, _, _)) {
                        set.insert(repetition);
                        had_character_union = true;
                    } else {
                        set.insert(element.clone());
                    }
                } else {
                    set.insert(element.clone());
                }
            }
            if !had_character_union {
                set.insert(this_character.clone());
            }
            RegularExpression::Alternation(set.into_iter().collect())
        } else {
            panic!("Not character and alternation")
        }
    }

    fn opunion_character_and_concat(
        this_character: &RegularExpression,
        that_concat: &RegularExpression,
    ) -> RegularExpression {
        if let (RegularExpression::Character(_), RegularExpression::Concat(that_elements)) =
            (this_character, that_concat)
        {
            if that_elements.len() == 1 && that_elements[0] == *this_character {
                this_character.clone()
            } else {
                Self::opunion_common_affixes(this_character, that_concat)
            }
        } else {
            panic!("Not character and concat")
        }
    }

    fn opunion_concat_and_repetition(
        this_concat: &RegularExpression,
        that_repetition: &RegularExpression,
    ) -> RegularExpression {
        if let (
            RegularExpression::Concat(_),
            RegularExpression::Repetition(that_regex, that_min, that_max_opt),
        ) = (this_concat, that_repetition)
        {
            if this_concat == &**that_regex && *that_min <= 2 {
                RegularExpression::Repetition(
                    that_regex.clone(),
                    cmp::min(1, *that_min),
                    *that_max_opt,
                )
            } else {
                Self::opunion_common_affixes(this_concat, that_repetition)
            }
        } else {
            panic!("Not concat and repetition")
        }
    }

    fn opunion_concat_and_alternation(
        this_concat: &RegularExpression,
        that_alternation: &RegularExpression,
    ) -> RegularExpression {
        if let (RegularExpression::Concat(_), RegularExpression::Alternation(that_elements)) =
            (this_concat, that_alternation)
        {
            let mut set = BTreeSet::new();

            let mut had_concat_union = false;
            for element in that_elements {
                if matches!(element, RegularExpression::Repetition(_, _, _)) {
                    let repetition = Self::opunion_concat_and_repetition(this_concat, element);
                    if matches!(repetition, RegularExpression::Repetition(_, _, _)) {
                        set.insert(repetition);
                        had_concat_union = true;
                    } else {
                        set.insert(element.clone());
                    }
                } else {
                    set.insert(element.clone());
                }
            }
            if !had_concat_union {
                set.insert(this_concat.clone());
            }
            RegularExpression::Alternation(set.into_iter().collect())
        } else {
            panic!("Not concat and alternation")
        }
    }

    fn opunion_repetition_and_repetition(
        this_repetition: &RegularExpression,
        that_repetition: &RegularExpression,
    ) -> RegularExpression {
        if let (
            RegularExpression::Repetition(this_regex, this_min, this_max_opt),
            RegularExpression::Repetition(that_regex, that_min, that_max_opt),
        ) = (this_repetition, that_repetition)
        {
            if this_regex == that_regex {
                if let (Some(this_max), Some(that_max)) = (this_max_opt, that_max_opt) {
                    if this_min <= that_max && that_min <= this_max
                        || this_max + 1 == *that_min
                        || that_max + 1 == *this_min
                    {
                        return RegularExpression::Repetition(
                            this_regex.clone(),
                            cmp::min(*this_min, *that_min),
                            Some(cmp::max(*this_max, *that_max)),
                        );
                    }
                } else {
                    return RegularExpression::Repetition(
                        this_regex.clone(),
                        cmp::min(*this_min, *that_min),
                        None,
                    );
                }
            }

            let mut alternate = vec![this_repetition.clone(), that_repetition.clone()];
            alternate.sort_unstable();
            RegularExpression::Alternation(alternate)
        } else {
            panic!("Not repetition")
        }
    }

    fn opunion_repetition_and_alternation(
        this_repetition: &RegularExpression,
        that_alternation: &RegularExpression,
    ) -> RegularExpression {
        if let (
            RegularExpression::Repetition(this_regex, this_min, this_max_opt),
            RegularExpression::Alternation(that_elements),
        ) = (this_repetition, that_alternation)
        {
            if that_alternation == &**this_regex && *this_min <= 2 {
                RegularExpression::Repetition(
                    this_regex.clone(),
                    cmp::min(1, *this_min),
                    *this_max_opt,
                )
            } else {
                let mut set = BTreeSet::new();

                let mut had_repetition_union = false;
                for element in that_elements {
                    if matches!(element, RegularExpression::Repetition(_, _, _)) {
                        let repetition =
                            Self::opunion_repetition_and_repetition(this_repetition, element);
                        if matches!(repetition, RegularExpression::Repetition(_, _, _)) {
                            set.insert(repetition);
                            had_repetition_union = true;
                        } else {
                            set.insert(element.clone());
                        }
                    } else if matches!(element, RegularExpression::Character(_)) {
                        let repetition =
                            Self::opunion_character_and_repetition(element, this_repetition);
                        if matches!(repetition, RegularExpression::Repetition(_, _, _)) {
                            set.insert(repetition);
                            had_repetition_union = true;
                        } else {
                            set.insert(element.clone());
                        }
                    } else if matches!(element, RegularExpression::Concat(_)) {
                        let repetition =
                            Self::opunion_concat_and_repetition(element, this_repetition);
                        if matches!(repetition, RegularExpression::Repetition(_, _, _)) {
                            set.insert(repetition);
                            had_repetition_union = true;
                        } else {
                            set.insert(element.clone());
                        }
                    } else {
                        set.insert(element.clone());
                    }
                }
                if !had_repetition_union {
                    set.insert(this_repetition.clone());
                }
                RegularExpression::Alternation(set.into_iter().collect())
            }
        } else {
            panic!("Not repetition and alternation")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union() -> Result<(), String> {
        assert_union("(a+|a+b)", "a+b?");

        Ok(())
    }

    fn assert_union(input: &str, output: &str) {
        let input = RegularExpression::new(input).unwrap();

        assert_eq!(output, input.to_string());
    }
}
