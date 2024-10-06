use super::*;

impl RegularExpression {
    pub fn concat(&self, other: &RegularExpression, append_back: bool) -> RegularExpression {
        if self.is_empty() || other.is_empty() {
            return RegularExpression::new_empty();
        } else if self.is_empty_string() {
            return other.clone();
        } else if other.is_empty_string() {
            return self.clone();
        }

        match (self, other) {
            (RegularExpression::Concat(_), RegularExpression::Concat(_)) => {
                if append_back {
                    Self::opconcat_concat_and_concat(self, other)
                } else {
                    Self::opconcat_concat_and_concat(other, self)
                }
            }
            (RegularExpression::Concat(_), _) => {
                if append_back {
                    Self::opconcat_concat_and_other(self, other)
                } else {
                    Self::opconcat_other_and_concat(other, self)
                }
            }
            (_, RegularExpression::Concat(_)) => {
                if append_back {
                    Self::opconcat_other_and_concat(self, other)
                } else {
                    Self::opconcat_concat_and_other(other, self)
                }
            }
            (_, _) => {
                if append_back {
                    Self::opconcat_other_and_other(self, other)
                } else {
                    Self::opconcat_other_and_other(other, self)
                }
            }
        }
    }

    fn opconcat_other_and_other(
        this: &RegularExpression,
        that: &RegularExpression,
    ) -> RegularExpression {
        if let Some(merged) = Self::opconcat_can_be_merged(this, that) {
            merged
        } else {
            let mut vec = VecDeque::with_capacity(2);
            vec.push_back(this.clone());
            vec.push_back(that.clone());
            RegularExpression::Concat(vec)
        }
    }

    fn opconcat_other_and_concat(
        this: &RegularExpression,
        that: &RegularExpression,
    ) -> RegularExpression {
        if let RegularExpression::Concat(that_elements) = that {
            if that_elements.is_empty() {
                return this.clone();
            }

            if let Some(merged) = Self::opconcat_can_be_merged(this, that) {
                return merged;
            }

            let mut vec = that_elements.clone();
            let that_index = 0;

            if let Some(merged) = Self::opconcat_can_be_merged(this, &that_elements[that_index]) {
                vec[that_index] = merged;
            } else {
                vec.push_front(this.clone());
            }

            if vec.len() == 1 {
                vec[0].clone()
            } else {
                RegularExpression::Concat(vec)
            }
        } else {
            panic!("Not concat")
        }
    }

    fn opconcat_concat_and_other(
        this: &RegularExpression,
        that: &RegularExpression,
    ) -> RegularExpression {
        if let RegularExpression::Concat(this_elements) = this {
            if this_elements.is_empty() {
                return that.clone();
            }

            if let Some(merged) = Self::opconcat_can_be_merged(this, that) {
                return merged;
            }

            let mut vec = this_elements.clone();
            let this_index = this_elements.len() - 1;

            if let Some(merged) = Self::opconcat_can_be_merged(&this_elements[this_index], that) {
                vec[this_index] = merged;
            } else {
                vec.push_back(that.clone());
            }

            if vec.len() == 1 {
                vec[0].clone()
            } else {
                RegularExpression::Concat(vec)
            }
        } else {
            panic!("Not concat")
        }
    }

    fn opconcat_concat_and_concat(
        this: &RegularExpression,
        that: &RegularExpression,
    ) -> RegularExpression {
        if let (
            RegularExpression::Concat(this_elements),
            RegularExpression::Concat(that_elements),
        ) = (this, that)
        {
            if this_elements.is_empty() {
                return RegularExpression::Concat(that_elements.clone());
            } else if that_elements.is_empty() {
                return RegularExpression::Concat(this_elements.clone());
            }

            if let Some(merged) = Self::opconcat_can_be_merged(this, that) {
                return merged;
            }

            let mut vec = this_elements.clone();
            let (this_index, that_index) = (this_elements.len() - 1, 0);

            if let Some(merged) =
                Self::opconcat_can_be_merged(&this_elements[this_index], &that_elements[that_index])
            {
                vec[this_index] = merged;
                vec.extend(that_elements.iter().skip(1).cloned());
            } else {
                vec.extend(that_elements.iter().cloned());
            }

            if vec.len() == 1 {
                vec[0].clone()
            } else {
                RegularExpression::Concat(vec)
            }
        } else {
            panic!("Not concat")
        }
    }

    fn opconcat_can_be_merged(
        this: &RegularExpression,
        that: &RegularExpression,
    ) -> Option<RegularExpression> {
        if let (
            RegularExpression::Repetition(this_regex, _, this_max_opt),
            RegularExpression::Repetition(that_regex, that_min, _),
        ) = (this, that)
        {
            if let (
                RegularExpression::Character(this_range),
                RegularExpression::Character(that_range),
            ) = (*this_regex.clone(), *that_regex.clone())
            {
                if this_range.contains_all(&that_range) && that_min == &0 && this_max_opt.is_none() {
                    return Some(this.clone());
                }
            }
        }

        if this == that {
            if let (
                RegularExpression::Repetition(this_regex, this_min, this_max_opt),
                RegularExpression::Repetition(_, that_min, that_max_opt),
            ) = (this, that)
            {
                let new_min = this_min + that_min;
                let new_max_opt =
                    if let (Some(this_max), Some(that_max)) = (this_max_opt, that_max_opt) {
                        Some(this_max + that_max)
                    } else {
                        None
                    };
                Some(RegularExpression::Repetition(
                    this_regex.clone(),
                    new_min,
                    new_max_opt,
                ))
            } else {
                Some(RegularExpression::Repetition(
                    Box::new(this.clone()),
                    2,
                    Some(2),
                ))
            }
        } else if let RegularExpression::Repetition(this_regex, this_min, this_max_opt) = this {
            if let Some(RegularExpression::Repetition(merged_regex, merged_min, merged_max_opt)) =
                Self::opconcat_can_be_merged(this_regex, that)
            {
                let new_min = this_min + merged_min - 1;
                let new_max_opt =
                    if let (Some(this_max), Some(merged_max)) = (this_max_opt, merged_max_opt) {
                        Some(this_max + merged_max - 1)
                    } else {
                        None
                    };
                Some(RegularExpression::Repetition(
                    merged_regex,
                    new_min,
                    new_max_opt,
                ))
            } else {
                None
            }
        } else if let RegularExpression::Repetition(that_regex, that_min, that_max_opt) = that {
            if let Some(RegularExpression::Repetition(merged_regex, merged_min, merged_max_opt)) =
                Self::opconcat_can_be_merged(this, that_regex)
            {
                let new_min = merged_min + that_min - 1;
                let new_max_opt =
                    if let (Some(merged_max), Some(that_max)) = (merged_max_opt, that_max_opt) {
                        Some(merged_max + that_max - 1)
                    } else {
                        None
                    };
                Some(RegularExpression::Repetition(
                    merged_regex,
                    new_min,
                    new_max_opt,
                ))
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat() -> Result<(), String> {
        assert_concat(".*a?", ".*");
        assert_concat(".{2,3}.{4,9}", ".{6,12}");

        Ok(())
    }

    fn assert_concat(input: &str, output: &str) {
        let input = RegularExpression::new(input).unwrap();

        assert_eq!(output, input.to_string());
    }
}
