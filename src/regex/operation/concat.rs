use super::*;

impl RegularExpression {
    /// Returns a regular expression that is the concatenation of all expressions in `regexes`.
    #[tracing::instrument(level = "trace", skip_all)]
    pub fn concat_all<'a, I: IntoIterator<Item = &'a RegularExpression>>(
        regexes: I,
    ) -> RegularExpression {
        let mut result = RegularExpression::new_empty_string();

        for other in regexes {
            result = result.concat(other, true);
        }

        result
    }

    /// Returns a new regular expression representing the concatenation of `self` and `other`; `append_back` determines their order.
    #[tracing::instrument(level = "trace", skip(self, other), fields(append_back = append_back))]
    pub fn concat(&self, other: &RegularExpression, append_back: bool) -> RegularExpression {
        if self.is_empty() || other.is_empty() {
            return RegularExpression::new_empty();
        } else if self.is_empty_string() {
            return other.clone();
        } else if other.is_empty_string() {
            return self.clone();
        }

        let (front, back) = if append_back {
            (self, other)
        } else {
            (other, self)
        };

        match (front, back) {
            (RegularExpression::Concat(..), RegularExpression::Concat(..)) => {
                Self::opconcat_concat_and_concat(front, back)
            }
            (RegularExpression::Concat(..), _) => Self::opconcat_concat_and_other(front, back),
            (_, RegularExpression::Concat(..)) => Self::opconcat_other_and_concat(front, back),
            (_, _) => Self::opconcat_other_and_other(front, back),
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

            // Clone the surviving elements only: the boundary element is
            // either replaced by the merge or kept alongside `this`.
            let mut vec: VecDeque<RegularExpression>;
            if let Some(merged) = Self::opconcat_can_be_merged(this, &that_elements[0]) {
                vec = that_elements.iter().skip(1).cloned().collect();
                vec.push_front(merged);
            } else {
                vec = that_elements.clone();
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

            // Clone the surviving elements only (see opconcat_other_and_concat).
            let this_index = this_elements.len() - 1;
            let mut vec: VecDeque<RegularExpression>;
            if let Some(merged) = Self::opconcat_can_be_merged(&this_elements[this_index], that) {
                vec = this_elements.iter().take(this_index).cloned().collect();
                vec.push_back(merged);
            } else {
                vec = this_elements.clone();
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

            // Clone the surviving elements only (see opconcat_other_and_concat).
            let (this_index, that_index) = (this_elements.len() - 1, 0);
            let mut vec: VecDeque<RegularExpression>;
            if let Some(merged) =
                Self::opconcat_can_be_merged(&this_elements[this_index], &that_elements[that_index])
            {
                vec = this_elements.iter().take(this_index).cloned().collect();
                vec.push_back(merged);
                vec.extend(that_elements.iter().skip(1).cloned());
            } else {
                vec = this_elements.clone();
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

    /// Merges the bounds of two adjacent repetitions of the same expression,
    /// `r{a,b}r{c,d}` → `r{a+c,b+d}`. Returns `None` ("cannot be merged",
    /// falling back to plain concatenation) when an addition would overflow.
    fn merge_repetition_bounds(
        this_min: u32,
        this_max_opt: &Option<u32>,
        that_min: u32,
        that_max_opt: &Option<u32>,
    ) -> Option<(u32, Option<u32>)> {
        let new_min = this_min.checked_add(that_min)?;
        let new_max_opt = if let (Some(this_max), Some(that_max)) = (this_max_opt, that_max_opt) {
            Some(this_max.checked_add(*that_max)?)
        } else {
            None
        };
        Some((new_min, new_max_opt))
    }

    fn opconcat_can_be_merged(
        this: &RegularExpression,
        that: &RegularExpression,
    ) -> Option<RegularExpression> {
        if this == that {
            if let (
                RegularExpression::Repetition(this_regex, this_min, this_max_opt),
                RegularExpression::Repetition(_, that_min, that_max_opt),
            ) = (this, that)
            {
                let (new_min, new_max_opt) = Self::merge_repetition_bounds(
                    *this_min,
                    this_max_opt,
                    *that_min,
                    that_max_opt,
                )?;
                Some(this_regex.repeat(new_min, new_max_opt))
            } else {
                Some(this.repeat(2, Some(2)))
            }
        } else if let (
            RegularExpression::Repetition(this_regex, this_min, this_max_opt),
            RegularExpression::Repetition(that_regex, that_min, that_max_opt),
        ) = (this, that)
        {
            if this_regex == that_regex {
                let (new_min, new_max_opt) = Self::merge_repetition_bounds(
                    *this_min,
                    this_max_opt,
                    *that_min,
                    that_max_opt,
                )?;
                Some(this_regex.repeat(new_min, new_max_opt))
            } else if let (
                RegularExpression::Character(this_range),
                RegularExpression::Character(that_range),
            ) = (&**this_regex, &**that_regex)
            {
                if this_range.contains_all(that_range) && that_min == &0 && this_max_opt.is_none() {
                    Some(this.clone())
                } else {
                    None
                }
            } else {
                None
            }
        } else if let RegularExpression::Repetition(this_regex, this_min, this_max_opt) = this {
            if **this_regex == *that {
                let (new_min, new_max_opt) =
                    Self::merge_repetition_bounds(*this_min, this_max_opt, 1, &Some(1))?;
                Some(this_regex.repeat(new_min, new_max_opt))
            } else {
                None
            }
        } else if let RegularExpression::Repetition(that_regex, that_min, that_max_opt) = that {
            if **that_regex == *this {
                let (new_min, new_max_opt) =
                    Self::merge_repetition_bounds(*that_min, that_max_opt, 1, &Some(1))?;
                Some(that_regex.repeat(new_min, new_max_opt))
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

    // Merging adjacent repetitions whose summed bounds would overflow must
    // fall back to plain concatenation instead of overflowing.
    #[test]
    fn concat_merge_bound_overflow_falls_back_to_concat() {
        let a = RegularExpression::new("a").unwrap();
        let big = RegularExpression::Repetition(Box::new(a), u32::MAX, None);
        let result = big.concat(&big, true);
        assert!(matches!(
            &result,
            RegularExpression::Concat(parts) if parts.len() == 2
        ));
    }

    #[test]
    fn test_concat() -> Result<(), String> {
        assert_concat("xxx", "x{3}");

        assert_concat("[a-z]+a?", "[a-z]+");
        assert_concat("(x{3})*x{1,2}", "(x{3})*x{1,2}");
        assert_concat(".*a?", ".*");
        assert_concat(".{2,3}.{4,9}", ".{6,12}");

        Ok(())
    }

    fn assert_concat(input: &str, output: &str) {
        let input = RegularExpression::new(input).unwrap();

        assert_eq!(output, input.to_string());
    }
}
