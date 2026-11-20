use super::*;

impl RegularExpression {
    /// Returns the concatenation of `regexes`, in order, simplified.
    #[tracing::instrument(level = "trace", skip_all)]
    pub fn concat_all<'a, I: IntoIterator<Item = &'a RegularExpression>>(
        regexes: I,
    ) -> RegularExpression {
        let mut result = RegularExpression::new_empty_string();

        for other in regexes {
            result.concat_mut(other.clone());
        }

        result
    }

    /// Returns the concatenation of `self` and `other`, simplified: `self`
    /// comes first if `append_back` is `true`, `other` if it is `false`.
    #[tracing::instrument(level = "trace", skip(self, other), fields(append_back = append_back))]
    pub fn concat(&self, other: &RegularExpression, append_back: bool) -> RegularExpression {
        let (front, back) = if append_back {
            (self, other)
        } else {
            (other, self)
        };

        let mut result = front.clone();
        result.concat_mut(back.clone());
        result
    }

    /// Appends `that` to `self`, simplified: [`concat`](Self::concat) in
    /// place. A fold of `concat` copies the expression accumulated so far at
    /// each step, which takes quadratic time; a fold of `concat_mut` does
    /// not.
    pub(crate) fn concat_mut(&mut self, that: RegularExpression) {
        if self.is_simplified_empty() || that.is_simplified_empty() {
            *self = RegularExpression::new_empty();
            return;
        } else if that.is_simplified_empty_string() {
            return;
        } else if self.is_simplified_empty_string() {
            *self = that;
            return;
        }

        // The two as a whole, as `(ab)*` and `ab`.
        if let Some(merged) = Self::opconcat_can_be_merged(self, &that) {
            *self = merged;
            return;
        }

        let this = std::mem::replace(self, RegularExpression::new_empty_string());
        let mut elements = match (this, that) {
            (RegularExpression::Concat(mut elements), RegularExpression::Concat(mut others)) => {
                // Only the elements that meet can be merged.
                if let Some(first) = others.pop_front() {
                    Self::push_concat_element(&mut elements, first);
                }
                elements.extend(others);
                elements
            }
            (RegularExpression::Concat(mut elements), that) => {
                Self::push_concat_element(&mut elements, that);
                elements
            }
            (this, RegularExpression::Concat(mut elements)) => {
                match elements
                    .front()
                    .and_then(|first| Self::opconcat_can_be_merged(&this, first))
                {
                    Some(merged) => elements[0] = merged,
                    None => elements.push_front(this),
                }
                elements
            }
            (this, that) => VecDeque::from([this, that]),
        };
        *self = match elements.len() {
            1 => elements.pop_front().expect("len() == 1"),
            _ => RegularExpression::Concat(elements),
        };
    }

    /// Appends `that` to the fold accumulator `elements`, merging with the
    /// back element when possible: the in-place equivalent of
    /// `acc = acc.concat(&that, true)` for a char-by-char fold.
    ///
    /// `that` must be neither the empty language nor the empty string (both
    /// are handled by `concat`'s degenerate checks, not here), and a merge
    /// between the whole accumulated `Concat` and a single element is
    /// structurally impossible, so only the back element needs checking.
    pub(crate) fn push_concat_element(
        elements: &mut VecDeque<RegularExpression>,
        that: RegularExpression,
    ) {
        if let Some(back) = elements.back()
            && let Some(merged) = Self::opconcat_can_be_merged(back, &that)
        {
            *elements.back_mut().expect("back() was Some") = merged;
            return;
        }
        elements.push_back(that);
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
