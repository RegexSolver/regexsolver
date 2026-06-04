use std::collections::BTreeSet;

use super::*;

impl RegularExpression {
    /// Returns a regular expression matching the union of `self` and `other`.
    pub fn union(&self, other: &RegularExpression) -> RegularExpression {
        Self::union_all([self, other])
    }

    /// Returns a regular expression that is the union of all expressions in `patterns`.
    pub fn union_all<'a, I: IntoIterator<Item = &'a RegularExpression>>(
        patterns: I,
    ) -> RegularExpression {
        let mut result: Cow<'a, RegularExpression> = Cow::Owned(RegularExpression::new_empty());

        for other in patterns {
            result = result.union_(other);

            if result.is_total() {
                break;
            }
        }

        result.into_owned()
    }

    fn union_<'a>(&self, other: &'a RegularExpression) -> Cow<'a, RegularExpression> {
        if self.is_total() || other.is_total() {
            return Cow::Owned(RegularExpression::new_total());
        } else if self.is_empty() {
            return Cow::Borrowed(other);
        } else if other.is_empty() || self == other {
            return Cow::Owned(self.clone());
        } else if other.is_empty_string() {
            return Cow::Owned(self.repeat(0, Some(1)));
        } else if self.is_empty_string() {
            return Cow::Owned(other.repeat(0, Some(1)));
        }
        Cow::Owned(match (self, other) {
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
                let mut new_alternation = Cow::Borrowed(other);
                for self_element in self_elements {
                    new_alternation = new_alternation.union_(self_element);
                }

                new_alternation.into_owned()
            }
        })
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
                that_regex.repeat(cmp::min(1, *that_min), *that_max_opt)
            } else {
                let mut alternate = vec![this_character.clone(), that_repetition.clone()];
                alternate.sort_unstable();
                RegularExpression::Alternation(alternate)
            }
        } else {
            panic!("Not character and repetition {this_character:?} {that_repetition:?}")
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
                    Cow::Owned(RegularExpression::Alternation(alternate_elements))
                } else {
                    self_regex.union_(&other_regex)
                }
            } else {
                Cow::Owned(self_regex.repeat(0, Some(1)))
            }
        } else if !other_regex.is_empty_string() {
            Cow::Owned(other_regex.repeat(0, Some(1)))
        } else {
            Cow::Owned(RegularExpression::new_empty_string())
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
                that_regex.repeat(cmp::min(1, *that_min), *that_max_opt)
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
                        return this_regex.repeat(
                            cmp::min(*this_min, *that_min),
                            Some(cmp::max(*this_max, *that_max)),
                        );
                    }
                } else {
                    // At least one side is unbounded. The union collapses to
                    // r{min(m1,m2),} only when the ranges overlap or are
                    // adjacent — i.e. the unbounded side starts no later than
                    // one past the bounded side's end. Otherwise there is a
                    // gap (e.g. a? ∪ a{3,} must NOT become a*).
                    let mergeable = match (this_max_opt, that_max_opt) {
                        (None, None) => true,
                        (Some(this_max), None) => *that_min <= this_max.saturating_add(1),
                        (None, Some(that_max)) => *this_min <= that_max.saturating_add(1),
                        (Some(_), Some(_)) => unreachable!("handled above"),
                    };
                    if mergeable {
                        return this_regex.repeat(cmp::min(*this_min, *that_min), None);
                    }
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
                this_regex.repeat(cmp::min(1, *this_min), *this_max_opt)
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

    // Regression: with an unbounded side, the repetition merge used to fire
    // unconditionally, so `a? ∪ a{3,}` collapsed to `a*` even though `a{2}` is
    // in neither operand. The merge is only sound when the unbounded range
    // starts no later than one past the bounded range's end.
    #[test]
    fn union_does_not_merge_gapped_repetitions() {
        let union = |x: &str, y: &str| {
            RegularExpression::parse(x, false)
                .unwrap()
                .union(&RegularExpression::parse(y, false).unwrap())
                .to_string()
        };

        // Gapped: must stay alternations.
        assert_eq!("(a?|a{3,})", union("a?", "a{3,}"));
        assert_eq!("(a?|a{3,})", union("a{3,}", "a?"));
        assert_eq!("(a{2}|a{5,})", union("a{2}", "a{5,}"));

        // Overlapping or adjacent: still merge.
        assert_eq!("a*", union("a?", "a{2,}"));
        assert_eq!("a{2,}", union("a{2}", "a{3,}"));
        assert_eq!("a*", union("a*", "a{3,}"));
        assert_eq!("a{3,}", union("a{3,}", "a{5,}"));
    }

    #[test]
    fn test_union() -> Result<(), String> {
        assert_union("(a+|a+b)", "a+b?");
        assert_union("(a+|a*)", "a*");
        assert_union("(a?|a{0,2})", "a{0,2}");
        assert_union("(a{2,4}|a{1,3})", "a{1,4}");
        assert_union("(a{1,2}|a{3,4})", "a{1,4}");
        assert_union("(a{3,4}|a{1,2})", "a{1,4}");

        Ok(())
    }

    fn assert_union(input: &str, output: &str) {
        let input = RegularExpression::new(input).unwrap();

        assert_eq!(output, input.to_string());
    }
}
