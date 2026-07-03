use std::collections::BTreeSet;

use super::*;

impl RegularExpression {
    /// Returns a regular expression matching the union of `self` and `other`.
    #[tracing::instrument(level = "trace", skip_all)]
    pub fn union(&self, other: &RegularExpression) -> RegularExpression {
        Self::union_all([self, other])
    }

    /// Returns a regular expression that is the union of all expressions in `regexes`.
    ///
    /// Folding through [`union`](Self::union) directly would clone the whole
    /// accumulated alternation once per operand (quadratic in the number of
    /// operands); the accumulator below applies the same element-merging
    /// rules in place instead.
    #[tracing::instrument(level = "trace", skip_all)]
    pub fn union_all<'a, I: IntoIterator<Item = &'a RegularExpression>>(
        regexes: I,
    ) -> RegularExpression {
        let mut accumulator = UnionAccumulator::new();

        for other in regexes {
            accumulator.push(other);

            if accumulator.is_total() {
                break;
            }
        }

        accumulator.finish()
    }

    /// Deadline-honoring variant of [`union_all`](Self::union_all) used by
    /// the parser: folding an alternation branch into the accumulated result
    /// can be expensive (e.g. common-affix extraction against a large
    /// accumulated alternation), so the fold checks the execution profile
    /// once per operand and aborts with `OperationTimeOutError` past the
    /// deadline.
    pub(crate) fn union_all_bounded<'a, I: IntoIterator<Item = &'a RegularExpression>>(
        regexes: I,
    ) -> Result<RegularExpression, EngineError> {
        let execution_profile = ExecutionProfile::get();
        let mut accumulator = UnionAccumulator::new();

        for other in regexes {
            execution_profile.assert_not_timed_out()?;
            accumulator.push(other);

            if accumulator.is_total() {
                break;
            }
        }

        Ok(accumulator.finish())
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
            (RegularExpression::Character(..), RegularExpression::Repetition(..)) => {
                Self::opunion_character_and_repetition(self, other)
            }
            (RegularExpression::Character(..), RegularExpression::Concat(..)) => {
                Self::opunion_character_and_concat(self, other)
            }
            (RegularExpression::Character(..), RegularExpression::Alternation(..)) => {
                Self::opunion_character_and_alternation(self, other)
            }
            (RegularExpression::Repetition(..), RegularExpression::Character(..)) => {
                Self::opunion_character_and_repetition(other, self)
            }
            (RegularExpression::Repetition(..), RegularExpression::Repetition(..)) => {
                Self::opunion_repetition_and_repetition(self, other)
            }
            (RegularExpression::Repetition(..), RegularExpression::Concat(..)) => {
                Self::opunion_concat_and_repetition(other, self)
            }
            (RegularExpression::Repetition(..), RegularExpression::Alternation(..)) => {
                Self::opunion_repetition_and_alternation(self, other)
            }
            (RegularExpression::Concat(..), RegularExpression::Character(..)) => {
                Self::opunion_character_and_concat(other, self)
            }
            (RegularExpression::Concat(..), RegularExpression::Repetition(..)) => {
                Self::opunion_concat_and_repetition(self, other)
            }
            (RegularExpression::Concat(..), RegularExpression::Concat(..)) => {
                Self::opunion_common_affixes(self, other)
            }
            (RegularExpression::Concat(..), RegularExpression::Alternation(..)) => {
                Self::opunion_concat_and_alternation(self, other)
            }
            (RegularExpression::Alternation(..), RegularExpression::Character(..)) => {
                Self::opunion_character_and_alternation(other, self)
            }
            (RegularExpression::Alternation(..), RegularExpression::Repetition(..)) => {
                Self::opunion_repetition_and_alternation(other, self)
            }
            (RegularExpression::Alternation(..), RegularExpression::Concat(..)) => {
                Self::opunion_concat_and_alternation(other, self)
            }
            (RegularExpression::Alternation(self_elements), RegularExpression::Alternation(..)) => {
                let mut accumulator = UnionAccumulator::seeded_from(other);
                for self_element in self_elements {
                    accumulator.push(self_element);
                }

                accumulator.finish()
            }
        })
    }

    fn opunion_character_and_repetition(
        this_character: &RegularExpression,
        that_repetition: &RegularExpression,
    ) -> RegularExpression {
        if let (
            RegularExpression::Character(..),
            RegularExpression::Repetition(that_regex, that_min, that_max_opt),
        ) = (this_character, that_repetition)
        {
            if this_character == &**that_regex
                && *that_min <= 2
                && that_max_opt.is_none_or(|that_max| that_max >= 1)
            {
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
                } else if matches!(element, RegularExpression::Repetition(..)) {
                    let repetition =
                        Self::opunion_character_and_repetition(this_character, element);
                    if matches!(repetition, RegularExpression::Repetition(..)) {
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
        if let (RegularExpression::Character(..), RegularExpression::Concat(that_elements)) =
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
            RegularExpression::Concat(..),
            RegularExpression::Repetition(that_regex, that_min, that_max_opt),
        ) = (this_concat, that_repetition)
        {
            // See `opunion_character_and_repetition`: the merge is only sound
            // when the repetition admits at least one copy.
            if this_concat == &**that_regex
                && *that_min <= 2
                && that_max_opt.is_none_or(|that_max| that_max >= 1)
            {
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
        if let (RegularExpression::Concat(..), RegularExpression::Alternation(that_elements)) =
            (this_concat, that_alternation)
        {
            let mut set = BTreeSet::new();

            let mut had_concat_union = false;
            for element in that_elements {
                if matches!(element, RegularExpression::Repetition(..)) {
                    let repetition = Self::opunion_concat_and_repetition(this_concat, element);
                    if matches!(repetition, RegularExpression::Repetition(..)) {
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
                        || this_max.saturating_add(1) == *that_min
                        || that_max.saturating_add(1) == *this_min
                    {
                        return this_regex.repeat(
                            cmp::min(*this_min, *that_min),
                            Some(cmp::max(*this_max, *that_max)),
                        );
                    }
                } else {
                    // At least one side is unbounded. The union collapses to
                    // r{min(m1,m2),} only when the ranges overlap or are
                    // adjacent (i.e. the unbounded side starts no later than
                    // one past the bounded side's end). Otherwise there is a
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
            // See `opunion_character_and_repetition`: the merge is only sound
            // when the repetition admits at least one copy.
            if that_alternation == &**this_regex
                && *this_min <= 2
                && this_max_opt.is_none_or(|this_max| this_max >= 1)
            {
                this_regex.repeat(cmp::min(1, *this_min), *this_max_opt)
            } else {
                let mut set = BTreeSet::new();

                let mut had_repetition_union = false;
                for element in that_elements {
                    if matches!(element, RegularExpression::Repetition(..)) {
                        let repetition =
                            Self::opunion_repetition_and_repetition(this_repetition, element);
                        if matches!(repetition, RegularExpression::Repetition(..)) {
                            set.insert(repetition);
                            had_repetition_union = true;
                        } else {
                            set.insert(element.clone());
                        }
                    } else if matches!(element, RegularExpression::Character(..)) {
                        let repetition =
                            Self::opunion_character_and_repetition(element, this_repetition);
                        if matches!(repetition, RegularExpression::Repetition(..)) {
                            set.insert(repetition);
                            had_repetition_union = true;
                        } else {
                            set.insert(element.clone());
                        }
                    } else if matches!(element, RegularExpression::Concat(..)) {
                        let repetition =
                            Self::opunion_concat_and_repetition(element, this_repetition);
                        if matches!(repetition, RegularExpression::Repetition(..)) {
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

/// Incremental accumulator behind [`RegularExpression::union_all`] and the
/// alternation ∪ alternation arm of `union_`.
///
/// While the accumulated result is not a multi-element alternation, operands
/// fold through the pairwise `union_` rules unchanged. Once it is one, the
/// exact per-element merge rules of the `opunion_*_and_alternation` helpers
/// are applied in place on a `BTreeSet` instead of rebuilding (and
/// re-cloning) the whole alternation for every operand, which made folding an
/// n-branch alternation quadratic in n.
enum UnionAccumulator {
    Expression(RegularExpression),
    Alternation(AlternationSet),
}

/// The elements of an accumulated alternation, with counts of the element
/// kinds an incoming operand could merge with, so pushes that cannot merge
/// with anything (the common case: long alternations of distinct literals)
/// skip the merge scan entirely.
#[derive(Default)]
struct AlternationSet {
    elements: BTreeSet<RegularExpression>,
    characters: usize,
    repetitions: usize,
}

impl AlternationSet {
    fn from_elements(elements: impl IntoIterator<Item = RegularExpression>) -> Self {
        let mut set = AlternationSet::default();
        for element in elements {
            set.insert(element);
        }
        set
    }

    fn insert(&mut self, element: RegularExpression) {
        let is_character = matches!(element, RegularExpression::Character(..));
        let is_repetition = matches!(element, RegularExpression::Repetition(..));
        if self.elements.insert(element) {
            self.characters += usize::from(is_character);
            self.repetitions += usize::from(is_repetition);
        }
    }

    fn remove(&mut self, element: &RegularExpression) {
        if self.elements.remove(element) {
            match element {
                RegularExpression::Character(..) => self.characters -= 1,
                RegularExpression::Repetition(..) => self.repetitions -= 1,
                _ => {}
            }
        }
    }

    /// Applies the `(old, new)` element merges collected by a scan; when no
    /// merge happened, the incoming operand joins the alternation as its own
    /// element, mirroring the `had_*_union` bookkeeping of the
    /// `opunion_*_and_alternation` helpers.
    fn apply(
        &mut self,
        replacements: Vec<(RegularExpression, RegularExpression)>,
        fallback: &RegularExpression,
    ) {
        if replacements.is_empty() {
            self.insert(fallback.clone());
        } else {
            for (old, _) in &replacements {
                self.remove(old);
            }
            for (_, new) in replacements {
                self.insert(new);
            }
        }
    }
}

impl UnionAccumulator {
    fn new() -> Self {
        Self::Expression(RegularExpression::new_empty())
    }

    /// Starts from an existing expression, exactly as if it had been pushed
    /// onto a fresh accumulator.
    fn seeded_from(regex: &RegularExpression) -> Self {
        let mut accumulator = Self::new();
        accumulator.push(regex);
        accumulator
    }

    fn from_expression(regex: RegularExpression) -> Self {
        match regex {
            // `new_empty()` is the zero-element alternation and must stay an
            // `Expression` so the `is_empty` arm of `union_` keeps firing;
            // singleton alternations behave like their element under the
            // pairwise rules and are left untouched too.
            RegularExpression::Alternation(elements) if elements.len() > 1 => {
                Self::Alternation(AlternationSet::from_elements(elements))
            }
            regex => Self::Expression(regex),
        }
    }

    fn push(&mut self, other: &RegularExpression) {
        match self {
            Self::Expression(regex) => {
                let result = regex.union_(other).into_owned();
                *self = Self::from_expression(result);
            }
            Self::Alternation(set) => {
                if let RegularExpression::Alternation(other_elements) = other {
                    // The `union_` checks that apply to an alternation
                    // operand: ∅ contributes nothing, and neither does an
                    // operand equal to the accumulator.
                    if other_elements.is_empty()
                        || (other_elements.len() == set.elements.len()
                            && other_elements.iter().eq(set.elements.iter()))
                    {
                        return;
                    }
                    // Mirror the alternation ∪ alternation arm of `union_`:
                    // restart from `other` and re-push the accumulated
                    // elements (a collapse mid-way, e.g. to Σ*, then keeps
                    // folding the remaining elements through the pairwise
                    // rules, exactly like the original left-fold).
                    let previous = std::mem::take(set);
                    *self = Self::from_expression(other.clone());
                    for element in previous.elements {
                        self.push(&element);
                    }
                } else if let Some(collapsed) = Self::push_into_set(set, other) {
                    *self = Self::from_expression(collapsed);
                }
            }
        }
    }

    /// Unions a non-alternation operand into the accumulated elements in
    /// place. Returns `Some(result)` when the union collapses to something
    /// other than the updated alternation (Σ*, or the whole alternation under
    /// a quantifier).
    fn push_into_set(
        set: &mut AlternationSet,
        other: &RegularExpression,
    ) -> Option<RegularExpression> {
        // The checks at the top of `union_`, specialized to a multi-element
        // alternation accumulator (which is structurally never ∅, {""}, nor
        // total).
        if other.is_total() {
            return Some(RegularExpression::new_total());
        }
        if other.is_empty() {
            return None;
        }
        if other.is_empty_string() {
            // L ∪ {""} = L?, exactly like the empty-string arm of `union_`.
            return Some(Self::materialize(set).repeat(0, Some(1)));
        }

        match other {
            RegularExpression::Character(..) => {
                Self::union_character_into_set(set, other);
                None
            }
            RegularExpression::Repetition(base, min, max_opt) => {
                // The whole accumulator is the repetition's base:
                // `(a|b) ∪ (a|b){1,2}` collapses to `(a|b){1,2}`. Mirrors
                // `opunion_repetition_and_alternation`, including the
                // `max >= 1` guard that keeps `r{0,0}` (= {""}) from
                // absorbing the alternation.
                if *min <= 2
                    && max_opt.is_none_or(|max| max >= 1)
                    && matches!(&**base, RegularExpression::Alternation(elements)
                        if elements.len() == set.elements.len()
                            && elements.iter().eq(set.elements.iter()))
                {
                    return Some(base.repeat(cmp::min(1, *min), *max_opt));
                }
                Self::union_repetition_into_set(set, other);
                None
            }
            RegularExpression::Concat(..) => {
                Self::union_concat_into_set(set, other);
                None
            }
            RegularExpression::Alternation(..) => unreachable!("handled by push"),
        }
    }

    /// Mirrors `opunion_character_and_alternation`.
    fn union_character_into_set(set: &mut AlternationSet, this_character: &RegularExpression) {
        let RegularExpression::Character(this_range) = this_character else {
            unreachable!("Not character")
        };

        if set.characters == 0 && set.repetitions == 0 {
            set.insert(this_character.clone());
            return;
        }

        let mut replacements = Vec::new();
        for element in &set.elements {
            match element {
                RegularExpression::Character(range) => {
                    replacements.push((
                        element.clone(),
                        RegularExpression::Character(this_range.union(range)),
                    ));
                }
                RegularExpression::Repetition(..) => {
                    let repetition = RegularExpression::opunion_character_and_repetition(
                        this_character,
                        element,
                    );
                    if matches!(repetition, RegularExpression::Repetition(..)) {
                        replacements.push((element.clone(), repetition));
                    }
                }
                _ => {}
            }
        }
        set.apply(replacements, this_character);
    }

    /// Mirrors the element loop of `opunion_repetition_and_alternation`.
    fn union_repetition_into_set(set: &mut AlternationSet, this_repetition: &RegularExpression) {
        let mut replacements = Vec::new();
        for element in &set.elements {
            let merged = match element {
                RegularExpression::Repetition(..) => {
                    RegularExpression::opunion_repetition_and_repetition(this_repetition, element)
                }
                RegularExpression::Character(..) => {
                    RegularExpression::opunion_character_and_repetition(element, this_repetition)
                }
                RegularExpression::Concat(..) => {
                    RegularExpression::opunion_concat_and_repetition(element, this_repetition)
                }
                _ => continue,
            };
            if matches!(merged, RegularExpression::Repetition(..)) {
                replacements.push((element.clone(), merged));
            }
        }
        set.apply(replacements, this_repetition);
    }

    /// Mirrors the element loop of `opunion_concat_and_alternation`.
    fn union_concat_into_set(set: &mut AlternationSet, this_concat: &RegularExpression) {
        if set.repetitions == 0 {
            set.insert(this_concat.clone());
            return;
        }

        let mut replacements = Vec::new();
        for element in &set.elements {
            if matches!(element, RegularExpression::Repetition(..)) {
                let merged = RegularExpression::opunion_concat_and_repetition(this_concat, element);
                if matches!(merged, RegularExpression::Repetition(..)) {
                    replacements.push((element.clone(), merged));
                }
            }
        }
        set.apply(replacements, this_concat);
    }

    fn materialize(set: &AlternationSet) -> RegularExpression {
        RegularExpression::Alternation(set.elements.iter().cloned().collect())
    }

    fn is_total(&self) -> bool {
        matches!(self, Self::Expression(regex) if regex.is_total())
    }

    fn finish(self) -> RegularExpression {
        match self {
            Self::Expression(regex) => regex,
            Self::Alternation(set) => {
                RegularExpression::Alternation(set.elements.into_iter().collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // With an unbounded side, merging two repetitions is only sound when the
    // unbounded range starts no later than one past the bounded range's end;
    // otherwise there is a gap (e.g. `a? ∪ a{3,}` must not become `a*`, since
    // `a{2}` is in neither operand).
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

    // Merging `r ∪ r{0,0}` must keep both operands (= `r?`), not collapse to
    // `r{0,0}` (= {""}) and silently drop the other operand.
    #[test]
    fn union_with_zero_repetition_keeps_both_operands() {
        let equivalent = |result: &RegularExpression, expected: &str| {
            let expected = RegularExpression::new(expected).unwrap();
            result
                .to_automaton()
                .unwrap()
                .equivalent(&expected.to_automaton().unwrap())
                .unwrap()
        };

        // Character ∪ repetition.
        let a = RegularExpression::new("a").unwrap();
        let a_zero = RegularExpression::Repetition(Box::new(a.clone()), 0, Some(0));
        assert!(equivalent(&a.union(&a_zero), "a?"));
        assert!(equivalent(&a_zero.union(&a), "a?"));

        // Concat ∪ repetition.
        let ab = RegularExpression::new("ab").unwrap();
        let ab_zero = RegularExpression::Repetition(Box::new(ab.clone()), 0, Some(0));
        assert!(equivalent(&ab.union(&ab_zero), "(ab)?"));

        // Alternation ∪ repetition of the whole alternation.
        let alt = RegularExpression::new("(ab|cd)").unwrap();
        let alt_zero = RegularExpression::Repetition(Box::new(alt.clone()), 0, Some(0));
        assert!(equivalent(&alt.union(&alt_zero), "(ab|cd)?"));

        // Degenerate bounds (`a{2,0}` is the empty language) must not absorb
        // the other operand either. The result cannot be converted (degenerate
        // bounds are rejected by `to_automaton`), so check structurally that
        // `a` is still there.
        let a_degenerate = RegularExpression::Repetition(Box::new(a.clone()), 2, Some(0));
        let result = a.union(&a_degenerate);
        if let RegularExpression::Alternation(elements) = &result {
            assert!(elements.contains(&a), "`a` was dropped from {result}");
        } else {
            panic!("expected an alternation, got {result}");
        }
    }

    // Extracting common affixes from repetitions with degenerate hand-built
    // bounds (`a{5,2}`, the empty language) must not underflow when the other
    // side is unbounded with the same minimum.
    #[test]
    fn union_does_not_underflow_on_degenerate_repetition_bounds() {
        let a = RegularExpression::new("a").unwrap();
        let unbounded = RegularExpression::Concat(VecDeque::from([RegularExpression::Repetition(
            Box::new(a.clone()),
            5,
            None,
        )]));
        let degenerate = RegularExpression::Repetition(Box::new(a), 5, Some(2));

        // Must not panic, in either order, and must keep both operands
        // (degenerate bounds are rejected by `to_automaton`, so the check is
        // structural).
        for result in [unbounded.union(&degenerate), degenerate.union(&unbounded)] {
            if let RegularExpression::Alternation(elements) = &result {
                assert_eq!(2, elements.len(), "unexpected shape: {result}");
            } else {
                panic!("expected an alternation, got {result}");
            }
        }
    }

    // `union_all` folds through the incremental accumulator; distinct
    // branches must all be kept and stay queryable.
    #[test]
    fn union_all_keeps_distinct_branches() {
        // Vary the first and last character so no common affix is extracted
        // and the branches genuinely accumulate as alternation elements.
        let letter = |i: usize| (b'a' + (i % 26) as u8) as char;
        let branch = |i: usize| format!("{}{i:03}{}", letter(i), letter(i + 1));
        let branches: Vec<_> = (0..500)
            .map(|i| RegularExpression::new(&branch(i)).unwrap())
            .collect();
        let union = RegularExpression::union_all(branches.iter());

        if let RegularExpression::Alternation(elements) = &union {
            assert_eq!(500, elements.len());
        } else {
            panic!("expected an alternation, got {union}");
        }
        let automaton = union.to_automaton().unwrap();
        assert!(automaton.is_match(&branch(0)));
        assert!(automaton.is_match(&branch(42)));
        assert!(automaton.is_match(&branch(499)));
        assert!(!automaton.is_match("zzz"));
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
