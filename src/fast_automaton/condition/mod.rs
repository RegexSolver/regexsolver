use std::hash::Hash;

use fast_bit_vec::FastBitVec;
use regex_charclass::{CharacterClass, char::Char};

use crate::{CharRange, error::EngineError};

use super::spanning_set::SpanningSet;
/// The [`ConditionConverter`](converter::ConditionConverter), which converts a
/// [`Condition`] from one spanning set to another.
pub mod converter;
mod fast_bit_vec;

/// The characters of a transition of a [`FastAutomaton`](crate::FastAutomaton):
/// a set of bases of the automaton's [`SpanningSet`], held as one bit per base.
/// The union, intersection and complement of conditions are bitwise operations
/// (see [this
/// article](https://alexvbrdn.me/post/optimizing-transition-conditions-automaton-representation)).
///
/// A condition only has a meaning with the spanning set it was built over, and
/// two conditions that are combined must share it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Condition(FastBitVec);

impl std::fmt::Display for Condition {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Hash for Condition {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Condition {
    /// Returns the condition that matches no character, over `spanning_set`.
    #[inline]
    pub fn empty(spanning_set: &SpanningSet) -> Self {
        Self(FastBitVec::from_elem(
            spanning_set.spanning_ranges_with_rest_len(),
            false,
        ))
    }

    /// Returns the condition that matches every character, over `spanning_set`.
    #[inline]
    pub fn total(spanning_set: &SpanningSet) -> Self {
        Self(FastBitVec::from_elem(
            spanning_set.spanning_ranges_with_rest_len(),
            true,
        ))
    }

    /// Converts `range` to a condition over `spanning_set`.
    ///
    /// Only the bases that lie entirely inside `range` are kept: when
    /// `spanning_set` cannot express `range`, the condition matches fewer
    /// characters than `range`. Compare [`to_range`](Self::to_range) with
    /// `range` to detect it, or let
    /// [`FastAutomaton::add_transition_from_range`](crate::fast_automaton::FastAutomaton::add_transition_from_range)
    /// extend the spanning set.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::ConditionInvalidRange`] if `range` is not empty
    /// and no base lies entirely inside it.
    pub fn from_range(range: &CharRange, spanning_set: &SpanningSet) -> Result<Self, EngineError> {
        if range.is_empty() {
            return Ok(Self::empty(spanning_set));
        } else if range.is_total() {
            return Ok(Self::total(spanning_set));
        }

        let mut cond = Self::empty(spanning_set);

        for (i, base) in spanning_set.spanning_ranges_with_rest().enumerate() {
            if range.contains_all(base) {
                cond.0.set(i, true);
            }
        }

        if cond.is_empty() {
            return Err(EngineError::ConditionInvalidRange);
        }

        Ok(cond)
    }

    /// Returns the condition whose only set bit is base `i` of `spanning_set`
    /// (bit `i` corresponds to `spanning_ranges_with_rest()`'s element `i`,
    /// the rest range first when present).
    #[inline]
    pub(crate) fn single_base(i: usize, spanning_set: &SpanningSet) -> Self {
        let mut cond = Self::empty(spanning_set);
        cond.insert_base(i);
        cond
    }

    /// Adds base `i` to the condition (see [`single_base`](Self::single_base)).
    #[inline]
    pub(crate) fn insert_base(&mut self, i: usize) {
        self.0.set(i, true);
    }

    /// Returns the characters the condition matches, read over `spanning_set`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::IncompatibleSpanningSet`] if the condition was
    /// built over a spanning set with another number of bases.
    pub fn to_range(&self, spanning_set: &SpanningSet) -> Result<CharRange, EngineError> {
        // A condition only carries meaning relative to the spanning set it
        // was built from; a width mismatch means they differ, so reject it
        // rather than index out of bounds or silently drop bits.
        if self.0.len() != spanning_set.spanning_ranges_with_rest_len() {
            return Err(EngineError::IncompatibleSpanningSet);
        }

        let mut range = CharRange::empty();

        for (i, base) in spanning_set.spanning_ranges_with_rest().enumerate() {
            if self.0.get(i) {
                range = range.union(base);
            }
        }

        Ok(range)
    }

    /// Returns the condition that matches the characters of `self` and those of
    /// `other`. Both must share the same spanning set.
    #[inline]
    pub fn union(&self, other: &Condition) -> Self {
        let mut new_cond = self.clone();
        new_cond.0.union(&other.0);
        new_cond
    }

    /// Adds the characters of `other` to `self`. Both must share the same
    /// spanning set.
    #[inline]
    pub fn union_with(&mut self, other: &Condition) {
        self.0.union(&other.0);
    }

    /// Returns the condition that matches the characters that `self` and
    /// `other` have in common. Both must share the same spanning set.
    #[inline]
    pub fn intersection(&self, other: &Condition) -> Self {
        let mut new_cond = self.clone();
        new_cond.0.intersection(&other.0);
        new_cond
    }

    /// Returns the condition that matches the characters that `self` does not.
    #[inline]
    pub fn complement(&self) -> Self {
        let mut new_cond = self.clone();
        new_cond.0.complement();
        new_cond
    }

    /// Returns the condition that matches the characters of `self` that `other`
    /// does not match. Both must share the same spanning set.
    #[inline]
    pub fn difference(&self, other: &Condition) -> Self {
        let mut new_cond = self.clone();
        new_cond.0.difference(&other.0);
        new_cond
    }

    /// Returns `true` if every character of `self` is in `other`. Both must
    /// share the same spanning set.
    #[inline]
    pub(crate) fn is_subset_of(&self, other: &Condition) -> bool {
        self.0.is_subset(&other.0)
    }

    /// Makes the condition match no character.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }

    /// Returns `true` if `self` and `other` have a character in common. Both
    /// must share the same spanning set.
    #[inline]
    pub fn has_intersection(&self, other: &Condition) -> bool {
        self.0.has_intersection(&other.0)
    }

    /// Returns `true` if the condition, read over `spanning_set`, matches
    /// `character`, given as a Unicode scalar value. A value that is not a
    /// scalar value is not matched.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::IncompatibleSpanningSet`] if the condition was
    /// built over a spanning set with another number of bases.
    #[inline]
    pub fn has_character(
        &self,
        character: &u32,
        spanning_set: &SpanningSet,
    ) -> Result<bool, EngineError> {
        let Some(character) = Char::from_u32(*character) else {
            return Ok(false);
        };
        if self.0.len() != spanning_set.spanning_ranges_with_rest_len() {
            return Err(EngineError::IncompatibleSpanningSet);
        }

        // Bit `i` corresponds to `spanning_ranges_with_rest()[i]` (the rest
        // range first, when present). Testing set bits directly avoids
        // materializing the union of their ranges (`to_range` clones and
        // unions every base) on the `is_match` hot path.
        let mut i = 0;
        if !spanning_set.rest().is_empty() {
            if self.0.get(i) && spanning_set.rest().contains(character) {
                return Ok(true);
            }
            i += 1;
        }
        for base in spanning_set.spanning_ranges() {
            if self.0.get(i) && base.contains(character) {
                return Ok(true);
            }
            i += 1;
        }
        Ok(false)
    }

    /// Returns `true` if the condition matches no character.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.empty()
    }

    /// Returns `true` if the condition matches every character.
    #[inline]
    pub fn is_total(&self) -> bool {
        self.0.total()
    }

    /// Returns the number of characters the condition matches, read over
    /// `spanning_set`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::IncompatibleSpanningSet`] if the condition was
    /// built over a spanning set with another number of bases.
    #[inline]
    pub fn cardinality(&self, spanning_set: &SpanningSet) -> Result<u32, EngineError> {
        if self.0.len() != spanning_set.spanning_ranges_with_rest_len() {
            return Err(EngineError::IncompatibleSpanningSet);
        }

        // The bases are disjoint, so the cardinality of their union is the
        // sum of their cardinalities.
        let mut cardinality = 0u32;
        for (i, base) in spanning_set.spanning_ranges_with_rest().enumerate() {
            if self.0.get(i) {
                cardinality += base.get_cardinality();
            }
        }
        Ok(cardinality)
    }

    /// Returns the bits of the condition, one per base of its spanning set: the
    /// rest first, when it is not empty, then the ranges in order.
    #[inline]
    pub fn binary_representation(&self) -> Vec<bool> {
        self.0.bits()
    }

    /// Iterates the indices of the set bits (i.e., the bases the condition
    /// covers) in ascending order, without allocating.
    #[inline]
    pub(crate) fn iter_set_bits(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter_set_bits()
    }
}

#[cfg(test)]
mod tests {
    use converter::ConditionConverter;
    use regex_charclass::irange::range::AnyRange;

    use super::*;

    fn spanning_set() -> SpanningSet {
        let ranges = vec![
            CharRange::new_from_range(Char::new('\u{0}')..=Char::new('\u{2}')),
            CharRange::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}')),
            CharRange::new_from_range(Char::new('\u{9}')..=Char::new('\u{9}')),
        ];

        SpanningSet::compute_spanning_set(&ranges)
    }

    fn get_test_cases_range() -> Vec<CharRange> {
        vec![
            CharRange::empty(),
            CharRange::total(),
            CharRange::new_from_range(Char::new('\u{0}')..=Char::new('\u{2}')),
            CharRange::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}')),
            CharRange::new_from_ranges(&[
                AnyRange::from(Char::new('\u{0}')..=Char::new('\u{2}')),
                AnyRange::from(Char::new('\u{4}')..=Char::new('\u{6}')),
            ]),
            CharRange::new_from_range(Char::new('\u{9}')..=Char::new('\u{9}')),
        ]
    }

    // Evaluating a condition against a spanning set it was not built from
    // must report the incompatibility rather than panic (too short) or
    // silently drop bits (too long).
    #[test]
    fn to_range_rejects_incompatible_spanning_set() {
        let small = SpanningSet::compute_spanning_set(&[CharRange::new_from_range(
            Char::new('a')..=Char::new('a'),
        )]);
        let large = spanning_set();

        let condition = Condition::total(&small);
        assert_eq!(
            condition.to_range(&large),
            Err(EngineError::IncompatibleSpanningSet)
        );

        let condition = Condition::total(&large);
        assert_eq!(
            condition.to_range(&small),
            Err(EngineError::IncompatibleSpanningSet)
        );
    }

    // `ConditionConverter::convert` must report an error, not panic, on a
    // condition that was not built over its source spanning set.
    #[test]
    fn convert_rejects_incompatible_condition() {
        let small = SpanningSet::compute_spanning_set(&[CharRange::new_from_range(
            Char::new('a')..=Char::new('a'),
        )]);
        let merged = small.merge(&spanning_set());
        let converter = ConditionConverter::new(&small, &merged).unwrap();

        let foreign = Condition::total(&merged);
        assert_eq!(
            converter.convert(&foreign),
            Err(EngineError::IncompatibleSpanningSet)
        );
    }

    #[test]
    fn test_empty_total() -> Result<(), String> {
        let spanning_set = spanning_set();
        let empty = Condition::empty(&spanning_set);
        assert!(empty.is_empty());
        assert_eq!(
            vec![false, false, false, false],
            empty.binary_representation()
        );
        let total = Condition::total(&spanning_set);
        assert!(total.is_total());
        assert_eq!(vec![true, true, true, true], total.binary_representation());

        assert_eq!(CharRange::empty(), empty.to_range(&spanning_set).unwrap());
        assert_eq!(CharRange::total(), total.to_range(&spanning_set).unwrap());

        assert_eq!(
            empty,
            Condition::from_range(&CharRange::empty(), &spanning_set).unwrap()
        );
        assert_eq!(
            total,
            Condition::from_range(&CharRange::total(), &spanning_set).unwrap()
        );

        assert_eq!(empty, total.complement());
        assert_eq!(total, empty.complement());

        let spanning_set = SpanningSet::new_total();
        let empty = Condition::empty(&spanning_set);
        let total = Condition::total(&spanning_set);

        assert_eq!(CharRange::empty(), empty.to_range(&spanning_set).unwrap());
        assert_eq!(CharRange::total(), total.to_range(&spanning_set).unwrap());

        assert_eq!(
            empty,
            Condition::from_range(&CharRange::empty(), &spanning_set).unwrap()
        );
        assert_eq!(vec![false], empty.binary_representation());

        assert_eq!(
            total,
            Condition::from_range(&CharRange::total(), &spanning_set).unwrap()
        );
        assert_eq!(vec![true], total.binary_representation());

        assert_eq!(empty, total.complement());
        assert_eq!(total, empty.complement());

        Ok(())
    }

    #[test]
    fn test_from_to_range() -> Result<(), String> {
        let spanning_set = spanning_set();

        for range in get_test_cases_range() {
            assert_range_conversion_to_range(&range, &spanning_set);
            assert_range_conversion_to_range(&range.complement(), &spanning_set);
        }

        Ok(())
    }

    fn assert_range_conversion_to_range(range: &CharRange, spanning_set: &SpanningSet) {
        let condition = Condition::from_range(range, spanning_set).unwrap();
        let range_from_condition = condition.to_range(spanning_set).unwrap();
        assert_eq!(range, &range_from_condition);

        let range_from_condition = condition.complement().to_range(spanning_set).unwrap();

        assert_eq!(range.complement(), range_from_condition);
    }

    #[test]
    fn test_project_to() -> Result<(), String> {
        let current_spanning_set = spanning_set();

        let ranges = vec![
            CharRange::new_from_range(Char::new('\u{0}')..=Char::new('\u{1}')),
            CharRange::new_from_range(Char::new('\u{2}')..=Char::new('\u{2}')),
            CharRange::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}')),
            CharRange::new_from_range(Char::new('\u{5}')..=Char::new('\u{6}')),
            CharRange::new_from_range(Char::new('\u{9}')..=Char::new('\u{9}')),
        ];
        let new_spanning_set = SpanningSet::compute_spanning_set(&ranges);
        let condition_converter =
            ConditionConverter::new(&current_spanning_set, &new_spanning_set).unwrap();

        for range in get_test_cases_range() {
            assert_project_to(
                &range,
                &current_spanning_set,
                &new_spanning_set,
                &condition_converter,
            );
            assert_project_to(
                &range.complement(),
                &current_spanning_set,
                &new_spanning_set,
                &condition_converter,
            );
        }

        Ok(())
    }

    fn assert_project_to(
        range: &CharRange,
        currently_used_spanning_set: &SpanningSet,
        newly_used_spanning_set: &SpanningSet,
        condition_converter: &ConditionConverter,
    ) {
        let condition = Condition::from_range(range, currently_used_spanning_set).unwrap();
        let projected_condition = condition_converter.convert(&condition).unwrap();

        assert_eq!(
            range,
            &condition.to_range(currently_used_spanning_set).unwrap()
        );
        assert_eq!(
            range,
            &projected_condition
                .to_range(newly_used_spanning_set)
                .unwrap()
        );

        let expected_condition = Condition::from_range(range, newly_used_spanning_set).unwrap();
        assert_eq!(expected_condition, projected_condition);
    }

    #[test]
    fn test_union_intersection_complement() -> Result<(), String> {
        let used_characters = spanning_set();

        for range_1 in get_test_cases_range() {
            for range_2 in get_test_cases_range() {
                assert_union_intersection_complement(&range_1, &range_2, &used_characters);
                assert_union_intersection_complement(
                    &range_1.complement(),
                    &range_2,
                    &used_characters,
                );
                assert_union_intersection_complement(
                    &range_1,
                    &range_2.complement(),
                    &used_characters,
                );
                assert_union_intersection_complement(
                    &range_1.complement(),
                    &range_2.complement(),
                    &used_characters,
                );
            }
        }

        Ok(())
    }

    fn assert_union_intersection_complement(
        range_1: &CharRange,
        range_2: &CharRange,
        used_characters: &SpanningSet,
    ) {
        let condition_1 = Condition::from_range(range_1, used_characters).unwrap();
        let condition_2 = Condition::from_range(range_2, used_characters).unwrap();

        assert_eq!(
            Condition::empty(used_characters),
            condition_1.intersection(&condition_1.complement())
        );

        assert_eq!(
            Condition::empty(used_characters),
            condition_2.intersection(&condition_2.complement())
        );

        let condition_union = condition_1.union(&condition_2);

        let condition_intersection_complement = condition_1
            .complement()
            .intersection(&condition_2.complement())
            .complement();

        assert_eq!(condition_union, condition_intersection_complement);
    }

    #[test]
    fn test_1() -> Result<(), String> {
        let ranges = vec![
            CharRange::new_from_range(Char::new('\u{0}')..=Char::new('\u{9}')),
            CharRange::new_from_range(Char::new('\u{B}')..=Char::new('\u{63}')),
            CharRange::new_from_range(Char::new('\u{65}')..=Char::new('\u{10FFFF}')),
        ];
        let spanning_set = SpanningSet::compute_spanning_set(&ranges);
        println!("{:?}", spanning_set);

        let range1 = CharRange::new_from_ranges(&[
            AnyRange::from(Char::new('\u{0}')..=Char::new('\u{9}')),
            AnyRange::from(Char::new('\u{B}')..=Char::new('\u{63}')),
            AnyRange::from(Char::new('\u{65}')..=Char::new('\u{10FFFF}')),
        ]);
        let condition1 = Condition::from_range(&range1, &spanning_set).unwrap();
        assert_eq!(range1, condition1.to_range(&spanning_set).unwrap());

        let range2 = CharRange::new_from_range(Char::new('\u{B}')..=Char::new('\u{63}'));
        let condition2 = Condition::from_range(&range2, &spanning_set).unwrap();
        assert_eq!(range2, condition2.to_range(&spanning_set).unwrap());

        let union_condition = condition1.union(&condition2);
        let union_range = union_condition.to_range(&spanning_set).unwrap();

        assert_eq!(range1, union_range);

        let complement = union_condition.complement();
        assert_eq!(
            union_range.complement(),
            complement.to_range(&spanning_set).unwrap()
        );

        Ok(())
    }
}
