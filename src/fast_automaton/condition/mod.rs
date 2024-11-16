use std::hash::Hash;

use crate::Range;
use fast_bit_vec::FastBitVec;
use regex_charclass::{char::Char, CharacterClass};

use crate::error::EngineError;

use super::spanning_set::SpanningSet;
pub mod converter;
mod fast_bit_vec;

/// Contains the condition of a transition in a [`crate::FastAutomaton`]
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
    #[inline]
    pub fn empty(spanning_set: &SpanningSet) -> Self {
        Self(FastBitVec::from_elem(
            spanning_set.spanning_ranges_with_rest_len(),
            false,
        ))
    }

    #[inline]
    pub fn total(spanning_set: &SpanningSet) -> Self {
        Self(FastBitVec::from_elem(
            spanning_set.spanning_ranges_with_rest_len(),
            true,
        ))
    }

    pub fn from_range(range: &Range, spanning_set: &SpanningSet) -> Result<Self, EngineError> {
        if range.is_empty() {
            return Ok(Self::empty(spanning_set));
        } else if range.is_total() {
            return Ok(Self::total(spanning_set));
        }

        let mut cond = Self::empty(spanning_set);

        for (i, base) in spanning_set
            .get_spanning_ranges_with_rest()
            .iter()
            .enumerate()
        {
            if range.contains_all(base) {
                cond.0.set(i, true);
            }
        }

        if cond.is_empty() {
            return Err(EngineError::ConditionInvalidRange);
        }

        Ok(cond)
    }

    pub fn to_range(&self, spanning_set: &SpanningSet) -> Result<Range, EngineError> {
        let mut range = Range::empty();

        for (i, base) in spanning_set
            .get_spanning_ranges_with_rest()
            .iter()
            .enumerate()
        {
            if let Some(has) = self.0.get(i) {
                if has {
                    range = range.union(base);
                }
            } else {
                return Err(EngineError::ConditionIndexOutOfBound);
            }
        }

        Ok(range)
    }

    #[inline]
    pub fn union(&self, cond: &Condition) -> Self {
        let mut new_cond = self.clone();
        new_cond.0.union(&cond.0);
        new_cond
    }

    #[inline]
    pub fn intersection(&self, cond: &Condition) -> Self {
        let mut new_cond = self.clone();
        new_cond.0.intersection(&cond.0);
        new_cond
    }

    #[inline]
    pub fn complement(&self) -> Self {
        let mut new_cond = self.clone();
        new_cond.0.complement();
        new_cond
    }

    #[inline]
    pub fn difference(&self, cond: &Condition) -> Self {
        let mut new_cond = self.clone();
        let subtrahend = cond.complement();
        new_cond.0.intersection(&subtrahend.0);
        new_cond
    }

    #[inline]
    pub fn has_intersection(&self, cond: &Condition) -> bool {
        self.0.has_intersection(&cond.0)
    }

    #[inline]
    pub fn has_character(
        &self,
        character: &u32,
        spanning_set: &SpanningSet,
    ) -> Result<bool, EngineError> {
        if let Some(character) = Char::from_u32(*character) {
            Ok(self.to_range(spanning_set)?.contains(character))
        } else {
            Ok(false)
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.empty()
    }

    #[inline]
    pub fn is_total(&self) -> bool {
        self.0.total()
    }

    #[inline]
    pub fn get_cardinality(&self, spanning_set: &SpanningSet) -> Result<u32, EngineError> {
        Ok(self.to_range(spanning_set)?.get_cardinality())
    }

    pub fn get_bits(&self) -> Vec<bool> {
        self.0.get_bits()
    }
}

#[cfg(test)]
mod tests {
    use converter::ConditionConverter;
    use regex_charclass::irange::range::AnyRange;

    use super::*;

    fn get_spanning_set() -> SpanningSet {
        let ranges = vec![
            Range::new_from_range(Char::new('\u{0}')..=Char::new('\u{2}')),
            Range::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}')),
            Range::new_from_range(Char::new('\u{9}')..=Char::new('\u{9}')),
        ];

        SpanningSet::compute_spanning_set(&ranges)
    }

    fn get_test_cases_range() -> Vec<Range> {
        vec![
            Range::empty(),
            Range::total(),
            Range::new_from_range(Char::new('\u{0}')..=Char::new('\u{2}')),
            Range::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}')),
            Range::new_from_ranges(&[
                AnyRange::from(Char::new('\u{0}')..=Char::new('\u{2}')),
                AnyRange::from(Char::new('\u{4}')..=Char::new('\u{6}')),
            ]),
            Range::new_from_range(Char::new('\u{9}')..=Char::new('\u{9}')),
        ]
    }

    #[test]
    fn test_empty_total() -> Result<(), String> {
        let spanning_set = get_spanning_set();
        let empty = Condition::empty(&spanning_set);
        //println!("{empty}");
        assert!(empty.is_empty());
        assert_eq!(vec![false, false, false, false], empty.get_bits());
        let total = Condition::total(&spanning_set);
        //println!("{total}");
        assert!(total.is_total());
        assert_eq!(vec![true, true, true, true], total.get_bits());

        assert_eq!(Range::empty(), empty.to_range(&spanning_set).unwrap());
        assert_eq!(Range::total(), total.to_range(&spanning_set).unwrap());

        assert_eq!(
            empty,
            Condition::from_range(&Range::empty(), &spanning_set).unwrap()
        );
        assert_eq!(
            total,
            Condition::from_range(&Range::total(), &spanning_set).unwrap()
        );

        assert_eq!(empty, total.complement());
        assert_eq!(total, empty.complement());

        let spanning_set = SpanningSet::new_total();
        let empty = Condition::empty(&spanning_set);
        let total = Condition::total(&spanning_set);

        assert_eq!(Range::empty(), empty.to_range(&spanning_set).unwrap());
        assert_eq!(Range::total(), total.to_range(&spanning_set).unwrap());

        assert_eq!(
            empty,
            Condition::from_range(&Range::empty(), &spanning_set).unwrap()
        );
        assert_eq!(vec![false], empty.get_bits());

        assert_eq!(
            total,
            Condition::from_range(&Range::total(), &spanning_set).unwrap()
        );
        assert_eq!(vec![true], total.get_bits());

        assert_eq!(empty, total.complement());
        assert_eq!(total, empty.complement());

        Ok(())
    }

    #[test]
    fn test_from_to_range() -> Result<(), String> {
        let spanning_set = get_spanning_set();

        for range in get_test_cases_range() {
            assert_range_convertion_to_range(&range, &spanning_set);
            assert_range_convertion_to_range(&range.complement(), &spanning_set);
        }

        Ok(())
    }

    fn assert_range_convertion_to_range(range: &Range, spanning_set: &SpanningSet) {
        let condition = Condition::from_range(range, spanning_set).unwrap();
        let range_from_condition = condition.to_range(spanning_set).unwrap();
        assert_eq!(range, &range_from_condition);

        let range_from_condition = condition.complement().to_range(spanning_set).unwrap();

        assert_eq!(range.complement(), range_from_condition);
    }

    #[test]
    fn test_project_to() -> Result<(), String> {
        let current_spanning_set = get_spanning_set();

        let ranges = vec![
            Range::new_from_range(Char::new('\u{0}')..=Char::new('\u{1}')),
            Range::new_from_range(Char::new('\u{2}')..=Char::new('\u{2}')),
            Range::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}')),
            Range::new_from_range(Char::new('\u{5}')..=Char::new('\u{6}')),
            Range::new_from_range(Char::new('\u{9}')..=Char::new('\u{9}')),
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
        range: &Range,
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
        let used_characters = get_spanning_set();

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
        range_1: &Range,
        range_2: &Range,
        used_characters: &SpanningSet,
    ) {
        let condition_1 = Condition::from_range(range_1, used_characters).unwrap();
        let condition_2 = Condition::from_range(range_2, used_characters).unwrap();

        assert_eq!(
            Condition::empty(&used_characters),
            condition_1.intersection(&condition_1.complement())
        );

        assert_eq!(
            Condition::empty(&used_characters),
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
            Range::new_from_range(Char::new('\u{0}')..=Char::new('\u{9}')),
            Range::new_from_range(Char::new('\u{B}')..=Char::new('\u{63}')),
            Range::new_from_range(Char::new('\u{65}')..=Char::new('\u{10FFFF}')),
        ];
        let spanning_set = SpanningSet::compute_spanning_set(&ranges);
        println!("{:?}", spanning_set);

        let range1 = Range::new_from_ranges(&[
            AnyRange::from(Char::new('\u{0}')..=Char::new('\u{9}')),
            AnyRange::from(Char::new('\u{B}')..=Char::new('\u{63}')),
            AnyRange::from(Char::new('\u{65}')..=Char::new('\u{10FFFF}')),
        ]);
        let condition1 = Condition::from_range(&range1, &spanning_set).unwrap();
        assert_eq!(range1, condition1.to_range(&spanning_set).unwrap());

        let range2 = Range::new_from_range(Char::new('\u{B}')..=Char::new('\u{63}'));
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
