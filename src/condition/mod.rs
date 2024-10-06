use std::hash::Hash;

use crate::Range;
use fast_bit_vec::FastBitVec;
use regex_charclass::{char::Char, CharacterClass};

use crate::{error::EngineError, used_bases::UsedBases};
mod fast_bit_vec;

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
    pub fn empty(used_bases: &UsedBases) -> Self {
        Self(FastBitVec::from_elem(used_bases.elements_len(), false))
    }

    #[inline]
    pub fn total(used_bases: &UsedBases) -> Self {
        Self(FastBitVec::from_elem(used_bases.elements_len(), true))
    }

    pub fn from_range(range: &Range, used_bases: &UsedBases) -> Result<Self, EngineError> {
        if range.is_empty() {
            return Ok(Self::empty(used_bases));
        } else if range.is_total() {
            return Ok(Self::total(used_bases));
        }

        let mut cond = Self::empty(used_bases);

        for (i, base) in used_bases.get_elements().iter().enumerate() {
            if range.contains_all(base) {
                cond.0.set(i, true);
            }
        }

        if cond.is_empty() {
            return Err(EngineError::ConditionInvalidRange);
        }

        Ok(cond)
    }

    pub fn to_range(&self, used_bases: &UsedBases) -> Result<Range, EngineError> {
        let mut range = Range::empty();

        for (i, base) in used_bases.get_elements().iter().enumerate() {
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

    pub fn project_to(
        &self,
        currently_used_bases: &UsedBases,
        newly_used_bases: &UsedBases,
    ) -> Result<Self, EngineError> {
        if currently_used_bases == newly_used_bases {
            Ok(self.clone())
        } else {
            let range = self.to_range(currently_used_bases)?;
            Self::from_range(&range, newly_used_bases)
        }
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
        used_bases: &UsedBases,
    ) -> Result<bool, EngineError> {
        if let Some(character) = Char::from_u32(*character) {
            Ok(self.to_range(used_bases)?.contains(character))
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
    pub fn get_cardinality(&self, used_bases: &UsedBases) -> Result<u32, EngineError> {
        Ok(self.to_range(used_bases)?.get_cardinality())
    }
}

#[cfg(test)]
mod tests {
    use regex_charclass::irange::range::AnyRange;

    use super::*;

    fn get_used_bases() -> UsedBases {
        let ranges = vec![
            Range::new_from_range(Char::new('\0')..=Char::new('\u{2}')),
            Range::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}')),
            Range::new_from_range(Char::new('\u{9}')..=Char::new('\u{9}')),
        ];

        UsedBases::compute_used_bases(&ranges)
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
        let used_bases = get_used_bases();
        let empty = Condition::empty(&used_bases);
        assert!(empty.is_empty());
        let total = Condition::total(&used_bases);
        println!("{total}");
        assert!(total.is_total());

        assert_eq!(Range::empty(), empty.to_range(&used_bases).unwrap());
        assert_eq!(Range::total(), total.to_range(&used_bases).unwrap());

        assert_eq!(
            empty,
            Condition::from_range(&Range::empty(), &used_bases).unwrap()
        );
        assert_eq!(
            total,
            Condition::from_range(&Range::total(), &used_bases).unwrap()
        );

        assert_eq!(empty, total.complement());
        assert_eq!(total, empty.complement());

        let used_bases = UsedBases::new_total();
        let empty = Condition::empty(&used_bases);
        let total = Condition::total(&used_bases);

        assert_eq!(Range::empty(), empty.to_range(&used_bases).unwrap());
        assert_eq!(Range::total(), total.to_range(&used_bases).unwrap());

        assert_eq!(
            empty,
            Condition::from_range(&Range::empty(), &used_bases).unwrap()
        );
        assert_eq!(
            total,
            Condition::from_range(&Range::total(), &used_bases).unwrap()
        );

        assert_eq!(empty, total.complement());
        assert_eq!(total, empty.complement());

        Ok(())
    }

    #[test]
    fn test_from_to_range() -> Result<(), String> {
        let used_bases = get_used_bases();

        for range in get_test_cases_range() {
            assert_range_convertion_to_range(&range, &used_bases);
            assert_range_convertion_to_range(&range.complement(), &used_bases);
        }

        Ok(())
    }

    fn assert_range_convertion_to_range(range: &Range, used_bases: &UsedBases) {
        let condition = Condition::from_range(range, used_bases).unwrap();
        let range_from_condition = condition.to_range(used_bases).unwrap();
        assert_eq!(range, &range_from_condition);

        let range_from_condition = condition.complement().to_range(used_bases).unwrap();

        assert_eq!(range.complement(), range_from_condition);
    }

    #[test]
    fn test_project_to() -> Result<(), String> {
        let currently_used_bases = get_used_bases();

        let ranges = vec![
            Range::new_from_range(Char::new('\u{0}')..=Char::new('\u{1}')),
            Range::new_from_range(Char::new('\u{2}')..=Char::new('\u{3}')),
            Range::new_from_range(Char::new('\u{4}')..=Char::new('\u{5}')),
            Range::new_from_range(Char::new('\u{6}')..=Char::new('\u{7}')),
            Range::new_from_range(Char::new('\u{9}')..=Char::new('\u{9}')),
        ];
        let newly_used_bases = UsedBases::compute_used_bases(&ranges);

        for range in get_test_cases_range() {
            assert_project_to(&range, &currently_used_bases, &newly_used_bases);
            assert_project_to(
                &range.complement(),
                &currently_used_bases,
                &newly_used_bases,
            );
        }

        Ok(())
    }

    fn assert_project_to(
        range: &Range,
        currently_used_characters: &UsedBases,
        newly_used_characters: &UsedBases,
    ) {
        let condition = Condition::from_range(range, currently_used_characters).unwrap();
        let projected_condition = condition
            .project_to(currently_used_characters, newly_used_characters)
            .unwrap();

        let expected_condition = Condition::from_range(range, newly_used_characters).unwrap();
        assert_eq!(expected_condition, projected_condition);
    }

    #[test]
    fn test_union_intersection_complement() -> Result<(), String> {
        let used_characters = get_used_bases();

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
        used_characters: &UsedBases,
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
        let used_bases = UsedBases::compute_used_bases(&ranges);
        println!("{:?}", used_bases);

        let range1 = Range::new_from_ranges(&[
            AnyRange::from(Char::new('\u{0}')..=Char::new('\u{9}')),
            AnyRange::from(Char::new('\u{B}')..=Char::new('\u{63}')),
            AnyRange::from(Char::new('\u{65}')..=Char::new('\u{10FFFF}')),
        ]);
        let condition1 = Condition::from_range(&range1, &used_bases).unwrap();
        assert_eq!(range1, condition1.to_range(&used_bases).unwrap());

        let range2 = Range::new_from_range(Char::new('\u{B}')..=Char::new('\u{63}'));
        let condition2 = Condition::from_range(&range2, &used_bases).unwrap();
        assert_eq!(range2, condition2.to_range(&used_bases).unwrap());

        let union_condition = condition1.union(&condition2);
        let union_range = union_condition.to_range(&used_bases).unwrap();

        assert_eq!(range1, union_range);

        let complement = union_condition.complement();
        assert_eq!(
            union_range.complement(),
            complement.to_range(&used_bases).unwrap()
        );

        Ok(())
    }
}
