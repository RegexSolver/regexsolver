use ahash::HashMapExt;
use nohash_hasher::IntMap;

use crate::{error::EngineError, fast_automaton::spanning_set::SpanningSet};

use super::Condition;

/// Converter to project [`Condition`] on a [`SpanningSet`].
pub struct ConditionConverter<'a, 'b> {
    from_spanning_set: &'a SpanningSet,
    to_spanning_set: &'b SpanningSet,
    equivalence_map: Vec<Vec<usize>>,
}

impl<'a, 'b> ConditionConverter<'a, 'b> {
    /// Build a converter to project [`Condition`] from `from_spanning_set` to `to_spanning_set`.
    ///
    /// Currently this method does not check that the provided [`SpanningSet`] are actually convertible.
    pub fn new(
        from_spanning_set: &'a SpanningSet,
        to_spanning_set: &'b SpanningSet,
    ) -> Result<Self, EngineError> {
        let mut to_base_map =
            IntMap::with_capacity(to_spanning_set.spanning_ranges_with_rest_len());
        for (i, base) in to_spanning_set
            .get_spanning_ranges_with_rest()
            .into_iter()
            .enumerate()
        {
            to_base_map.insert(i, base);
        }

        let mut equivalence_map: Vec<Vec<usize>> =
            Vec::with_capacity(from_spanning_set.get_number_of_spanning_ranges() + 1);
        for from_base in from_spanning_set.get_spanning_ranges_with_rest().iter() {
            let mut index = Vec::with_capacity(1);
            for (i, to_base) in &to_base_map {
                if from_base == to_base || from_base.has_intersection(to_base) {
                    index.push(*i);
                }
            }
            index.iter().for_each(|i| {
                to_base_map.remove(i);
            });
            equivalence_map.push(index);
        }

        Ok(ConditionConverter {
            from_spanning_set,
            to_spanning_set,
            equivalence_map,
        })
    }

    /// Project the given [`Condition`] from `from_spanning_set` to `to_spanning_set`.
    ///
    /// If `from_spanning_set` is not convertible to `to_spanning_set` or if the given [`Condition`] is not based on `from_spanning_set`,
    /// the resulting [`Condition`] will not have any relevance.
    pub fn convert(&self, condition: &Condition) -> Result<Condition, EngineError> {
        let mut new_condition = Condition::empty(self.to_spanning_set);
        for (from_index, to_indexes) in self.equivalence_map.iter().enumerate() {
            if let Some(has) = condition.0.get(from_index) {
                if has && !to_indexes.is_empty() {
                    to_indexes.iter().for_each(|&to_index| {
                        new_condition.0.set(to_index, true);
                    });
                }
            } else {
                return Err(EngineError::ConditionIndexOutOfBound);
            }
        }

        Ok(new_condition)
    }

    /// Returns `from_spanning_set`.
    pub fn get_from_spanning_set(&self) -> &'a SpanningSet {
        self.from_spanning_set
    }

    /// Returns `to_spanning_set`.
    pub fn get_to_spanning_set(&self) -> &'b SpanningSet {
        self.to_spanning_set
    }
}

#[cfg(test)]
mod tests {
    use regex_charclass::{char::Char, irange::range::AnyRange};

    use crate::Range;

    use super::*;

    fn get_from_spanning_set() -> SpanningSet {
        let ranges = vec![
            Range::new_from_range(Char::new('\0')..=Char::new('\u{2}')),
            Range::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}')),
            Range::new_from_range(Char::new('\u{9}')..=Char::new('\u{9}')),
        ];

        SpanningSet::compute_spanning_set(&ranges)
    }

    fn get_to_spanning_set() -> SpanningSet {
        let ranges = vec![
            Range::new_from_range(Char::new('\0')..=Char::new('\u{1}')),
            Range::new_from_range(Char::new('\u{2}')..=Char::new('\u{2}')),
            Range::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}')),
            Range::new_from_range(Char::new('\u{9}')..=Char::new('\u{9}')),
            Range::new_from_range(Char::new('\u{20}')..=Char::new('\u{22}')),
        ];

        SpanningSet::compute_spanning_set(&ranges)
    }

    #[test]
    fn test_convert() -> Result<(), String> {
        let from_spanning_set = get_from_spanning_set();
        let to_spanning_set = get_to_spanning_set();

        let converter = ConditionConverter::new(&from_spanning_set, &to_spanning_set).unwrap();

        let empty = Condition::empty(&from_spanning_set);
        assert!(converter.convert(&empty).unwrap().is_empty());

        let total = Condition::total(&from_spanning_set);
        assert!(converter.convert(&total).unwrap().is_total());

        let range = Range::new_from_range(Char::new('\0')..=Char::new('\u{2}'));
        let condition = Condition::from_range(&range, &from_spanning_set).unwrap();
        assert_eq!(
            range,
            converter
                .convert(&condition)
                .unwrap()
                .to_range(&to_spanning_set)
                .unwrap()
        );

        let range = Range::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}'));
        let condition = Condition::from_range(&range, &from_spanning_set).unwrap();
        assert_eq!(
            range,
            converter
                .convert(&condition)
                .unwrap()
                .to_range(&to_spanning_set)
                .unwrap()
        );

        let range = Range::new_from_ranges(&[
            AnyRange::from(Char::new('\u{4}')..=Char::new('\u{6}')),
            AnyRange::from(Char::new('\u{9}')..=Char::new('\u{9}')),
        ]);
        let condition = Condition::from_range(&range, &from_spanning_set).unwrap();
        assert_eq!(
            range,
            converter
                .convert(&condition)
                .unwrap()
                .to_range(&to_spanning_set)
                .unwrap()
        );

        Ok(())
    }
}
