use ahash::HashMapExt;

use crate::{IntMap, error::EngineError, fast_automaton::spanning_set::SpanningSet};

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
    /// Two directions are legitimate: refinement (a merged spanning set
    /// before a binary operation) and coarsening (a recomputed minimal
    /// spanning set, where bases no transition uses fold into the rest). The
    /// pair is therefore not validated here; instead [`convert`](Self::convert)
    /// asserts in debug builds that each projection preserves the
    /// condition's character range.
    pub fn new(
        from_spanning_set: &'a SpanningSet,
        to_spanning_set: &'b SpanningSet,
    ) -> Result<Self, EngineError> {
        let mut to_base_map =
            IntMap::with_capacity(to_spanning_set.spanning_ranges_with_rest_len());
        for (i, base) in to_spanning_set
            .spanning_ranges_with_rest()
            .into_iter()
            .enumerate()
        {
            to_base_map.insert(i, base);
        }

        let mut equivalence_map: Vec<Vec<usize>> =
            Vec::with_capacity(from_spanning_set.number_of_spanning_ranges() + 1);
        for from_base in from_spanning_set.spanning_ranges_with_rest().iter() {
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
    /// Returns [`EngineError::IncompatibleSpanningSet`] if the given
    /// [`Condition`] was not built over `from_spanning_set`.
    pub fn convert(&self, condition: &Condition) -> Result<Condition, EngineError> {
        if condition.0.len() != self.from_spanning_set.spanning_ranges_with_rest_len() {
            return Err(EngineError::IncompatibleSpanningSet);
        }
        let mut new_condition = Condition::empty(self.to_spanning_set);
        for (from_index, to_indexes) in self.equivalence_map.iter().enumerate() {
            if condition.0.get(from_index) && !to_indexes.is_empty() {
                to_indexes.iter().for_each(|&to_index| {
                    new_condition.0.set(to_index, true);
                });
            }
        }

        // The one invariant every legitimate use (refining and coarsening
        // alike) must uphold: the projection denotes the same character set.
        // A violation means a condition referenced a base the target spanning
        // set cannot express, causing silent language corruption in release.
        debug_assert_eq!(
            condition
                .to_range(self.from_spanning_set)
                .expect("the length was checked above"),
            new_condition
                .to_range(self.to_spanning_set)
                .expect("the condition was built over the target spanning set"),
            "the projection changed the condition's character range"
        );

        Ok(new_condition)
    }

    /// Returns `from_spanning_set`.
    pub fn from_spanning_set(&self) -> &'a SpanningSet {
        self.from_spanning_set
    }

    /// Returns `to_spanning_set`.
    pub fn to_spanning_set(&self) -> &'b SpanningSet {
        self.to_spanning_set
    }
}

#[cfg(test)]
mod tests {
    use crate::CharRange;
    use regex_charclass::{char::Char, irange::range::AnyRange};

    use super::*;

    fn from_spanning_set() -> SpanningSet {
        let ranges = vec![
            CharRange::new_from_range(Char::new('\0')..=Char::new('\u{2}')),
            CharRange::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}')),
            CharRange::new_from_range(Char::new('\u{9}')..=Char::new('\u{9}')),
        ];

        SpanningSet::compute_spanning_set(&ranges)
    }

    fn to_spanning_set() -> SpanningSet {
        let ranges = vec![
            CharRange::new_from_range(Char::new('\0')..=Char::new('\u{1}')),
            CharRange::new_from_range(Char::new('\u{2}')..=Char::new('\u{2}')),
            CharRange::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}')),
            CharRange::new_from_range(Char::new('\u{9}')..=Char::new('\u{9}')),
            CharRange::new_from_range(Char::new('\u{20}')..=Char::new('\u{22}')),
        ];

        SpanningSet::compute_spanning_set(&ranges)
    }

    #[test]
    fn test_convert() -> Result<(), String> {
        let from_spanning_set = from_spanning_set();
        let to_spanning_set = to_spanning_set();

        let converter = ConditionConverter::new(&from_spanning_set, &to_spanning_set).unwrap();

        let empty = Condition::empty(&from_spanning_set);
        assert!(converter.convert(&empty).unwrap().is_empty());

        let total = Condition::total(&from_spanning_set);
        assert!(converter.convert(&total).unwrap().is_total());

        let range = CharRange::new_from_range(Char::new('\0')..=Char::new('\u{2}'));
        let condition = Condition::from_range(&range, &from_spanning_set).unwrap();
        assert_eq!(
            range,
            converter
                .convert(&condition)
                .unwrap()
                .to_range(&to_spanning_set)
                .unwrap()
        );

        let range = CharRange::new_from_range(Char::new('\u{4}')..=Char::new('\u{6}'));
        let condition = Condition::from_range(&range, &from_spanning_set).unwrap();
        assert_eq!(
            range,
            converter
                .convert(&condition)
                .unwrap()
                .to_range(&to_spanning_set)
                .unwrap()
        );

        let range = CharRange::new_from_ranges(&[
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
