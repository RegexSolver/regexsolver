use std::slice::Iter;

use ahash::AHashSet;

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

use crate::CharRange;

/// Contains a set of [`CharRange`] that span all the transition of a [`crate::FastAutomaton`].
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanningSet(Vec<CharRange>, CharRange);

impl SpanningSet {
    pub fn new_empty() -> Self {
        SpanningSet(vec![], CharRange::total())
    }

    pub fn new_total() -> Self {
        SpanningSet(vec![CharRange::total()], CharRange::empty())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty() && self.1.is_total()
    }

    pub fn is_total(&self) -> bool {
        self.0.len() == 1 && self.0[0].is_total() && self.1.is_empty()
    }

    pub(crate) fn spanning_ranges_with_rest_len(&self) -> usize {
        if self.1.is_empty() {
            self.0.len()
        } else {
            self.0.len() + 1
        }
    }

    pub(crate) fn get_spanning_ranges_with_rest(&self) -> Vec<CharRange> {
        if self.1.is_empty() {
            self.0.clone()
        } else {
            let mut elements = vec![self.1.clone()];
            elements.extend(self.0.clone());
            elements
        }
    }

    pub fn get_spanning_ranges(&self) -> Iter<CharRange> {
        self.0.iter()
    }

    pub fn get_number_of_spanning_ranges(&self) -> usize {
        self.0.len()
    }

    pub fn get_spanning_range(&self, i: usize) -> Option<&CharRange> {
        self.0.get(i)
    }

    pub fn get_rest(&self) -> &CharRange {
        &self.1
    }

    /// Compute a new minimal spanning set by merging the provided spanning set.
    pub fn merge(&self, other: &Self) -> Self {
        let mut ranges = Vec::with_capacity(self.0.len() + other.0.len());
        ranges.extend_from_slice(&self.0);
        ranges.extend_from_slice(&other.0);

        Self::compute_spanning_set(&ranges)
    }

    /// Compute a new minimal spanning set for the provided ranges.
    pub fn compute_spanning_set(ranges: &[CharRange]) -> Self {
        let mut spanning_ranges: Vec<CharRange> = ranges.to_vec();
        spanning_ranges.sort_unstable();
        spanning_ranges.dedup();

        let mut new_spanning_ranges = AHashSet::with_capacity(spanning_ranges.len());
        let mut changed = true;
        while changed {
            new_spanning_ranges.clear();
            changed = false;
            while let Some(set) = spanning_ranges.pop() {
                if let Some(index) = spanning_ranges
                    .iter()
                    .position(|other_set| set != *other_set && set.has_intersection(other_set))
                {
                    let other_set = spanning_ranges.swap_remove(index);
                    let intersection_set = set.intersection(&other_set);
                    new_spanning_ranges.insert(intersection_set);
                    let difference_set = set.difference(&other_set);
                    if !difference_set.is_empty() {
                        new_spanning_ranges.insert(difference_set);
                    }
                    let difference_set = other_set.difference(&set);
                    if !difference_set.is_empty() {
                        new_spanning_ranges.insert(difference_set);
                    }
                    changed = true;
                } else if !set.is_empty() {
                    new_spanning_ranges.insert(set);
                }
            }
            spanning_ranges = new_spanning_ranges.iter().cloned().collect();
        }

        spanning_ranges.sort_unstable();

        let mut total = CharRange::empty();
        for base in &spanning_ranges {
            total = total.union(base);
        }

        SpanningSet(spanning_ranges, total.complement())
    }
}
