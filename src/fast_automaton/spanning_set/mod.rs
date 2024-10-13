use std::slice::Iter;

use ahash::AHashSet;
use serde::{Deserialize, Serialize};
use regex_charclass::{char::Char, irange::RangeSet};

/// Contains a set of [`RangeSet<Char>`] that span all the transition of a [`crate::FastAutomaton`]. 
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct SpanningSet(Vec<RangeSet<Char>>, RangeSet<Char>);

impl SpanningSet {
    pub fn new_empty() -> Self {
        SpanningSet(vec![], RangeSet::total())
    }

    pub fn new_total() -> Self {
        SpanningSet(vec![RangeSet::total()], RangeSet::empty())
    }

    pub(crate) fn spanning_ranges_with_rest_len(&self) -> usize {
        if self.1.is_empty() {
            self.0.len()
        } else {
            self.0.len() + 1
        }
    }

    pub(crate) fn get_spanning_ranges_with_rest(&self) -> Vec<RangeSet<Char>> {
        if self.1.is_empty() {
            self.0.clone()
        } else {
            let mut elements = vec![self.1.clone()];
            elements.extend(self.0.clone());
            elements
        }
    }

    pub fn get_spanning_ranges(&self) -> Iter<RangeSet<Char>> {
        self.0.iter()
    }

    pub fn get_number_of_spanning_ranges(&self) -> usize {
        self.0.len()
    }

    pub fn get_spanning_range(&self, i: usize) -> Option<&RangeSet<Char>> {
        self.0.get(i)
    }

    pub fn get_rest(&self) -> &RangeSet<Char> {
        &self.1
    }

    pub fn merge(&self, other: &Self) -> Self {
        let mut ranges = Vec::with_capacity(self.0.len() + other.0.len());
        ranges.extend_from_slice(&self.0);
        ranges.extend_from_slice(&other.0);

        Self::compute_spanning_set(&ranges)
    }

    pub fn compute_spanning_set(ranges: &[RangeSet<Char>]) -> Self {
        let mut spanning_ranges: Vec<RangeSet<Char>> = ranges.to_vec();
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
                    let subtraction_set = set.difference(&other_set);
                    if !subtraction_set.is_empty() {
                        new_spanning_ranges.insert(subtraction_set);
                    }
                    let subtraction_set = other_set.difference(&set);
                    if !subtraction_set.is_empty() {
                        new_spanning_ranges.insert(subtraction_set);
                    }
                    changed = true;
                } else if !set.is_empty() {
                    new_spanning_ranges.insert(set);
                }
            }
            spanning_ranges = new_spanning_ranges.iter().cloned().collect();
        }

        spanning_ranges.sort_unstable();

        let mut total = RangeSet::empty();
        for base in &spanning_ranges {
            total = total.union(base);
        }

        SpanningSet(spanning_ranges, total.complement())
    }
}
