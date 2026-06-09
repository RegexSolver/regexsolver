use std::slice::Iter;

use ahash::AHashSet;

use crate::CharRange;

/// A set of [`CharRange`] that spans all transitions of a [`crate::FastAutomaton`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanningSet(Vec<CharRange>, CharRange);

impl SpanningSet {
    /// Creates a spanning set from explicit disjoint `ranges` plus the `rest`
    /// range covering every character they don't. The caller is responsible for
    /// these invariants; prefer [`compute_spanning_set`](Self::compute_spanning_set),
    /// which derives a minimal, well-formed set from arbitrary ranges.
    pub fn new(ranges: Vec<CharRange>, rest: CharRange) -> Self {
        SpanningSet(ranges, rest)
    }

    /// Creates the spanning set of an automaton with no transitions: no
    /// explicit ranges, with the rest covering all characters.
    pub fn new_empty() -> Self {
        SpanningSet(vec![], CharRange::total())
    }

    /// Creates the spanning set with a single range covering all characters and
    /// an empty rest.
    pub fn new_total() -> Self {
        SpanningSet(vec![CharRange::total()], CharRange::empty())
    }

    /// Returns `true` if this is the empty spanning set (no explicit ranges;
    /// see [`new_empty`](Self::new_empty)).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty() && self.1.is_total()
    }

    /// Returns `true` if this is the total spanning set (one all-covering
    /// range; see [`new_total`](Self::new_total)).
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

    /// Returns an iterator over the explicit (non-rest) ranges in the spanning set.
    pub fn get_spanning_ranges(&self) -> Iter<'_, CharRange> {
        self.0.iter()
    }

    /// Returns the number of explicit (non-rest) ranges in the spanning set.
    pub fn get_number_of_spanning_ranges(&self) -> usize {
        self.0.len()
    }

    /// Returns the explicit range at index `i`, or `None` if out of bounds.
    pub fn get_spanning_range(&self, i: usize) -> Option<&CharRange> {
        self.0.get(i)
    }

    /// Returns the "rest" range covering all characters not in any explicit range.
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
