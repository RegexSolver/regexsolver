use std::slice::Iter;

use ahash::AHashSet;
use serde::{Deserialize, Serialize};

use crate::Range;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct UsedBases(Vec<Range>, Range);

impl UsedBases {
    pub fn new_empty() -> Self {
        UsedBases(vec![], Range::total())
    }

    pub fn new_total() -> Self {
        UsedBases(vec![Range::total()], Range::empty())
    }

    pub fn elements_len(&self) -> usize {
        if self.1.is_empty() {
            self.0.len()
        } else {
            self.0.len() + 1
        }
    }

    pub fn get_elements(&self) -> Vec<Range> {
        if self.1.is_empty() {
            self.0.clone()
        } else {
            let mut elements = vec![self.1.clone()];
            elements.extend(self.0.clone());
            elements
        }
    }

    pub fn get_bases(&self) -> Iter<Range> {
        self.0.iter()
    }

    pub fn get_number_of_bases(&self) -> usize {
        self.0.len()
    }

    pub fn get_base(&self, i: usize) -> Option<&Range> {
        self.0.get(i)
    }

    pub fn get_rest(&self) -> &Range {
        &self.1
    }

    pub fn merge(&self, other: &Self) -> Self {
        let mut ranges = Vec::with_capacity(self.0.len() + other.0.len());
        ranges.extend_from_slice(&self.0);
        ranges.extend_from_slice(&other.0);

        Self::compute_used_bases(&ranges)
    }

    pub fn compute_used_bases(ranges: &[Range]) -> Self {
        let mut bases: Vec<Range> = ranges.to_vec();
        bases.sort_unstable();
        bases.dedup();

        let mut new_bases = AHashSet::with_capacity(bases.len());
        let mut changed = true;
        while changed {
            new_bases.clear();
            changed = false;
            while let Some(set) = bases.pop() {
                if let Some(index) = bases
                    .iter()
                    .position(|other_set| set != *other_set && set.has_intersection(other_set))
                {
                    let other_set = bases.swap_remove(index);
                    let intersection_set = set.intersection(&other_set);
                    new_bases.insert(intersection_set);
                    let subtraction_set = set.difference(&other_set);
                    if !subtraction_set.is_empty() {
                        new_bases.insert(subtraction_set);
                    }
                    let subtraction_set = other_set.difference(&set);
                    if !subtraction_set.is_empty() {
                        new_bases.insert(subtraction_set);
                    }
                    changed = true;
                } else if !set.is_empty() {
                    new_bases.insert(set);
                }
            }
            bases = new_bases.iter().cloned().collect();
        }

        bases.sort_unstable();

        let mut total = Range::empty();
        for base in &bases {
            total = total.union(base);
        }

        UsedBases(bases, total.complement())
    }
}
