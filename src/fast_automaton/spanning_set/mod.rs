use std::slice::Iter;

use ahash::AHashMap;
use regex_charclass::irange::RangeSet;

use super::{from_scalar, scalar};
use crate::CharRange;

/// Converts merged, ascending scalar segments (see [`scalar`](super::scalar))
/// back into a [`CharRange`]. The segments are disjoint and non-adjacent, so
/// the flat bound list is already the canonical representation the set
/// operations produce.
fn segments_to_range(segments: &[(u32, u32)]) -> CharRange {
    let mut bounds = Vec::with_capacity(segments.len() * 2);
    for &(start, end) in segments {
        bounds.push(from_scalar(start).expect("segment bounds are valid scalars"));
        bounds.push(from_scalar(end).expect("segment bounds are valid scalars"));
    }
    RangeSet(bounds)
}

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

    pub(crate) fn spanning_ranges_with_rest(&self) -> impl Iterator<Item = &CharRange> {
        std::iter::once(&self.1)
            .filter(|rest| !rest.is_empty())
            .chain(self.0.iter())
    }

    /// Returns an iterator over the explicit (non-rest) ranges in the spanning set.
    pub fn spanning_ranges(&self) -> Iter<'_, CharRange> {
        self.0.iter()
    }

    /// Returns the number of explicit (non-rest) ranges in the spanning set.
    pub fn number_of_spanning_ranges(&self) -> usize {
        self.0.len()
    }

    /// Returns the explicit range at index `i`, or `None` if out of bounds.
    pub fn spanning_range(&self, i: usize) -> Option<&CharRange> {
        self.0.get(i)
    }

    /// Returns the "rest" range covering all characters not in any explicit range.
    pub fn rest(&self) -> &CharRange {
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
    ///
    /// The bases are the *atoms* of the inputs: the classes of characters
    /// covered by exactly the same subset of input ranges. They are computed
    /// with one endpoint sweep over the surrogate-free scalar space,
    /// O(E log E) in the total number of input intervals.
    pub fn compute_spanning_set(ranges: &[CharRange]) -> Self {
        let mut inputs: Vec<&CharRange> = ranges.iter().filter(|r| !r.is_empty()).collect();
        inputs.sort_unstable();
        inputs.dedup();

        // The two dominant real-world calls: an empty automaton, and
        // `add_transition_from_range` extending the set by a single range.
        match inputs.as_slice() {
            [] => return SpanningSet(vec![], CharRange::total()),
            [input] => return SpanningSet(vec![(*input).clone()], input.complement()),
            _ => {}
        }

        // Interval [lo, hi] becomes the half-open [scalar(lo), scalar(hi)+1).
        let interval_count: usize = inputs.iter().map(|input| input.0.len() / 2).sum();
        let mut events: Vec<Event> = Vec::with_capacity(interval_count * 2);
        for (input_index, input) in inputs.iter().enumerate() {
            for pair in input.0.chunks_exact(2) {
                events.push(Event {
                    position: scalar(pair[0]),
                    input_index: input_index as u32,
                    is_start: true,
                });
                events.push(Event {
                    position: scalar(pair[1]) + 1,
                    input_index: input_index as u32,
                    is_start: false,
                });
            }
        }
        events.sort_unstable_by_key(|event| event.position);

        let (mut spanning_ranges, covered) = if inputs.len() <= 64 {
            Self::sweep_small(&events)
        } else {
            Self::sweep_wide(inputs.len(), &events)
        };
        spanning_ranges.sort_unstable();

        SpanningSet(spanning_ranges, segments_to_range(&covered).complement())
    }

    /// The sweep for at most 64 inputs (virtually every call): the signature
    /// fits one `u64`, so tagged segments are grouped into atoms with a plain
    /// sort instead of a hash map.
    fn sweep_small(events: &[Event]) -> (Vec<CharRange>, Vec<(u32, u32)>) {
        let mut active = 0u64;
        let mut tagged: Vec<(u64, u32, u32)> = Vec::with_capacity(events.len());
        let mut covered: Vec<(u32, u32)> = Vec::new();

        let mut i = 0;
        let mut previous_position = 0u32;
        while i < events.len() {
            let position = events[i].position;
            if active != 0 && previous_position < position {
                let segment = (previous_position, position - 1);
                push_merging_adjacent(&mut covered, segment);
                tagged.push((active, segment.0, segment.1));
            }
            while i < events.len() && events[i].position == position {
                active ^= 1 << events[i].input_index;
                i += 1;
            }
            previous_position = position;
        }

        // Group by signature; the sweep emitted segments in ascending
        // position order, which the (signature, position) sort preserves
        // within each group, so adjacent segments merge as in `sweep_wide`.
        tagged.sort_unstable();
        let mut atoms: Vec<CharRange> = Vec::new();
        let mut i = 0;
        while i < tagged.len() {
            let signature = tagged[i].0;
            let mut segments: Vec<(u32, u32)> = Vec::new();
            while i < tagged.len() && tagged[i].0 == signature {
                push_merging_adjacent(&mut segments, (tagged[i].1, tagged[i].2));
                i += 1;
            }
            atoms.push(segments_to_range(&segments));
        }
        (atoms, covered)
    }

    /// The sweep for more than 64 inputs: the signature is a bit vector and
    /// atoms are grouped through a hash map.
    fn sweep_wide(input_count: usize, events: &[Event]) -> (Vec<CharRange>, Vec<(u32, u32)>) {
        let mut active = vec![0u64; input_count.div_ceil(64)];
        let mut active_count = 0usize;
        let mut atoms: AHashMap<Vec<u64>, Vec<(u32, u32)>> = AHashMap::new();
        let mut covered: Vec<(u32, u32)> = Vec::new();

        let mut i = 0;
        let mut previous_position = 0u32;
        while i < events.len() {
            let position = events[i].position;
            if active_count > 0 && previous_position < position {
                let segment = (previous_position, position - 1);
                push_merging_adjacent(&mut covered, segment);
                if let Some(segments) = atoms.get_mut(&active) {
                    push_merging_adjacent(segments, segment);
                } else {
                    atoms.insert(active.clone(), vec![segment]);
                }
            }
            while i < events.len() && events[i].position == position {
                let event = &events[i];
                let (word, bit) = ((event.input_index / 64) as usize, event.input_index % 64);
                if event.is_start {
                    active[word] |= 1 << bit;
                    active_count += 1;
                } else {
                    active[word] &= !(1 << bit);
                    active_count -= 1;
                }
                i += 1;
            }
            previous_position = position;
        }

        (
            atoms
                .values()
                .map(|segments| segments_to_range(segments))
                .collect(),
            covered,
        )
    }
}

/// A boundary of one input interval during the sweep: the covering-input
/// signature is constant between consecutive event positions.
struct Event {
    position: u32,
    input_index: u32,
    is_start: bool,
}

/// Appends `segment` to an ascending segment list, extending the last entry
/// instead when they touch.
#[inline]
fn push_merging_adjacent(segments: &mut Vec<(u32, u32)>, segment: (u32, u32)) {
    match segments.last_mut() {
        Some(last) if last.1 + 1 == segment.0 => last.1 = segment.1,
        _ => segments.push(segment),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashSet;
    use proptest::prelude::*;

    /// The pairwise intersection-splitting fixpoint `compute_spanning_set`
    /// replaced, kept as the oracle the sweep is fuzzed against.
    fn compute_spanning_set_reference(ranges: &[CharRange]) -> SpanningSet {
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
            spanning_ranges = new_spanning_ranges.drain().collect();
        }

        spanning_ranges.sort_unstable();

        let mut total = CharRange::empty();
        for base in &spanning_ranges {
            total = total.union(base);
        }

        SpanningSet(spanning_ranges, total.complement())
    }

    /// A random `CharRange` in canonical representation (built by unioning
    /// single intervals, which merges overlaps and adjacency) — the form
    /// every real caller passes in.
    fn arb_char_range() -> impl Strategy<Value = CharRange> {
        proptest::collection::vec((0u32..=0x10F7FF, 0u32..=0x10F7FF), 1..4).prop_map(|pairs| {
            let mut range = CharRange::empty();
            for (a, b) in pairs {
                let (low, high) = if a <= b { (a, b) } else { (b, a) };
                let interval = CharRange::new_from_range(
                    from_scalar(low).unwrap()..=from_scalar(high).unwrap(),
                );
                range = range.union(&interval);
            }
            range
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn sweep_matches_the_fixpoint_reference(
            inputs in proptest::collection::vec(arb_char_range(), 0..6)
        ) {
            let expected = compute_spanning_set_reference(&inputs);
            let actual = SpanningSet::compute_spanning_set(&inputs);
            prop_assert_eq!(expected, actual);
        }
    }

    // More than 64 inputs exercises the wide (bit-vector signature) sweep;
    // the wide ranges force overlaps across many inputs at once.
    #[test]
    fn sweep_wide_matches_the_reference() {
        let single = |low: u32, high: u32| {
            CharRange::new_from_range(from_scalar(low).unwrap()..=from_scalar(high).unwrap())
        };

        let mut inputs = Vec::new();
        for i in 0..70u32 {
            inputs.push(single(i * 10, i * 10 + 5));
        }
        inputs.push(single(3, 400));
        inputs.push(single(250, 699));

        assert_eq!(
            compute_spanning_set_reference(&inputs),
            SpanningSet::compute_spanning_set(&inputs)
        );
    }

    // Inputs whose intervals touch the extremes and the surrogate hole.
    #[test]
    fn sweep_handles_boundary_ranges() {
        use regex_charclass::char::Char;

        let cases: Vec<Vec<CharRange>> = vec![
            vec![],
            vec![CharRange::empty()],
            vec![CharRange::total()],
            vec![CharRange::total(), CharRange::total()],
            vec![
                CharRange::new_from_range(Char::new('\u{0}')..=Char::new('\u{D7FF}')),
                CharRange::new_from_range(Char::new('\u{E000}')..=Char::new('\u{10FFFF}')),
            ],
            vec![
                CharRange::new_from_range(Char::new('\u{D000}')..=Char::new('\u{F000}')),
                CharRange::new_from_range(Char::new('\u{E000}')..=Char::new('\u{E000}')),
            ],
            vec![
                CharRange::new_from_range(Char::new('\u{10FFFF}')..=Char::new('\u{10FFFF}')),
                CharRange::new_from_range(Char::new('\u{0}')..=Char::new('\u{0}')),
            ],
        ];
        for inputs in cases {
            assert_eq!(
                compute_spanning_set_reference(&inputs),
                SpanningSet::compute_spanning_set(&inputs),
                "inputs: {inputs:?}"
            );
        }
    }
}
