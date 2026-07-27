use crate::{EngineError, execution_profile::ExecutionProfile};
use ahash::{AHashSet, RandomState};
use indexmap::IndexSet;

use super::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::Range;

/// The order in which [`FastAutomaton::generate_strings`] walks a language.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GenerationOrder {
    /// Shortest strings first, each path expanded from the low end of its
    /// character ranges before the next path is visited: `.*abc.*` yields
    /// `abc`, `abc\u{0}`, `abc\u{1}`, ... This sweeps the language in a stable
    /// order, and is the cheapest way to page through it with `offset`.
    #[default]
    Exhaustive,
    /// Samples the language instead of sweeping it: a few strings per path
    /// before moving to the next one, with representative characters (`a`,
    /// `0`, `A`, ` `, ...) spread over each range rather than always its first
    /// one. `.*abc.*` yields `abc`, `0abc`, `aabc`, `abc0`, ... — the shapes
    /// the pattern allows, instead of a million variations of one of them,
    /// which is what makes it usable to derive test cases.
    ///
    /// Shape comes first: the strings cover every path the automaton holds
    /// before any path is asked for a second one, so a `limit` smaller than
    /// the number of shapes is spent entirely on distinct shapes, and only a
    /// larger one starts varying the characters within them.
    ///
    /// Deterministic, and pages with `offset` like [`Exhaustive`](Self::Exhaustive).
    /// Each pass takes twice as many strings per path as the previous one, so
    /// a finite language is still enumerated in full given a large enough
    /// `limit`; those repeated passes make it slower than `Exhaustive`.
    Sampled,
}

/// The characters a sampled string reaches for first, in order of preference:
/// one per kind of character a range is usually built from, so that an early
/// sample lands on a letter, a digit or a space rather than on `\u{0}`. The
/// list is rotated by position, so neighbouring characters of a sample differ.
const SAMPLE_CHARS: [char; 10] = [
    'a',
    '0',
    'A',
    ' ',
    '_',
    '~',
    '\n',
    '\u{e9}',
    '\u{4e2d}',
    '\u{1f600}',
];

/// The block of code points `char` cannot hold: [`Char`] values skip it, so
/// scalar values have to be shifted down past it to be counted.
const SURROGATES: Range<u32> = 0xD800..0xE000;

#[derive(Clone, Eq, PartialEq)]
struct QueueItem {
    score: usize,
    depth: usize,
    state: usize,
    ranges: Vec<CharRange>,
    hash: u64,
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .cmp(&self.score)
            .then_with(|| self.depth.cmp(&other.depth))
            .then_with(|| self.state.cmp(&other.state))
            .then_with(|| self.hash.cmp(&other.hash))
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// What [`FastAutomaton::generate_strings`] is allowed to generate: the order
/// to walk the language in, and the characters it may use.
///
/// [`GenerationOrder`] converts into it, so an order can be passed on its own
/// wherever options are expected.
///
/// # Examples
///
/// ```
/// use regexsolver::{CharRange, Term, fast_automaton::{GenerationOptions, GenerationOrder}};
/// use regexsolver::regex_charclass::char::Char;
///
/// let term = Term::from_pattern(".{2}").unwrap();
///
/// // An order on its own.
/// let strings = term.generate_strings(3, 0, GenerationOrder::Sampled).unwrap();
///
/// // Sampled, and restricted to lowercase letters.
/// let lowercase = CharRange::new_from_range(Char::new('a')..=Char::new('z'));
/// let options = GenerationOptions::from(GenerationOrder::Sampled).with_charset(lowercase);
///
/// let strings = term.generate_strings(3, 0, options).unwrap();
/// assert!(strings.iter().all(|s| s.chars().all(|c| c.is_ascii_lowercase())));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GenerationOptions {
    order: GenerationOrder,
    charset: Option<CharRange>,
}

impl GenerationOptions {
    /// Default options: [`GenerationOrder::Exhaustive`], over every character
    /// the automaton allows.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy of these options walking the language in `order`.
    pub fn with_order(mut self, order: GenerationOrder) -> Self {
        self.order = order;
        self
    }

    /// Returns a copy of these options restricted to `charset`.
    ///
    /// Only strings made entirely of those characters are generated: a path
    /// through a transition that `charset` rules out is dropped whole, never
    /// shortened. Restricting to characters the automaton never matches
    /// generates nothing.
    ///
    /// A [`CharRange`] is built from bounds, or out of a character class
    /// pattern through [`RegularExpression`](crate::regex::RegularExpression):
    ///
    /// ```
    /// use regexsolver::{CharRange, regex::RegularExpression};
    /// use regexsolver::regex_charclass::char::Char;
    ///
    /// let printable = CharRange::new_from_range(Char::new(' ')..=Char::new('~'));
    ///
    /// let no_controls = match RegularExpression::new("\\P{C}").unwrap() {
    ///     RegularExpression::Character(charset) => charset,
    ///     other => panic!("not a character class: {other}"),
    /// };
    /// ```
    pub fn with_charset(mut self, charset: CharRange) -> Self {
        self.charset = Some(charset);
        self
    }

    /// The order the language is walked in.
    pub fn order(&self) -> GenerationOrder {
        self.order
    }

    /// The characters generation is restricted to, `None` when it is not.
    pub fn charset(&self) -> Option<&CharRange> {
        self.charset.as_ref()
    }
}

impl From<GenerationOrder> for GenerationOptions {
    fn from(order: GenerationOrder) -> Self {
        GenerationOptions {
            order,
            charset: None,
        }
    }
}

impl FastAutomaton {
    /// Generates up to `limit` distinct strings matched by the automaton under
    /// the given [`GenerationOptions`], skipping the first `offset` strings.
    ///
    /// `options` is a [`GenerationOrder`] on its own, or a full
    /// [`GenerationOptions`] to also restrict the characters used.
    ///
    /// Strings are only guaranteed to be distinct **within a single call**:
    /// the offset fast-skips by counting paths, and in a non-deterministic
    /// automaton the same string can be reached through several paths, so
    /// calls with different offsets may repeat strings (or skip some).
    /// [`determinize`](Self::determinize) (and ideally
    /// [`minimize`](Self::minimize)) first to make pages disjoint. Offsets are
    /// also only consistent between calls made with the same options.
    #[tracing::instrument(level = "debug", skip(self, options), fields(states = self.number_of_states(), deterministic=self.is_deterministic(), limit=limit, offset=offset, order=tracing::field::Empty, charset=tracing::field::Empty))]
    pub fn generate_strings(
        &self,
        limit: usize,
        offset: usize,
        options: impl Into<GenerationOptions>,
    ) -> Result<Vec<String>, EngineError> {
        let options = options.into();

        let span = tracing::Span::current();
        span.record("order", tracing::field::debug(options.order));
        span.record(
            "charset",
            tracing::field::debug(options.charset.as_ref().map(|charset| charset.to_regex())),
        );

        self.generate(limit, offset, &options)
    }

    /// [`generate_strings`](Self::generate_strings) over borrowed options, for
    /// the callers holding them across several batches.
    pub(crate) fn generate(
        &self,
        limit: usize,
        offset: usize,
        options: &GenerationOptions,
    ) -> Result<Vec<String>, EngineError> {
        if self.is_empty() || limit == 0 {
            return Ok(vec![]);
        }

        let mut generation = Generation::new(self, limit, offset, options.charset())?;

        match options.order {
            GenerationOrder::Exhaustive => {
                generation.walk(self, None)?;
            }
            GenerationOrder::Sampled => {
                // A pass takes at most `window` combinations per path, so no
                // single path can spend the whole `limit` on itself. The
                // windows double and pick up where the previous one stopped:
                // a pass that exhausts the automaton without filling `limit`
                // is followed by one digging deeper into the same paths, until
                // a pass finds nothing left to cover.
                let mut window = 0..1;
                loop {
                    let covered = generation.walk(self, Some(&window))?;
                    if covered == 0 || generation.is_full() {
                        break;
                    }
                    window = window.end..window.end.saturating_mul(2).saturating_add(1);
                }
            }
        }

        Ok(generation.strings.into_iter().collect())
    }
}

/// The state of a single [`FastAutomaton::generate_strings`] call, shared by
/// every pass a [`GenerationOrder::Sampled`] generation makes over the
/// automaton.
struct Generation<'a> {
    /// Number of transitions from each state to the nearest accept state;
    /// `usize::MAX` for the states that cannot reach one.
    distances: Vec<usize>,
    /// Length of the longest string the automaton matches.
    max_len: usize,
    limit: usize,
    offset: usize,
    strings: IndexSet<String, RandomState>,
    visited: AHashSet<(State, usize, u64)>,
    /// The characters each transition condition stands for, the charset
    /// already taken out. A condition the charset leaves nothing of is absent,
    /// which is what makes its transition impassable.
    ranges: AHashMap<&'a Condition, CharRange>,
    execution_profile: ExecutionProfile,
}

impl<'a> Generation<'a> {
    fn new(
        automaton: &'a FastAutomaton,
        limit: usize,
        offset: usize,
        charset: Option<&CharRange>,
    ) -> Result<Self, EngineError> {
        let num_states = automaton.transitions.len();

        // What every transition condition leaves once the charset is taken
        // out. Resolved up front rather than as the walk reaches them, so that
        // the search below already knows which transitions are impassable.
        let mut ranges: AHashMap<&Condition, CharRange> = AHashMap::with_capacity(num_states);
        for state in automaton.states() {
            for (cond, _) in automaton.transitions_from(state) {
                if ranges.contains_key(cond) {
                    continue;
                }
                let range = cond.to_range(&automaton.spanning_set)?;
                ranges.insert(
                    cond,
                    match charset {
                        Some(charset) => range.intersection(charset),
                        None => range,
                    },
                );
            }
        }

        // REVERSE BFS: the exact distance from every state to an accept state,
        // which drives the A* search and prunes the states that never accept.
        // A state the charset leaves no way out of is one of those dead ends.
        let mut incoming = vec![vec![]; num_states];
        let mut dist_q = VecDeque::new();
        let mut distances = vec![usize::MAX; num_states];

        for state in automaton.states() {
            if automaton.is_accepted(state) {
                distances[state] = 0;
                dist_q.push_back(state);
            }
            for (cond, &to_state) in automaton.transitions_from(state) {
                if !ranges[cond].is_empty() {
                    incoming[to_state].push(state);
                }
            }
        }

        while let Some(state) = dist_q.pop_front() {
            let d = distances[state];
            for &prev in &incoming[state] {
                if distances[prev] == usize::MAX {
                    distances[prev] = d + 1;
                    dist_q.push_back(prev);
                }
            }
        }

        let (_, max) = automaton.length();

        Ok(Generation {
            distances,
            max_len: max.unwrap_or(u32::MAX) as usize,
            limit,
            offset,
            strings: IndexSet::with_capacity_and_hasher(limit, RandomState::default()),
            visited: AHashSet::with_capacity(num_states),
            ranges,
            execution_profile: ExecutionProfile::get(),
        })
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.strings.len() >= self.limit
    }

    /// A* SEARCH: walks the automaton once, shortest path first, emitting the
    /// strings of every accepting path it pops until `limit` strings are
    /// collected or the automaton runs out of paths.
    ///
    /// `window` restricts each path to the combinations whose index falls
    /// inside it; `None` takes them all. Returns how many combinations the
    /// pass covered, the ones `offset` skipped included.
    fn walk(
        &mut self,
        automaton: &'a FastAutomaton,
        window: Option<&Range<usize>>,
    ) -> Result<usize, EngineError> {
        let start_state = automaton.start_state();

        // If the start state can't reach an accept state, exit immediately
        if self.distances[start_state] == usize::MAX {
            return Ok(0);
        }

        self.visited.clear();
        ();
        let mut covered = 0usize;

        let mut q = BinaryHeap::new();
        q.push(QueueItem {
            score: self.distances[start_state],
            depth: 0,
            state: start_state,
            ranges: vec![],
            hash: 0u64,
        });

        while let Some(QueueItem {
            score: _,
            depth: current_depth,
            state,
            mut ranges,
            hash: h,
        }) = q.pop()
        {
            self.execution_profile.assert_not_timed_out()?;

            if automaton.is_accepted(state) {
                covered = covered.saturating_add(match window {
                    Some(window) => self.emit_sampled(&ranges, window)?,
                    None => self.emit_all(&ranges)?,
                });

                if self.is_full() {
                    break;
                }
            }

            if current_depth >= self.max_len {
                continue;
            }

            let next_depth = current_depth + 1;
            let mut valid_transitions = Vec::new();

            for (cond, &to_state) in automaton.transitions_from(state) {
                // DEAD-END PRUNING: Instantly kill paths that cannot accept
                if self.distances[to_state] == usize::MAX {
                    continue;
                }

                // ...and the transitions the charset closed off.
                let range = &self.ranges[cond];
                if range.is_empty() {
                    continue;
                }

                let hash = path_mix(h, mix64(state as u64 ^ mix64(to_state as u64)));

                if self.visited.insert((to_state, next_depth, hash)) {
                    valid_transitions.push((to_state, range.clone(), hash));
                }
            }

            // Vector Reuse Optimization
            if let Some((last_state, last_range, last_hash)) = valid_transitions.pop() {
                for (to_state, range, hash) in valid_transitions {
                    let mut new_ranges = ranges.clone();
                    new_ranges.push(range);
                    q.push(QueueItem {
                        score: next_depth + self.distances[to_state], // A* Score Formula
                        depth: next_depth,
                        state: to_state,
                        ranges: new_ranges,
                        hash,
                    });
                }

                ranges.push(last_range);
                q.push(QueueItem {
                    score: next_depth + self.distances[last_state], // A* Score Formula
                    depth: next_depth,
                    state: last_state,
                    ranges,
                    hash: last_hash,
                });
            }
        }

        Ok(covered)
    }

    /// Emits every combination of `ranges` that `offset` does not skip, in
    /// ascending character order. Returns the number of combinations the path
    /// holds.
    fn emit_all(&mut self, ranges: &[CharRange]) -> Result<usize, EngineError> {
        let range_lengths: Vec<usize> = ranges
            .iter()
            .map(|r| r.get_cardinality() as usize)
            .collect();

        let mut total_combinations = 1usize;
        for &len in &range_lengths {
            total_combinations = total_combinations.saturating_mul(len);
        }

        if self.offset >= total_combinations {
            self.offset -= total_combinations;
            return Ok(total_combinations);
        }

        let mut current_str = String::with_capacity(ranges.len());
        self.emit_combinations(ranges, &range_lengths, 0, &mut current_str)?;
        Ok(total_combinations)
    }

    fn emit_combinations(
        &mut self,
        ranges: &[CharRange],
        range_lengths: &[usize],
        depth: usize,
        current_str: &mut String,
    ) -> Result<(), EngineError> {
        if depth == ranges.len() {
            if self.offset > 0 {
                self.offset -= 1;
            } else {
                self.strings.insert(current_str.clone());
            }
            return Ok(());
        }

        // Calculate combinations for the remaining suffix of ranges
        let mut sub_combinations = 1usize;
        for &len in &range_lengths[depth + 1..] {
            sub_combinations = sub_combinations.saturating_mul(len);
        }

        for ch in ranges[depth].iter() {
            self.execution_profile.assert_not_timed_out()?;

            // If skipping this character's subtree fits within the remaining offset
            if self.offset >= sub_combinations {
                self.offset -= sub_combinations;
                continue;
            }

            current_str.push(ch.to_char());
            self.emit_combinations(ranges, range_lengths, depth + 1, current_str)?;
            current_str.pop();

            if self.is_full() {
                break;
            }
        }

        Ok(())
    }

    /// Emits the combinations of `ranges` whose index falls inside `window`,
    /// spread over the ranges by the sampling permutation. Returns how many of
    /// them the window covered, the ones `offset` skipped included.
    fn emit_sampled(
        &mut self,
        ranges: &[CharRange],
        window: &Range<usize>,
    ) -> Result<usize, EngineError> {
        let range_lengths: Vec<u128> = ranges.iter().map(|r| r.get_cardinality() as u128).collect();

        // `None` once the product stops fitting: such a path holds more
        // combinations than a window will ever reach into.
        let total_combinations = range_lengths
            .iter()
            .try_fold(1u128, |total, &len| total.checked_mul(len));
        let bound =
            total_combinations.map_or(usize::MAX, |total| total.min(usize::MAX as u128) as usize);

        let covered = window.end.min(bound) - window.start.min(bound);
        if self.offset >= covered {
            self.offset -= covered;
            return Ok(covered);
        }

        // The permutation needs `index * multiplier` to stay within `u128`;
        // beyond that the mixed-radix digits alone give enough variety, since
        // consecutive indices already differ from their first character on.
        let scramble = total_combinations
            .filter(|&total| total <= u64::MAX as u128)
            .map(|total| (scramble_multiplier(total), total));

        let first = window.start.min(bound) + self.offset;
        self.offset = 0;

        for index in first..window.end.min(bound) {
            self.execution_profile.assert_not_timed_out()?;

            let combination = match scramble {
                Some((multiplier, total)) => (index as u128 * multiplier) % total,
                None => index as u128,
            };

            let string = sample_string(ranges, &range_lengths, combination)?;
            self.strings.insert(string);

            if self.is_full() {
                break;
            }
        }

        Ok(covered)
    }
}

/// Builds the combination of `ranges` at index `combination`, read as a
/// mixed-radix number whose least significant digit is the first character:
/// consecutive combinations then differ from their first character on, instead
/// of only in their last one.
fn sample_string(
    ranges: &[CharRange],
    range_lengths: &[u128],
    mut combination: u128,
) -> Result<String, EngineError> {
    let mut string = String::with_capacity(ranges.len());

    for (position, (range, &length)) in ranges.iter().zip(range_lengths).enumerate() {
        let index = (combination % length) as u32;
        combination /= length;
        string.push(sample_char(range, position, index)?.to_char());
    }

    Ok(string)
}

/// Returns the character `range` holds at `index` in sampling order: the
/// [`SAMPLE_CHARS`] it contains first, rotated by `position` so that adjacent
/// characters of a sample differ, then the rest of the range in order.
///
/// The mapping is a permutation of the range, which is what keeps the sampled
/// strings distinct and `offset` exact.
fn sample_char(range: &CharRange, position: usize, index: u32) -> Result<Char, EngineError> {
    let mut sampled = [0u32; SAMPLE_CHARS.len()];
    let mut count = 0;

    for rotation in 0..SAMPLE_CHARS.len() {
        let ch = Char::new(SAMPLE_CHARS[(position + rotation) % SAMPLE_CHARS.len()]);
        if !range.contains(ch) {
            continue;
        }
        if count as u32 == index {
            return Ok(ch);
        }
        sampled[count] = ordinal_of(range, ch);
        count += 1;
    }

    // Past the sample characters: take the `index - count`-th character of the
    // range that is not one of them, so none is handed out twice.
    let sampled = &mut sampled[..count];
    sampled.sort_unstable();
    let mut ordinal = index - count as u32;
    for &taken in sampled.iter() {
        if taken > ordinal {
            break;
        }
        ordinal += 1;
    }

    char_at(range, ordinal).ok_or(EngineError::InvalidCharacterInRegex)
}

/// The `(low, high)` scalar bounds of the intervals `range` is made of.
fn intervals(range: &CharRange) -> impl Iterator<Item = (u32, u32)> + '_ {
    range
        .0
        .chunks_exact(2)
        .map(|bounds| (scalar(bounds[0]), scalar(bounds[1])))
}

/// The number of characters `range` holds before `target`, which it contains.
fn ordinal_of(range: &CharRange, target: Char) -> u32 {
    let target = scalar(target);
    let mut ordinal = 0;

    for (low, high) in intervals(range) {
        if target <= high {
            return ordinal + (target - low);
        }
        ordinal += high - low + 1;
    }

    ordinal
}

/// The character `range` holds at `ordinal`, `None` past its cardinality.
fn char_at(range: &CharRange, mut ordinal: u32) -> Option<Char> {
    for (low, high) in intervals(range) {
        let length = high - low + 1;
        if ordinal < length {
            return from_scalar(low + ordinal);
        }
        ordinal -= length;
    }

    None
}

/// The index of `ch` among all the characters, the surrogate block excluded.
#[inline]
fn scalar(ch: Char) -> u32 {
    let code = ch.to_u32();
    if code >= SURROGATES.end {
        code - (SURROGATES.end - SURROGATES.start)
    } else {
        code
    }
}

/// The inverse of [`scalar`].
#[inline]
fn from_scalar(index: u32) -> Option<Char> {
    Char::from_u32(if index >= SURROGATES.start {
        index + (SURROGATES.end - SURROGATES.start)
    } else {
        index
    })
}

/// A stride coprime with `total`, close to its golden-ratio fraction, so that
/// consecutive sample indices land far apart in the combination space instead
/// of walking it in order. Being coprime keeps `index * stride % total` a
/// permutation, so no combination comes up twice.
fn scramble_multiplier(total: u128) -> u128 {
    if total < 3 {
        return 1;
    }

    let mut multiplier = ((total as f64) * 0.618_033_988_749_895) as u128;
    multiplier = multiplier.clamp(1, total - 1);

    for _ in 0..64 {
        if gcd(multiplier, total) == 1 {
            return multiplier;
        }
        multiplier += 1;
        if multiplier >= total {
            multiplier = 1;
        }
    }

    1
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    // splitmix64
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
fn path_mix(h: u64, x: u64) -> u64 {
    h.wrapping_mul(0x9E3779B97F4A7C15).rotate_left(7) ^ x
}

#[cfg(test)]
mod tests {
    use super::{GenerationOptions, GenerationOrder, char_at, sample_char};
    use crate::CharRange;
    use crate::cardinality::Cardinality;
    use crate::{fast_automaton::FastAutomaton, regex::RegularExpression};
    use regex::Regex;
    use regex_charclass::{CharacterClass, char::Char, irange::range::AnyRange};

    const ORDERS: [GenerationOrder; 2] = [GenerationOrder::Exhaustive, GenerationOrder::Sampled];

    /// Every set of options the generation tests run through: both orders,
    /// each of them once unrestricted and once over printable ASCII, which is
    /// narrow enough to close off transitions in most of the patterns.
    fn all_options() -> Vec<GenerationOptions> {
        ORDERS
            .into_iter()
            .flat_map(|order| {
                [
                    GenerationOptions::from(order),
                    GenerationOptions::from(order).with_charset(printable_ascii()),
                ]
            })
            .collect()
    }

    fn printable_ascii() -> CharRange {
        CharRange::new_from_range_char(' '..='~')
    }

    #[test]
    fn test_generate_strings_1() -> Result<(), String> {
        let automaton = RegularExpression::parse("((aad|..e.*|e.z)*|q)", false)
            .unwrap()
            .to_automaton()
            .unwrap();

        let automaton = automaton.determinize().unwrap();
        for order in ORDERS {
            println!("{:?}", automaton.generate_strings(30, 0, order).unwrap());
        }

        Ok(())
    }

    #[test]
    fn test_generate_strings_2() -> Result<(), String> {
        let automaton = RegularExpression::parse("(abc|de){2}", true)
            .unwrap()
            .to_automaton()
            .unwrap();

        let automaton = automaton.determinize().unwrap();
        for order in ORDERS {
            let strings = automaton.generate_strings(2, 0, order).unwrap();
            assert_eq!(2, strings.len());

            let strings = automaton.generate_strings(2, 2, order).unwrap();
            assert_eq!(2, strings.len());
        }

        Ok(())
    }

    #[test]
    fn test_generate_strings_3() -> Result<(), String> {
        assert_generate_strings(r"<([A-Za-z][A-Za-z0-9]*)[^>]*?/>", 500);
        assert_generate_strings("a{100}[a-z]", 100);
        assert_generate_strings("(ab|cd)e", 100);
        assert_generate_strings("[a-z]+", 100);
        assert_generate_strings("[a-z]+@", 100);
        assert_generate_strings("ù", 1000);

        assert_generate_strings("[0-9]+[A-Z]*", 500);
        assert_generate_strings("a+(ba+)*", 200);
        assert_generate_strings("((a|bc)*|d)", 200);
        assert_generate_strings(".*", 50);
        assert_generate_strings("(ac|ads|a)*", 200);
        assert_generate_strings("((aad|ads|a)*|q)", 200);

        assert_generate_strings(
            r"john[!#-'\*\+\-/-9=\?\^-\u{007e}]*(\.[!#-'\*\+\-/-9=\?\^-\u{007e}](\.?[!#-'\*\+\-/-9=\?\^-\u{007e}])*)?\.?doe@example\.com",
            1000,
        );

        assert_generate_strings("(?:A+(?:\\.[AB]+)*|\"(?:C|\\\\D)*\")@", 500);
        assert_generate_strings(
            "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@",
            500,
        );
        assert_generate_strings("((aad|ads|a)*abc.*uif(aad|ads|x)*|q)", 1000);

        Ok(())
    }

    #[test]
    fn test_generate_strings_offset() -> Result<(), String> {
        assert_generate_strings_offset(".{900}");
        assert_generate_strings_offset("[a-z]+");
        assert_generate_strings_offset("[a-z]+@");

        assert_generate_strings_offset("[0-9]+[A-Z]*");
        assert_generate_strings_offset("a+(ba+)*");
        assert_generate_strings_offset("((a|bc)*|d)");
        assert_generate_strings_offset(".*");
        assert_generate_strings_offset("(ac|ads|a)*");
        assert_generate_strings_offset("((aad|ads|a)*|q)");

        assert_generate_strings_offset(
            r"john[!#-'\*\+\-/-9=\?\^-\u{007e}]*(\.[!#-'\*\+\-/-9=\?\^-\u{007e}](\.?[!#-'\*\+\-/-9=\?\^-\u{007e}])*)?\.?doe@example\.com",
        );

        assert_generate_strings_offset("(?:A+(?:\\.[AB]+)*|\"(?:C|\\\\D)*\")@");
        assert_generate_strings_offset(
            "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@",
        );
        assert_generate_strings_offset("((aad|ads|a)*abc.*uif(aad|ads|x)*|q)");

        Ok(())
    }

    /// The sampled order exists so that a pattern's *shapes* get covered: what
    /// the exhaustive order spends a million strings on (one path, every
    /// character of its last range) has to fit in a handful of them.
    #[test]
    fn test_generate_strings_sampled_covers_the_whole_pattern() {
        let automaton = automaton_of(".*abc.*").determinize().unwrap().into_owned();

        let exhaustive = automaton
            .generate_strings(20, 0, GenerationOrder::Exhaustive)
            .unwrap();
        assert!(
            exhaustive.iter().all(|s| s.starts_with("abc")),
            "the exhaustive order stays on the first path it finds: {exhaustive:?}"
        );

        let sampled = automaton
            .generate_strings(20, 0, GenerationOrder::Sampled)
            .unwrap();
        assert!(
            sampled.iter().any(|s| !s.starts_with("abc")),
            "the sampled order has to reach the strings with a prefix before `abc`: {sampled:?}"
        );
        assert!(
            sampled.iter().any(|s| !s.ends_with("abc")),
            "the sampled order has to reach the strings with a suffix after `abc`: {sampled:?}"
        );
        assert!(
            sampled.iter().any(|s| s.len() > 5),
            "the sampled order has to reach longer strings too: {sampled:?}"
        );
    }

    /// Sampling still enumerates a finite language in full, given the room:
    /// the passes dig deeper into every path until nothing is left to cover.
    #[test]
    fn test_generate_strings_sampled_is_exhaustive_in_the_limit() {
        let automaton = automaton_of("[a-z][0-9]");

        let mut sampled = automaton
            .generate_strings(1000, 0, GenerationOrder::Sampled)
            .unwrap();
        sampled.sort();

        let mut expected: Vec<String> = ('a'..='z')
            .flat_map(|letter| ('0'..='9').map(move |digit| format!("{letter}{digit}")))
            .collect();
        expected.sort();

        assert_eq!(expected, sampled);
    }

    /// The sample characters are what a range is reached for first, so a
    /// pattern made of wide ranges samples as readable text rather than as
    /// control characters.
    #[test]
    fn test_generate_strings_sampled_prefers_representative_characters() {
        let automaton = automaton_of(".{3}");

        let sampled = automaton
            .generate_strings(1, 0, GenerationOrder::Sampled)
            .unwrap();

        assert_eq!(vec!["a0A".to_string()], sampled);
    }

    /// Sampling hands out each character of a range exactly once: anything else
    /// and a path would repeat a string, or `offset` would drift off its page.
    #[test]
    fn test_sample_char_is_a_permutation_of_the_range() {
        let ranges = [
            CharRange::new_from_range_char('a'..='z'),
            // Several intervals, one of them straddling the surrogate hole.
            CharRange::new_from_ranges(&[
                AnyRange::from(Char::new('0')..=Char::new('9')),
                AnyRange::from(Char::new('\u{d7fe}')..=Char::new('\u{e001}')),
            ]),
            // Past the hole, around one of the sample characters.
            CharRange::new_from_range_char('\u{1f5ff}'..='\u{1f601}'),
        ];

        for range in ranges {
            let expected: Vec<char> = range.iter().map(|ch| ch.to_char()).collect();

            for position in 0..3 {
                let mut sampled: Vec<char> = (0..range.get_cardinality())
                    .map(|index| sample_char(&range, position, index).unwrap().to_char())
                    .collect();
                sampled.sort_unstable();

                assert_eq!(expected, sampled, "range {range}, position {position}");
            }
        }
    }

    /// A charset rules out whole paths, not single characters: a path that
    /// needs a ruled-out character is dropped even when it only needs it
    /// several transitions in, and the strings around it still come out.
    #[test]
    fn test_generate_strings_charset_drops_the_paths_it_closes() {
        let automaton = automaton_of("(a[0-9]b|xyz)");
        let letters = CharRange::new_from_range_char('a'..='z');

        for order in ORDERS {
            let options = GenerationOptions::from(order).with_charset(letters.clone());

            assert_eq!(
                vec!["xyz".to_string()],
                automaton.generate_strings(10, 0, options).unwrap(),
                "{order:?}"
            );
        }
    }

    /// A charset that leaves the pattern nothing is an empty page, not an
    /// error: the search prunes the start state like any other dead end.
    #[test]
    fn test_generate_strings_charset_can_leave_nothing() {
        let automaton = automaton_of("[0-9]+");
        let letters = CharRange::new_from_range_char('a'..='z');

        for order in ORDERS {
            let options = GenerationOptions::from(order).with_charset(letters.clone());

            assert!(
                automaton
                    .generate_strings(10, 0, options)
                    .unwrap()
                    .is_empty(),
                "{order:?}"
            );
        }
    }

    /// The charset narrows the ranges the strings are built from, so what is
    /// generated is the language the pattern and the charset agree on.
    #[test]
    fn test_generate_strings_charset_narrows_every_range() {
        let automaton = automaton_of(".{2}");
        let vowels = CharRange::new_from_ranges(&[
            AnyRange::from(Char::new('a')..=Char::new('a')),
            AnyRange::from(Char::new('e')..=Char::new('e')),
        ]);

        for order in ORDERS {
            let options = GenerationOptions::from(order).with_charset(vowels.clone());

            let mut strings = automaton.generate_strings(10, 0, options).unwrap();
            strings.sort();

            assert_eq!(vec!["aa", "ae", "ea", "ee"], strings, "{order:?}");
        }
    }

    /// The surrogate block is not a character, so the whole alphabet is one
    /// character shorter than its last code point suggests.
    #[test]
    fn test_char_at_walks_over_the_surrogate_block() {
        let total = CharRange::total();

        assert_eq!('\u{0}', char_at(&total, 0).unwrap().to_char());
        assert_eq!('\u{d7ff}', char_at(&total, 0xd7ff).unwrap().to_char());
        assert_eq!('\u{e000}', char_at(&total, 0xd800).unwrap().to_char());

        let cardinality = total.get_cardinality();
        assert_eq!(
            '\u{10ffff}',
            char_at(&total, cardinality - 1).unwrap().to_char()
        );
        assert!(char_at(&total, cardinality).is_none());
    }

    fn automaton_of(regex: &str) -> FastAutomaton {
        RegularExpression::parse(regex, false)
            .unwrap()
            .to_automaton()
            .unwrap()
    }

    fn assert_generate_strings_offset(regex: &str) {
        println!("regex: {regex}");
        let automaton = automaton_of(regex);

        for options in all_options() {
            // Generate 30 strings at once
            let all_strings = automaton.generate_strings(30, 0, options.clone()).unwrap();

            // Generate the same 30 strings in chunks of 10
            let chunk1 = automaton.generate_strings(10, 0, options.clone()).unwrap();
            let chunk2 = automaton.generate_strings(10, 10, options.clone()).unwrap();
            let chunk3 = automaton.generate_strings(10, 20, options.clone()).unwrap();

            assert_eq!(
                all_strings.len(),
                30,
                "Should generate exactly 30 strings ({options:?})"
            );
            assert_eq!(chunk1.len(), 10, "{options:?}");
            assert_eq!(chunk2.len(), 10, "{options:?}");
            assert_eq!(chunk3.len(), 10, "{options:?}");

            // Combine the chunks
            let mut combined = chunk1;
            combined.extend(chunk2);
            combined.extend(chunk3);

            // Prove that generating in chunks perfectly matches the bulk generation
            assert_eq!(
                all_strings, combined,
                "Chunked generation did not match bulk generation ({options:?})"
            );

            let cardinality = automaton.cardinality().unwrap();

            // A charset only ever leaves fewer strings than the automaton
            // holds, so its count is past the end of the restricted language
            // too.
            if let Cardinality::Integer(count) = cardinality {
                let empty_chunk = automaton
                    .generate_strings(10, count as usize, options.clone())
                    .unwrap();
                assert!(
                    empty_chunk.is_empty(),
                    "Chunk past limits should be empty ({options:?})"
                );
            }
        }
    }

    fn assert_generate_strings(regex: &str, number: usize) {
        println!(":{}", regex);
        let automaton = automaton_of(regex);

        let re = Regex::new(&format!("(?s)^{}$", regex)).unwrap();

        for options in all_options() {
            let strings = automaton
                .generate_strings(number, 0, options.clone())
                .unwrap();
            println!("nb of strings ({options:?}): {}/{}", strings.len(), number);
            assert!(number >= strings.len());

            let distinct: std::collections::HashSet<_> = strings.iter().collect();
            assert_eq!(
                distinct.len(),
                strings.len(),
                "the same string came up twice ({options:?})"
            );

            for string in strings {
                if let Some(charset) = options.charset() {
                    assert!(
                        string.chars().all(|ch| charset.contains(Char::new(ch))),
                        "'{string}' uses characters outside the charset"
                    );
                }
                if !re.is_match(&string) {
                    for byte in string.as_bytes() {
                        print!("{:02x} ", byte);
                    }
                    panic!("'{string}' ({options:?})")
                }
            }
        }
    }
}
