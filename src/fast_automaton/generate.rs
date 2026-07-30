use crate::{EngineError, execution_profile::ExecutionProfile};
use ahash::RandomState;
use indexmap::IndexSet;

use super::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::Range;

/// Each transition condition's index into the range pool the generation
/// resolved, the charset already taken out; `None` for the conditions the
/// charset leaves nothing of.
type RangeIds<'a> = AHashMap<&'a Condition, Option<u32>>;

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
    /// before moving to the next one, so `.*abc.*` yields `abc` and a string
    /// for each of the other shapes (`abc\u{0}`, `\u{0}abc`, ...) — the shapes
    /// the pattern allows, instead of a million variations of one of them,
    /// which is what makes it usable to derive test cases.
    ///
    /// Shape comes first: the strings cover every path the automaton holds
    /// before any path is asked for a second one, so a `limit` smaller than
    /// the number of shapes is spent entirely on distinct shapes, and only a
    /// larger one starts varying the characters within them.
    ///
    /// Within a path, characters come in the same ascending order
    /// [`Exhaustive`](Self::Exhaustive) uses: the order chooses which strings
    /// come first, never the characters they are made of. To generate from
    /// specific characters, restrict generation with
    /// [`GenerationOptions::with_charset`].
    ///
    /// Deterministic, and pages with `offset` like [`Exhaustive`](Self::Exhaustive).
    /// Each pass takes twice as many strings per path as the previous one, so
    /// a finite language is still enumerated in full given a large enough
    /// `limit`; those repeated passes make it slower than `Exhaustive`.
    Sampled,
}

/// The most strings to reserve room for up front. `limit` is caller-controlled
/// and huge values (up to `usize::MAX`) are legitimate ways to ask for
/// everything, so it cannot size the allocation on its own; past this hint the
/// set grows as it fills.
const STRINGS_CAPACITY_LIMIT: usize = 1 << 12;

/// How much a [`PathCache`] may hold — a finite language can still have far
/// more paths than fit in memory. Past these, recording gives up and the later
/// sampled passes search the automaton again: time spent instead of memory.
const CACHE_IDS_LIMIT: usize = 1 << 20;
const CACHE_PATHS_LIMIT: usize = 1 << 17;

#[derive(Clone, Eq, PartialEq)]
struct QueueItem {
    score: usize,
    depth: usize,
    state: usize,
    /// The path's transitions as indices into [`Generation::range_pool`]
    ranges: Vec<u32>,
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .cmp(&self.score)
            .then_with(|| self.depth.cmp(&other.depth))
            .then_with(|| self.state.cmp(&other.state))
            .then_with(|| self.ranges.cmp(&other.ranges))
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

        // Serializing the charset is not free: only when the span is recorded.
        let span = tracing::Span::current();
        if !span.is_disabled() {
            span.record("order", tracing::field::debug(options.order));
            span.record(
                "charset",
                tracing::field::debug(options.charset.as_ref().map(|charset| charset.to_regex())),
            );
        }

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
                generation.walk(self, None, None)?;
            }
            GenerationOrder::Sampled => {
                // A pass takes at most `window` combinations per path, so no
                // single path can spend the whole `limit` on itself. The
                // windows double and pick up where the previous one stopped:
                // a pass that exhausts the automaton without filling `limit`
                // is followed by one digging deeper into the same paths, until
                // a pass finds nothing left to cover. Exhausting the automaton
                // also proves the paths finite and leaves them in `cache`, so
                // the passes after it replay them instead of searching again.
                let mut window = 0..1;
                let mut cache = PathCache::new();
                loop {
                    let covered = if cache.complete {
                        generation.replay(&cache, &window)?
                    } else {
                        generation.walk(self, Some(&window), Some(&mut cache))?
                    };
                    if covered == 0 || generation.emitter.is_full() {
                        break;
                    }
                    window = window.end..window.end.saturating_mul(2).saturating_add(1);
                }
            }
        }

        Ok(generation.emitter.strings.into_iter().collect())
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
    /// The characters each transition stands for, resolved once: the paths
    /// refer to them by index (see [`QueueItem::ranges`]).
    range_pool: Vec<CharRange>,
    /// Each transition condition's index into
    /// [`range_pool`](Self::range_pool), the charset already taken out. A
    /// condition the charset leaves nothing of holds `None`, which is what
    /// makes its transition impassable.
    range_ids: RangeIds<'a>,
    emitter: Emitter,
}

/// Turns the paths the search pops into strings: the half of a [`Generation`]
/// the emission mutates, kept apart from the search data so that a borrow of
/// the range pool can live alongside it.
struct Emitter {
    limit: usize,
    offset: usize,
    strings: IndexSet<String, RandomState>,
    execution_profile: ExecutionProfile,
}

/// The accepting paths a sampled pass popped, in pop order — flat, path `i`
/// being `ids[starts[i]..starts[i + 1]]`. A pass that runs out of paths has
/// recorded all of them, and the passes after it replay the cache instead of
/// searching the automaton again.
struct PathCache {
    ids: Vec<u32>,
    starts: Vec<usize>,
    /// The cache holds every path of the automaton and can stand in for it.
    complete: bool,
    /// Recording outgrew [`CACHE_IDS_LIMIT`]/[`CACHE_PATHS_LIMIT`] and gave
    /// up; the cache stays empty and every pass searches.
    overflowed: bool,
}

impl PathCache {
    fn new() -> Self {
        PathCache {
            ids: vec![],
            starts: vec![0],
            complete: false,
            overflowed: false,
        }
    }

    fn record(&mut self, path: &[u32]) {
        if self.overflowed {
            return;
        }
        if self.ids.len().saturating_add(path.len()) > CACHE_IDS_LIMIT
            || self.starts.len() > CACHE_PATHS_LIMIT
        {
            self.overflowed = true;
            self.ids = vec![];
            self.starts = vec![0];
            return;
        }

        self.ids.extend_from_slice(path);
        self.starts.push(self.ids.len());
    }

    fn paths(&self) -> impl Iterator<Item = &[u32]> {
        self.starts
            .windows(2)
            .map(|window| &self.ids[window[0]..window[1]])
    }
}

/// What every transition condition leaves once the charset is taken out.
/// Resolved up front rather than as the walk reaches them, so that the search
/// already knows which transitions are impassable: a pool of the non-empty
/// ranges, and each condition's index into it — `None` for the conditions the
/// charset leaves nothing of.
fn resolve_ranges<'a>(
    automaton: &'a FastAutomaton,
    charset: Option<&CharRange>,
) -> Result<(Vec<CharRange>, RangeIds<'a>), EngineError> {
    let mut range_pool: Vec<CharRange> = Vec::new();
    let mut range_ids: RangeIds = AHashMap::with_capacity(automaton.transitions.len());

    for state in automaton.states() {
        for (cond, _) in automaton.transitions_from(state) {
            let std::collections::hash_map::Entry::Vacant(vacant) = range_ids.entry(cond) else {
                continue;
            };
            let range = cond.to_range(&automaton.spanning_set)?;
            let range = match charset {
                Some(charset) => range.intersection(charset),
                None => range,
            };
            let id = if range.is_empty() {
                None
            } else {
                range_pool.push(range);
                Some((range_pool.len() - 1) as u32)
            };
            vacant.insert(id);
        }
    }

    Ok((range_pool, range_ids))
}

/// REVERSE BFS: the exact distance from every state to an accept state, which
/// drives the A* search and prunes the states that never accept; `usize::MAX`
/// for the states that cannot reach one. A state the charset leaves no way out
/// of (no id in `range_ids`) is one of those dead ends.
fn distances_to_accept(automaton: &FastAutomaton, range_ids: &RangeIds) -> Vec<usize> {
    let num_states = automaton.transitions.len();
    let mut incoming = vec![vec![]; num_states];
    let mut dist_q = VecDeque::new();
    let mut distances = vec![usize::MAX; num_states];

    for state in automaton.states() {
        if automaton.is_accepted(state) {
            distances[state] = 0;
            dist_q.push_back(state);
        }
        for (cond, &to_state) in automaton.transitions_from(state) {
            if range_ids[cond].is_some() {
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

    distances
}

/// The ranges of a path's transitions, looked up from the pool.
fn resolve<'p>(pool: &'p [CharRange], path: &[u32]) -> Vec<&'p CharRange> {
    path.iter().map(|&id| &pool[id as usize]).collect()
}

impl<'a> Generation<'a> {
    fn new(
        automaton: &'a FastAutomaton,
        limit: usize,
        offset: usize,
        charset: Option<&CharRange>,
    ) -> Result<Self, EngineError> {
        let (range_pool, range_ids) = resolve_ranges(automaton, charset)?;
        let distances = distances_to_accept(automaton, &range_ids);
        let (_, max) = automaton.length();

        Ok(Generation {
            distances,
            max_len: max.unwrap_or(u32::MAX) as usize,
            range_pool,
            range_ids,
            emitter: Emitter {
                limit,
                offset,
                strings: IndexSet::with_capacity_and_hasher(
                    limit.min(STRINGS_CAPACITY_LIMIT),
                    RandomState::default(),
                ),
                execution_profile: ExecutionProfile::get(),
            },
        })
    }

    /// A* SEARCH: walks the automaton once, shortest path first, emitting the
    /// strings of every accepting path it pops until `limit` strings are
    /// collected or the automaton runs out of paths.
    ///
    /// `window` restricts each path to the combinations whose index falls
    /// inside it; `None` takes them all. `cache`, when given, records the
    /// accepting paths in pop order, and running out of paths marks it
    /// complete: [`replay`](Self::replay) then stands in for the next passes.
    /// Returns how many combinations the pass covered, the ones `offset`
    /// skipped included.
    fn walk(
        &mut self,
        automaton: &'a FastAutomaton,
        window: Option<&Range<usize>>,
        mut cache: Option<&mut PathCache>,
    ) -> Result<usize, EngineError> {
        let start_state = automaton.start_state();

        // If the start state can't reach an accept state, exit immediately
        if self.distances[start_state] == usize::MAX {
            return Ok(0);
        }

        let mut covered = 0usize;

        let mut q = BinaryHeap::new();
        q.push(QueueItem {
            score: self.distances[start_state],
            depth: 0,
            state: start_state,
            ranges: vec![],
        });

        while let Some(QueueItem {
            score: _,
            depth: current_depth,
            state,
            ranges,
        }) = q.pop()
        {
            self.emitter.execution_profile.assert_not_timed_out()?;

            if automaton.is_accepted(state) {
                if let Some(cache) = cache.as_deref_mut() {
                    cache.record(&ranges);
                }

                let resolved = resolve(&self.range_pool, &ranges);
                covered = covered.saturating_add(match window {
                    Some(window) => self.emitter.emit_sampled(&resolved, window)?,
                    None => self.emitter.emit_all(&resolved)?,
                });

                if self.emitter.is_full() {
                    break;
                }
            }

            if current_depth >= self.max_len {
                continue;
            }

            self.expand(automaton, &mut q, current_depth + 1, state, ranges);
        }

        // An empty queue means every path was popped, so a recording cache
        // now holds them all.
        if q.is_empty()
            && let Some(cache) = cache
            && !cache.overflowed
        {
            cache.complete = true;
        }

        Ok(covered)
    }

    /// Queues every passable one-transition extension of a popped path, the
    /// last one taking over the path's own vector instead of cloning it.
    fn expand(
        &self,
        automaton: &'a FastAutomaton,
        q: &mut BinaryHeap<QueueItem>,
        next_depth: usize,
        state: State,
        mut ranges: Vec<u32>,
    ) {
        let mut valid_transitions = Vec::new();

        for (cond, &to_state) in automaton.transitions_from(state) {
            // DEAD-END PRUNING: Instantly kill paths that cannot accept
            if self.distances[to_state] == usize::MAX {
                continue;
            }

            // ...and the transitions the charset closed off.
            let Some(range_id) = self.range_ids[cond] else {
                continue;
            };

            valid_transitions.push((to_state, range_id));
        }

        // Vector Reuse Optimization
        if let Some((last_state, last_id)) = valid_transitions.pop() {
            for (to_state, range_id) in valid_transitions {
                let mut new_ranges = ranges.clone();
                new_ranges.push(range_id);
                q.push(QueueItem {
                    score: next_depth + self.distances[to_state], // A* Score Formula
                    depth: next_depth,
                    state: to_state,
                    ranges: new_ranges,
                });
            }

            ranges.push(last_id);
            q.push(QueueItem {
                score: next_depth + self.distances[last_state], // A* Score Formula
                depth: next_depth,
                state: last_state,
                ranges,
            });
        }
    }

    /// Emits `window` from every path of a complete [`PathCache`], in the
    /// order the search popped them: what a [`walk`](Self::walk) pass would
    /// do, minus the search.
    fn replay(&mut self, cache: &PathCache, window: &Range<usize>) -> Result<usize, EngineError> {
        let mut covered = 0usize;

        for path in cache.paths() {
            self.emitter.execution_profile.assert_not_timed_out()?;

            let resolved = resolve(&self.range_pool, path);
            covered = covered.saturating_add(self.emitter.emit_sampled(&resolved, window)?);

            if self.emitter.is_full() {
                break;
            }
        }

        Ok(covered)
    }
}

impl Emitter {
    #[inline]
    fn is_full(&self) -> bool {
        self.strings.len() >= self.limit
    }

    /// Emits every combination of `ranges` that `offset` does not skip, in
    /// ascending character order. Returns the number of combinations the path
    /// holds.
    fn emit_all(&mut self, ranges: &[&CharRange]) -> Result<usize, EngineError> {
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

        self.emit_combinations(ranges, &range_lengths)?;
        Ok(total_combinations)
    }

    /// Walks the combinations depth-first over an explicit stack of range
    /// cursors, one per position: recursing per character would overflow the
    /// stack on the paths thousands of transitions long.
    fn emit_combinations(
        &mut self,
        ranges: &[&CharRange],
        range_lengths: &[usize],
    ) -> Result<(), EngineError> {
        if ranges.is_empty() {
            // A single-combination path: `emit_all` either skipped it whole or
            // arrived here with nothing left of the offset.
            debug_assert_eq!(0, self.offset);
            self.strings.insert(String::new());
            return Ok(());
        }

        // Combinations under a single character at each position: the product
        // of the range lengths past it.
        let mut sub_combinations = vec![1usize; ranges.len()];
        for position in (0..ranges.len() - 1).rev() {
            sub_combinations[position] =
                sub_combinations[position + 1].saturating_mul(range_lengths[position + 1]);
        }

        let mut current_str = String::with_capacity(ranges.len());
        let mut cursors = Vec::with_capacity(ranges.len());
        cursors.push(self.descend(ranges[0], sub_combinations[0]));

        while let Some(cursor) = cursors.last_mut() {
            let next = cursor.next();
            let position = cursors.len() - 1;

            let Some(ch) = next else {
                // The range is exhausted: back up to the previous position and
                // move it to its next character.
                cursors.pop();
                current_str.pop();
                continue;
            };

            self.execution_profile.assert_not_timed_out()?;

            current_str.push(ch.to_char());
            if position + 1 == ranges.len() {
                // A full combination; whatever `offset` had left to skip was
                // consumed by the descents, so this string is on the page.
                self.strings.insert(current_str.clone());
                current_str.pop();

                if self.is_full() {
                    break;
                }
            } else {
                cursors.push(self.descend(ranges[position + 1], sub_combinations[position + 1]));
            }
        }

        Ok(())
    }

    /// A cursor over `range`, opened on the first combination `offset` does
    /// not skip: the subtrees of `sub_combinations` strings each before it are
    /// stepped over in one division, not walked character by character.
    fn descend<'r>(&mut self, range: &'r CharRange, sub_combinations: usize) -> RangeCursor<'r> {
        // Past the first emitted string the offset is zero and every cursor
        // starts at its range's first character. `skip` stays within the
        // range: the offset was left smaller than the previous position's
        // subtree, which this whole range spans. (A saturated subtree count
        // under-skips into the first character, never past the range.)
        let skip = self.offset / sub_combinations;
        self.offset -= skip * sub_combinations;
        RangeCursor::new(range, skip as u32)
    }

    /// Emits the combinations of `ranges` whose index falls inside `window`,
    /// in the ascending order [`emit_all`](Self::emit_all) walks them in.
    /// Returns how many of them the window covered, the ones `offset` skipped
    /// included.
    fn emit_sampled(
        &mut self,
        ranges: &[&CharRange],
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

        let first = window.start.min(bound) + self.offset;
        self.offset = 0;

        for index in first..window.end.min(bound) {
            self.execution_profile.assert_not_timed_out()?;

            let string = sample_string(ranges, &range_lengths, index as u128)?;
            self.strings.insert(string);

            if self.is_full() {
                break;
            }
        }

        Ok(covered)
    }
}

/// Builds the combination of `ranges` at index `combination`, read as a
/// mixed-radix number whose least significant digit is the last character,
/// each digit indexing its range in ascending order: combinations come out in
/// the lexicographic order [`Emitter::emit_all`] walks them in.
///
/// The mapping is a bijection over the combinations, which is what keeps the
/// sampled strings distinct and `offset` exact.
fn sample_string(
    ranges: &[&CharRange],
    range_lengths: &[u128],
    mut combination: u128,
) -> Result<String, EngineError> {
    let mut chars = Vec::with_capacity(ranges.len());

    for (&range, &length) in ranges.iter().zip(range_lengths).rev() {
        let index = (combination % length) as u32;
        combination /= length;
        let ch = char_at(range, index).ok_or(EngineError::InvalidCharacterInRegex)?;
        chars.push(ch.to_char());
    }

    chars.reverse();
    Ok(chars.into_iter().collect())
}

/// A cursor over the characters a [`CharRange`] holds, in order, opened at an
/// arbitrary ordinal: what the range's own iterator cannot do, and what lets
/// [`Emitter::descend`] skip an offset in one step.
struct RangeCursor<'a> {
    /// The `(low, high)` bound pairs the range is made of.
    bounds: &'a [Char],
    /// Index of the current pair's low bound; past `bounds` once exhausted.
    pair: usize,
    /// [`scalar`] of the next character to hand out.
    next_scalar: u32,
}

impl<'a> RangeCursor<'a> {
    /// A cursor whose first character is `range`'s `ordinal`-th; exhausted
    /// from the start when `ordinal` is past the range's cardinality.
    fn new(range: &'a CharRange, mut ordinal: u32) -> Self {
        let bounds = range.0.as_slice();
        let mut pair = 0;
        let mut next_scalar = 0;

        while pair < bounds.len() {
            let (low, high) = (scalar(bounds[pair]), scalar(bounds[pair + 1]));
            let length = high - low + 1;
            if ordinal < length {
                next_scalar = low + ordinal;
                break;
            }
            ordinal -= length;
            pair += 2;
        }

        RangeCursor {
            bounds,
            pair,
            next_scalar,
        }
    }

    fn next(&mut self) -> Option<Char> {
        if self.pair >= self.bounds.len() {
            return None;
        }

        let ch = from_scalar(self.next_scalar);
        if self.next_scalar == scalar(self.bounds[self.pair + 1]) {
            // Past the current interval: on to the next one.
            self.pair += 2;
            if self.pair < self.bounds.len() {
                self.next_scalar = scalar(self.bounds[self.pair]);
            }
        } else {
            self.next_scalar += 1;
        }

        ch
    }
}

/// The character `range` holds at `ordinal`, `None` past its cardinality.
fn char_at(range: &CharRange, ordinal: u32) -> Option<Char> {
    RangeCursor::new(range, ordinal).next()
}

#[cfg(test)]
mod tests {
    use super::{GenerationOptions, GenerationOrder, RangeCursor, char_at};
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

    /// The order chooses which strings come first, never the characters they
    /// are made of: a sampled string reaches for the same low end of each
    /// range the exhaustive order starts from, and a charset is how specific
    /// characters are asked for.
    #[test]
    fn test_generate_strings_sampled_uses_the_same_characters_as_exhaustive() {
        let automaton = automaton_of(".{3}");

        let sampled = automaton
            .generate_strings(1, 0, GenerationOrder::Sampled)
            .unwrap();
        let exhaustive = automaton
            .generate_strings(1, 0, GenerationOrder::Exhaustive)
            .unwrap();
        assert_eq!(exhaustive, sampled);

        let lowercase = CharRange::new_from_range_char('a'..='z');
        let options = GenerationOptions::from(GenerationOrder::Sampled).with_charset(lowercase);
        assert_eq!(
            vec!["aaa".to_string()],
            automaton.generate_strings(1, 0, options).unwrap()
        );
    }

    /// A single-path language has only one shape, so there is nothing for the
    /// sampled order to spread over: within a path the characters come in the
    /// lexicographic order the exhaustive order walks.
    #[test]
    fn test_generate_strings_sampled_matches_exhaustive_on_a_single_path() {
        let automaton = automaton_of("[a-z]{2}");

        let exhaustive = automaton
            .generate_strings(10, 0, GenerationOrder::Exhaustive)
            .unwrap();
        let sampled = automaton
            .generate_strings(10, 0, GenerationOrder::Sampled)
            .unwrap();

        assert_eq!(exhaustive, sampled);
    }

    /// The strongest form of "the order chooses which strings come first,
    /// never the characters": on a finite language, sampled and exhaustive
    /// enumerate exactly the same set — including through nondeterministic
    /// automata, multi-interval charsets, and ranges straddling the surrogate
    /// hole.
    #[test]
    fn test_generate_strings_sampled_and_exhaustive_agree_as_sets() {
        let multi_interval = CharRange::new_from_ranges(&[
            AnyRange::from(Char::new('x')..=Char::new('z')),
            AnyRange::from(Char::new('0')..=Char::new('1')),
        ]);
        let surrogate_straddle = CharRange::new_from_range_char('\u{d7fe}'..='\u{e001}');

        let cases: Vec<(&str, Option<CharRange>)> = vec![
            ("(a|bc){0,3}", None),
            // Nondeterministic: "ab" is reachable through both branches.
            ("(ab|a)b{0,2}", None),
            ("[a-e]{0,2}[0-3]?", None),
            (".{0,2}", Some(multi_interval)),
            (".", Some(surrogate_straddle)),
        ];

        for (pattern, charset) in cases {
            for determinize in [false, true] {
                let automaton = if determinize {
                    automaton_of(pattern).determinize().unwrap().into_owned()
                } else {
                    automaton_of(pattern)
                };

                let options = |order| match &charset {
                    Some(charset) => GenerationOptions::from(order).with_charset(charset.clone()),
                    None => GenerationOptions::from(order),
                };

                let mut exhaustive = automaton
                    .generate_strings(100_000, 0, options(GenerationOrder::Exhaustive))
                    .unwrap();
                let mut sampled = automaton
                    .generate_strings(100_000, 0, options(GenerationOrder::Sampled))
                    .unwrap();

                exhaustive.sort();
                sampled.sort();
                assert_eq!(
                    exhaustive, sampled,
                    "{pattern:?} (determinized: {determinize})"
                );
            }
        }
    }

    /// Sampled pages stay consistent at every chunk size, not only at the
    /// window boundaries: an offset landing mid-window or mid-path continues
    /// exactly where the previous page stopped.
    #[test]
    fn test_generate_strings_sampled_pages_at_any_boundary() {
        for pattern in ["(a|bc){0,3}", "[a-c]{1,3}", "(x|yy)(0|11)?"] {
            // Deterministic and minimal, so pages are exactly disjoint.
            let mut automaton = automaton_of(pattern).determinize().unwrap().into_owned();
            automaton.minimize().unwrap();

            let bulk = automaton
                .generate_strings(60, 0, GenerationOrder::Sampled)
                .unwrap();

            for chunk in 1..=7usize {
                let mut paged = vec![];
                loop {
                    let page = automaton
                        .generate_strings(chunk, paged.len(), GenerationOrder::Sampled)
                        .unwrap();
                    if page.is_empty() {
                        break;
                    }
                    paged.extend(page);
                    if paged.len() >= bulk.len() {
                        break;
                    }
                }
                paged.truncate(bulk.len());

                assert_eq!(bulk, paged, "{pattern:?} at chunk size {chunk}");
            }
        }
    }

    /// A path holding more combinations than `u128` fits still samples the
    /// ascending sequence: the decode consumes the index from the last
    /// position, so the positions it never reaches keep their range's first
    /// character.
    #[test]
    fn test_generate_strings_sampled_orders_huge_paths() {
        let automaton = automaton_of(".{40}");

        let sampled = automaton
            .generate_strings(3, 0, GenerationOrder::Sampled)
            .unwrap();

        assert_eq!(
            vec![
                "\u{0}".repeat(40),
                format!("{}\u{1}", "\u{0}".repeat(39)),
                format!("{}\u{2}", "\u{0}".repeat(39)),
            ],
            sampled
        );
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

    /// The offset steps over whole subtrees at once: a page from deep inside a
    /// large language comes back without walking everything before it.
    #[test]
    fn test_generate_strings_offset_reaches_deep_pages() {
        let automaton = automaton_of("[a-z]{5}");
        let total = 26usize.pow(5);

        let strings = automaton
            .generate_strings(2, total - 2, GenerationOrder::Exhaustive)
            .unwrap();

        assert_eq!(vec!["zzzzy".to_string(), "zzzzz".to_string()], strings);
    }

    /// The cursor hands out exactly the characters past its opening ordinal,
    /// across intervals and over the surrogate hole, like the plain iterator.
    #[test]
    fn test_range_cursor_opens_at_any_ordinal() {
        let range = CharRange::new_from_ranges(&[
            AnyRange::from(Char::new('0')..=Char::new('9')),
            AnyRange::from(Char::new('\u{d7fe}')..=Char::new('\u{e001}')),
        ]);

        for start in 0..=range.get_cardinality() {
            let expected: Vec<char> = range
                .iter()
                .skip(start as usize)
                .map(|c| c.to_char())
                .collect();

            let mut cursor = RangeCursor::new(&range, start);
            let mut walked = vec![];
            while let Some(ch) = cursor.next() {
                walked.push(ch.to_char());
            }

            assert_eq!(expected, walked, "start {start}");
        }
    }

    /// A huge `limit` means "everything the language holds", not "reserve this
    /// much memory": it must not size an allocation before generation starts.
    #[test]
    fn test_generate_strings_limit_does_not_preallocate() {
        let automaton = automaton_of("[ab]{2}");

        for order in ORDERS {
            let mut strings = automaton.generate_strings(usize::MAX, 0, order).unwrap();
            strings.sort();

            assert_eq!(vec!["aa", "ab", "ba", "bb"], strings, "{order:?}");
        }
    }

    /// Combination emission walks an explicit stack, not the call stack: a
    /// path tens of thousands of transitions long emits without overflowing.
    #[test]
    fn test_generate_strings_very_long_string() {
        let automaton = automaton_of("[ab]{20000}");

        for order in ORDERS {
            let strings = automaton.generate_strings(2, 0, order).unwrap();
            assert_eq!(2, strings.len(), "{order:?}");
            for string in &strings {
                assert_eq!(20_000, string.len(), "{order:?}");
            }
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
