use crate::{EngineError, execution_profile::ExecutionProfile};
use indexmap::IndexSet;

#[cfg(feature = "ahash")]
use ahash::{AHashMap as HashMap, RandomState};
#[cfg(not(feature = "ahash"))]
use std::collections::hash_map::RandomState;

use super::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::Range;

/// Each transition condition's index into the range pool the generation
/// resolved, the charset already taken out; `None` for the conditions the
/// charset leaves nothing of.
type RangeIds<'a> = HashMap<&'a Condition, Option<u32>>;

/// How [`FastAutomaton::generate_strings`] schedules the *paths* of a
/// language: one at a time, or interleaved so that every shape the pattern
/// allows is covered early. Orthogonal to [`CharacterOrder`], which chooses
/// the strings within each path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PathOrder {
    /// One path at a time, shortest first, expanded in full before the next
    /// path is visited: `.*abc.*` yields `abc`, `abc\u{0}`, `abc\u{1}`, ...
    /// The cheapest way to page through a whole language with `offset`.
    #[default]
    Sweep,
    /// A few strings per path before moving to the next one, so `.*abc.*`
    /// yields `abc` and a string for each of the other shapes (`abc\u{0}`,
    /// `\u{0}abc`, ...) instead of a million variations of one of them.
    ///
    /// Shape comes first: the strings cover every path the automaton holds
    /// before any path is asked for a second one, so a `limit` smaller than
    /// the number of shapes is spent entirely on distinct shapes, and only a
    /// larger one starts varying the characters within them.
    ///
    /// Deterministic, and pages with `offset` like [`Sweep`](Self::Sweep).
    /// Each pass takes twice as many strings per path as the previous one, so
    /// a finite language is still enumerated in full given a large enough
    /// `limit`; those repeated passes make it slower than `Sweep`.
    Interleave,
    /// [`Interleave`](Self::Interleave), with same-length paths visited in an
    /// order drawn by the seed ([`GenerationOptions::with_seed`], 0 by
    /// default) instead of a fixed one. Shorter paths still come first, since
    /// the search has to stay shortest-first to emit anything at all on an
    /// infinite language, so the seed only draws among paths of equal length.
    ///
    /// The draw within a length is a randomized cascade rather than a uniform
    /// shuffle: the seed randomizes the pop order of the underlying
    /// shortest-first search, and a path only becomes available once its whole
    /// prefix chain has popped, so a shape branching off an already-visited
    /// path leads more often than one sharing nothing with it. Coverage is
    /// untouched, every shape still coming before any shape's second string.
    ///
    /// Independent of [`CharacterOrder`]: shuffled paths over
    /// [`Ascending`](CharacterOrder::Ascending) characters yield each drawn
    /// shape's smallest witness; pair with [`CharacterOrder::Shuffled`] for
    /// fully random-looking test cases. Deterministic for a given seed, and
    /// pages with `offset` like the other orders; offsets are only consistent
    /// between calls sharing the seed.
    Shuffled,
}

/// Which strings of a path [`FastAutomaton::generate_strings`] reaches for
/// first: its character combinations in ascending order, or a seeded shuffle
/// of them. Orthogonal to [`PathOrder`], which schedules the paths
/// themselves.
///
/// Neither changes *what* is generated: on a finite language every
/// combination of the two axes enumerates exactly the same strings, given the
/// `limit`. To generate from specific characters, restrict generation with
/// [`GenerationOptions::with_charset`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CharacterOrder {
    /// Each position expanded from the low end of its character range first:
    /// `[a-z]{8}` yields `aaaaaaaa`, `aaaaaaab`, ... A stable, documented
    /// order: the smallest witnesses of a path come first.
    #[default]
    Ascending,
    /// A seeded permutation of each path's combinations, so `[a-z]{8}` yields
    /// something like `sjtwsive` rather than `aaaaaaaa`. To draw the *shapes*
    /// by the seed too, pair with [`PathOrder::Shuffled`].
    ///
    /// Random in look only: the seed ([`GenerationOptions::with_seed`], 0 by
    /// default) picks one fixed permutation, so generation is reproducible and
    /// pages with `offset` like [`Ascending`](Self::Ascending). The
    /// permutation is a bijection, so it repeats no more strings across
    /// offsets than `Ascending` does. Offsets are only consistent between
    /// calls sharing the seed, and unlike `Ascending`'s the exact sequence is
    /// implementation-defined: it may change between releases.
    Shuffled,
}

/// The most strings to reserve room for up front. `limit` is caller-controlled
/// and huge values (up to `usize::MAX`) are legitimate ways to ask for
/// everything, so it cannot size the allocation on its own; past this hint the
/// set grows as it fills.
const STRINGS_CAPACITY_LIMIT: usize = 1 << 12;

/// How much a [`PathCache`] may hold: a finite language can still have far
/// more paths than fit in memory. Past these, recording gives up and the later
/// interleave passes search the automaton again, spending time instead of
/// memory.
const CACHE_IDS_LIMIT: usize = 1 << 20;
const CACHE_PATHS_LIMIT: usize = 1 << 17;

/// Salts the seed into [`Generation::shape_key`], so the shape draw and the
/// [`Permuter`]'s character draw are independent functions of the same seed.
const SHAPE_KEY_SALT: u64 = 0x517C_C1B7_2722_0A95;

#[derive(Clone, Eq, PartialEq)]
struct QueueItem {
    score: usize,
    /// Seeded tie-break between items of equal score, 0 unless paths are
    /// [`PathOrder::Shuffled`]: what draws the shapes a small `limit` reaches
    /// (see [`Generation::tie`]).
    tie: u64,
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
            .then_with(|| self.tie.cmp(&other.tie))
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

/// What [`FastAutomaton::generate_strings`] is allowed to generate: how paths
/// are scheduled ([`PathOrder`]), how the strings within them are ordered
/// ([`CharacterOrder`], with the seed behind the `Shuffled` modes of both
/// axes), the characters it may use, and the string lengths it is confined
/// to.
///
/// Either axis converts into it, so one can be passed on its own wherever
/// options are expected and the other keeps its default; so does a
/// `(PathOrder, CharacterOrder)` pair.
///
/// # Examples
///
/// ```
/// use regexsolver::{CharRange, Term, fast_automaton::{CharacterOrder, GenerationOptions, PathOrder}};
/// use regexsolver::regex_charclass::char::Char;
///
/// let term = Term::from_pattern(".{2}").unwrap();
///
/// // An axis on its own.
/// let strings = term.generate_strings(3, 0, PathOrder::Interleave).unwrap();
///
/// // Both axes, restricted to lowercase letters.
/// let lowercase = CharRange::new_from_range(Char::new('a')..=Char::new('z'));
/// let options = GenerationOptions::from((PathOrder::Interleave, CharacterOrder::Shuffled))
///     .with_charset(lowercase)
///     .with_seed(42);
///
/// let strings = term.generate_strings(3, 0, options).unwrap();
/// assert!(strings.iter().all(|s| s.chars().all(|c| c.is_ascii_lowercase())));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GenerationOptions {
    paths: PathOrder,
    characters: CharacterOrder,
    charset: Option<CharRange>,
    seed: u64,
    min_length: usize,
    max_length: Option<usize>,
}

impl GenerationOptions {
    /// Default options: [`PathOrder::Sweep`] over
    /// [`CharacterOrder::Ascending`] combinations, using every character and
    /// string length the automaton allows.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy of these options scheduling paths in `paths` order.
    pub fn with_paths(mut self, paths: PathOrder) -> Self {
        self.paths = paths;
        self
    }

    /// Returns a copy of these options ordering each path's strings in
    /// `characters` order.
    pub fn with_characters(mut self, characters: CharacterOrder) -> Self {
        self.characters = characters;
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

    /// Returns a copy of these options drawing [`PathOrder::Shuffled`]'s
    /// path draws and [`CharacterOrder::Shuffled`]'s permutation from `seed`;
    /// generation using neither ignores it.
    ///
    /// The default seed is 0, a fixed seed rather than a random one, so two
    /// calls with the same options generate the same strings and `offset`
    /// pages through them consistently. Change the seed to draw a different
    /// sequence of strings from the same pattern.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Returns a copy of these options generating only strings at least
    /// `min_length` characters long: the shorter strings the automaton
    /// matches are left out of the enumeration, `offset` never counting
    /// them. 0 by default, which keeps every string.
    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.min_length = min_length;
        self
    }

    /// Returns a copy of these options generating only strings at most
    /// `max_length` characters long: the longer strings the automaton
    /// matches are left out of the enumeration, `offset` never counting
    /// them. Unbounded by default; without a bound, a deep `offset` into a
    /// looping language (`.*`) pages into arbitrarily long strings, so set one
    /// when the offset is not under your control.
    ///
    /// A bound below `min_length` leaves nothing to generate.
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// The order paths are scheduled in.
    pub fn paths(&self) -> PathOrder {
        self.paths
    }

    /// The order each path's strings come out in.
    pub fn characters(&self) -> CharacterOrder {
        self.characters
    }

    /// The characters generation is restricted to, `None` when it is not.
    pub fn charset(&self) -> Option<&CharRange> {
        self.charset.as_ref()
    }

    /// The seed behind [`PathOrder::Shuffled`] and
    /// [`CharacterOrder::Shuffled`].
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The shortest string generation may emit.
    pub fn min_length(&self) -> usize {
        self.min_length
    }

    /// The longest string generation may emit, `None` when unbounded.
    pub fn max_length(&self) -> Option<usize> {
        self.max_length
    }
}

impl From<PathOrder> for GenerationOptions {
    fn from(paths: PathOrder) -> Self {
        GenerationOptions {
            paths,
            ..Default::default()
        }
    }
}

impl From<CharacterOrder> for GenerationOptions {
    fn from(characters: CharacterOrder) -> Self {
        GenerationOptions {
            characters,
            ..Default::default()
        }
    }
}

impl From<(PathOrder, CharacterOrder)> for GenerationOptions {
    fn from((paths, characters): (PathOrder, CharacterOrder)) -> Self {
        GenerationOptions {
            paths,
            characters,
            ..Default::default()
        }
    }
}

impl FastAutomaton {
    /// Generates up to `limit` distinct strings matched by the automaton under
    /// the given [`GenerationOptions`], skipping the first `offset` strings.
    ///
    /// `options` is a [`PathOrder`] or [`CharacterOrder`] on its own (or a
    /// pair of them), or a full [`GenerationOptions`] to also set the seed
    /// and restrict the characters and string lengths used.
    ///
    /// Strings are only guaranteed to be distinct **within a single call**:
    /// the offset fast-skips by counting paths, and in a non-deterministic
    /// automaton the same string can be reached through several paths, so
    /// calls with different offsets may repeat strings (or skip some).
    /// [`determinize`](Self::determinize) (and ideally
    /// [`minimize`](Self::minimize)) first to make pages disjoint. Offsets are
    /// also only consistent between calls made with the same options.
    ///
    /// [`GenerationOptions::with_min_length`] and
    /// [`with_max_length`](GenerationOptions::with_max_length) confine the
    /// enumeration to a band of string lengths. Without a max, a deep
    /// `offset` into a looping language (`.*`) pages into arbitrarily long
    /// strings. Generation runs under the active [`ExecutionProfile`]: its
    /// timeout aborts with [`EngineError::OperationTimeOutError`].
    #[tracing::instrument(level = "debug", skip(self, options), fields(states = self.number_of_states(), deterministic=self.is_deterministic(), limit=limit, offset=offset, paths=tracing::field::Empty, characters=tracing::field::Empty, charset=tracing::field::Empty, seed=tracing::field::Empty, min_length=tracing::field::Empty, max_length=tracing::field::Empty))]
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
            span.record("paths", tracing::field::debug(options.paths));
            span.record("characters", tracing::field::debug(options.characters));
            span.record(
                "charset",
                tracing::field::debug(options.charset.as_ref().map(|charset| charset.to_regex())),
            );
            span.record("seed", options.seed);
            span.record("min_length", options.min_length as u64);
            span.record("max_length", tracing::field::debug(options.max_length));
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

        let mut generation = Generation::new(self, limit, offset, options)?;

        match options.paths {
            PathOrder::Sweep => {
                // The ascending sweep walks each path's combinations with
                // cursors; the shuffled one has to index them through the
                // permutation, which a window over everything is.
                let window =
                    (options.characters == CharacterOrder::Shuffled).then_some(0..usize::MAX);
                generation.walk(self, window.as_ref(), None)?;
            }
            PathOrder::Interleave | PathOrder::Shuffled => {
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
/// every pass an interleaving generation makes over the automaton.
struct Generation<'a> {
    /// Number of transitions from each state to the nearest accept state;
    /// `usize::MAX` for the states that cannot reach one.
    distances: Vec<usize>,
    /// Length of the longest string generation may emit: what the automaton
    /// matches, capped at the options'
    /// [`max_length`](GenerationOptions::max_length).
    max_len: usize,
    /// Length of the shortest string generation may emit
    /// ([`GenerationOptions::min_length`]); the search still walks the
    /// shorter accepting paths, since they lead to long enough ones, it just
    /// does not emit them.
    min_len: usize,
    /// The characters each transition stands for, resolved once: the paths
    /// refer to them by index (see [`QueueItem::ranges`]).
    range_pool: Vec<CharRange>,
    /// Each transition condition's index into
    /// [`range_pool`](Self::range_pool), the charset already taken out. A
    /// condition the charset leaves nothing of holds `None`, which is what
    /// makes its transition impassable.
    range_ids: RangeIds<'a>,
    /// Seeds [`QueueItem::tie`] under [`PathOrder::Shuffled`]; `None` leaves
    /// every tie at 0 and the pop order to the deterministic fallback.
    shape_key: Option<u64>,
    emitter: Emitter,
}

/// Turns the paths the search pops into strings: the half of a [`Generation`]
/// the emission mutates, kept apart from the search data so that a borrow of
/// the range pool can live alongside it.
struct Emitter {
    limit: usize,
    offset: usize,
    /// Scrambles each path's combination indices for
    /// [`CharacterOrder::Shuffled`]; `None` walks them in ascending order.
    permuter: Option<Permuter>,
    strings: IndexSet<String, RandomState>,
    execution_profile: ExecutionProfile,
}

/// The accepting paths an interleave pass popped, in pop order, flattened so
/// that path `i` is `ids[starts[i]..starts[i + 1]]`. A pass that runs out of paths has
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
/// ranges, and each condition's index into it, `None` for the conditions the
/// charset leaves nothing of.
fn resolve_ranges<'a>(
    automaton: &'a FastAutomaton,
    charset: Option<&CharRange>,
) -> Result<(Vec<CharRange>, RangeIds<'a>), EngineError> {
    let mut range_pool: Vec<CharRange> = Vec::new();
    let mut range_ids: RangeIds = HashMap::with_capacity(automaton.transitions.len());

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
        options: &GenerationOptions,
    ) -> Result<Self, EngineError> {
        let (range_pool, range_ids) = resolve_ranges(automaton, options.charset())?;
        let distances = distances_to_accept(automaton, &range_ids);
        let (_, max) = automaton.length();
        let execution_profile = ExecutionProfile::get();

        // The options' length bounds: without a max, a deep offset into a
        // looping language would page into arbitrarily long strings.
        let max_len =
            (max.unwrap_or(u32::MAX) as usize).min(options.max_length().unwrap_or(usize::MAX));

        Ok(Generation {
            distances,
            max_len,
            min_len: options.min_length(),
            range_pool,
            range_ids,
            shape_key: (options.paths == PathOrder::Shuffled)
                .then(|| mix(options.seed ^ SHAPE_KEY_SALT)),
            emitter: Emitter {
                limit,
                offset,
                permuter: (options.characters == CharacterOrder::Shuffled)
                    .then(|| Permuter::new(options.seed)),
                strings: IndexSet::with_capacity_and_hasher(
                    limit.min(STRINGS_CAPACITY_LIMIT),
                    RandomState::default(),
                ),
                execution_profile,
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
            tie: 0,
            depth: 0,
            state: start_state,
            ranges: vec![],
        });

        while let Some(QueueItem {
            score: _,
            tie,
            depth: current_depth,
            state,
            ranges,
        }) = q.pop()
        {
            self.emitter.execution_profile.assert_not_timed_out()?;

            // A path shorter than `min_len` is walked, since its extensions
            // are long enough, but never emitted, recorded, or counted.
            if automaton.is_accepted(state) && current_depth >= self.min_len {
                if let Some(cache) = cache.as_deref_mut() {
                    cache.record(&ranges);
                }

                let resolved = resolve(&self.range_pool, &ranges);
                covered = covered.saturating_add(match window {
                    Some(window) => self.emitter.emit_window(&resolved, &ranges, window)?,
                    None => self.emitter.emit_all(&resolved)?,
                });

                if self.emitter.is_full() {
                    break;
                }
            }

            if current_depth >= self.max_len {
                continue;
            }

            self.expand(automaton, &mut q, current_depth + 1, state, tie, ranges);
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
        tie: u64,
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
                    tie: self.tie(tie, range_id, to_state),
                    depth: next_depth,
                    state: to_state,
                    ranges: new_ranges,
                });
            }

            let tie = self.tie(tie, last_id, last_state);
            ranges.push(last_id);
            q.push(QueueItem {
                score: next_depth + self.distances[last_state], // A* Score Formula
                tie,
                depth: next_depth,
                state: last_state,
                ranges,
            });
        }
    }

    /// The tie of a path extended by `range_id` into `to_state`: the parent's
    /// tie folded with a seeded hash of the transition, so equal-score paths
    /// pop in an order the seed draws, the cascade documented on
    /// [`PathOrder::Shuffled`]. Without a [`shape_key`](Self::shape_key) it is
    /// 0, falling through to the deterministic tie-breaks.
    fn tie(&self, parent: u64, range_id: u32, to_state: State) -> u64 {
        match self.shape_key {
            Some(key) => mix(parent ^ mix(key ^ ((range_id as u64) << 32) ^ to_state as u64)),
            None => 0,
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
            covered = covered.saturating_add(self.emitter.emit_window(&resolved, path, window)?);

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
    /// in the ascending order [`emit_all`](Self::emit_all) walks them in, or,
    /// with a [`permuter`](Self::permuter), the path's own seeded permutation
    /// of it (`path` holds the transition ids the ranges were resolved from).
    /// Returns how many of them the window covered, the ones `offset` skipped
    /// included.
    fn emit_window(
        &mut self,
        ranges: &[&CharRange],
        path: &[u32],
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

        let tweak = self
            .permuter
            .as_ref()
            .map_or(0, |permuter| permuter.path_tweak(path));

        for index in first..window.end.min(bound) {
            self.execution_profile.assert_not_timed_out()?;

            // The permutation reorders `[0, bound)` onto itself, so the
            // window still covers `covered` distinct combinations, just not
            // the ascending ones.
            let combination = match &self.permuter {
                Some(permuter) => permuter.permute(index as u128, bound as u128, tweak),
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

/// A seeded family of permutations of `[0, bound)` for any `bound`, one per
/// `tweak`, evaluated point by point: a tweaked Feistel network over the
/// smallest even-width binary domain holding `bound`, cycle-walked back into
/// it. Being a bijection (at any fixed tweak) is what keeps
/// [`CharacterOrder::Shuffled`] strings distinct and `offset` exact, exactly
/// like the ascending order it stands in for; being a fixed function of the
/// seed is what lets a page be generated without materializing (or even
/// visiting) the combinations around it.
struct Permuter {
    keys: [u64; 4],
    tweak_key: u64,
}

impl Permuter {
    fn new(seed: u64) -> Self {
        // SplitMix64: one independent-looking round key per Feistel round,
        // nearby seeds included.
        let mut state = seed;
        let mut keys = [0u64; 4];
        for key in &mut keys {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            *key = mix(state);
        }
        let tweak_key = mix(state.wrapping_add(0x9E37_79B9_7F4A_7C15));
        Permuter { keys, tweak_key }
    }

    /// A path's tweak: its transition ids folded through the seeded hash,
    /// picking the path's own permutation out of the family. Without it,
    /// same-shape alternation branches (`[a-z]{4}|[A-Z]{4}`) would emit the
    /// same combination indices and mirror each other's strings.
    fn path_tweak(&self, path: &[u32]) -> u64 {
        path.iter()
            .fold(self.tweak_key, |acc, &id| mix(acc ^ id as u64))
    }

    /// Where the `tweak`'s permutation of `[0, bound)` sends `index`; `index`
    /// must be below `bound`.
    fn permute(&self, index: u128, bound: u128, tweak: u64) -> u128 {
        debug_assert!(index < bound);
        if bound <= 1 {
            return index;
        }

        let bits = 128 - (bound - 1).leading_zeros();
        let half = bits.div_ceil(2);
        let mask = (1u128 << half) - 1;

        // CYCLE-WALKING: encrypt until the value falls back under `bound`.
        // The walk follows the cycle `index` itself sits on, so it terminates
        // (on `index`, at worst), and distinct indices never land on the same
        // value, whether they sit on distinct cycles or ahead of one another
        // on the same one. The domain is under `4 * bound`, so it takes a few steps.
        let mut value = index;
        loop {
            value = self.encrypt(value, half, mask, tweak);
            if value < bound {
                return value;
            }
        }
    }

    /// One pass of the 4-round Feistel network: a bijection over
    /// `[0, 2^(2 * half))` for any fixed `tweak`, `half` at most 64.
    fn encrypt(&self, value: u128, half: u32, mask: u128, tweak: u64) -> u128 {
        let mut left = value >> half;
        let mut right = value & mask;
        for &key in &self.keys {
            let round = (mix(right as u64 ^ key ^ tweak) as u128) & mask;
            (left, right) = (right, left ^ round);
        }
        (left << half) | right
    }
}

/// SplitMix64's finalizer: the avalanche behind the [`Permuter`]'s round keys
/// and round function.
fn mix(value: u64) -> u64 {
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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
    use super::{CharacterOrder, GenerationOptions, PathOrder, Permuter, RangeCursor, char_at};
    use crate::CharRange;
    use crate::cardinality::Cardinality;
    use crate::{fast_automaton::FastAutomaton, regex::RegularExpression};
    use regex::Regex;
    use regex_charclass::{CharacterClass, char::Char, irange::range::AnyRange};

    const AXES: [(PathOrder, CharacterOrder); 6] = [
        (PathOrder::Sweep, CharacterOrder::Ascending),
        (PathOrder::Sweep, CharacterOrder::Shuffled),
        (PathOrder::Interleave, CharacterOrder::Ascending),
        (PathOrder::Interleave, CharacterOrder::Shuffled),
        (PathOrder::Shuffled, CharacterOrder::Ascending),
        (PathOrder::Shuffled, CharacterOrder::Shuffled),
    ];

    /// Every set of options the generation tests run through: all four axis
    /// combinations, each of them once unrestricted and once over printable
    /// ASCII, which is narrow enough to close off transitions in most of the
    /// patterns.
    fn all_options() -> Vec<GenerationOptions> {
        AXES.into_iter()
            .flat_map(|axes| {
                [
                    GenerationOptions::from(axes),
                    GenerationOptions::from(axes).with_charset(printable_ascii()),
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
        for axes in AXES {
            println!("{:?}", automaton.generate_strings(30, 0, axes).unwrap());
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
        for axes in AXES {
            let strings = automaton.generate_strings(2, 0, axes).unwrap();
            assert_eq!(2, strings.len());

            let strings = automaton.generate_strings(2, 2, axes).unwrap();
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

    /// The interleave order exists so that a pattern's *shapes* get covered:
    /// what the sweep spends a million strings on (one path, every character
    /// of its last range) has to fit in a handful of them.
    #[test]
    fn test_generate_strings_interleave_covers_the_whole_pattern() {
        let automaton = automaton_of(".*abc.*").determinize().unwrap().into_owned();

        let swept = automaton.generate_strings(20, 0, PathOrder::Sweep).unwrap();
        assert!(
            swept.iter().all(|s| s.starts_with("abc")),
            "the sweep stays on the first path it finds: {swept:?}"
        );

        let interleaved = automaton
            .generate_strings(20, 0, PathOrder::Interleave)
            .unwrap();
        assert!(
            interleaved.iter().any(|s| !s.starts_with("abc")),
            "the interleave order has to reach the strings with a prefix before `abc`: {interleaved:?}"
        );
        assert!(
            interleaved.iter().any(|s| !s.ends_with("abc")),
            "the interleave order has to reach the strings with a suffix after `abc`: {interleaved:?}"
        );
        assert!(
            interleaved.iter().any(|s| s.len() > 5),
            "the interleave order has to reach longer strings too: {interleaved:?}"
        );
    }

    /// Interleaving still enumerates a finite language in full, given the
    /// room: the passes dig deeper into every path until nothing is left to
    /// cover.
    #[test]
    fn test_generate_strings_interleave_is_exhaustive_in_the_limit() {
        let automaton = automaton_of("[a-z][0-9]");

        let mut interleaved = automaton
            .generate_strings(1000, 0, PathOrder::Interleave)
            .unwrap();
        interleaved.sort();

        let mut expected: Vec<String> = ('a'..='z')
            .flat_map(|letter| ('0'..='9').map(move |digit| format!("{letter}{digit}")))
            .collect();
        expected.sort();

        assert_eq!(expected, interleaved);
    }

    /// The path order chooses which strings come first, never the characters
    /// they are made of: an interleaved string reaches for the same low end
    /// of each range the sweep starts from, and a charset is how specific
    /// characters are asked for.
    #[test]
    fn test_generate_strings_interleave_uses_the_same_characters_as_sweep() {
        let automaton = automaton_of(".{3}");

        let interleaved = automaton
            .generate_strings(1, 0, PathOrder::Interleave)
            .unwrap();
        let swept = automaton.generate_strings(1, 0, PathOrder::Sweep).unwrap();
        assert_eq!(swept, interleaved);

        let lowercase = CharRange::new_from_range_char('a'..='z');
        let options = GenerationOptions::from(PathOrder::Interleave).with_charset(lowercase);
        assert_eq!(
            vec!["aaa".to_string()],
            automaton.generate_strings(1, 0, options).unwrap()
        );
    }

    /// A single-path language has only one shape, so there is nothing for the
    /// interleave order to spread over: within a path the characters come in
    /// the lexicographic order the sweep walks.
    #[test]
    fn test_generate_strings_interleave_matches_sweep_on_a_single_path() {
        let automaton = automaton_of("[a-z]{2}");

        let swept = automaton.generate_strings(10, 0, PathOrder::Sweep).unwrap();
        let interleaved = automaton
            .generate_strings(10, 0, PathOrder::Interleave)
            .unwrap();

        assert_eq!(swept, interleaved);
    }

    /// The strongest form of "the axes choose which strings come first, never
    /// what is generated": on a finite language, every axis combination, at
    /// any seed, enumerates exactly the same set, including through
    /// nondeterministic automata, multi-interval charsets, and ranges
    /// straddling the surrogate hole.
    #[test]
    fn test_generate_strings_all_axes_agree_as_sets() {
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

                let options = |axes, seed| {
                    let options = GenerationOptions::from(axes).with_seed(seed);
                    match &charset {
                        Some(charset) => options.with_charset(charset.clone()),
                        None => options,
                    }
                };

                let mut baseline = automaton
                    .generate_strings(100_000, 0, options(AXES[0], 0))
                    .unwrap();
                baseline.sort();

                for axes in &AXES[1..] {
                    for seed in [0, 1, 42] {
                        let mut strings = automaton
                            .generate_strings(100_000, 0, options(*axes, seed))
                            .unwrap();
                        strings.sort();

                        assert_eq!(
                            baseline, strings,
                            "{pattern:?} {axes:?} seed {seed} (determinized: {determinize})"
                        );
                    }
                }
            }
        }
    }

    /// Pages stay consistent at every chunk size under every axis
    /// combination, not only at the window boundaries: an offset landing
    /// mid-window or mid-path continues exactly where the previous page
    /// stopped, never repeating a string.
    #[test]
    fn test_generate_strings_pages_at_any_boundary() {
        for pattern in ["(a|bc){0,3}", "[a-c]{1,3}", "(x|yy)(0|11)?"] {
            // Deterministic and minimal, so pages are exactly disjoint.
            let mut automaton = automaton_of(pattern).determinize().unwrap().into_owned();
            automaton.minimize().unwrap();

            for axes in AXES {
                let options = GenerationOptions::from(axes).with_seed(7);
                assert_pages_match_bulk(&automaton, &options, 1..=7);
            }
        }
    }

    /// Rebuilds a 60-string bulk page chunk by chunk at each of the given
    /// chunk sizes, and asserts every rebuild matches the bulk exactly.
    fn assert_pages_match_bulk(
        automaton: &FastAutomaton,
        options: &GenerationOptions,
        chunks: impl IntoIterator<Item = usize>,
    ) {
        let bulk = automaton.generate_strings(60, 0, options.clone()).unwrap();

        for chunk in chunks {
            let mut paged = vec![];
            loop {
                let page = automaton
                    .generate_strings(chunk, paged.len(), options.clone())
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

            assert_eq!(bulk, paged, "{options:?} at chunk size {chunk}");
        }
    }

    /// A path holding more combinations than `u128` fits still emits the
    /// ascending sequence: the decode consumes the index from the last
    /// position, so the positions it never reaches keep their range's first
    /// character.
    #[test]
    fn test_generate_strings_interleave_orders_huge_paths() {
        let automaton = automaton_of(".{40}");

        let interleaved = automaton
            .generate_strings(3, 0, PathOrder::Interleave)
            .unwrap();

        assert_eq!(
            vec![
                "\u{0}".repeat(40),
                format!("{}\u{1}", "\u{0}".repeat(39)),
                format!("{}\u{2}", "\u{0}".repeat(39)),
            ],
            interleaved
        );
    }

    /// Shuffling keeps the interleave order's shape-first coverage, and
    /// actually looks random: the strings of a wide range are spread over it,
    /// not clustered at its low end the way ascending generation starts.
    #[test]
    fn test_generate_strings_shuffled_covers_shapes_and_spreads_characters() {
        let automaton = automaton_of(".*abc.*").determinize().unwrap().into_owned();

        let strings = automaton
            .generate_strings(20, 0, (PathOrder::Shuffled, CharacterOrder::Shuffled))
            .unwrap();
        assert!(
            strings.iter().any(|s| !s.starts_with("abc"))
                && strings.iter().any(|s| !s.ends_with("abc")),
            "shuffled paths still have to cover every shape of the pattern: {strings:?}"
        );

        let strings = automaton_of("[a-z]{20}")
            .generate_strings(5, 0, CharacterOrder::Shuffled)
            .unwrap();
        assert!(
            strings
                .iter()
                .any(|s| s.chars().filter(|&ch| ch > 'm').count() > 5),
            "the strings have to reach past the low end of the range: {strings:?}"
        );
    }

    /// The seed is fixed, so shuffled generation is reproducible; a different
    /// seed draws a different sequence of strings from the same pattern.
    #[test]
    fn test_generate_strings_shuffled_is_seeded() {
        let automaton = automaton_of("[a-z]{8}");
        let options = |seed| GenerationOptions::from(CharacterOrder::Shuffled).with_seed(seed);

        let strings = automaton.generate_strings(10, 0, options(42)).unwrap();
        assert_eq!(
            strings,
            automaton.generate_strings(10, 0, options(42)).unwrap()
        );
        assert_ne!(
            strings,
            automaton.generate_strings(10, 0, options(43)).unwrap()
        );
    }

    /// Shuffled paths draw the *shapes* by seed, independently of the
    /// characters: over ascending characters, which same-length paths a small
    /// `limit` reaches depends on the seed instead of always being the same
    /// ones, and the whole language still comes out, whatever the seed.
    #[test]
    fn test_generate_strings_shuffled_paths_draw_shapes_by_seed() {
        let automaton = automaton_of("(aa|bb|cc|dd|ee|ff|gg|hh)");
        let options = |seed| GenerationOptions::from(PathOrder::Shuffled).with_seed(seed);

        let first = automaton.generate_strings(3, 0, options(1)).unwrap();
        assert_eq!(first, automaton.generate_strings(3, 0, options(1)).unwrap());
        assert!(
            (2..20).any(|seed| automaton.generate_strings(3, 0, options(seed)).unwrap() != first),
            "no seed reordered the shapes: {first:?}"
        );

        let mut all = automaton.generate_strings(100, 0, options(1)).unwrap();
        all.sort();
        assert_eq!(vec!["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh"], all);
    }

    /// The axes stay independent the other way around too: shuffling the
    /// characters leaves the shape order alone. On single-combination paths
    /// there is nothing for the character permutation to reorder, so
    /// interleaved generation comes out identical with and without it.
    #[test]
    fn test_generate_strings_shuffled_characters_leave_the_shape_order_alone() {
        let automaton = automaton_of("(aa|bb|cc|dd|ee|ff|gg|hh)");

        let ascending = automaton
            .generate_strings(8, 0, PathOrder::Interleave)
            .unwrap();
        let shuffled = automaton
            .generate_strings(8, 0, (PathOrder::Interleave, CharacterOrder::Shuffled))
            .unwrap();

        assert_eq!(ascending, shuffled);
    }

    /// Shuffled paths over ascending characters: a seed-drawn order of
    /// shapes, each shown as its smallest witness.
    #[test]
    fn test_generate_strings_shuffled_paths_keep_ascending_witnesses() {
        let automaton = automaton_of("(aa|bb|cc)[0-9]");

        for seed in [0, 1, 42] {
            let options = GenerationOptions::from(PathOrder::Shuffled).with_seed(seed);
            let mut strings = automaton.generate_strings(3, 0, options).unwrap();

            // Whatever order the seed drew the three shapes in, the first
            // string of each is the low end of its ranges.
            strings.sort();
            assert_eq!(vec!["aa0", "bb0", "cc0"], strings, "seed {seed}");
        }
    }

    /// `with_max_length` bounds the generated string length: a deep offset
    /// into `.*` pages within the bound instead of into arbitrarily long
    /// strings, and comes back quickly, whatever the axes.
    #[test]
    fn test_generate_strings_max_length_bounds_deep_offsets() {
        let automaton = automaton_of(".*");

        for axes in AXES {
            let options = GenerationOptions::from(axes).with_max_length(5);
            let strings = automaton
                .generate_strings(5, 1_000_000_000, options)
                .unwrap();

            assert!(!strings.is_empty(), "{axes:?}");
            for string in &strings {
                assert!(string.chars().count() <= 5, "{axes:?}: {string:?}");
            }
        }
    }

    /// The bounded language is a well-defined finite set: `(ab)*` under a
    /// max of 5 stops at `abab`, and the page past it is empty, not endless.
    #[test]
    fn test_generate_strings_max_length_truncates_the_language() {
        let automaton = automaton_of("(ab)*");

        for axes in AXES {
            let options = GenerationOptions::from(axes).with_max_length(5);
            let mut strings = automaton.generate_strings(100, 0, options.clone()).unwrap();
            strings.sort();
            assert_eq!(vec!["", "ab", "abab"], strings, "{axes:?}");

            let past_the_end = automaton.generate_strings(10, 3, options).unwrap();
            assert!(past_the_end.is_empty(), "{axes:?}: {past_the_end:?}");
        }
    }

    /// The bound is exactly what it says, a length: a finite language keeps
    /// every string within it and loses every string past it.
    #[test]
    fn test_generate_strings_max_length_applies_to_finite_languages_too() {
        let automaton = automaton_of("(ab){1,2}");

        for axes in AXES {
            let mut strings = automaton
                .generate_strings(10, 0, GenerationOptions::from(axes).with_max_length(4))
                .unwrap();
            strings.sort();
            assert_eq!(vec!["ab", "abab"], strings, "{axes:?}");

            assert_eq!(
                vec!["ab".to_string()],
                automaton
                    .generate_strings(10, 0, GenerationOptions::from(axes).with_max_length(3))
                    .unwrap(),
                "{axes:?}"
            );
        }
    }

    /// `with_min_length` leaves the short strings out: the enumeration
    /// starts at the bound, and an empty band generates nothing.
    #[test]
    fn test_generate_strings_min_length_skips_short_strings() {
        let automaton = automaton_of("(ab)*");

        for axes in AXES {
            let options = GenerationOptions::from(axes)
                .with_min_length(3)
                .with_max_length(8);
            let mut strings = automaton.generate_strings(100, 0, options).unwrap();
            strings.sort();
            assert_eq!(vec!["abab", "ababab", "abababab"], strings, "{axes:?}");

            let empty_band = GenerationOptions::from(axes)
                .with_min_length(5)
                .with_max_length(3);
            assert!(
                automaton
                    .generate_strings(10, 0, empty_band)
                    .unwrap()
                    .is_empty(),
                "{axes:?}"
            );
        }
    }

    /// Length bounds compose with paging: `offset` never counts the strings
    /// outside the band, so pages of the bounded language stay consistent at
    /// any chunk size.
    #[test]
    fn test_generate_strings_length_bounds_page_consistently() {
        // Deterministic and minimal, so pages are exactly disjoint.
        let mut automaton = automaton_of("(a|bc){0,3}")
            .determinize()
            .unwrap()
            .into_owned();
        automaton.minimize().unwrap();

        for axes in AXES {
            let options = GenerationOptions::from(axes)
                .with_seed(7)
                .with_min_length(2)
                .with_max_length(4);

            let bulk = automaton.generate_strings(60, 0, options.clone()).unwrap();
            assert!(
                bulk.iter().all(|s| (2..=4).contains(&s.chars().count())),
                "{axes:?}: {bulk:?}"
            );

            assert_pages_match_bulk(&automaton, &options, [1, 3]);
        }
    }

    /// The permuter maps `[0, bound)` onto itself one-to-one for any bound
    /// and tweak, which is what "distinct strings across offsets" rests on.
    #[test]
    fn test_permuter_is_a_bijection() {
        for seed in [0, 1, 42] {
            let permuter = Permuter::new(seed);
            for bound in [1u128, 2, 3, 7, 26, 100, 4096, 100_003] {
                for tweak in [0, permuter.path_tweak(&[3, 1, 4])] {
                    let mut images: Vec<u128> = (0..bound)
                        .map(|index| permuter.permute(index, bound, tweak))
                        .collect();
                    images.sort_unstable();

                    assert!(
                        images.iter().enumerate().all(|(i, &v)| i as u128 == v),
                        "seed {seed}, bound {bound}, tweak {tweak}"
                    );
                }
            }
        }
    }

    /// Every path draws its own permutation: alternation branches of the same
    /// shape emit unrelated strings instead of mirroring each other's
    /// combination indices ("knyn" next to "KNYN").
    #[test]
    fn test_generate_strings_shuffled_decorrelates_same_shape_branches() {
        let automaton = automaton_of("([a-z]{6}|[A-Z]{6})");

        for seed in [0, 1, 42] {
            let options = GenerationOptions::from(CharacterOrder::Shuffled)
                .with_paths(PathOrder::Interleave)
                .with_seed(seed);
            let strings = automaton.generate_strings(2, 0, options).unwrap();

            let [first, second] = strings.as_slice() else {
                panic!("expected one string per branch: {strings:?}");
            };
            assert_ne!(
                first.to_lowercase(),
                second.to_lowercase(),
                "seed {seed}: the branches drew the same combination"
            );
        }
    }

    /// A charset rules out whole paths, not single characters: a path that
    /// needs a ruled-out character is dropped even when it only needs it
    /// several transitions in, and the strings around it still come out.
    #[test]
    fn test_generate_strings_charset_drops_the_paths_it_closes() {
        let automaton = automaton_of("(a[0-9]b|xyz)");
        let letters = CharRange::new_from_range_char('a'..='z');

        for axes in AXES {
            let options = GenerationOptions::from(axes).with_charset(letters.clone());

            assert_eq!(
                vec!["xyz".to_string()],
                automaton.generate_strings(10, 0, options).unwrap(),
                "{axes:?}"
            );
        }
    }

    /// A charset that leaves the pattern nothing is an empty page, not an
    /// error: the search prunes the start state like any other dead end.
    #[test]
    fn test_generate_strings_charset_can_leave_nothing() {
        let automaton = automaton_of("[0-9]+");
        let letters = CharRange::new_from_range_char('a'..='z');

        for axes in AXES {
            let options = GenerationOptions::from(axes).with_charset(letters.clone());

            assert!(
                automaton
                    .generate_strings(10, 0, options)
                    .unwrap()
                    .is_empty(),
                "{axes:?}"
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

        for axes in AXES {
            let options = GenerationOptions::from(axes).with_charset(vowels.clone());

            let mut strings = automaton.generate_strings(10, 0, options).unwrap();
            strings.sort();

            assert_eq!(vec!["aa", "ae", "ea", "ee"], strings, "{axes:?}");
        }
    }

    /// The offset steps over whole subtrees at once: a page from deep inside a
    /// large language comes back without walking everything before it.
    #[test]
    fn test_generate_strings_offset_reaches_deep_pages() {
        let automaton = automaton_of("[a-z]{5}");
        let total = 26usize.pow(5);

        let strings = automaton
            .generate_strings(2, total - 2, PathOrder::Sweep)
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

        for axes in AXES {
            let mut strings = automaton.generate_strings(usize::MAX, 0, axes).unwrap();
            strings.sort();

            assert_eq!(vec!["aa", "ab", "ba", "bb"], strings, "{axes:?}");
        }
    }

    /// Combination emission walks an explicit stack, not the call stack: a
    /// path tens of thousands of transitions long emits without overflowing.
    #[test]
    fn test_generate_strings_very_long_string() {
        let automaton = automaton_of("[ab]{20000}");

        for axes in AXES {
            let strings = automaton.generate_strings(2, 0, axes).unwrap();
            assert_eq!(2, strings.len(), "{axes:?}");
            for string in &strings {
                assert_eq!(20_000, string.len(), "{axes:?}");
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
