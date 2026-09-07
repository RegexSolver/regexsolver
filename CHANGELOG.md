# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-09-06

A maintenance release covering dependencies and packaging. The public API is unchanged, and no operation returns a different result.

### Changed
- Updated `regex-charclass` to 1.2 and `regex-syntax` to 0.8.11. The `regex-charclass` `ucd-16` feature is now named explicitly, so its Unicode tables match the ones `regex-syntax` parses with. If the two disagree, a character class such as `\p{Greek}` is printed back as a raw list of ranges instead of its name.
- The crate is now `#![forbid(unsafe_code)]`. It already contained no `unsafe`.
- The README is an overview rather than a reference. The full semantics list, the Unicode version in force, the hand-built-automaton and AST walkthroughs, and two of the three `ExecutionProfile` examples now live in the crate documentation, which already carried most of them.
- `Cargo.toml` publishes from an `include` allowlist rather than an `exclude` denylist, and declares the docs.rs metadata that builds the documentation with all features enabled.

### Added
- `CONTRIBUTING.md` and `SECURITY.md`.
- `tests/public_api.rs`, covering the crate as a dependent sees it.
- The README's `rust` blocks are compiled and run as doctests, replacing `tests/readme_examples.rs`, which duplicated them by hand.
- README sections for the minimum supported Rust version and contributing.

### Fixed
- The README's automaton image pointed at the `v1` branch; it now points at `main`.

## [1.0.0] - 2026-08-08

This is a major redesign of the public API around the `Term` enum (wrapping either a `RegularExpression` or a `FastAutomaton`), which dispatches each operation to the cheaper representation when possible. See the crate-level docs for the architecture. Almost the entire public surface changed; highlights below.

### Added
- New `Term` constructors and conversions: `new_empty`, `new_total`, `new_empty_string`, `from_pattern`, `from_regex(RegularExpression)`, `from_automaton(FastAutomaton)`, plus `From<RegularExpression>`, `From<FastAutomaton>`, `FromStr`, `Display`, and `Default` (= `new_empty`).
- New `Term` operations: `concat`, `complement`, `determinize`, `minimize`, `matches`, `is_deterministic`, `is_minimal`, `is_finite`, `to_pattern`, and `iter_strings` (lazy `StringGenerator` iterator).
- `RegularExpression::MAX_NESTING_DEPTH` and a depth check in `to_automaton` that returns the new `EngineError::RegexTooDeeplyNested` for hand-built trees nested past the limit, instead of overflowing the stack.
- `to_automaton` returns `Cow` to avoid unnecessary cloning; `to_regex` returns `Result<Cow<…>, EngineError>` (see Changed: it is now fallible).
- `FastAutomaton` gained corresponding low-level constructors/operations (`new_empty`, `new_total`, `new_empty_string`, `determinize`, `minimize` using Hopcroft's algorithm, `is_minimal`, `unaccept`, `print_dot`, `try_add_transition`) and inspection helpers (`states`, `direct_states`, `transitions_from`, `transitions_to_vec`, `has_transition`, ...).
- New `EngineError` variants: `InvalidRepetitionBounds`, `IncompatibleSpanningSet`, `DeterministicAutomatonRequired`, `UnsupportedRegexFeature`, `RegexTooDeeplyNested`; the enum is now `#[non_exhaustive]`.
- `tracing` instrumentation on the core `Term`, `FastAutomaton`, and `RegularExpression` operations (concat, union, intersection, difference, complement, repeat, determinize, minimize, equivalence/subset checks, string generation, conversions). No-op unless a `tracing` subscriber is installed.
- Parallel (Rayon-backed) variants of union/intersection for >3 operands, gated behind the default-on `parallel` feature, with sequential fallbacks for `--no-default-features`.
- `regex_charclass` is re-exported at the crate root, so automata can be built by hand (`Char`, `CharRange`) without adding a separately version-matched dependency.
- `NoHashHasher`, the crate-owned no-op hasher behind the `IntSet` state-id sets (replaces the `nohash-hasher` dependency, keeping 0.x types out of the public API).
- `EngineError` implements `Clone`; `StringGenerator` implements `Debug` and `FusedIterator`; `ExecutionProfileBuilder` implements `Debug` and `Clone`.
- `PathOrder` and `CharacterOrder`, the two independent axes `generate_strings`/`iter_strings` enumerate a language along. `PathOrder::Sweep` is the previous behaviour: shortest strings first, one path expanded in full before the next. `PathOrder::Interleave` covers every shape the automaton holds before asking any of them for a second string, so `.*abc.*` yields `abc`, `abc\u{0}`, `\u{0}abc`, ... instead of a million variations of `abc\u{0}`. `PathOrder::Shuffled` interleaves and additionally visits same-length shapes in a seed-drawn order. `CharacterOrder::Ascending` (the default) expands each position from the low end of its character range; `CharacterOrder::Shuffled` draws each path's combinations through a seeded permutation instead, so `[a-z]{8}` yields something like `sjtwsive` rather than `aaaaaaaa` (the exact shuffled sequence is implementation-defined). Every combination of the axes enumerates the same strings, stays deterministic (the seed defaults to 0), and pages with `offset` the same way.
- `GenerationOptions`, what `generate_strings`/`iter_strings` may generate: the two axes, the seed behind the `Shuffled` modes (`with_seed(u64)`), an optional charset (`with_charset(CharRange)`) and optional length bounds (`with_min_length`/`with_max_length`). A path needing a ruled-out character is dropped whole rather than shortened, and `offset` never counts the strings outside the length band. Without a max, a deep `offset` into a looping language like `.*` pages into arbitrarily long strings.

### Changed
- `Term::to_regex`/`to_pattern` and `FastAutomaton::to_regex` are now fallible (`Result<_, EngineError>`) and honor the `ExecutionProfile` timeout, since state elimination can grow super-polynomially on adversarial automata. `Term`'s `Display` therefore renders a pattern only for regex-backed terms; automaton-backed terms display as Graphviz DOT (use `to_pattern` for a parseable pattern).
- `ExecutionProfile` redesigned as an immutable, thread-local-aware config built via the new `ExecutionProfileBuilder`, governing execution timeouts, state-count limits, and an `implicit_determinization` toggle.
- `union`/`intersection`/`concat` now take `impl IntoIterator<Item = impl Borrow<Term>>` instead of `&[Term]`, so `&[a, b]`, `[&a, &b]`, and `Vec<Term>` all work without cloning.
- `repeat` now takes `impl RangeBounds<u32>` (e.g. `3..6`, `..=2`) instead of explicit min/max parameters.
- `generate_strings` now takes `(limit, offset, options)`: pagination instead of a single `count`, plus the `GenerationOptions` to generate under (the two enumeration axes, a seed, and optionally a charset and length bounds). A `PathOrder`, a `CharacterOrder`, or a `(PathOrder, CharacterOrder)` pair converts into options, so any of them can be passed on its own. `iter_strings` takes the same `options`.
- `is_empty`, `is_total`, and `is_empty_string` now return `Result<bool, EngineError>` instead of `bool`.
- `are_equivalent`/`is_subset_of` renamed to `equivalent`/`subset`.
- `subtraction` renamed to `difference`, kept single-operand by design.
- Renamed `FastAutomaton::as_dot` to `to_dot` (old printing `to_dot` is now `print_dot`), matching the crate's `to_*` convention for allocating conversions (`to_pattern`, `to_regex`, `to_automaton`, `to_range`).
- Renamed `get_*` accessors to drop the `get_` prefix, per the Rust API Guidelines' C-GETTER convention: `Term::get_length` to `length`, `Term::get_cardinality` to `cardinality`, `FastAutomaton::get_length` to `length`, `FastAutomaton::get_cardinality` to `cardinality`, `FastAutomaton::get_number_of_states` to `number_of_states`, `FastAutomaton::get_condition` to `condition`, `FastAutomaton::get_start_state` to `start_state`, `FastAutomaton::get_accept_states` to `accept_states`, `FastAutomaton::get_spanning_set` to `spanning_set`, `FastAutomaton::get_live_states` to `live_states`, `FastAutomaton::get_spanning_bases` to `spanning_bases`, `RegularExpression::get_length` to `length`, `RegularExpression::get_cardinality` to `cardinality`, `SpanningSet::get_spanning_ranges` to `spanning_ranges`, `SpanningSet::get_number_of_spanning_ranges` to `number_of_spanning_ranges`, `SpanningSet::get_spanning_range` to `spanning_range`, `SpanningSet::get_rest` to `rest`, `Condition::get_cardinality` to `cardinality`, `Condition::get_binary_representation` to `binary_representation`, `ConditionConverter::get_from_spanning_set`/`get_to_spanning_set` to `from_spanning_set`/`to_spanning_set`.
- Error messages follow the std convention (lowercase, no trailing punctuation) so they compose cleanly when wrapped by callers.
- Edition bumped to 2024 and `Cargo.toml` metadata (`description`, `categories`) updated.

### Removed
- The `serde` feature and all serialization, FAIR (base85) encoding, encryption, and compression support (`serde`, `ciborium`, `z85`, `aes-gcm-siv`, `sha2`, `flate2` dependencies).
- `Term::get_details` and the `Details` type.
- The `tokenizer` module.
- Unused `log`, `rand`, and `lazy_static` dependencies, and the `regex` crate dependency (now dev-only, used by integration tests).
- The `nohash-hasher` dependency (replaced by the crate-owned `NoHashHasher`; see Added).
- `EngineError` variants `AutomatonShouldBeDeterministic`, `TooMuchTerms`, `ConditionIndexOutOfBound`, `TokenError`, and the `is_server_error` method.
- The `max_number_of_terms` execution-profile limit (no longer enforced).

## Earlier releases

Releases prior to 1.0.0 (`v0.1.0` through `v0.3.1`) predate this changelog; see the [GitHub tags](https://github.com/RegexSolver/regexsolver/tags) and commit history for details.

[Unreleased]: https://github.com/RegexSolver/regexsolver/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/RegexSolver/regexsolver/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/RegexSolver/regexsolver/compare/v0.3.1...v1.0.0
