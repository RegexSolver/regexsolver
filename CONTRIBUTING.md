# Contributing to regexsolver

Bug reports, questions and pull requests are all welcome.

## Reporting a bug

Open an issue with the smallest pattern that reproduces the problem, the operation you called, what you expected and what you got. If the input is an automaton rather than a pattern, `to_dot()` prints it in a form that can be pasted into an issue.

For a result that is wrong rather than an error or a hang, please name a string you believe is on the wrong side of the answer. A single one is enough to turn the report into a test.

## Before opening a pull request

Run what CI runs:

```sh
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo clippy --locked --no-default-features --all-targets -- -D warnings
cargo test --locked --all-targets
cargo test --locked --no-default-features --all-targets
cargo test --locked --doc
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
```

Please also:

- **Add a test.** Unit tests live next to the code; tests that drive the crate the way a dependent would live in `tests/`, with `tests/public_api.rs` as the place for the public surface itself.
- **Prefer a property test for anything about languages.** `tests/proptest_strategies.rs` generates random DFAs, NFAs and regular expressions with a documented coverage guarantee, and already expresses most of the invariants worth checking: an operation agreeing with a brute-force oracle over a small alphabet, a conversion round-tripping, a result agreeing with the `regex` crate. Most changes only need a new case.
- **Document new public items.** `missing_docs` is warned on, and doc examples are run as tests.
- **Keep the MSRV.** The crate builds with Rust 1.88 and later, verified in CI. Raising it is a breaking change.
- **Update `CHANGELOG.md`** under an `## [Unreleased]` heading.

## Nothing may hang, and nothing may panic

The crate is meant to run untrusted patterns, and that shapes what an acceptable implementation is.

Every loop that can grow with the input has to be bounded by the thread-local `ExecutionProfile`: call `assert_not_timed_out()` inside it, and `assert_max_number_of_states()` wherever states are created, propagating the `EngineError` each returns. Determinization and automaton-to-regex conversion are where the growth is worse than polynomial, so they are where this matters most.

Recursion over a `RegularExpression` has to account for the tree being hand-built rather than parsed: the variants are public and freely constructible, so a caller can nest them past what the stack holds. `MAX_NESTING_DEPTH` and the check in `to_automaton` are the guard; `Display` is iterative for the same reason.

A malformed input is an `EngineError`, never a panic. `unwrap()` in library code is acceptable only where the invariant is local and obvious. Where it is an internal invariant that a caller cannot violate, prefer `debug_assert!` plus a defined fallback, the way `is_match` handles a condition that does not belong to its spanning set.

## The Unicode version has to match `regex-syntax`

Patterns are parsed by `regex-syntax`, whose Unicode tables are compiled in, and character classes are named back into `\p{...}` form by `regex-charclass`, whose tables are selected by a `ucd-*` feature. The two have to be the same UCD version, or a class survives the round trip as a raw list of ranges instead of its name. `Cargo.toml` pins `regex-charclass`'s `ucd-16` for this reason, and `tests/public_api.rs` fails if they drift. Bumping `regex-syntax` past a Unicode release means bumping that feature in the same commit.

## Benchmarks

`cargo bench` measures the operation families over named inputs: realistic patterns at three sizes, plus the `(a|b)*a(a|b){N}` family whose minimal DFA has 2^N states, which is the worst case of subset construction. Numbers move a lot with machine load, so compare runs on an otherwise idle machine before claiming a change is faster.

`cargo test --test state_elimination_quality -- --ignored --nocapture` reports how large the patterns produced by automaton-to-regex conversion are over a corpus, which is the number to watch when changing a state-elimination heuristic.

## Releasing

Publishing is automated. Push a `v*` tag matching the version in `Cargo.toml`, and the release workflow checks formatting, clippy, tests, docs, the MSRV and a dry-run publish before it uploads to crates.io.
