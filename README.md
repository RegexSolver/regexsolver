# RegexSolver

[![Crates.io Version](https://img.shields.io/crates/v/regexsolver)](https://crates.io/crates/regexsolver)
[![docs.rs](https://img.shields.io/docsrs/regexsolver)](https://docs.rs/regexsolver)
[![CI](https://github.com/RegexSolver/regexsolver/actions/workflows/rust.yml/badge.svg)](https://github.com/RegexSolver/regexsolver/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/crates/l/regexsolver)](LICENSE)
[![MSRV](https://img.shields.io/crates/msrv/regexsolver)](#minimum-supported-rust-version)

The `regex` crate tells you whether a *string* matches a pattern. **RegexSolver treats patterns as the sets of strings they match**, so you can intersect, subtract, compare, complement and enumerate them, and get the result back as a regex.

```rust
use regexsolver::{Term, fast_automaton::PathOrder};

let a: Term = "(ab|xy){2}".parse().unwrap();
let b: Term = ".*xy".parse().unwrap();

// Which strings match BOTH patterns? Get the answer as a regex:
let both = a.intersection([&b]).unwrap();
assert_eq!(both.to_pattern().unwrap(), "(ab|xy)xy");

// Test a concrete string against the result (matching is anchored):
assert!(both.matches("abxy").unwrap());

// ...and sample them:
assert_eq!(both.generate_strings(2, 0, PathOrder::Sweep).unwrap(), ["xyxy", "abxy"]);
```

## Use cases

- **Safe migrations** - `old_rule.subset(&new_rule)?`: does the new validation pattern accept *everything* the old one did?
- **Test-data generation** - `term.generate_strings(100, 0, (PathOrder::Shuffled, CharacterOrder::Shuffled))?`: realistic-looking strings matching any pattern, spread over the cases the pattern allows, reproducible by seed and pageable.
- **Rule analysis**: find shadowed or overlapping routes, firewall rules, and validators with `intersection` / `difference`.
- **Equivalence proofs** - `a.equivalent(&b)?`: show that two differently-written patterns match exactly the same strings.
- **Input bounds** - `term.length()`: the shortest and longest string a pattern can match, for instance to size a database column.
- **Pattern simplification**: every operation returns a `Term` you can turn back into a regex pattern with `to_pattern()`.

Under the hood, every pattern compiles to a finite automaton:

<p align="center"><img src="https://raw.githubusercontent.com/RegexSolver/regexsolver/refs/heads/main/assets/automaton.svg" alt="the minimal automaton for text containing abc then def"/></p>
<p align="center"><sub>The minimal automaton for <code>.*abc.*def.*</code> (text containing <code>abc</code> then <code>def</code>), rendered from this library's <code>to_dot()</code>.</sub></p>

## Install

```bash
cargo add regexsolver
```

The default `parallel` feature runs unions and intersections of more than 3 operands, and parts of the automaton-to-regex conversion, on [rayon](https://crates.io/crates/rayon). Turn it off for a leaner dependency tree on single-threaded workloads:

```toml
regexsolver = { version = "1", default-features = false }
```

Two runnable examples ship with the crate:

```bash
git clone https://github.com/RegexSolver/regexsolver && cd regexsolver

# How do two patterns relate? (equivalence, subsets, intersection, differences)
cargo run --example relate -- "(ab|xy){2}" ".*xy"

# Generate n sample strings matching a pattern
cargo run --example generate -- "[a-z]{2}[0-9]" 20
```

## Semantics

RegexSolver implements **pure regular languages**, which differs from typical regex engines in two ways:

- **Everything is anchored**: `abc` matches the string "abc", not "xabc" or "abcx". Patterns describe *whole strings*.
- **`.` matches any character**, including line feed (`\n`).

Patterns are parsed with [regex-syntax](https://docs.rs/regex-syntax/latest/regex_syntax/). Constructs that don't change the language as a set of strings, such as ungreedy markers and redundant `^`/`$` anchors, are accepted and ignored. Constructs a regular language can't express (backreferences, lookaround, inline flags, word boundaries) return an `EngineError` instead of silently changing what the pattern means. The [crate documentation](https://docs.rs/regexsolver) covers each case, and the Unicode version in force.

## API overview

[`Term`](https://docs.rs/regexsolver/latest/regexsolver/enum.Term.html) is the type you'll interact with: it wraps either a regular expression or an automaton and picks the best representation for each operation. The essentials:

| Method | Description |
| -------- | ------- |
| `Term::from_pattern(pattern)` | Parses a pattern into a term. |
| `intersection(&self, terms)` / `union(&self, terms)` | Set operations over any number of terms. |
| `difference(&self, other)` / `complement(&self)` | What `self` matches and `other` doesn't / everything `self` doesn't match. |
| `concat(&self, terms)` / `repeat(&self, range)` | Sequence and repeat languages; `range` is any Rust range expression (`2..=5`, `1..`, `..3`, ...). |
| `equivalent(&self, other)` / `subset(&self, other)` | Compare languages. |
| `is_empty()` / `is_total()` / `length()` / `cardinality()` | Analyze a language: matches nothing? everything? string lengths? how many strings? |
| `generate_strings(limit, offset, options)` | Enumerate matching strings eagerly (call `determinize()` or `minimize()` once first when paginating). |
| `iter_strings(options)` | Lazy iterator equivalent; computes the deterministic automaton once and yields strings in batches. `options.with_min_length(n)`/`.with_max_length(n)` confine the walk to a band of lengths; with a max, even an infinite language yields a finite iterator. |
| `to_pattern()` / `to_automaton()` / `to_regex()` | Convert back out. |

All fallible operations return `Result<_, EngineError>`.

`Term` is a thin layer over two public types you can drop down to whenever you need them:

- [`FastAutomaton`](https://docs.rs/regexsolver/latest/regexsolver/fast_automaton/struct.FastAutomaton.html) builds, manipulates and analyzes automata directly. Everything `Term` does is available on it, plus low-level construction (`new_state`, `accept`, `add_transition_from_range`, `add_epsilon_transition`, ...) and inspection (`states`, `transitions_from`, `to_dot`, ...).
- [`RegularExpression`](https://docs.rs/regexsolver/latest/regexsolver/regex/enum.RegularExpression.html) is the parsed pattern itself: a plain AST enum (`Character` / `Repetition` / `Concat` / `Alternation`) you can walk to lint or analyze patterns, alongside the simplifying combinators and the analyses.

## Execution limits

Automaton operations can blow up on adversarial inputs, so the engine is built to run untrusted patterns safely: a thread-local `ExecutionProfile` caps runtime and state explosion, and controls when the engine may determinize on its own. Hitting a limit returns a specific `EngineError` instead of hanging or panicking.

```rust
use regexsolver::{Term, execution_profile::ExecutionProfileBuilder, error::EngineError, fast_automaton::GenerationOptions};

let term = Term::from_pattern(".*abc.*cdef.*sqdsqf.*").unwrap();

let execution_profile = ExecutionProfileBuilder::new()
	.execution_timeout(50) // limit in milliseconds
	.build();

// Asking for 100 million strings cannot finish within the budget, so the
// generation aborts instead of running to completion.
execution_profile.run(|| {
	assert_eq!(EngineError::OperationTimeOutError, term.generate_strings(100_000_000, 0, GenerationOptions::new()).unwrap_err());
});
```

The same profile caps how many states an automaton may hold (`max_number_of_states`, failing with `EngineError::AutomatonHasTooManyStates`), and can refuse implicit determinization outright (`implicit_determinization(false)`), so the potentially exponential subset construction only ever runs through an explicit `determinize()` call. Both are documented with examples on [`ExecutionProfile`](https://docs.rs/regexsolver/latest/regexsolver/execution_profile/struct.ExecutionProfile.html).

## Implementation

- Patterns are parsed with [regex-syntax](https://docs.rs/regex-syntax/latest/regex_syntax/) and simplified into a small regular-expression AST; set operations run on finite automata; results convert back to patterns via state elimination.
- Transition labels are bitvectors over a per-automaton "spanning set" of disjoint character ranges, making label union/intersection/complement O(1): see [Optimizing Automaton Representation with Transition Conditions](https://alexvbrdn.me/post/optimizing-transition-conditions-automaton-representation).
- Correctness is cross-validated against the `regex` crate and exercised by property-based tests over randomly generated automata and expressions, with brute-force oracles for the analyses.

## Other languages

This engine also backs a hosted API, with clients for other languages. They call that service rather than embedding the crate, so they need an API token from the [developer console](https://console.regexsolver.com/):

- [regexsolver-java](https://github.com/RegexSolver/regexsolver-java)
- [regexsolver-js](https://github.com/RegexSolver/regexsolver-js)
- [regexsolver-python](https://github.com/RegexSolver/regexsolver-python)

Our [guide](https://docs.regexsolver.com/) covers getting started with them.

## Minimum supported Rust version

`regexsolver` builds with Rust 1.88 and later, verified in CI. Raising this version is a breaking change.

## Contributing

Bug reports and pull requests are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md). Notable changes are recorded in [CHANGELOG.md](CHANGELOG.md), and the security policy is in [SECURITY.md](SECURITY.md).

## License

Licensed under the [MIT License](LICENSE).
