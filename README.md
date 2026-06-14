# RegexSolver

[![Crates.io Version](https://img.shields.io/crates/v/regexsolver)](https://crates.io/crates/regexsolver)
[![docs.rs](https://img.shields.io/docsrs/regexsolver)](https://docs.rs/regexsolver)
[![CI](https://github.com/RegexSolver/regexsolver/actions/workflows/rust.yml/badge.svg)](https://github.com/RegexSolver/regexsolver/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/crates/l/regexsolver)](LICENSE)

The `regex` crate tells you whether a *string* matches a pattern. **RegexSolver treats patterns as the sets of strings they match** — so you can intersect, subtract, compare, complement, and enumerate them, and get the result back as a regex.

```rust
use regexsolver::Term;

let a: Term = "(ab|xy){2}".parse()?;
let b: Term = ".*xy".parse()?;

// Which strings match BOTH patterns? Get the answer as a regex:
let both = a.intersection([&b])?;
assert_eq!(both.to_pattern(), "(ab|xy)xy");

// Test a concrete string against the result (matching is anchored):
assert!(both.matches("abxy")?);

// ...and sample them:
assert_eq!(both.generate_strings(2, 0)?, ["xyxy", "abxy"]);
```

## What would you use this for?

- **Safe migrations** - `old_rule.subset(&new_rule)?`: does the new validation pattern accept *everything* the old one did?
- **Test-data generation** - `term.generate_strings(100, 0)?`: produce strings matching any pattern, with pagination.
- **Rule analysis**: find shadowed or overlapping routes, firewall rules, and validators with `intersection` / `difference`.
- **Equivalence proofs** - `a.equivalent(&b)?`: show that two differently-written patterns match exactly the same strings.
- **Pattern simplification**: every operation returns a `Term` you can turn back into a regex pattern with `to_pattern()`.

Under the hood, every pattern compiles to a finite automaton:

<p align="center"><img src="https://raw.githubusercontent.com/RegexSolver/regexsolver/refs/heads/v1/assets/automaton.svg" alt="the minimal automaton of (ab|cd)*"/></p>
<p align="center"><sub><code>(ab|cd)*</code> compiled to its minimal automaton, generated with this library's <code>to_dot()</code></sub></p>

## Try it

```bash
git clone https://github.com/RegexSolver/regexsolver && cd regexsolver

# How do two patterns relate? (equivalence, subsets, intersection, differences)
cargo run --example relate -- "(ab|xy){2}" ".*xy"

# Generate n sample strings matching a pattern
cargo run --example generate -- "[a-z]{2}[0-9]" 20
```

Or in your own project:

```bash
cargo add regexsolver
```

By default the `parallel` feature is enabled: unions/intersections of more than 3 operands and parts of the automaton-to-regex conversion run on [rayon](https://crates.io/crates/rayon). Disable it for a leaner dependency tree on single-threaded workloads:

```toml
regexsolver = { version = "1", default-features = false }
```

## Semantics in 30 seconds

RegexSolver implements **pure regular languages**, which differs from typical regex engines in two ways:

- **Everything is anchored**: `abc` matches the string "abc", not "xabc" or "abcx". Patterns describe *whole strings*.
- **`.` matches any character**, including line feed (`\n`).

The rest follows from regular-language theory:

- **Backreferences** (`\1`, `\2`, ...) go beyond regular languages and return an error, as do **lookahead/lookbehind** assertions (`(?=...)`, `(?<=...)`).
- **All quantifiers are greedy**: ungreedy markers (`*?`, `+?`, `??`) are ignored as *sets of strings*, `a*` and `a*?` are the same language.
- **The empty language** (matches no string at all) is written `[]` (empty character class). This is distinct from the empty string `""`.

RegexSolver is based on the [regex-syntax](https://docs.rs/regex-syntax/0.8.5/regex_syntax/) library for parsing patterns. Unsupported features are parsed but ignored; they do not raise an error unless they affect semantics that cannot be represented (e.g., backreferences). This allows for some flexibility in writing regular expressions, but it is important to be aware of the unsupported features to avoid unexpected behavior.

## A tour of the API

[`Term`](https://docs.rs/regexsolver/latest/regexsolver/enum.Term.html) is the type you'll interact with: it wraps either a regular expression or an automaton and picks the best representation for each operation. The essentials:

| Method | Description |
| -------- | ------- |
| `Term::from_pattern(pattern)` | Parses a pattern into a term. |
| `intersection(&self, terms)` / `union(&self, terms)` | Set operations over any number of terms. |
| `difference(&self, other)` / `complement(&self)` | What `self` matches and `other` doesn't / everything `self` doesn't match. |
| `concat(&self, terms)` / `repeat(&self, range)` | Sequence and repeat languages; `range` is any Rust range expression (`2..=5`, `1..`, `..3`, ...). |
| `equivalent(&self, other)` / `subset(&self, other)` | Compare languages. |
| `is_empty()` / `is_total()` / `length()` / `cardinality()` | Analyze a language: matches nothing? everything? string lengths? how many strings? |
| `generate_strings(limit, offset)` | Enumerate matching strings eagerly (call `minimize()` once first when paginating). |
| `iter_strings()` | Lazy iterator equivalent; computes the automaton once and yields strings in batches. |
| `to_pattern()` / `to_automaton()` / `to_regex()` | Convert back out. |

All fallible operations return `Result<_, EngineError>`.

### Building automata by hand

`FastAutomaton` is used to directly build, manipulate and analyze automata. To convert an automaton to a `RegularExpression` the method `to_regex()` can be used.

States are created with `new_state()` and transitions with `add_transition_from_range`, which labels the transition with a plain `CharRange`:

```rust
use regexsolver::CharRange;
use regexsolver::fast_automaton::FastAutomaton;
use regex_charclass::char::Char;

// Build an automaton matching "[a-c][0-9]*" by hand:
let mut automaton = FastAutomaton::new_empty();
let s1 = automaton.new_state();
automaton.accept(s1);

let a_to_c = CharRange::new_from_range(Char::new('a')..=Char::new('c'));
let digits = CharRange::new_from_range(Char::new('0')..=Char::new('9'));
automaton.add_transition_from_range(0, s1, &a_to_c)?;
automaton.add_transition_from_range(s1, s1, &digits)?;

assert!(automaton.is_match("b42"));
assert_eq!(automaton.to_regex().to_string(), "[a-c][0-9]*");
```

Internally, transition labels are bitvector `Condition`s over the automaton's `SpanningSet` of disjoint character ranges, that is what makes label union/intersection/complement O(1) ([article](https://alexvbrdn.me/post/optimizing-transition-conditions-automaton-representation)). `add_transition_from_range` maintains that representation for you; for full manual control over conditions and spanning sets, see the [`add_transition` documentation](https://docs.rs/regexsolver/latest/regexsolver/fast_automaton/struct.FastAutomaton.html#method.add_transition).

Everything `Term` does is also available directly on [`FastAutomaton`](https://docs.rs/regexsolver/latest/regexsolver/fast_automaton/struct.FastAutomaton.html), including `determinize`, `minimize`, the set operations, `equivalent`/`subset`, the analyses, `generate_strings`, `to_regex`, plus low-level construction (`new_state`, `accept`, `add_epsilon_transition`, ...) and inspection (`states`, `transitions_from`, `to_dot`, ...).

### Working with patterns as ASTs

`RegularExpression` is the parsed pattern itself: a plain AST enum (`Character` / `Repetition` / `Concat` / `Alternation`) you can analyze and walk directly. Set operations like intersection and difference live on `FastAutomaton` (or, more conveniently, on `Term`); convert with `to_automaton()`.

```rust
use regexsolver::cardinality::Cardinality;
use regexsolver::regex::RegularExpression;

// A validation pattern for an order id, e.g. "ORD-2024-12345".
let pattern = RegularExpression::new("ORD-20[0-9]{2}-[0-9]{4,6}")?;

// How long can matching ids get? Size your database column accordingly.
assert_eq!(pattern.length(), (Some(13), Some(15)));

// How many distinct ids does the pattern allow?
assert_eq!(pattern.cardinality(), Cardinality::Integer(111_000_000));

// The AST is a plain enum: walk it to lint patterns, e.g. reject
// validation rules that accept unboundedly long input.
fn has_unbounded_repetition(regex: &RegularExpression) -> bool {
    match regex {
        RegularExpression::Character(_) => false,
        RegularExpression::Repetition(inner, _, max) => {
            max.is_none() || has_unbounded_repetition(inner)
        }
        RegularExpression::Concat(parts) => parts.iter().any(has_unbounded_repetition),
        RegularExpression::Alternation(parts) => parts.iter().any(has_unbounded_repetition),
    }
}
assert!(!has_unbounded_repetition(&pattern));
assert!(has_unbounded_repetition(&RegularExpression::new(".*@example\\.com")?));
```

The variants are freely constructible too; a hand-built repetition whose maximum is below its minimum denotes no valid language and is rejected with `EngineError::InvalidRepetitionBounds` when converted by `to_automaton()`.

Parsing (`new`, `parse`), the simplifying combinators (`concat`, `union`, `repeat`, `simplify`) and the analyses (`length`, `cardinality`, `evaluate_complexity`) are documented on [`RegularExpression`](https://docs.rs/regexsolver/latest/regexsolver/regex/enum.RegularExpression.html).

## Bound Execution

Automaton operations can blow up on adversarial inputs, so the engine is built to run untrusted patterns safely: a thread-local `ExecutionProfile` caps runtime and state explosion, and controls when the engine may determinize or minimize on its own. Hitting a limit returns a specific `EngineError` instead of hanging or panicking.

### Time-Bounded Execution

```rust
use regexsolver::{Term, execution_profile::{ExecutionProfile, ExecutionProfileBuilder}, error::EngineError};

let term = Term::from_pattern(".*abc.*cdef.*sqdsqf.*")?;

let execution_profile = ExecutionProfileBuilder::new()
	.execution_timeout(5) // limit in milliseconds
	.build();

// We run the operation with the defined limitation
execution_profile.run(|| {
	assert_eq!(EngineError::OperationTimeOutError, term.generate_strings(1000, 1_000_000).unwrap_err());
});
```

### State-Limited Execution

```rust
use regexsolver::{Term, execution_profile::{ExecutionProfile, ExecutionProfileBuilder}, error::EngineError};

let term1 = Term::from_pattern(".*abcdef.*")?;
let term2 = Term::from_pattern(".*defabc.*")?;

let execution_profile = ExecutionProfileBuilder::new()
	.max_number_of_states(5) // we set the limit
	.build();

// We run the operation with the defined limitation
execution_profile.run(|| {
	assert_eq!(EngineError::AutomatonHasTooManyStates, term1.intersection(&[term2]).unwrap_err());
});
```

### Disabling Implicit Determinization

`FastAutomaton` operations that require a deterministic automaton (`minimize`, `complement`, `difference`, `equivalent`, `subset`, `cardinality`, ...) determinize a non-deterministic input on their own by default. Since subset construction can blow up exponentially, this can be disabled: those operations then return `EngineError::DeterministicAutomatonRequired` instead, and determinization only happens through an explicit `determinize()` call. Deterministic inputs are always accepted, and the whole `Term` API keeps working since that layer manages the underlying representation itself, so its determinizations count as explicit.

```rust
use regexsolver::execution_profile::ExecutionProfileBuilder;
use regexsolver::error::EngineError;

let execution_profile = ExecutionProfileBuilder::new()
	.implicit_determinization(false) // default is true
	.build();

// `nfa` is any non-deterministic FastAutomaton
execution_profile.run(|| {
	assert_eq!(EngineError::DeterministicAutomatonRequired, nfa.clone().minimize().unwrap_err());

	// Determinizing explicitly is always allowed.
	let mut dfa = nfa.determinize().unwrap().into_owned();
	assert!(dfa.minimize().is_ok());
});
```

## How it works

- Patterns are parsed with [regex-syntax](https://docs.rs/regex-syntax/latest/regex_syntax/) and simplified into a small regular-expression AST; set operations run on finite automata; results convert back to patterns via state elimination.
- Transition labels are bitvectors over a per-automaton "spanning set" of disjoint character ranges, making label union/intersection/complement O(1): see [Optimizing Automaton Representation with Transition Conditions](https://alexvbrdn.me/post/optimizing-transition-conditions-automaton-representation).
- Correctness is cross-validated against the `regex` crate and exercised by property-based tests over randomly generated automata and expressions, with brute-force oracles for the analyses.

## Cross-Language Support

If you want to use this library with other programming languages, we provide a wide range of wrappers:
- [regexsolver-java](https://github.com/RegexSolver/regexsolver-java)
- [regexsolver-js](https://github.com/RegexSolver/regexsolver-js)
- [regexsolver-python](https://github.com/RegexSolver/regexsolver-python)

For more information about how to use the wrappers, you can refer to our [guide](https://docs.regexsolver.com/getting-started.html).

## License

This project is licensed under the MIT License.
