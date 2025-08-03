# RegexSolver
[![Crates.io Version](https://img.shields.io/crates/v/regexsolver)](https://crates.io/crates/regexsolver)

**RegexSolver** is a high-performance Rust library for building, combining, and analyzing regular expressions and finite automata. Ideal for constraint solvers, code generators, test-case generators, and any use case requiring rich regex/automaton operations.

## Table of Contents

 - [Installation](#installation)
 - [Example](#example)
 - [Key Concepts & Limitations](#key-concepts--limitations)
 - [API](#api)
    - [Term](#term)
    - [FastAutomaton](#fastautomaton)
    - [RegularExpression](#regularexpression)
 - [Bound Execution](#bound-execution)
 - [Cross-Language Support](#cross-language-support)
 - [License](#license)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
regexsolver = "1"
```

## Example

```rust
use regexsolver::Term;

// Create terms from regex
let t1 = Term::from_pattern("abc.*").unwrap();
let t2 = Term::from_pattern(".*xyz").unwrap();

// Concatenate
let concat = t1.concat(&[t2]).unwrap();
assert_eq!(concat.to_pattern().unwrap(), "abc.*xyz");

// Union
let union = t1.union(&[Term::from_pattern("fgh").unwrap()]).unwrap();
assert_eq!(union.to_pattern().unwrap(), "(abc.*|fgh)");

// Intersection
let inter = Term::from_pattern("(ab|xy){2}")
    .unwrap()
    .intersection(&[Term::from_pattern(".*xy").unwrap()])
    .unwrap(); // (ab|xy)xy
assert_eq!(inter.to_pattern().unwrap(), "(ab|xy)xy");

// Subtraction
let diff = Term::from_pattern("a*")
    .unwrap()
    .subtraction(&Term::from_pattern("").unwrap())
    .unwrap();
assert_eq!(diff.to_pattern().unwrap(), "a+");

// Repetition
let rep = Term::from_pattern("abc")
    .unwrap()
    .repeat(2, Some(4))
    .unwrap();
assert_eq!(rep.to_pattern().unwrap(), "(abc){2,4}");

// Analyze
assert_eq!(rep.get_length(), (Some(6), Some(12)));
assert!(!rep.is_empty());

// Generate examples
let samples = Term::from_pattern("(x|y){1,3}")
    .unwrap()
    .generate_strings(5)
    .unwrap();
println!("Some matches: {:?}", samples);

// Equivalence & subset
let a = Term::from_pattern("a+").unwrap();
let b = Term::from_pattern("a*").unwrap();
assert!(!a.are_equivalent(&b).unwrap());
assert!(a.is_subset_of(&b).unwrap());
```

## Key Concepts & Limitations

RegexSolver supports a subset of regular expressions that adhere to the principles of regular languages. Here are the key characteristics and limitations of the regular expressions supported by RegexSolver:
- **Anchored Expressions:** All regular expressions in RegexSolver are anchored. This means that the expressions are treated as if they start and end at the boundaries of the input text. For example, the expression `abc` will match the string "abc" but not "xabc" or "abcx".
- **Lookahead/Lookbehind:** RegexSolver does not support lookahead (`(?=...)`) or lookbehind (`(?<=...)`) assertions. Using them would return an error.
- **Greedy/Ungreedy Quantifiers:** The concept of ungreedy (`*?`, `+?`, `??`) quantifiers is not supported. All quantifiers are treated as greedy. For example, `a*` or `a*?` will match the longest possible sequence of "a"s.
- **Line Feed and Dot:** RegexSolver handle every characters the same way. The dot character `.` matches every possible unicode characters including the line feed (`\n`).
- **Pure Regular Expressions:** RegexSolver focuses on pure regular expressions as defined in regular language theory. This means features that extend beyond regular languages, such as backreferences (`\1`, `\2`, etc.), are not supported. Any use of backreference would return an error.
- **Empty Regular Expressions:** An empty regular expression is denoted by `[]`, which represents a pattern that matches no input, not even an empty string.

RegexSolver is based on the [regex-syntax](https://docs.rs/regex-syntax/0.8.5/regex_syntax/) library for parsing patterns. As a result, unsupported features supported by the parser will be parsed but ignored. This allows for some flexibility in writing regular expressions, but it is important to be aware of the unsupported features to avoid unexpected behavior.

## API

### Term

`Term` is an enum designed to represent either a regular expression or a compiled automaton. This unified representation enables seamless and efficient execution of set operations across multiple instances. It's particularly valuable when working  with both regular expressions and automata, allowing operations to be performed transparently regardless of the underlying representation.

#### Build
| Method | Return | Description |
| -------- | ------- | ------- |
| `from_automaton(automaton: FastAutomaton)` | `Term` | Creates a new `Term` holding the provided `FastAutomaton`. |
| `from_pattern(pattern: &str)` | `Result<Term, EngineError>` | Parses the provided pattern and returns a new `Term` holding the resulting `RegularExpression`. |
| `from_regex(regex: RegularExpression)` | `Term` | Creates a new `Term` holding the provided `RegularExpression`. |
| `new_empty()` | `Term` | Creates a term that matches the empty language. |
| `new_empty_string()` | `Term` | Creates a term that only matches the empty string `""`. |
| `new_total()` | `Term` | Creates a term that matches all possible strings. |

#### Manipulate
| Method | Return | Description |
| -------- | ------- | ------- |
| `concat(&self, terms: &[Term])` | `Result<Term, EngineError>` | Computes the concatenation of the given terms. |
| `difference(&self, subtrahend: &Term)` | `Result<Term, EngineError>` | Alias for `subtraction`. |
| `intersection(&self, terms: &[Term])` | `Result<Term, EngineError>` | Computes the intersection of the given terms. |
| `repeat(&self, min: u32, max_opt: Option<u32>)` | `Result<Term, EngineError>` | Computes the repetition of the current term between `min` and `max_opt` times; if `max_opt` is `None`, the repetition is unbounded. |
| `subtraction(&self, subtrahend: &Term)` | `Result<Term, EngineError>` | Computes the difference between `self` and the given subtrahend. |
| `union(&self, terms: &[Term])` | `Result<Term, EngineError>` | Computes the union of the given terms. |

#### Analyze
| Method | Return | Description |
| -------- | ------- | ------- |
| `are_equivalent(&self, term: &Term)` | `Result<bool, EngineError>` | Returns `true` if both terms accept the same language. |
| `generate_strings(&self, count: usize)` | `Result<Vec<String>, EngineError>` | Generates `count` strings matched by the term. |
| `get_cardinality()` | `Result<Cardinality<u32>, EngineError>` | Returns the cardinality of the term (i.e., the number of possible matched strings). |
| `get_length(&self)` | `(Option<u32>, Option<u32>)` | Returns the minimum and maximum length of matched strings. |
| `is_empty(&self)` | `bool` | Checks if the term matches the empty language. |
| `is_empty_string(&self)` | `bool` | Checks if the term matches only the empty string `""`. |
| `is_subset_of(&self, term: &Term)` | `Result<bool, EngineError>` | Returns `true` if all strings matched by the current term are also matched by the given term. |
| `is_total(&self)` | `bool` | Checks if the term matches all possible strings. |
| `to_automaton(&self)` | `Result<Cow<FastAutomaton>, EngineError>` | Converts the term to a `FastAutomaton`. |
| `to_pattern(&self)` | `Option<String>` | Converts the term to a regular expression pattern; returns `None` if conversion isn’t possible. |
| `to_regex(&self)` | `Option<Cow<RegularExpression>>` | Converts the term to a RegularExpression; returns `None` if conversion isn’t possible. |


### FastAutomaton

`FastAutomaton` is used to directly build, manipulate and analyze automata. To convert an automaton to a `RegularExpression` the method `to_regex()` can be used. Not all automaton can be converted to a regular expression.

When building or modifying an automaton you might come to use the method `add_transition(&mut self, from_state: State, to_state: State, new_cond: &Condition)`. This method accepts a `Condition` rather than a raw character set. To build a `Condition`, call:
```rust
Condition::from_range(&range, &spanning_set);
```
where `spanning_set` is the automaton's current `SpanningSet`. The `CharRange` you pass must be fully covered by that spanning set. If it isn't, you have two options:

1. Merge an existing spanning set with another:
```rust
let new_set = SpanningSet::merge(&old_set, &other_set);
```

2. Recompute from a list of ranges:
```rust
let new_set = SpanningSet::compute_spanning_set(&[range_set1, range_set2, …]);
```

After constructing `new_set`, apply it to the automaton:
```rust
fast_automaton.apply_new_spanning_set(&new_set);
```

This design allows us to perform unions, intersections, and complements of transition conditions in O(1) time, but it does add some complexity to automaton construction. For more details, you can check [this article](https://alexvbrdn.me/post/optimizing-transition-conditions-automaton-representation).

#### Build
| Method | Return | Description |
| -------- | ------- | ------- |
| `accept(&mut self, state: State)` | | Marks the provided state as an accepting (final) state. |
| `add_epsilon_transition(&mut self, from_state: State, to_state: State)` | | Creates a new epsilon transition between the two states. |
| `add_transition(&mut self, from_state: State, to_state: State, new_cond: &Condition)` | | Creates a new transition with the given condition; the condition must follow the automaton’s current spanning set. |
| `apply_new_spanning_set(&mut self, new_spanning_set: &SpanningSet)` | `Result<(), EngineError>` | Applies the provided spanning set and projects all existing conditions onto it. |
| `new_empty()` | `FastAutomaton` | Creates an automaton that matches the empty language. |
| `new_empty_string()` | `FastAutomaton` | Creates an automaton that only matches the empty string `""`. |
| `new_from_range(range: &CharRange)` | `Result<FastAutomaton, EngineError>` | Creates an automaton that matches one of the characters in the given `CharRange`. |
| `new_state(&mut self)` | `State` | Creates a new state and returns its identifier. |
| `new_total()` | `FastAutomaton` | Creates an automaton that matches all possible strings. |
| `remove_state(&mut self, state: State)` | | Removes the state and all its connected transitions; panics if it's a start state. |
| `remove_states(&mut self, states: &IntSet<State>)` | | Removes the given states and their connected transitions; panics if any is a start state. |

#### Manipulate
| Method | Return | Description |
| -------- | ------- | ------- |
| `complement(&mut self)` | `Result<(), EngineError>` | Complements the automaton; it must be deterministic. |
| `concat(&self, other: &FastAutomaton)` | `Result<FastAutomaton, EngineError>` | Returns a new `FastAutomaton` representing the concatenation of `self` and `other`. |
| `concat_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(automatons: I)` | `Result<FastAutomaton, EngineError>` | Returns a new `FastAutomaton` representing the concatenation of all automata in the given iterator. |
| `determinize(&self)` | `Result<Cow<FastAutomaton>, EngineError>` | Determinizes the automaton and returns the result as a new `FastAutomaton`. |
| `intersection(&self, other: &FastAutomaton)` | `Result<FastAutomaton, EngineError>` | Returns a new `FastAutomaton` representing the intersection of `self` and `other`. |
| `intersection_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(automatons: I)` | `Result<FastAutomaton, EngineError>` | Returns a new `FastAutomaton` that is the intersection of all automatons in the given iterator. |
| `intersection_all_par<'a, I: IntoParallelIterator<Item = &'a FastAutomaton>>(automatons: I)` | `Result<FastAutomaton, EngineError>` | Returns a new `FastAutomaton` that is the intersection of all automatons in the given parallel iterator. |
| `repeat(&self, min: u32, max_opt: Option<u32>)` | `Result<FastAutomaton, EngineError>` | Computes the repetition of the automaton between `min` and `max_opt` times; if `max_opt` is `None`, the repetition is unbounded. |
| `subtraction(&self, other: &FastAutomaton)` | `Result<FastAutomaton, EngineError>` | Returns a new `FastAutomaton` representing the substraction of `self` and `other`. |
| `union(&self, other: &FastAutomaton)` | `Result<FastAutomaton, EngineError>` | Returns a new `FastAutomaton` representing the union of `self` and `other`. |
| `union_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(automatons: I)` | `Result<FastAutomaton, EngineError>` | Returns a new `FastAutomaton` that is the union of all automatons in the given iterator. |
| `union_all_par<'a, I: IntoParallelIterator<Item = &'a FastAutomaton>>(automatons: I)` | `Result<FastAutomaton, EngineError>` | Returns a new `FastAutomaton` that is the union of all automatons in the given parallel iterator. |

#### Analyze
| Method | Return | Description |
| -------- | ------- | ------- |
| `all_states_iter(&self)` | `impl Iterator<Item = State>` | Returns an iterator over the automaton’s states. |
| `all_states_vec(&self)` | `Vec<State>` | Returns a vector containing the automaton’s states. |
| `are_equivalent(&self, other: &FastAutomaton)` | `Result<bool, EngineError>` | Returns `true` if both automata accept the same language. |
| `direct_states_iter(&self, state: &State)` | `impl Iterator<Item = State>` | Returns an iterator over states directly reachable from the given state in one transition. |
| `direct_states_vec(&self, state: &State)` | `Vec<State>` | Returns a vector of states directly reachable from the given state in one transition. |
| `does_transition_exists(&self, from_state: State, to_state: State)` | `bool` | Returns `true` if there is a directed transition from `from_state` to `to_state`. |
| `generate_strings(&self, count: usize)` | `Result<AHashSet<String>, EngineError>` | Generates `count` strings matched by the automaton. |
| `get_accept_states(&self)` | `&IntSet<State>` | Returns a reference to the set of accept (final) states. |
| `get_cardinality(&self)` | `Cardinality<u32>` | Returns the cardinality of the automaton (i.e., the number of possible matched strings). |
| `get_condition(&self, from_state: State, to_state: State)` | `Option<&Condition>` | Returns a reference to the condition of the directed transition between the two states, if any. |
| `get_condition_mut(&mut self, from_state: State, to_state: State)` | `Option<&Condition>` | Returns a mutable reference to the condition of the directed transition between the two states, if any. |
| `get_length(&self)` | `(Option<u32>, Option<u32>)` | Returns the minimum and maximum length of matched strings. |
| `get_reacheable_states(&self)` | `IntSet<State>` | Returns the set of all states reachable from the start state. |
| `get_spanning_set(&self)` | `&SpanningSet` | Returns a reference to the automaton's spanning set. |
| `get_start_state(&self)` | `State` | Returns the start state. |
| `has_intersection(&self, other: &FastAutomaton)` | `Result<bool, EngineError>` | Returns `true` if the two automata have a non-empty intersection. |
| `has_state(&self, state: State)` | `bool` | Returns `true` if the automaton contains the given state. |
| `is_accepted(&self, state: &State)` | `bool` | Returns `true` if the given state is one of the accept states. |
| `is_cyclic(&self)` | `bool` | Returns `true` if the automaton contains at least one cycle. |
| `is_determinitic(&self)` | `bool` | Returns `true` if the automaton is deterministic. |
| `is_empty(&self)` | `bool` | Checks if the automaton matches the empty language. |
| `is_empty_string(&self)` | `bool` | Checks if the automaton only matches the empty string `""`. |
| `is_subset_of(&self, other: &FastAutomaton)` | `Result<bool, EngineError>` | Returns `true` if all strings accepted by `self` are also accepted by `other`. |
| `is_total(&self)` | `bool` | Checks if the automaton matches all possible strings. |
| `state_in_degree(&self, state: State)` | `usize` | Returns the number of transitions to the provided state. |
| `state_out_degree(&self, state: State)` | `usize` | Returns the number of transitions from the provided state. |
| `to_regex(&self)` | `Option<RegularExpression>` | Attempts to convert the automaton to a `RegularExpression`; returns `None` if no equivalent pattern are found. |
| `transitions_from_into_iter(&self, state: State)` | `impl Iterator<Item = TransitionTo>` | Returns an owned iterator over transitions from the given state. |
| `transitions_from_iter(&self, state: State)` | `impl Iterator<Item = (&Condition, &State)>` | Returns an iterator over transitions from the given state. |
| `transitions_from_iter_mut(&mut self, state: State)` | `impl Iterator<Item = (&mut Condition, &State)>` | Returns a mutable iterator over transitions from the given state. |
| `transitions_from_vec(&self, state: State)` | `Vec<TransitionTo>` | Returns a vector of transitions from the given state. |
| `transitions_to_vec(&self, state: State)` | `Vec<TransitionFrom>` | Returns a vector of transitions to the given state. |


### RegularExpression

`RegularExpression` is used to directly build, manipulate and analyze regular expression patterns. Not all the set operations are available, for more advanced operation such as intersection, subtraction/difference and complement it is necessary to convert in to a `FastAutomaton` with the method `to_automaton()`.

#### Build
| Method | Return | Description |
| -------- | ------- | ------- |
| `concat(&self, other: &RegularExpression, append_back: bool)` | `RegularExpression` | Returns a new regular expression representing the concatenation of `self` and `other`; `append_back` determines their order. |
| `new(pattern: &str)` | `Result<RegularExpression, EngineError>` | Parses the provided pattern and returns the resulting `RegularExpression`. |
| `new_empty()` | `RegularExpression` | Creates a regular expression that matches the empty language. |
| `new_empty_string()` | `RegularExpression` | Creates a regular expression that matches only the empty string `""`. |
| `new_total()` | `RegularExpression` | Creates a regular expression that matches all possible strings. |
| `repeat(&self, min: u32, max_opt: Option<u32>)` | `RegularExpression` | Returns the repetition of the expression between `min` and `max_opt` times; if `max_opt` is `None`, the repetition is unbounded. |
| `simplify(&self)` | `RegularExpression` | Returns a simplified version by eliminating redundant constructs and applying canonical reductions. |
| `union(&self, other: &RegularExpression)` | `RegularExpression` | Returns a regular expression matching the union of `self` and `other`. |
| `union_all<'a, I: IntoIterator<Item = &'a RegularExpression>>(patterns: I)` | `RegularExpression` | Returns a regular expression that is the union of all expressions in `patterns`. |

#### Analyze
| Method | Return | Description |
| -------- | ------- | ------- |
| `get_cardinality(&self)` | `Cardinality<u32>` | Returns the cardinality of the regular expression (i.e., the number of possible matched strings). |
| `get_length(&self)` | `(Option<u32>, Option<u32>)` | Returns the minimum and maximum length of possible matched strings. |
| `is_empty(&self)` | `bool` | Checks if the regular expression matches the empty language. |
| `is_empty_string(&self)` | `bool` | Checks if the regular expression only matches the empty string `""`. |
| `is_total(&self)` | `bool` | Checks if the regular expression matches all possible strings. |
| `to_automaton(&self)` | `Result<FastAutomaton, EngineError>` | Converts the regular expression to an equivalent `FastAutomaton`. |

## Bound Execution

By default, all operations run without limits. For heavy or untrusted patterns, use a thread local `ExecutionProfile` to cap execution time and maximum number of states in used automata.

### Time-Bounded Execution

```rust
use regexsolver::{Term, execution_profile::{ExecutionProfile, ExecutionProfileBuilder}, error::EngineError};

let term = Term::from_pattern(".*abc.*cdef.*sqdsqf.*").unwrap();

let execution_profile = ExecutionProfileBuilder::new()
	.execution_timeout(5) // We set the limit (5ms)
	.build();

// We run the operation with the defined limitation
execution_profile.run(|| {
	assert_eq!(EngineError::OperationTimeOutError, term.generate_strings(1000).unwrap_err());
});
```

### State-Limited Execution

```rust
use regexsolver::{Term, execution_profile::{ExecutionProfile, ExecutionProfileBuilder}, error::EngineError};

let term1 = Term::from_pattern(".*abcdef.*").unwrap();
let term2 = Term::from_pattern(".*defabc.*").unwrap();

let execution_profile = ExecutionProfileBuilder::new()
	.max_number_of_states(5) // We set the limit
	.build();

// We run the operation with the defined limitation
execution_profile.run(|| {
	assert_eq!(EngineError::AutomatonHasTooManyStates, term1.intersection(&[term2]).unwrap_err());
});
```

## Cross-Language Support


If you want to use this library with other programming languages, we provide a wide range of wrappers:
- [regexsolver-java](https://github.com/RegexSolver/regexsolver-java)
- [regexsolver-js](https://github.com/RegexSolver/regexsolver-js)
- [regexsolver-python](https://github.com/RegexSolver/regexsolver-python)

For more information about how to use the wrappers, you can refer to our [getting started guide](https://docs.regexsolver.com/getting-started.html).

## License

This project is licensed under the MIT License.
