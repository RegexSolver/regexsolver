# RegexSolver
[![Crates.io Version](https://img.shields.io/crates/v/regexsolver)](https://crates.io/crates/regexsolver)

**RegexSolver** is a high-performance Rust library for building, combining, and analyzing regular expressions and finite automata. Ideal for constraint solvers, code generators, test-case generators, and any use case requiring rich regex/automaton operations.

## Table of Contents

 - [Installation](#installation)
 - [Example](#example)
 - [Key Concepts & Limitations](#key-concepts-limitations)
 - [API](#api)
    - [Term](#term)
    - [FastAutomaton](#fastautomaton)
    - [RegularExpression](#regularexpression)
 - [Error Handling](#error-handling)
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
assert_eq!(concat.to_string(), "abc.*xyz");

// Union
let union = t1.union(&[Term::from_pattern("fgh").unwrap()]).unwrap();
assert_eq!(union.to_string(), "(abc.*|fgh)");

// Intersection
let inter = Term::from_pattern("(ab|xy){2}")
    .unwrap()
    .intersection(&[Term::from_pattern(".*xy").unwrap()])
    .unwrap(); // (ab|xy)xy
assert_eq!(inter.to_string(), "(ab|xy)xy");

// Subtraction
let diff = Term::from_pattern("a*")
    .unwrap()
    .subtraction(&Term::from_pattern("").unwrap())
    .unwrap();
assert_eq!(diff.to_string(), "a+");

// Repetition
let rep = Term::from_pattern("abc").unwrap().repeat(2, Some(4)).unwrap();
assert_eq!(rep.to_string(), "(abc){2,4}");

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
| `new_empty()` | `Term` | Creates a term that matches the empty language. |
| `new_total()` | `Term` | Creates a term that matches all possible strings. |
| `new_empty_string()` | `Term` | Creates a term that only match the empty string `""`. |
| `from_pattern(pattern: &str)` | `Result<Term, EngineError>` | Parses the provided pattern and return a new `Term` holding the resulting `RegularExpression`. |
| `from_pattern(regex: RegularExpression)` | `Term` | Creates a new `Term` holding the provided `RegularExpression`. |
| `from_automaton(automaton: FastAutomaton)` | `Term` | Creates a new `Term` holding the provided `FastAutomaton`. |

#### Manipulate
| Method | Return | Description |
| -------- | ------- | ------- |
| `concat(&self, terms: &[Term])` | `Result<Term, EngineError>` | Computes the concatenation of the given collection of terms. Returns the resulting term. |
| `union(&self, terms: &[Term])` | `Result<Term, EngineError>` | Computes the union of the given collection of terms. Returns the resulting term. |
| `intersection(&self, terms: &[Term])` | `Result<Term, EngineError>` | Computes the intersection of the given collection of terms. Returns the resulting term. |
| `subtraction(&self, subtrahend: &Term)` | `Result<Term, EngineError>` | Computes the subtraction/difference of the two given terms. Returns the resulting term. |
| `difference(&self, subtrahend: &Term)` | `Result<Term, EngineError>` | See `self.subtraction(subtrahend: &Term)`. |
| `repeat(&self, min: u32, max_opt: Option<u32>)` | `Result<Term, EngineError>` | Returns the repetition of the current term, between `min` and `max_opt` times. If `max_opt` is `None`, the repetition is unbounded. |

#### Analyze
| Method | Return | Description |
| -------- | ------- | ------- |
| `generate_strings(&self, count: usize)` | `Result<Vec<String>, EngineError>` | Generates the given count of strings matched by the given term. |
| `are_equivalent(&self, term: &Term)` | `Result<bool, EngineError>` | Computes whether the current term and the given term are equivalent. Returns `true` if both terms accept the same language. |
| `is_subset_of(&self, term: &Term)` | `Result<bool, EngineError>` | Computes whether the current term is a subset of the given term. Returns `true` if all strings matched by the current term are also matched by the given term. |
| `is_empty(&self)` | `bool` | Checks if the current term matches the empty language. |
| `is_total(&self)` | `bool` | Checks if the current term matches all possible strings. |
| `is_empty_string(&self)` | `bool` | Checks if the current term only match the empty string `""`. |
| `get_length(&self)` | `(Option<u32>, Option<u32>)` | Returns the minimum and maximum length of the possible matched strings. |
| `get_cardinality()` | `Result<Cardinality<u32>, EngineError>` | Returns the cardinality of the provided term (i.e. the number of the possible matched strings). |
| `to_automaton(&self)` | `Result<Cow<FastAutomaton>, EngineError>` | Converts the current `Term` to a `FastAutomaton`. |
| `to_regex(&self)` | `Option<Cow<RegularExpression>>` | Converts the current `Term` to a `RegularExpression`. Returns `None` if the automaton cannot be converted. |


### FastAutomaton

`FastAutomaton` is used to directly build, manipulate and analyze automata. To convert an automaton to a `RegularExpression` the method `to_regex()` can be used. Not all automaton can be converted to a regular expression.

When building or modifying an automaton you might come to use the method `add_transition(&mut self, from_state: State, to_state: State, new_cond: &Condition)`. This method accepts a `Condition` rather than a raw character set. To construct a Condition, call: To build a `Condition`, call:
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
| `new_empty()` | `FastAutomaton` | Create an automaton that matches the empty language. |
| `new_total()` | `FastAutomaton` | Create an automaton that matches all possible strings. |
| `new_empty_string()` | `FastAutomaton` | Create an automaton that only match the empty string `""`. |
| `new_from_range(range: &CharRange)` | `Result<FastAutomaton, EngineError>` | Create an automaton that matches one of the characters in the provided `CharRange`. |
| `new_state(&mut self)` | `State` | Create a new state in the automaton and returns its identifier. |
| `accept(&mut self, state: State)` | | Make the automaton accept the provided state as a valid final state. |
| `add_transition(&mut self, from_state: State, to_state: State, new_cond: &Condition)` | | Create a new transition between the two provided states with the given condition, the provided condition must follow the same spanning set as the rest of the automaton. |
| `add_epsilon_transition(&mut self, from_state: State, to_state: State)` | | Create a new epsilon transition between the two provided states. |
| `remove_state(&mut self, state: State)` | | Remove the provided state from the automaton. Remove all the transitions it is connected to. Panic if the state is used as a start state. |
| `remove_states(&mut self, states: &IntSet<State>)` | | Remove the provided states from the automaton. Remove all the transitions they are connected to. Panic if one of the state is used as a start state. |
| `apply_new_spanning_set(&mut self, new_spanning_set: &SpanningSet)` | `Result<(), EngineError>` | Apply the provided spanning set to the automaton and project all of its conditions on it. |

#### Manipulate
| Method | Return | Description |
| -------- | ------- | ------- |
| `union(&self, other: &FastAutomaton)` | `Result<FastAutomaton, EngineError>` |  |
| `union_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(automatons: I)` | `Result<FastAutomaton, EngineError>` |  |
| `union_all_par<'a, I: IntoParallelIterator<Item = &'a FastAutomaton>>(automatons: I)` | `Result<FastAutomaton, EngineError>` |  |
| `concat(&self, other: &FastAutomaton)` | `Result<FastAutomaton, EngineError>` |  |
| `concat_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(automatons: I)` | `Result<FastAutomaton, EngineError>` |  |
| `determinize(&self)` | `Result<FastAutomaton, EngineError>` |  |
| `intersection(&self, other: &FastAutomaton)` | `Result<FastAutomaton, EngineError>` |  |
| `intersection_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(automatons: I)` | `Result<FastAutomaton, EngineError>` |  |
| `intersection_all_par<'a, I: IntoParallelIterator<Item = &'a FastAutomaton>>(automatons: I)` | `Result<FastAutomaton, EngineError>` |  |
| `complement(&mut self)` | `Result<(), EngineError>` |  |
| `subtraction(&self, other: &FastAutomaton)` | `Result<FastAutomaton, EngineError>` |  |
| `repeat(&self, min: u32, max_opt: Option<u32>)` | `Result<FastAutomaton, EngineError>` |  |

#### Analyze
| Method | Return | Description |
| -------- | ------- | ------- |
| `state_in_degree(&self, state: State)` | `usize` | Returns the number of transitions to the provided state. |
| `state_out_degree(&self, state: State)` | `usize` | Returns the number of transitions from the provided state. |
| `all_states_iter(&self)` | `impl Iterator<Item = State>` | Returns an iterator of the states of the automaton. |
| `all_states_vec(&self)` | `Vec<State>` | Returns a vector containing the states of the automaton. |
| `direct_states_iter(&self, state: &State)` | `impl Iterator<Item = State>` | Returns an iterator over all states directly reachable from the given state in one transition. |
| `direct_states_vec(&self, state: &State)` | `Vec<State>` | Returns a vector containing all states directly reachable from the given state in one transition. |
| `transitions_to_vec(&self, state: State)` | `Vec<TransitionFrom>` | Returns a vector containing the transitions to the provided state. |
| `transitions_from_vec(&self, state: State)` | `Vec<TransitionTo>` | Returns a vector containing the transitions from the provided state. |
| `transitions_from_iter(&self, state: State)` | `impl Iterator<Item = (&Condition, &State)>` | Returns an iterator containing the transitions from the provided state. |
| `transitions_from_iter_mut(&mut self, state: State)` | `impl Iterator<Item = (&mut Condition, &State)>` | Returns a mutable iterator containing the transitions from the provided state. |
| `transitions_from_into_iter(&self, state: State)` | `impl Iterator<Item = TransitionTo>` | Returns an owned iterator containing the transitions from the provided state. |
| `does_transition_exists(&self, from_state: State, to_state: State)` | `bool` | Returns `true` if there is a directed transition between the two provided states. |
| `get_condition(&self, from_state: State, to_state: State)` | `Option<&Condition>` | Get a reference of the directed transtion's condition between the two provided states. |
| `get_condition_mut(&mut self, from_state: State, to_state: State)` | `Option<&Condition>` | Get a mutable reference of the directed transtion's condition between the two provided states. |
| `get_start_state(&self)` | `State` | Returns the start state of the automaton. |
| `get_accept_states(&self)` | `&IntSet<State>` | Get a reference to the set of accept (final) states of the automaton. |
| `get_spanning_set(&self)` | `&SpanningSet` | Returns a reference to the automaton's spanning set. |
| `is_accepted(&self, state: &State)` | `bool` | Returns `true` if the given `state` is one of the automaton's accept states. |
| `is_determinitic(&self)` | `bool` | Returns `true` if the automaton is deterministic. |
| `is_cyclic(&self)` | `bool` | Returns `true` if the automaton contains at least one cycle. |
| `has_state(&self, state: State)` | `bool` | Returns `true` if the automaton contains at least one cycle. |
| `to_regex(&self)` | `Option<RegularExpression>` | Try to convert the automaton to a `RegularExpression`. If it cannot find an equivalent pattern returns `None`. |
| `has_intersection(&self, other: &FastAutomaton)` | `Result<bool, EngineError>` | |


### RegularExpression

`RegularExpression` is used to directly build, manipulate and analyze regular expression patterns. Not all the set operations are available, for more advanced operation such as intersection, subtraction/difference and complement it is necessary to convert in to a `FastAutomaton` with the method `to_automaton()`.

## Error Handling

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
