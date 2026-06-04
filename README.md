# RegexSolver
[![Crates.io Version](https://img.shields.io/crates/v/regexsolver)](https://crates.io/crates/regexsolver)

**RegexSolver** is a Rust library for building, combining, and analyzing regular expressions and finite automata. It is designed for constraint solvers, test generators, and other systems that need advanced regex and automaton operations.

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
use regexsolver::error::EngineError;

fn main() -> Result<(), EngineError> {
    // Create terms from regex
    let t1 = Term::from_pattern("abc.*")?;
    let t2 = Term::from_pattern(".*xyz")?;

    // Concatenate
    let concat = t1.concat(&[t2])?;
    assert_eq!(concat.to_pattern(), "abc.*xyz");

    // Union
    let union = t1.union(&[Term::from_pattern("fgh")?])?;
    assert_eq!(union.to_pattern(), "(abc.*|fgh)");

    // Intersection
    let inter = Term::from_pattern("(ab|xy){2}")?
        .intersection(&[Term::from_pattern(".*xy")?])?;
    assert_eq!(inter.to_pattern(), "(ab|xy)xy");

    // Difference
    let diff = Term::from_pattern("a*")?
        .difference(&Term::from_pattern("")?)?;
    assert_eq!(diff.to_pattern(), "a+");

    // Repetition
    let rep = Term::from_pattern("abc")?
        .repeat(2, Some(4))?;
    assert_eq!(rep.to_pattern(), "(abc){2,4}");

    // Analyze
    assert_eq!(rep.get_length(), (Some(6), Some(12)));
    assert!(!rep.is_empty());

    // Generate examples
    let samples = Term::from_pattern("(x|y){1,3}")?
        .generate_strings(5, 0)?;
    println!("Some matches: {:?}", samples);

    // Equivalence & subset
    let a = Term::from_pattern("a+")?;
    let b = Term::from_pattern("a*")?;
    assert!(!a.equivalent(&b)?);
    assert!(a.subset(&b)?);

    Ok(())
}
```

## Key Concepts & Limitations

RegexSolver supports a subset of regular expressions that adhere to the principles of regular languages. Here are the key characteristics and limitations of the regular expressions supported by RegexSolver:
- **Anchored Expressions:** All regular expressions in RegexSolver are anchored. This means that the expressions are treated as if they start and end at the boundaries of the input text. For example, the expression `abc` will match the string "abc" but not "xabc" or "abcx".
- **Lookahead/Lookbehind:** RegexSolver does not support lookahead (`(?=...)`) or lookbehind (`(?<=...)`) assertions. Using them returns an error.
- **Pure Regular Expressions:** RegexSolver focuses on pure regular expressions as defined in regular language theory. This means features that extend beyond regular languages, such as backreferences (`\1`, `\2`, etc.), are not supported. Any use of backreference would return an error.
- **Greedy/Ungreedy Quantifiers:** The concept of ungreedy (`*?`, `+?`, `??`) quantifiers is not supported. All quantifiers are treated as greedy. For example, `a*` or `a*?` will match the longest possible sequence of "a"s.
- **Line Feed and Dot:** RegexSolver handles all characters the same way. The dot `.` matches any Unicode character including line feed (`\n`).
- **Empty Regular Expressions:** The empty language (matches no string) is represented by constructs like `[]` (empty character class). This is distinct from the empty string.

RegexSolver is based on the [regex-syntax](https://docs.rs/regex-syntax/0.8.5/regex_syntax/) library for parsing patterns. Unsupported features are parsed but ignored; they do not raise an error unless they affect semantics that cannot be represented (e.g., backreferences). This allows for some flexibility in writing regular expressions, but it is important to be aware of the unsupported features to avoid unexpected behavior.

## API

### Term

`Term` is an enum designed to represent either a regular expression or an automaton. Used when working with both regular expressions and automata, allowing operations to be performed transparently regardless of the underlying representation.

#### Build
| Method | Return | Description |
| -------- | ------- | ------- |
| `from_automaton(automaton: FastAutomaton)` | `Term` | Creates a new `Term` holding the provided `FastAutomaton`. |
| `from_pattern(pattern: &str)` | `Result<Term, EngineError>` | Parses and simplifies the provided pattern and returns a new `Term` holding the resulting `RegularExpression`. |
| `from_regex(regex: RegularExpression)` | `Term` | Creates a new `Term` holding the provided `RegularExpression`. |
| `new_empty()` | `Term` | Creates a term that matches the empty language. |
| `new_empty_string()` | `Term` | Creates a term that only matches the empty string `""`. |
| `new_total()` | `Term` | Creates a term that matches all possible strings. |

#### Manipulate
| Method | Return | Description |
| -------- | ------- | ------- |
| `concat(&self, terms: &[Term])` | `Result<Term, EngineError>` | Computes the concatenation of the given terms. |
| `difference(&self, other: &Term)` | `Result<Term, EngineError>` | Computes the difference between `self` and `other`. |
| `intersection(&self, terms: &[Term])` | `Result<Term, EngineError>` | Computes the intersection of the given terms. |
| `repeat(&self, min: u32, max_opt: Option<u32>)` | `Result<Term, EngineError>` | Computes the repetition of the current term between `min` and `max_opt` times; if `max_opt` is `None`, the repetition is unbounded. |
| `union(&self, terms: &[Term])` | `Result<Term, EngineError>` | Computes the union of the given terms. |

#### Analyze
| Method | Return | Description |
| -------- | ------- | ------- |
| `equivalent(&self, term: &Term)` | `Result<bool, EngineError>` | Returns `true` if both terms accept the same language. |
| `generate_strings(&self, count: usize, offset: usize)` | `Result<Vec<String>, EngineError>` | Generates `count` strings matched by the term, skipping the first `offset` strings. |
| `get_cardinality(&self)` | `Result<Cardinality<u32>, EngineError>` | Returns the cardinality of the term (i.e., the number of possible matched strings). |
| `get_length(&self)` | `(Option<u32>, Option<u32>)` | Returns the minimum and maximum length of matched strings. |
| `is_empty(&self)` | `bool` | Checks if the term matches the empty language. |
| `is_empty_string(&self)` | `bool` | Checks if the term matches only the empty string `""`. |
| `is_total(&self)` | `bool` | Checks if the term matches all possible strings. |
| `subset(&self, term: &Term)` | `Result<bool, EngineError>` | Returns `true` if all strings matched by the current term are also matched by the given term. |
| `to_automaton(&self)` | `Result<Cow<FastAutomaton>, EngineError>` | Converts the term to a `FastAutomaton`. |
| `to_pattern(&self)` | `String` | Converts the term to a regular expression pattern. |
| `to_regex(&self)` | `Cow<RegularExpression>` | Converts the term to a `RegularExpression`. |

### FastAutomaton

`FastAutomaton` is used to directly build, manipulate and analyze automata. To convert an automaton to a `RegularExpression` the method `to_regex()` can be used.

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
| `accept(&mut self, state: State)` | `()` | Marks the provided state as an accepting (final) state. |
| `add_epsilon_transition(&mut self, from_state: State, to_state: State)` | `()` | Creates a new epsilon transition between the two states. |
| `add_transition(&mut self, from_state: State, to_state: State, new_cond: &Condition)` | `()` | Creates a new transition with the given condition; the condition must follow the automaton’s current spanning set. |
| `apply_new_spanning_set(&mut self, new_spanning_set: &SpanningSet)` | `Result<(), EngineError>` | Applies the provided spanning set and projects all existing conditions onto it. |
| `new_empty()` | `FastAutomaton` | Creates an automaton that matches the empty language. |
| `new_empty_string()` | `FastAutomaton` | Creates an automaton that only matches the empty string `""`. |
| `new_from_range(range: &CharRange)` | `FastAutomaton` | Creates an automaton that matches one of the characters in the given `CharRange`. |
| `new_state(&mut self)` | `State` | Creates a new state and returns its identifier. |
| `new_total()` | `FastAutomaton` | Creates an automaton that matches all possible strings. |
| `remove_state(&mut self, state: State)` | `()` | Removes the state and its connected transitions; panics if it's a start state. |
| `remove_states(&mut self, states: &IntSet<State>)` | `()` | Removes the given states and their connected transitions; panics if any is a start state. |
| `remove_transition(&mut self, from_state: State, to_state: State)` | `()` | Removes the transition between the two provided states if it exists. |

#### Manipulate
| Method | Return | Description |
| -------- | ------- | ------- |
| `complement(&mut self)` | `Result<(), EngineError>` | Complements the automaton; it must be deterministic. |
| `concat(&self, other: &FastAutomaton)` | `Result<FastAutomaton, EngineError>` | Computes the concatenation between `self` and `other`. |
| `concat_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(automata: I)` | `Result<FastAutomaton, EngineError>` | Computes the concatenation of all automata in the given iterator. |
| `determinize(&self)` | `Result<Cow<FastAutomaton>, EngineError>` | Determinizes the automaton and returns the result. |
| `difference(&self, other: &FastAutomaton)` | `Result<FastAutomaton, EngineError>` | Computes the difference between `self` and `other`. |
| `has_intersection(&self, other: &FastAutomaton)` | `Result<bool, EngineError>` | Returns `true` if the two automata have a non-empty intersection. |
| `intersection(&self, other: &FastAutomaton)` | `Result<FastAutomaton, EngineError>` | Computes the intersection between `self` and `other`. |
| `intersection_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(automata: I)` | `Result<FastAutomaton, EngineError>` | Computes the intersection of all automata in the given iterator. |
| `intersection_all_par<'a, I: IntoParallelIterator<Item = &'a FastAutomaton>>(automata: I)` | `Result<FastAutomaton, EngineError>` | Computes in parallel the intersection of all automata in the given iterator. |
| `repeat(&self, min: u32, max_opt: Option<u32>)` | `Result<FastAutomaton, EngineError>` | Computes the repetition of the automaton between `min` and `max_opt` times; if `max_opt` is `None`, the repetition is unbounded. |
| `union(&self, other: &FastAutomaton)` | `Result<FastAutomaton, EngineError>` | Computes the union between `self` and `other`. |
| `union_all<'a, I: IntoIterator<Item = &'a FastAutomaton>>(automata: I)` | `Result<FastAutomaton, EngineError>` | Computes the union of all automata in the given iterator. |
| `union_all_par<'a, I: IntoParallelIterator<Item = &'a FastAutomaton>>(automata: I)` | `Result<FastAutomaton, EngineError>` | Computes in parallel the union of all automata in the given iterator. |

#### Analyze
| Method | Return | Description |
| -------- | ------- | ------- |
| `as_dot(&self)` | `String` | Returns the automaton's DOT representation. |
| `direct_states(&self, state: State)` | `impl Iterator<Item = State>` | Returns an iterator over states directly reachable from the given state in one transition. |
| `direct_states_vec(&self, state: State)` | `Vec<State>` | Returns a vector of states directly reachable from the given state in one transition. |
| `equivalent(&self, other: &FastAutomaton)` | `Result<bool, EngineError>` | Returns `true` if both automata accept the same language. |
| `generate_strings(&self, count: usize, offset: usize)` | `Result<Vec<String>, EngineError>` | Generates `count` strings matched by the automaton, skipping the first `offset` strings. |
| `get_accept_states(&self)` | `&IntSet<State>` | Returns a reference to the set of accept (final) states. |
| `get_cardinality(&self)` | `Cardinality<u32>` | Returns the cardinality of the automaton (i.e., the number of possible matched strings). |
| `get_condition(&self, from_state: State, to_state: State)` | `Option<&Condition>` | Returns a reference to the condition of the directed transition between the two states, if any. |
| `get_length(&self)` | `(Option<u32>, Option<u32>)` | Returns the minimum and maximum length of matched strings. |
| `get_number_of_states(&self)` | `usize` | Returns the number of states in the automaton. |
| `get_live_states(&self)` | `IntSet<State>` | Returns the set of "live" states: those that can reach an accept state. |
| `get_spanning_set(&self)` | `&SpanningSet` | Returns a reference to the automaton's spanning set. |
| `get_start_state(&self)` | `State` | Returns the start state. |
| `has_state(&self, state: State)` | `bool` | Returns `true` if the automaton contains the given state. |
| `has_transition(&self, from_state: State, to_state: State)` | `bool` | Returns `true` if there is a directed transition from `from_state` to `to_state`. |
| `in_degree(&self, state: State)` | `usize` | Returns the number of transitions to the provided state. |
| `is_accepted(&self, state: State)` | `bool` | Returns `true` if the given state is one of the accept states. |
| `is_deterministic(&self)` | `bool` | Returns `true` if the automaton is deterministic. |
| `is_empty(&self)` | `bool` | Checks if the automaton matches the empty language. |
| `is_empty_string(&self)` | `bool` | Checks if the automaton only matches the empty string `""`. |
| `is_match(&self, string: &str)` | `bool` | Returns `true` if the automaton matches the given string. |
| `is_total(&self)` | `bool` | Checks if the automaton matches all possible strings. |
| `out_degree(&self, state: State)` | `usize` | Returns the number of transitions from the provided state. |
| `print_dot(&self)` | `()` | Prints the automaton's DOT representation. |
| `states(&self)` | `impl Iterator<Item = State>` | Returns an iterator over the automaton’s states. |
| `states_vec(&self)` | `Vec<State>` | Returns a vector containing the automaton’s states. |
| `subset(&self, other: &FastAutomaton)` | `Result<bool, EngineError>` | Returns `true` if all strings accepted by `self` are also accepted by `other`. |
| `to_regex(&self)` | `RegularExpression` | Converts the term to a `RegularExpression`. |
| `transitions_from(&self, state: State)` | `impl Iterator<Item = (&Condition, &State)>` | Returns an iterator over transitions from the given state. |
| `transitions_from_vec(&self, state: State)` | `Vec<(Condition, State)>` | Returns a vector of transitions from the given state. |
| `transitions_to_vec(&self, state: State)` | `Vec<(State, Condition)>` | Returns a vector of transitions to the given state. |


### RegularExpression

`RegularExpression` is used to directly build, manipulate and analyze regular expression patterns. Not all the set operations are available, for more advanced operation such as intersection, subtraction/difference and complement it is necessary to convert into a `FastAutomaton` with the method `to_automaton()`.

#### Build/Manipulate
| Method | Return | Description |
| -------- | ------- | ------- |
| `concat(&self, other: &RegularExpression, append_back: bool)` | `RegularExpression` | Returns a new regular expression representing the concatenation of `self` and `other`; `append_back` determines their order. |
| `concat_all<'a, I: IntoIterator<Item = &'a RegularExpression>>(patterns: I)` | `RegularExpression` | Returns a regular expression that is the concatenation of all expressions in `patterns`. |
| `new(pattern: &str)` | `Result<RegularExpression, EngineError>` | Parses and simplifies the provided pattern and returns the resulting `RegularExpression`. |
| `new_empty()` | `RegularExpression` | Creates a regular expression that matches the empty language. |
| `new_empty_string()` | `RegularExpression` | Creates a regular expression that matches only the empty string `""`. |
| `new_total()` | `RegularExpression` | Creates a regular expression that matches all possible strings. |
| `parse(pattern: &str, simplify: bool)` | `Result<RegularExpression, EngineError>` | Parses the provided pattern and returns the resulting `RegularExpression`. If `simplify` is `true`, the expression is simplified during parsing. |
| `repeat(&self, min: u32, max_opt: Option<u32>)` | `RegularExpression` | Computes the repetition of the automaton between `min` and `max_opt` times; if `max_opt` is `None`, the repetition is unbounded. |
| `simplify(&self)` | `RegularExpression` | Returns a simplified version by eliminating redundant constructs and applying canonical reductions. |
| `union(&self, other: &RegularExpression)` | `RegularExpression` | Returns a regular expression matching the union of `self` and `other`. |
| `union_all<'a, I: IntoIterator<Item = &'a RegularExpression>>(patterns: I)` | `RegularExpression` | Returns a regular expression that is the union of all expressions in `patterns`. |

#### Analyze
| Method | Return | Description |
| -------- | ------- | ------- |
| `evaluate_complexity(&self)` | `f64` | Returns a heuristic score for the readability of the pattern. |
| `get_cardinality(&self)` | `Cardinality<u32>` | Returns the cardinality of the regular expression (i.e., the number of possible matched strings). |
| `get_length(&self)` | `(Option<u32>, Option<u32>)` | Returns the minimum and maximum length of possible matched strings. |
| `is_empty(&self)` | `bool` | Checks if the regular expression matches the empty language. |
| `is_empty_string(&self)` | `bool` | Checks if the regular expression only matches the empty string `""`. |
| `is_total(&self)` | `bool` | Checks if the regular expression matches all possible strings. |
| `to_automaton(&self)` | `Result<FastAutomaton, EngineError>` | Converts the regular expression to an equivalent `FastAutomaton`. |


## Bound Execution

Use a thread-local `ExecutionProfile` to cap runtime or state explosion; hitting a limit returns a specific `EngineError`.

### Time-Bounded Execution

```rust
use regexsolver::{Term, execution_profile::{ExecutionProfile, ExecutionProfileBuilder}, error::EngineError};

let term = Term::from_pattern(".*abc.*cdef.*sqdsqf.*")?;

let execution_profile = ExecutionProfileBuilder::new()
	.execution_timeout(5) // limit in milliseconds
	.build();

// We run the operation with the defined limitation
execution_profile.run(|| {
	assert_eq!(EngineError::OperationTimeOutError, term.generate_strings(1000).unwrap_err());
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

## Cross-Language Support

If you want to use this library with other programming languages, we provide a wide range of wrappers:
- [regexsolver-java](https://github.com/RegexSolver/regexsolver-java)
- [regexsolver-js](https://github.com/RegexSolver/regexsolver-js)
- [regexsolver-python](https://github.com/RegexSolver/regexsolver-python)

For more information about how to use the wrappers, you can refer to our [guide](https://docs.regexsolver.com/getting-started.html).

## License

This project is licensed under the MIT License.
