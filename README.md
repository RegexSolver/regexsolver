# RegexSolver
[![Crates.io Version](https://img.shields.io/crates/v/regexsolver)](https://crates.io/crates/regexsolver)

**RegexSolver** is a high-performance Rust library for building, combining, and analyzing regular expressions and finite automata. Ideal for constraint solvers, code generators, test-case generators, and any use case requiring rich regex/automaton operations.

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
let t1 = Term::from_regex("abc.*").unwrap();
let t2 = Term::from_regex(".*xyz").unwrap();

// Concatenate
let concat = t1.concat(&[t2]).unwrap();
assert_eq!(concat.to_string(), "abc.*xyz");

// Union
let union = t1.union(&[Term::from_regex("fgh").unwrap()]).unwrap();
assert_eq!(union.to_string(), "(abc.*|fgh)");

// Intersection
let inter = Term::from_regex("(ab|xy){2}")
    .unwrap()
    .intersection(&[Term::from_regex(".*xy").unwrap()])
    .unwrap(); // (ab|xy)xy
assert_eq!(inter.to_string(), "(ab|xy)xy");

// Subtraction
let diff = Term::from_regex("a*")
    .unwrap()
    .subtraction(&Term::from_regex("").unwrap())
    .unwrap();
assert_eq!(diff.to_string(), "a+");

// Repetition
let rep = Term::from_regex("abc").unwrap().repeat(2, Some(4)).unwrap();
assert_eq!(rep.to_string(), "(abc){2,4}");

// Analyze
assert_eq!(rep.get_length(), (Some(6), Some(12)));
assert!(!rep.is_empty());

// Generate examples
let samples = Term::from_regex("(x|y){1,3}")
    .unwrap()
    .generate_strings(5)
    .unwrap();
println!("Some matches: {:?}", samples);

// Equivalence & subset
let a = Term::from_regex("a+").unwrap();
let b = Term::from_regex("a*").unwrap();
assert!(!a.are_equivalent(&b).unwrap());
assert!(a.is_subset_of(&b).unwrap());
```

## API

### Term

`Term` is an enum designed to represent either a regular expression pattern or a compiled automaton. This unified representation enables seamless and efficient execution of set operations across multiple instances. It's particularly valuable when working  with both regular expressions and automata, allowing operations to be performed transparently regardless of the underlying representation.

| Method | Return | Description |
| -------- | ------- | ------- |
| `Term::new_empty()` | `Term` | Create a term that matches the empty language. |
| `Term::new_total()` | `Term` | Create a term that matches all possible strings. |
| `Term::new_empty_string()` | `Term` | Create a term that only match the empty string `""`. |
| `Term::from_pattern(pattern: &str)` | `Result<Term, EngineError>` | Parse the provided pattern and return a new `Term` holding the resulting `RegularExpression`. |
| `Term::from_regex(regex: RegularExpression)` | `Term` | Create a new `Term` holding the provided `RegularExpression`. |
| `Term::from_automaton(automaton: FastAutomaton)` | `Term` | Create a new `Term` holding the provided `FastAutomaton`. |
| `self.concat(terms: &[Term])` | `Result<Term, EngineError>` | Compute the concatenation of the given collection of terms. Returns the resulting term. |
| `self.union(terms: &[Term])` | `Result<Term, EngineError>` | Compute the union of the given collection of terms. Returns the resulting term. |
| `self.intersection(terms: &[Term])` | `Result<Term, EngineError>` | Compute the intersection of the given collection of terms. Returns the resulting term. |
| `self.subtraction(subtrahend: &Term)` | `Result<Term, EngineError>` | Compute the subtraction/difference of the two given terms. Returns the resulting term. |
| `self.difference(subtrahend: &Term)` | `Result<Term, EngineError>` | See `self.subtraction(subtrahend: &Term)`. |
| `self.repeat(min: u32, max_opt: Option<u32>)` | `Result<Term, EngineError>` | Returns the repetition of the current term, between `min` and `max_opt` times. If `max_opt` is `None`, the repetition is unbounded. |
| `self.generate_strings(count: usize)` | `Result<Vec<String>, EngineError>` | Generate the given count of strings matched by the given term. |
| `self.are_equivalent(term: &Term)` | `Result<bool, EngineError>` | Compute whether the current term and the given term are equivalent. Returns `true` if both terms accept the same language. |
| `self.is_subset_of(term: &Term)` | `Result<bool, EngineError>` | Compute whether the current term is a subset of the given term. Returns `true` if all strings matched by the current term are also matched by the given term. |
| `self.is_empty()` | `bool` | Check if the current term matches the empty language. |
| `self.is_total()` | `bool` | Check if the current term matches all possible strings. |
| `self.is_empty_string()` | `bool` | Check if the current term only match the empty string `""`. |
| `self.get_length()` | `(Option<u32>, Option<u32>)` | Returns the minimum and maximum length of the possible matched strings. |
| `self.get_cardinality()` | `Result<Cardinality<u32>, EngineError>` | Returns the cardinality of the provided term (i.e. the number of the possible matched strings). |


### FastAutomaton

`FastAutomaton` is used to directly build, manipulate and analyze automata. To convert an automaton to a `RegularExpression` the method `to_regex()` can be used, not all automaton can be converted to a regular expression.



### RegularExpression

`RegularExpression` is used to directly build, manipulate and analyze regular expression patterns. Not all the set operations are available, for more advanced operation such as intersection, subtraction/difference and complement it is necessary to convert in to a `FastAutomaton` with the method `to_automaton()`.

## Error Handling

## Bound Execution

By default, all operations run without limits. For heavy or untrusted patterns, use a thread local `ExecutionProfile` to cap execution time and maximum number of states in used automata.

### Time-Bounded Execution

```rust
use regexsolver::{Term, execution_profile::{ExecutionProfile, ExecutionProfileBuilder}, error::EngineError};

let term = Term::from_regex(".*abc.*cdef.*sqdsqf.*").unwrap();

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

let term1 = Term::from_regex(".*abcdef.*").unwrap();
let term2 = Term::from_regex(".*defabc.*").unwrap();

let execution_profile = ExecutionProfileBuilder::new()
	.max_number_of_states(5) // We set the limit
	.build();

// We run the operation with the defined limitation
execution_profile.run(|| {
	assert_eq!(EngineError::AutomatonHasTooManyStates, term1.intersection(&[term2]).unwrap_err());
});
```



## Key Concepts & Limitations

RegexSolver supports a subset of regular expressions that adhere to the principles of regular languages. Here are the key characteristics and limitations of the regular expressions supported by RegexSolver:

- **Anchored Expressions:** All regular expressions in RegexSolver are anchored. This means that the expressions are treated as if they start and end at the boundaries of the input text. For example, the expression `abc` will match the string "abc" but not "xabc" or "abcx".
- **Lookahead/Lookbehind:** RegexSolver does not support lookahead (`(?=...)`) or lookbehind (`(?<=...)`) assertions. Using them would return an error.
- **Greedy/Ungreedy Quantifiers:** The concept of ungreedy (`*?`, `+?`, `??`) quantifiers is not supported. All quantifiers are treated as greedy. For example, `a*` or `a*?` will match the longest possible sequence of "a"s.
- **Line Feed and Dot:** RegexSolver handle every characters the same way. The dot character . matches every possible unicode characters including the line feed (`\n`).
- **Pure Regular Expressions:** RegexSolver focuses on pure regular expressions as defined in regular language theory. This means features that extend beyond regular languages, such as backreferences (`\1`, `\2`, etc.), are not supported. Any use of backreference would return an error.
- **Empty Regular Expressions:** An empty regular expression is denoted by `[]`, which represents a pattern that matches no input, not even an empty string.

RegexSolver is based on the [regex-syntax](https://docs.rs/regex-syntax/0.8.5/regex_syntax/) library for parsing expressions. As a result, unsupported features supported by the parser will be parsed but ignored. This allows for some flexibility in writing regular expressions, but it is important to be aware of the unsupported features to avoid unexpected behavior.

## Cross-Language Support


If you want to use this library with other programming languages, we provide a wide range of wrappers:
- [regexsolver-java](https://github.com/RegexSolver/regexsolver-java)
- [regexsolver-js](https://github.com/RegexSolver/regexsolver-js)
- [regexsolver-python](https://github.com/RegexSolver/regexsolver-python)

For more information about how to use the wrappers, you can refer to our [getting started guide](https://docs.regexsolver.com/getting-started.html).

## License

This project is licensed under the MIT License.
