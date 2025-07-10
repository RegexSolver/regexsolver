
# RegexSolver
[![Crates.io Version](https://img.shields.io/crates/v/regexsolver)](https://crates.io/crates/regexsolver)
 A high-performance Rust library for building, combining, and analyzing regular expressions and finite automata.
 
Ideal for constraint solvers, code generators, test-case generators, and any use case requiring rich regex/automaton operations at scale.

## Key Features
-  **Dual Representation**: Work interchangeably with regex syntax or compiled automata via the `Term` enum.
-  **Set Operations**: Concatenate, union, intersect, subtract, and repeat regex/automaton terms.
-  **Analysis & Properties**:
	- Compute language **cardinality**, **length bounds**, **emptiness**, and **totality**.
	- Check **equivalence** and **subset** relations between terms.
-  **String Generation**: Generate example strings matching a term, for testing or sampling.
-  **Performance & Tuning**: Pluggable `ExecutionProfile` to bound time and resource usage.

## Installation
Add the following line in your `Cargo.toml`:
```toml
[dependencies]
regexsolver = "1"
```
## Examples

```rust
use regexsolver::Term;

// Create terms from regex
let t1 = Term::from_regex("abc.*").unwrap();
let t2 = Term::from_regex(".*xyz").unwrap();

// Concatenate
let concat = t1.concat(&[t2]).unwrap();
assert_eq!(concat.to_string(), "abc.*xyz");

// Union
let union = t1.union(&[Term::from_regex("fgh").unwrap()]).unwrap(); // (abc.*|fgh)
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
let rep = Term::from_regex("abc").unwrap().repeat(2, Some(4)).unwrap(); // (abc){2,4}
assert_eq!(rep.to_string(), "(abc){2,4}");

// Analyze
let details = rep.get_details().unwrap();
assert_eq!(details.get_length(), &(Some(6), Some(12)));
assert!(!details.is_empty());

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

## Execution Profiles
By default, all operations run without limits. For heavy or untrusted patterns, use an `ExecutionProfile` to cap execution time and maximum number of states in used automata.

### Example: Limit the execution time
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

### Example: Limit the number of states
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