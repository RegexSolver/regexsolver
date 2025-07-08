
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
-  **Performance & Tuning**: Pluggable `ExecutionProfile` to bound cost and resource usage.

## Installation
Add the following line in your `Cargo.toml`:
```toml
[dependencies]
regexsolver = "1"
```
## Examples

```rust
// Create terms from regex
let t1 = Term::from_regex("abc.*")?;
let t2 = Term::from_regex(".*xyz")?;

// Concatenate
let concat = t1.concat(&[t2])?;
assert_eq!(concat.to_string(), "abc.*xyz");

// Union
let union = t1.union(&[Term::from_regex("fgh")?])?; // (abc.*|fgh)
assert_eq!(union.to_string(), "(abc.*|fgh)");

// Intersection
let inter = Term::from_regex("(ab|xy){2}")?.intersection(&[Term::from_regex(".*xy")?])?; // (ab|xy)xy
assert_eq!(inter.to_string(), "(ab|xy)xy");

// Subtraction
let diff = Term::from_regex("a*")?.subtraction(&Term::from_regex("")?)?;
assert_eq!(diff.to_string(), "a+");

// Repetition
let rep = Term::from_regex("abc")?.repeat(2, Some(4))?; // (abc){2,4}
assert_eq!(rep.to_string(), "(abc){2,4}");

// Analyze
let details = rep.get_details()?;
assert_eq!(details.get_length(), &(Some(6), Some(12)));
assert!(!details.is_empty());

// Generate examples
let samples = Term::from_regex("(x|y){1,3}")?.generate_strings(5)?;
println!("Some matches: {:?}", samples);

// Equivalence & subset
let a = Term::from_regex("a+")?;
let b = Term::from_regex("a*")?;
assert!(!a.are_equivalent(&b)?);
assert!(a.is_subset_of(&b)?);
```

## Execution Profiles
By default, all operations run without limits. For heavy or untrusted patterns, use an `ExecutionProfile` to cap time, memory or term count:
