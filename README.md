# RegexSolver

[![Crates.io Version](https://img.shields.io/crates/v/regexsolver)](https://crates.io/crates/regexsolver)

This repository contains the code of [RegexSolver](https://regexsolver.com/) engine.

For more information, you can check the library's [documentation](https://docs.rs/regexsolver/latest/regexsolver/).

If you want to use this library with other programming languages, we provide a wide range of wrappers:

- [regexsolver-java](https://github.com/RegexSolver/regexsolver-java)
- [regexsolver-js](https://github.com/RegexSolver/regexsolver-js)
- [regexsolver-python](https://github.com/RegexSolver/regexsolver-python)

For more information about how to use the wrappers, you can refer to our [getting started guide](https://docs.regexsolver.com/getting-started.html).

## Installation

Add the following line in your `Cargo.toml`:

```toml
[dependencies]
regexsolver = "0.2"
```

## Examples

### Union

```rust
use regexsolver::Term;

let term1 = Term::from_regex("abc").unwrap();
let term2 = Term::from_regex("de").unwrap();
let term3 = Term::from_regex("fghi").unwrap();

let union = term1.union(&[term2, term3]).unwrap();

if let Term::RegularExpression(regex) = union {
    println!("{}", regex.to_string()); // (abc|de|fghi)
}
```

### Intersection

```rust
use regexsolver::Term;

let term1 = Term::from_regex("(abc|de){2}").unwrap();
let term2 = Term::from_regex("de.*").unwrap();
let term3 = Term::from_regex(".*abc").unwrap();

let intersection = term1.intersection(&[term2, term3]).unwrap();

if let Term::RegularExpression(regex) = intersection {
    println!("{}", regex.to_string()); // deabc
}
```

### Difference/Subtraction

```rust
use regexsolver::Term;

let term1 = Term::from_regex("(abc|de)").unwrap();
let term2 = Term::from_regex("de").unwrap();

let subtraction = term1.subtraction(&term2).unwrap();

if let Term::RegularExpression(regex) = subtraction {
    assert_eq!("abc", regex.to_string());
}
```
