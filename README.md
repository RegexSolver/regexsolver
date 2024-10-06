# RegexSolver

[![Crates.io Version](https://img.shields.io/crates/v/regexsolver)](https://crates.io/crates/regexsolver)

This repository contains the code of RegexSolver engine.

## Installation

Add the following line in your `Cargo.toml`:

```toml
[dependencies]
regexsolver = "0.1"
```

## Examples

```rust
use regexsolver::{intersection, regex::RegularExpression, subtraction, Term};


let term1 = Term::RegularExpression(RegularExpression::new("(abc|de|fg){2,}").unwrap());
let term2 = Term::RegularExpression(RegularExpression::new("de.*").unwrap());
let term3 = Term::RegularExpression(RegularExpression::new(".*abc").unwrap());

let term4 = Term::RegularExpression(RegularExpression::new(".+(abc|de).+").unwrap());

let intersection = intersection(&[term1, term2, term3]).unwrap();

let result = subtraction(&intersection, &term4).unwrap();

if let Term::RegularExpression(regex) = result {
    println!("result={}", regex); // result=de(fg)*abc
}
```
