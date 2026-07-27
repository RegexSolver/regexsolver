//! Explore how two regex patterns relate as languages.
//!
//! ```text
//! cargo run --example relate -- "(abc|de){2}" ".*xy"
//! ```

use regexsolver::{Term, fast_automaton::GenerationOrder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(a), Some(b)) = (args.next(), args.next()) else {
        eprintln!("Usage: cargo run --example relate -- <pattern-a> <pattern-b>");
        std::process::exit(2);
    };

    let a_term = Term::from_pattern(&a)?;
    let b_term = Term::from_pattern(&b)?;

    println!("a = {a}");
    println!("b = {b}");
    println!();

    if a_term.equivalent(&b_term)? {
        println!("a and b match exactly the same strings.");
        return Ok(());
    }
    println!("equivalent:    no");
    println!("a subset of b: {}", a_term.subset(&b_term)?);
    println!("b subset of a: {}", b_term.subset(&a_term)?);
    println!();

    let intersection = a_term.intersection([&b_term])?;
    if intersection.is_empty()? {
        println!("a ∩ b = [] (no string matches both)");
    } else {
        println!("a ∩ b = {}", intersection.to_pattern()?);
        println!(
            "        e.g. {:?}",
            intersection.generate_strings(5, 0, GenerationOrder::Sampled)?
        );
    }

    let pattern_or_empty = |term: Term| -> Result<String, Box<dyn std::error::Error>> {
        Ok(if term.is_empty()? {
            "[]".to_string()
        } else {
            term.to_pattern()?
        })
    };
    println!("a - b = {}", pattern_or_empty(a_term.difference(&b_term)?)?);
    println!("b - a = {}", pattern_or_empty(b_term.difference(&a_term)?)?);

    Ok(())
}
