//! Generate strings matching a regex pattern.
//!
//! ```text
//! cargo run --example generate -- "[a-z]{2}[0-9]" 20
//! cargo run --example generate -- "[a-z]{2}[0-9]" 20 interleave
//! cargo run --example generate -- "[A-Z][a-z]+ [0-9]{4}" 20 shuffled
//! cargo run --example generate -- ".{4}" 20 interleave "[ -~]"
//! ```

use regexsolver::regex::RegularExpression;
use regexsolver::{
    Term,
    fast_automaton::{CharacterOrder, GenerationOptions, PathOrder},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(pattern) = args.next() else {
        eprintln!(
            "Usage: cargo run --example generate -- <pattern> [count] [sweep|interleave|shuffled] [charset]"
        );
        std::process::exit(2);
    };
    let count: usize = args.next().map(|c| c.parse()).transpose()?.unwrap_or(10);

    // `sweep` walks the language in order, `interleave` spreads the strings
    // over the shapes the pattern allows, and `shuffled` draws both the
    // shapes and their characters by seed (see `PathOrder`/`CharacterOrder`).
    let axes = match args.next().as_deref() {
        None | Some("sweep") => (PathOrder::Sweep, CharacterOrder::Ascending),
        Some("interleave") => (PathOrder::Interleave, CharacterOrder::Ascending),
        Some("shuffled") => (PathOrder::Shuffled, CharacterOrder::Shuffled),
        Some(other) => {
            eprintln!("Unknown order {other:?}, expected `sweep`, `interleave` or `shuffled`");
            std::process::exit(2);
        }
    };

    let mut options = GenerationOptions::from(axes);

    // A charset is a plain character class, e.g. "[ -~]" for printable ASCII:
    // only the strings made entirely of its characters are generated.
    if let Some(charset) = args.next() {
        match RegularExpression::new(&charset)? {
            RegularExpression::Character(charset) => options = options.with_charset(charset),
            _ => {
                eprintln!("The charset has to be a single character class, e.g. \"[ -~]\"");
                std::process::exit(2);
            }
        }
    }

    // Minimize once: pagination over the same minimized term yields
    // disjoint, consistent pages (see `Term::generate_strings`).
    let term = Term::from_pattern(&pattern)?.minimize()?;
    for string in term.generate_strings(count, 0, options)? {
        println!("{string:?}");
    }

    Ok(())
}
