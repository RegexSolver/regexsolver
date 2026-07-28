//! Generate strings matching a regex pattern.
//!
//! ```text
//! cargo run --example generate -- "[a-z]{2}[0-9]" 20
//! cargo run --example generate -- "[a-z]{2}[0-9]" 20 sampled
//! cargo run --example generate -- ".{4}" 20 sampled "[ -~]"
//! ```

use regexsolver::regex::RegularExpression;
use regexsolver::{
    Term,
    fast_automaton::{GenerationOptions, GenerationOrder},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(pattern) = args.next() else {
        eprintln!(
            "Usage: cargo run --example generate -- <pattern> [count] [exhaustive|sampled] [charset]"
        );
        std::process::exit(2);
    };
    let count: usize = args.next().map(|c| c.parse()).transpose()?.unwrap_or(10);

    // `exhaustive` sweeps the language in order, `sampled` spreads the strings
    // over the shapes the pattern allows (see `GenerationOrder`).
    let order = match args.next().as_deref() {
        None | Some("exhaustive") => GenerationOrder::Exhaustive,
        Some("sampled") => GenerationOrder::Sampled,
        Some(other) => {
            eprintln!("Unknown order {other:?}, expected `exhaustive` or `sampled`");
            std::process::exit(2);
        }
    };

    let mut options = GenerationOptions::from(order);

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
