//! Generate strings matching a regex pattern.
//!
//! ```text
//! cargo run --example generate -- "[a-z]{2}[0-9]" 20
//! ```

use regexsolver::Term;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(pattern) = args.next() else {
        eprintln!("Usage: cargo run --example generate -- <pattern> [count]");
        std::process::exit(2);
    };
    let count: usize = args.next().map(|c| c.parse()).transpose()?.unwrap_or(10);

    // Minimize once: pagination over the same minimized term yields
    // disjoint, consistent pages (see `Term::generate_strings`).
    let term = Term::from_pattern(&pattern)?.minimize()?;
    for string in term.generate_strings(count, 0)? {
        println!("{string:?}");
    }

    Ok(())
}
