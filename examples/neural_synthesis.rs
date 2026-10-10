//! Converts automata to regexes with state elimination and with a neural
//! synthesis model, side by side.
//!
//! ```text
//! cargo run --release --features neural-synthesis-hub --example neural_synthesis -- [model]
//! ```
//!
//! `model` is a directory holding `config.json` and `model.safetensors`, or a
//! Hugging Face Hub repository id (default:
//! `NeuralSynthesizer::DEFAULT_HUB_MODEL`). The model runs on the best
//! available device.

use std::{path::Path, time::Instant};

use regexsolver::{
    Term,
    neural_synthesis::{Device, NeuralSynthesizer},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| NeuralSynthesizer::DEFAULT_HUB_MODEL.to_string());
    let device = Device::best_available();
    let synthesizer = if Path::new(&model).is_dir() {
        NeuralSynthesizer::from_dir(&model, device)?
    } else {
        NeuralSynthesizer::from_hub(&model, device)?
    };
    println!("{} on {device:?}", synthesizer.model_name());

    for (a, b) in [
        (".*h(b|bh)", ".*[fi-z].*"),
        (".*abc.*", ".*def.*"),
        ("x*", "(xxx)*"),
        ("(ab|cd)*", "(abcd)*"),
        ("[a-z]{2,6}", ".*q.*"),
    ] {
        let difference = a.parse::<Term>()?.difference(&b.parse::<Term>()?)?;
        let start = Instant::now();
        let eliminated = difference.to_regex()?;
        let elapsed_eliminated = start.elapsed();
        let start = Instant::now();
        let synthesized = difference.to_regex_with(&synthesizer)?;
        let elapsed_synthesized = start.elapsed();
        println!("{a} - {b}");
        println!("  state elimination ({elapsed_eliminated:.1?}): {eliminated}");
        println!("  neural synthesis  ({elapsed_synthesized:.1?}): {synthesized}");
    }
    Ok(())
}
