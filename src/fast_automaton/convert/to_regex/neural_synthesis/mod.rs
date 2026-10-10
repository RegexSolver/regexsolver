//! Short, readable regexes for automata, proposed by learned models.
//!
//! [`FastAutomaton::to_regex`] converts an automaton by state elimination:
//! always correct, but often long and hard to read. A [`NeuralSynthesizer`]
//! runs a model that reads the automaton's minimal DFA and writes regexes for
//! it, usually much simpler ones. Every candidate is **checked** against the
//! automaton with an equivalence test and discarded unless it describes
//! exactly the same language, so a model can make a result simpler, never
//! wrong. A `NeuralSynthesizer` is a [`RegexSynthesizer`]: pass it to
//! [`FastAutomaton::to_regex_with`] or [`Term::to_regex_with`](crate::Term::to_regex_with).
//!
//! When the model has no correct regex for an automaton, or the automaton is
//! beyond its limits, the conversion splits the minimal DFA
//! into independent pieces (concatenations, unions, stars, and the paths
//! around a few hub states that break up large cycles), runs the model on
//! every piece small enough, in batches, and combines the simplest regex of
//! each piece. [`NeuralSynthesizer::with_decomposition`] turns this off.
//!
//! ```no_run
//! use regexsolver::{
//!     Term,
//!     neural_synthesis::{Device, NeuralSynthesizer},
//! };
//!
//! // A model release directory: config.json and model.safetensors.
//! let synthesizer = NeuralSynthesizer::from_dir("kleene1-9m-b16-t128", Device::Cpu)?;
//!
//! let a: Term = ".*h(b|bh)".parse()?;
//! let b: Term = ".*[fi-z].*".parse()?;
//! let difference = a.difference(&b)?;
//! println!("{}", difference.to_regex_with(&synthesizer)?);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Models
//!
//! A model is a pair of files: `config.json`, which names the model and its
//! format, and `model.safetensors`, its weights. They are not bundled with the
//! crate: load them from disk ([`NeuralSynthesizer::from_files`],
//! [`from_dir`](NeuralSynthesizer::from_dir),
//! [`from_bytes`](NeuralSynthesizer::from_bytes)), or, with the
//! `neural-synthesis-hub` feature, download them from the Hugging Face Hub
//! (`NeuralSynthesizer::from_hub`). The format is read from
//! `config.json`; the formats supported so far:
//!
//! - `kleene` v1, the Kleene models (`kleene1-9m-b16-t128`): a graph
//!   transformer over the DFA and a transformer decoder over the regex, for
//!   minimal DFAs of at most 16 states and 16 bases (the disjoint character
//!   ranges labelling the transitions).
//!
//! # Devices
//!
//! Models run with [candle](https://github.com/huggingface/candle) on the CPU
//! by default. The `neural-synthesis-cuda`, `-cudnn`, `-metal`, `-mkl` and
//! `-accelerate` features enable candle's backends; pick the device with
//! [`Device`].

use std::{fmt, path::Path};

use serde::Deserialize;

use crate::{
    error::EngineError,
    execution_profile::ExecutionProfile,
    fast_automaton::{FastAutomaton, RegexSynthesizer},
    regex::RegularExpression,
};

mod decompose;
#[cfg(all(feature = "neural-synthesis-hub", not(target_family = "wasm")))]
mod hub;
mod kleene;

/// Where a model runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// The CPU (faster with the `neural-synthesis-mkl` or
    /// `neural-synthesis-accelerate` feature).
    #[default]
    Cpu,
    /// The CUDA GPU with this ordinal; needs the `neural-synthesis-cuda`
    /// feature.
    Cuda(usize),
    /// The Metal GPU with this ordinal; needs the `neural-synthesis-metal`
    /// feature.
    Metal(usize),
}

impl Device {
    /// The first CUDA GPU if the crate was built with CUDA support and one is
    /// available, else the first Metal GPU likewise, else the CPU.
    pub fn best_available() -> Self {
        if candle_core::utils::cuda_is_available() {
            Device::Cuda(0)
        } else if candle_core::utils::metal_is_available() {
            Device::Metal(0)
        } else {
            Device::Cpu
        }
    }

    fn to_candle(self) -> Result<candle_core::Device, NeuralSynthesisError> {
        let device = match self {
            Device::Cpu => return Ok(candle_core::Device::Cpu),
            Device::Cuda(ordinal) => candle_core::Device::new_cuda(ordinal),
            Device::Metal(ordinal) => candle_core::Device::new_metal(ordinal),
        };
        device.map_err(|err| NeuralSynthesisError::Device(format!("{self:?}: {err}")))
    }
}

/// An error loading a model.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NeuralSynthesisError {
    /// A model file could not be read.
    Io(String),
    /// `config.json` is invalid.
    Config(String),
    /// `config.json` describes a model format this version of the crate
    /// cannot run.
    UnsupportedFormat {
        /// The `format` field.
        format: String,
        /// The `format_version` field.
        version: u32,
    },
    /// The weights do not match the model `config.json` describes.
    Weights(String),
    /// The device is not available, or the crate was built without its
    /// backend.
    Device(String),
    /// The model could not be downloaded from the Hugging Face Hub.
    Hub(String),
}

impl fmt::Display for NeuralSynthesisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuralSynthesisError::Io(err) => write!(f, "cannot read the model: {err}"),
            NeuralSynthesisError::Config(err) => write!(f, "invalid model config: {err}"),
            NeuralSynthesisError::UnsupportedFormat { format, version } => {
                write!(f, "unsupported model format: {format} v{version}")
            }
            NeuralSynthesisError::Weights(err) => write!(f, "invalid model weights: {err}"),
            NeuralSynthesisError::Device(err) => write!(f, "device unavailable: {err}"),
            NeuralSynthesisError::Hub(err) => write!(f, "cannot download the model: {err}"),
        }
    }
}

impl std::error::Error for NeuralSynthesisError {}

impl From<std::io::Error> for NeuralSynthesisError {
    fn from(err: std::io::Error) -> Self {
        NeuralSynthesisError::Io(err.to_string())
    }
}

/// How a model draws its candidates.
#[derive(Clone, Copy, Debug)]
struct Sampling {
    /// At least 1: the greedy decoding, then samples.
    candidates: usize,
    temperature: f64,
    seed: u64,
}

/// A model format: turns a minimal DFA into candidate regexes, which the
/// caller verifies.
trait Model: Send + Sync {
    /// The largest minimal DFA, in states, the model takes.
    fn max_states(&self) -> usize;

    /// The largest minimal DFA, in bases, the model takes.
    fn max_bases(&self) -> usize;

    /// Candidate regexes for each of `dfas`, DFAs from [`minimal_dfa`],
    /// decoded together, in decoding order (the greedy one first); none for
    /// a DFA beyond the model's limits. Candidates are not checked and may
    /// repeat; a DFA's candidates do not depend on the others.
    fn candidates(
        &self,
        dfas: &[&FastAutomaton],
        sampling: &Sampling,
    ) -> Result<Vec<Vec<RegularExpression>>, String>;
}

/// The fields of `config.json` every format has.
#[derive(serde::Deserialize)]
struct Header {
    name: String,
    format: String,
    format_version: u32,
}

/// A loaded model, which proposes regexes for automata. See the [module
/// documentation](self).
///
/// Loading parses the weights once; a `NeuralSynthesizer` is then cheap to
/// share between threads.
pub struct NeuralSynthesizer {
    name: String,
    model: Box<dyn Model>,
    sampling: Sampling,
    decomposition: bool,
}

impl fmt::Debug for NeuralSynthesizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeuralSynthesizer")
            .field("name", &self.name)
            .field("sampling", &self.sampling)
            .field("decomposition", &self.decomposition)
            .finish_non_exhaustive()
    }
}

impl NeuralSynthesizer {
    /// Default number of candidates per automaton: the greedy decoding plus
    /// seven samples.
    pub const DEFAULT_CANDIDATES: usize = 8;

    /// Loads a model from its `config.json` and `model.safetensors` files.
    pub fn from_files(
        config: impl AsRef<Path>,
        weights: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self, NeuralSynthesisError> {
        let config = std::fs::read(config)?;
        let weights = std::fs::read(weights)?;
        Self::from_bytes(&config, weights, device)
    }

    /// Loads a model from a directory holding `config.json` and
    /// `model.safetensors`.
    pub fn from_dir(dir: impl AsRef<Path>, device: Device) -> Result<Self, NeuralSynthesisError> {
        let dir = dir.as_ref();
        Self::from_files(
            dir.join("config.json"),
            dir.join("model.safetensors"),
            device,
        )
    }

    /// Loads a model from the contents of its `config.json` and
    /// `model.safetensors`.
    pub fn from_bytes(
        config: &[u8],
        weights: Vec<u8>,
        device: Device,
    ) -> Result<Self, NeuralSynthesisError> {
        let config: serde_json::Value = serde_json::from_slice(config)
            .map_err(|err| NeuralSynthesisError::Config(err.to_string()))?;
        let header = Header::deserialize(&config)
            .map_err(|err| NeuralSynthesisError::Config(err.to_string()))?;
        let model: Box<dyn Model> = match (header.format.as_str(), header.format_version) {
            (kleene::FORMAT, kleene::FORMAT_VERSION) => {
                Box::new(kleene::Kleene::new(&config, weights, &device.to_candle()?)?)
            }
            _ => {
                return Err(NeuralSynthesisError::UnsupportedFormat {
                    format: header.format,
                    version: header.format_version,
                });
            }
        };
        Ok(Self {
            name: header.name,
            model,
            sampling: Sampling {
                candidates: Self::DEFAULT_CANDIDATES,
                temperature: 1.0,
                seed: 0,
            },
            decomposition: true,
        })
    }

    /// Sets how many candidates are decoded per automaton (at least 1): the
    /// greedy decoding, then samples. More candidates find a correct regex
    /// more often, at the cost of decoding and checking them.
    pub fn with_candidates(mut self, candidates: usize) -> Self {
        self.sampling.candidates = candidates.max(1);
        self
    }

    /// Sets the sampling temperature of the candidates after the first
    /// (default 1). A temperature that is not positive samples greedily, so
    /// every candidate is the first.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.sampling.temperature = temperature;
        self
    }

    /// Sets the seed of the sampling (default 0). Proposals are
    /// deterministic for a given seed and device.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.sampling.seed = seed;
        self
    }

    /// Sets whether converting ([`FastAutomaton::to_regex_with`]) splits an
    /// automaton the model has no correct regex for, or is too large for,
    /// into independent pieces and runs the model on those (default `true`).
    /// Splitting finds simpler regexes for many more automata, at the cost of
    /// running the model on many pieces: up to seconds on the CPU for
    /// automata of dozens of states.
    pub fn with_decomposition(mut self, decomposition: bool) -> Self {
        self.decomposition = decomposition;
        self
    }

    /// The model's name, from its `config.json` (e.g. `kleene1-9m-b16-t128`).
    pub fn model_name(&self) -> &str {
        &self.name
    }

    /// The largest minimal DFA, in states, the model takes.
    pub fn max_states(&self) -> usize {
        self.model.max_states()
    }

    /// The largest minimal DFA, in bases (disjoint character ranges), the
    /// model takes.
    pub fn max_bases(&self) -> usize {
        self.model.max_bases()
    }

    /// Asks the model for a regex describing exactly the language of
    /// `automaton`.
    ///
    /// Returns the simplest candidate equivalent to the automaton, or `None`
    /// when its minimal DFA is beyond the model's limits or no candidate is
    /// correct. Candidates whose check fails (for instance on the execution
    /// profile's timeout) are skipped.
    ///
    /// Fails if `automaton` cannot be minimized within the execution
    /// profile, or if the profile's deadline passes during inference.
    pub fn propose(
        &self,
        automaton: &FastAutomaton,
    ) -> Result<Option<RegularExpression>, EngineError> {
        Ok(self.propose_batch(&[automaton])?.pop().flatten())
    }

    /// [`propose`](Self::propose) for several automata at once: the model
    /// decodes them together, which is much faster than one by one. The
    /// result for each automaton is the same as `propose`'s.
    pub fn propose_batch(
        &self,
        automata: &[&FastAutomaton],
    ) -> Result<Vec<Option<RegularExpression>>, EngineError> {
        let mut dfas = Vec::with_capacity(automata.len());
        for automaton in automata {
            dfas.push(minimal_dfa(automaton)?);
        }
        let eligible: Vec<usize> = (0..dfas.len())
            .filter(|i| dfas[*i].number_of_states() <= self.model.max_states())
            .collect();
        let mut proposals = vec![None; dfas.len()];
        if eligible.is_empty() {
            return Ok(proposals);
        }
        let inputs: Vec<&FastAutomaton> = eligible.iter().map(|i| &dfas[*i]).collect();
        let candidates = match self.model.candidates(&inputs, &self.sampling) {
            Ok(candidates) => candidates,
            Err(err) => {
                tracing::warn!("{} inference failed: {err}", self.name);
                return Ok(proposals);
            }
        };
        // Inference is bounded by the model's size, not by the input, but can
        // still outlast a short deadline.
        ExecutionProfile::get().assert_not_timed_out()?;

        for (i, candidates) in eligible.into_iter().zip(candidates) {
            proposals[i] = Self::check(&dfas[i], candidates);
        }
        Ok(proposals)
    }

    /// A regex for `automaton`, which the model has no correct regex for or
    /// cannot take, from its minimal DFA split into pieces the model converts
    /// (see the [module documentation](self)); `None` when decomposition is
    /// disabled. The result describes exactly the automaton's language.
    ///
    /// Fails if the execution profile's deadline passes.
    pub(crate) fn propose_decomposed(
        &self,
        automaton: &FastAutomaton,
    ) -> Result<Option<RegularExpression>, EngineError> {
        if !self.decomposition {
            return Ok(None);
        }
        let mut decomposer = decompose::Decomposer::new(Some(self));
        decomposer.whole_tried = true;
        let regex = decomposer.convert(automaton)?;
        tracing::debug!(stats = ?decomposer.stats, "decomposed");
        Ok(Some(regex))
    }

    /// The simplest of `candidates` equivalent to `dfa`.
    fn check(dfa: &FastAutomaton, candidates: Vec<RegularExpression>) -> Option<RegularExpression> {
        let mut unique: Vec<RegularExpression> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if !unique.contains(&candidate) {
                unique.push(candidate);
            }
        }
        let mut ranked: Vec<(f64, usize, RegularExpression)> = unique
            .into_iter()
            .enumerate()
            .map(|(i, regex)| (regex.evaluate_complexity(), i, regex))
            .collect();
        // The simplest first; ties keep the earliest (the greedy one first).
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

        for (_, _, regex) in ranked {
            match regex.to_automaton().and_then(|a| a.equivalent(dfa)) {
                Ok(true) => return Some(regex),
                Ok(false) => {}
                Err(err) => tracing::debug!("candidate {regex} not checked: {err}"),
            }
        }
        None
    }
}

impl RegexSynthesizer for NeuralSynthesizer {
    /// The model first proposes a regex for the whole automaton. When it has
    /// no correct one, or the automaton is beyond its limits, the automaton's
    /// minimal DFA is split into independent pieces (unless
    /// [`with_decomposition`](NeuralSynthesizer::with_decomposition) disables
    /// it), each converted by the model or by state elimination, whichever is
    /// simpler, and their regexes combined. Returns the simpler of that result
    /// and [`FastAutomaton::to_regex`]'s state elimination; state elimination
    /// alone if the model fails (for instance on the execution profile's
    /// deadline). The result always describes exactly the automaton's
    /// language.
    fn synthesize(&self, automaton: &FastAutomaton) -> Result<RegularExpression, EngineError> {
        let proposed = self.propose(automaton).and_then(|proposed| match proposed {
            Some(proposed) => Ok(Some(proposed)),
            None => self.propose_decomposed(automaton),
        });
        let proposed = match proposed {
            Ok(proposed) => proposed,
            Err(err) => {
                tracing::debug!("neural synthesis skipped: {err}");
                None
            }
        };
        let Some(proposed) = proposed else {
            return automaton.to_regex();
        };
        match automaton.to_regex() {
            Ok(eliminated) if eliminated.evaluate_complexity() < proposed.evaluate_complexity() => {
                Ok(eliminated)
            }
            // The proposal stands on its own if state elimination runs out
            // of budget.
            _ => Ok(proposed),
        }
    }
}

/// Determinizes, minimizes, and shrinks the spanning set to the coarsest one
/// the transitions allow, so the number of bases is a property of the
/// language rather than of the construction.
fn minimal_dfa(automaton: &FastAutomaton) -> Result<FastAutomaton, EngineError> {
    let mut dfa = automaton.determinize()?.into_owned();
    dfa.minimize()?;
    Ok(dfa)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model under `KLEENE_MODEL_DIR`, for the tests that need weights:
    /// `KLEENE_MODEL_DIR=<release dir> cargo test --release --features neural-synthesis neural_synthesis -- --ignored`
    fn kleene() -> NeuralSynthesizer {
        let dir = std::env::var("KLEENE_MODEL_DIR").expect("KLEENE_MODEL_DIR");
        NeuralSynthesizer::from_dir(dir, Device::Cpu).unwrap()
    }

    #[test]
    #[ignore = "needs the model weights (KLEENE_MODEL_DIR)"]
    fn to_regex_with_keeps_the_language() {
        let synthesizer = kleene();
        assert_eq!(synthesizer.model_name(), "kleene1-9m-b16-t128");
        assert_eq!(
            (synthesizer.max_states(), synthesizer.max_bases()),
            (16, 16)
        );
        for (a, b) in [
            (".*h(b|bh)", ".*[fi-z].*"),
            (".*abc.*", ".*def.*"),
            ("x*", "(xxx)*"),
            ("(ab|cd)*", "(abcd)*"),
            ("[a-z]{2,6}", ".*q.*"),
            ("(a|b)*abb", "a.*"),
        ] {
            let a = RegularExpression::new(a).unwrap().to_automaton().unwrap();
            let b = RegularExpression::new(b).unwrap().to_automaton().unwrap();
            let automaton = a.difference(&b).unwrap();
            let regex = automaton.to_regex_with(&synthesizer).unwrap();
            assert!(
                regex
                    .to_automaton()
                    .unwrap()
                    .equivalent(&automaton)
                    .unwrap(),
                "{regex}"
            );
            let eliminated = automaton.to_regex().unwrap();
            assert!(
                regex.evaluate_complexity() <= eliminated.evaluate_complexity(),
                "{regex} vs {eliminated}"
            );
        }
    }

    #[test]
    #[ignore = "needs the model weights (KLEENE_MODEL_DIR)"]
    fn decomposes_automata_beyond_the_model() {
        let synthesizer = kleene();
        let mut automaton = RegularExpression::new(".*abc.*")
            .unwrap()
            .to_automaton()
            .unwrap();
        for other in [".*def.*", ".*gh.*"] {
            let other = RegularExpression::new(other)
                .unwrap()
                .to_automaton()
                .unwrap();
            automaton = automaton.intersection(&other).unwrap();
        }
        // As a DFA, like the results of differences and complements.
        let automaton = minimal_dfa(&automaton).unwrap();
        assert!(automaton.number_of_states() > synthesizer.max_states());
        let eliminated = automaton.to_regex().unwrap();

        let regex = automaton.to_regex_with(&synthesizer).unwrap();
        assert!(
            regex
                .to_automaton()
                .unwrap()
                .equivalent(&automaton)
                .unwrap(),
            "{regex}"
        );
        assert!(regex.evaluate_complexity() < eliminated.evaluate_complexity());

        let undecomposed = automaton
            .to_regex_with(&synthesizer.with_decomposition(false))
            .unwrap();
        assert_eq!(undecomposed, eliminated);
    }

    #[test]
    #[ignore = "needs the model weights (KLEENE_MODEL_DIR)"]
    fn simplifies_the_model_card_example() {
        // `[a-egh]*hbh?`, as state elimination writes it.
        let automaton = RegularExpression::new(
            "([a-eg]|h+[ac-eg]|(h+b)+([a-eg]|h[ac-eg]|h{2,}[ac-eg]))*(h+b)+h?",
        )
        .unwrap()
        .to_automaton()
        .unwrap();
        let regex = kleene()
            .with_candidates(1)
            .propose(&automaton)
            .unwrap()
            .unwrap();
        assert_eq!(regex.to_string(), "[a-egh]*hbh?");
    }

    #[test]
    fn rejects_unknown_formats() {
        let config = br#"{"name": "x", "format": "other", "format_version": 1}"#;
        assert!(matches!(
            NeuralSynthesizer::from_bytes(config, vec![], Device::Cpu),
            Err(NeuralSynthesisError::UnsupportedFormat { format, version: 1 }) if format == "other"
        ));
        assert!(matches!(
            NeuralSynthesizer::from_bytes(b"not json", vec![], Device::Cpu),
            Err(NeuralSynthesisError::Config(_))
        ));
    }

    #[test]
    fn devices_need_their_backend() {
        assert!(Device::default().to_candle().unwrap().is_cpu());
        #[cfg(not(feature = "neural-synthesis-cuda"))]
        assert!(matches!(
            Device::Cuda(0).to_candle(),
            Err(NeuralSynthesisError::Device(_))
        ));
        #[cfg(not(feature = "neural-synthesis-cuda"))]
        assert_eq!(Device::best_available(), Device::Cpu);
    }
}
