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
//! A model can come with its success rate by size and difficulty of the DFA
//! (see [Models](#models)); the conversion does not run it on an automaton,
//! or a piece of one, whose rate is below a threshold
//! ([`NeuralSynthesizer::with_min_success`]): a piece that cannot be split
//! further is then converted by state elimination.
//!
//! When the model has no correct regex for an automaton, or the automaton is
//! beyond its limits or below the threshold, the conversion splits the minimal DFA
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
//! - `kleene` v1 and v2, the Kleene models (`kleene1-9m-b16-t128`,
//!   `kleene2-9m-b16-t128`): a graph transformer over the DFA and a
//!   transformer decoder over the regex, for minimal DFAs of at most 16 states
//!   and 16 bases (the disjoint character ranges labelling the transitions).
//!
//! `config.json` may give the model's success rate (`gate`): for each cell of
//! a grid over features of a minimal DFA, the share of the DFAs of a
//! calibration set for which one of the model's candidates is correct. The
//! features are `states`, `bases` (its spanning ranges) and `complexity` (that
//! of its state elimination regex); each axis lists the lower bounds of its
//! bins, and the rates are in row-major order, `null` for a cell without DFAs
//! (the model is then tried).
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
    CharRange, IntMap,
    error::EngineError,
    execution_profile::ExecutionProfile,
    fast_automaton::{FastAutomaton, RegexSynthesizer, State},
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
    /// At least 1: the greedy decoding (or the beam's hypotheses), then
    /// samples.
    candidates: usize,
    temperature: f64,
    seed: u64,
    /// The width of a beam search replacing the greedy decoding; 0 or 1:
    /// none.
    beam: usize,
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

/// A feature of a minimal DFA the success rate depends on.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Feature {
    /// Its number of states.
    States,
    /// Its number of bases: the ranges of its minimal spanning set.
    Bases,
    /// The complexity of its state elimination regex.
    Complexity,
}

#[derive(Clone, Debug, Deserialize)]
struct Axis {
    feature: Feature,
    /// The lower bounds of the bins, increasing; a value below the first is
    /// in the first bin.
    bins: Vec<f64>,
}

/// The model's success rate over a grid of features (`config.json` `gate`).
#[derive(Clone, Debug, Deserialize)]
struct Gate {
    axes: Vec<Axis>,
    /// Per cell, in row-major order over `axes`; `None` without data.
    rates: Vec<Option<f64>>,
}

impl Gate {
    fn validate(&self) -> Result<(), String> {
        let cells = self
            .axes
            .iter()
            .map(|axis| axis.bins.len())
            .product::<usize>();
        if self
            .axes
            .iter()
            .any(|axis| axis.bins.is_empty() || !axis.bins.is_sorted())
        {
            return Err("gate: every axis needs increasing bins".to_string());
        }
        if self.rates.len() != cells {
            return Err(format!(
                "gate: {} rates for {cells} cells",
                self.rates.len()
            ));
        }
        Ok(())
    }

    /// The success rate for `dfa`, a DFA from [`minimal_dfa`], whose state
    /// elimination regex is `eliminated` when already computed; `None` when
    /// its cell has no data.
    fn rate(
        &self,
        dfa: &FastAutomaton,
        eliminated: Option<&RegularExpression>,
    ) -> Result<Option<f64>, EngineError> {
        let mut cell = 0;
        for axis in &self.axes {
            let value = match axis.feature {
                Feature::States => dfa.number_of_states() as f64,
                Feature::Bases => dfa.spanning_set().number_of_spanning_ranges() as f64,
                Feature::Complexity => match eliminated {
                    Some(eliminated) => eliminated.evaluate_complexity(),
                    None => dfa.to_regex()?.evaluate_complexity(),
                },
            };
            let bin = axis.bins.partition_point(|lower| *lower <= value).max(1) - 1;
            cell = cell * axis.bins.len() + bin;
        }
        Ok(self.rates[cell])
    }

    /// Whether [`rate`](Self::rate) needs the state elimination regex.
    fn uses_complexity(&self) -> bool {
        self.axes
            .iter()
            .any(|axis| axis.feature == Feature::Complexity)
    }
}

/// A DFA from [`minimal_dfa`] to propose a regex for.
struct Target<'a> {
    dfa: &'a FastAutomaton,
    /// Its canonical form, which the minimal DFA of a correct candidate has.
    form: &'a CanonicalForm,
    /// Candidates more complex than this are not checked: they would not be
    /// kept.
    ceiling: Option<f64>,
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
    gate: Option<Gate>,
    min_success: f64,
}

impl fmt::Debug for NeuralSynthesizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeuralSynthesizer")
            .field("name", &self.name)
            .field("sampling", &self.sampling)
            .field("decomposition", &self.decomposition)
            .field("min_success", &self.min_success)
            .finish_non_exhaustive()
    }
}

impl NeuralSynthesizer {
    /// Default number of candidates per automaton: the hypotheses of the
    /// beam search ([`DEFAULT_BEAM`](Self::DEFAULT_BEAM)) alone, as samples
    /// added to a beam make the regexes hardly simpler.
    pub const DEFAULT_CANDIDATES: usize = 1;

    /// Default width of the beam search replacing the greedy decoding: four
    /// hypotheses find simpler regexes than four samples, in about the same
    /// time. Each doubling of the width roughly doubles the time.
    pub const DEFAULT_BEAM: usize = 4;

    /// Default success rate below which the model is not run on an automaton
    /// or a piece of its decomposition (see
    /// [`with_min_success`](Self::with_min_success)). Higher thresholds save
    /// little time: an automaton the model is not run on is decomposed, and
    /// the model runs on its pieces.
    pub const DEFAULT_MIN_SUCCESS: f64 = 0.05;

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
        let gate = match config.get("gate") {
            None | Some(serde_json::Value::Null) => None,
            Some(gate) => {
                let gate = Gate::deserialize(gate)
                    .map_err(|err| NeuralSynthesisError::Config(format!("gate: {err}")))?;
                gate.validate().map_err(NeuralSynthesisError::Config)?;
                Some(gate)
            }
        };
        let model: Box<dyn Model> = match (header.format.as_str(), header.format_version) {
            (kleene::FORMAT, version) if kleene::FORMAT_VERSIONS.contains(&version) => Box::new(
                kleene::Kleene::new(&config, version, weights, &device.to_candle()?)?,
            ),
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
                beam: Self::DEFAULT_BEAM,
            },
            decomposition: true,
            gate,
            min_success: Self::DEFAULT_MIN_SUCCESS,
        })
    }

    /// Sets how many candidates are decoded per automaton (at least 1): the
    /// greedy decoding, or the hypotheses of the beam search
    /// ([`with_beam`](Self::with_beam)), then samples (default
    /// [`DEFAULT_CANDIDATES`](Self::DEFAULT_CANDIDATES)). More candidates find
    /// a correct regex more often, at the cost of decoding and checking them.
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

    /// Sets the width of a beam search that replaces the greedy decoding
    /// (default [`DEFAULT_BEAM`](Self::DEFAULT_BEAM); 0 or 1 decodes
    /// greedily). Its hypotheses, up to `beam` of the most likely regexes, are
    /// checked with the [`with_candidates`](Self::with_candidates) `- 1`
    /// samples. A beam of width `w` decodes up to `w` sequences per automaton,
    /// about the cost of `w` candidates.
    pub fn with_beam(mut self, beam: usize) -> Self {
        self.sampling.beam = beam;
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

    /// Sets the success rate below which converting
    /// ([`FastAutomaton::to_regex_with`]) does not run the model on an
    /// automaton, going straight to its decomposition, nor on a piece of the
    /// decomposition, which is then split further or converted by state
    /// elimination (default [`DEFAULT_MIN_SUCCESS`](Self::DEFAULT_MIN_SUCCESS)).
    /// The rate is the model's, from its `config.json`, for DFAs like the
    /// automaton's (or the piece's) minimal DFA (see the [module
    /// documentation](self)); a model without one is always run, and so is
    /// every model with 0. Skipping saves the decoding and checking of
    /// candidates that are almost never correct.
    pub fn with_min_success(mut self, min_success: f64) -> Self {
        self.min_success = min_success;
        self
    }

    /// The model's success rate for `automaton`, from its `config.json`:
    /// how often, on a calibration set, one of its candidates was correct for
    /// DFAs like the automaton's minimal DFA. `None` when the model has no
    /// rate or none for such DFAs.
    ///
    /// Fails if `automaton` cannot be minimized, or its minimal DFA converted
    /// by state elimination, within the execution profile.
    pub fn success_rate(&self, automaton: &FastAutomaton) -> Result<Option<f64>, EngineError> {
        match &self.gate {
            Some(gate) => gate.rate(&minimal_dfa(automaton)?, None),
            None => Ok(None),
        }
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
        let mut forms = Vec::with_capacity(automata.len());
        for automaton in automata {
            let dfa = minimal_dfa(automaton)?;
            forms.push(canonical_form(&dfa)?);
            dfas.push(dfa);
        }
        let targets: Vec<Target> = dfas
            .iter()
            .zip(&forms)
            .map(|(dfa, form)| Target {
                dfa,
                form,
                ceiling: None,
            })
            .collect();
        self.propose_targets(&targets)
    }

    /// [`propose_batch`](Self::propose_batch) for [`Target`]s.
    fn propose_targets(
        &self,
        targets: &[Target],
    ) -> Result<Vec<Option<RegularExpression>>, EngineError> {
        let eligible: Vec<usize> = (0..targets.len())
            .filter(|i| targets[*i].dfa.number_of_states() <= self.model.max_states())
            .collect();
        let mut proposals = vec![None; targets.len()];
        if eligible.is_empty() {
            return Ok(proposals);
        }
        let inputs: Vec<&FastAutomaton> = eligible.iter().map(|i| targets[*i].dfa).collect();
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

        // The DFAs' checks are independent.
        let checks: Vec<(usize, Vec<RegularExpression>)> =
            eligible.into_iter().zip(candidates).collect();
        #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
        let checked: Vec<(usize, Option<RegularExpression>)> = {
            use rayon::prelude::*;
            let execution_profile = ExecutionProfile::get();
            checks
                .into_par_iter()
                .map(|(i, candidates)| {
                    let checked = execution_profile.apply(|| Self::check(&targets[i], candidates));
                    (i, checked)
                })
                .collect()
        };
        #[cfg(any(not(feature = "parallel"), target_family = "wasm"))]
        let checked: Vec<(usize, Option<RegularExpression>)> = checks
            .into_iter()
            .map(|(i, candidates)| (i, Self::check(&targets[i], candidates)))
            .collect();
        for (i, proposal) in checked {
            proposals[i] = proposal;
        }
        Ok(proposals)
    }

    /// The model's proposal for the minimal DFA of `automaton`, unless its
    /// success rate for it is below
    /// [`with_min_success`](Self::with_min_success)'s; when there is none,
    /// the composition of the regexes of its pieces (see the [module
    /// documentation](self)), unless decomposition is disabled. Also returns
    /// the state elimination regex of the minimal DFA, when computed on the
    /// way.
    ///
    /// Fails if the execution profile's deadline passes.
    fn propose_or_decompose(
        &self,
        automaton: &FastAutomaton,
    ) -> Result<(Option<RegularExpression>, Option<RegularExpression>), EngineError> {
        let dfa = minimal_dfa(automaton)?;
        let form = canonical_form(&dfa)?;
        let mut eliminated = None;
        if dfa.number_of_states() <= self.model.max_states() {
            if self.min_success > 0.0 && self.gate.as_ref().is_some_and(Gate::uses_complexity) {
                eliminated = Some(dfa.to_regex()?);
            }
            if self.below_min_success(&dfa, eliminated.as_ref())? {
                tracing::debug!(
                    "{}: not run on the whole automaton (success rate)",
                    self.name
                );
            } else {
                // No ceiling: a correct proposal, however complex, is
                // compared with state elimination rather than decomposed.
                let target = Target {
                    dfa: &dfa,
                    form: &form,
                    ceiling: None,
                };
                if let Some(proposed) = self.propose_targets(&[target])?.pop().flatten() {
                    return Ok((Some(proposed), eliminated));
                }
            }
        }
        if !self.decomposition {
            return Ok((None, eliminated));
        }
        let mut decomposer = decompose::Decomposer::new(Some(self));
        decomposer.whole_tried = true;
        let (regex, eliminated) = decomposer.convert_minimal(dfa, form, eliminated)?;
        tracing::debug!(stats = ?decomposer.stats, "decomposed");
        Ok((Some(regex), Some(eliminated)))
    }

    /// Whether the model's success rate for `dfa`, a DFA from [`minimal_dfa`]
    /// whose state elimination regex is `eliminated` when already computed,
    /// is below [`with_min_success`](Self::with_min_success)'s: the model is
    /// then not run on it.
    fn below_min_success(
        &self,
        dfa: &FastAutomaton,
        eliminated: Option<&RegularExpression>,
    ) -> Result<bool, EngineError> {
        match &self.gate {
            Some(gate) if self.min_success > 0.0 => Ok(gate
                .rate(dfa, eliminated)?
                .is_some_and(|rate| rate < self.min_success)),
            _ => Ok(false),
        }
    }

    /// The simplest of `candidates` describing the language of `target`, at
    /// most as complex as its ceiling.
    fn check(target: &Target, candidates: Vec<RegularExpression>) -> Option<RegularExpression> {
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

        // Built on the first candidate checked.
        let mut complement = None;
        for (complexity, _, regex) in ranked {
            if target.ceiling.is_some_and(|ceiling| complexity > ceiling) {
                break;
            }
            match Self::describes(&regex, target, &mut complement) {
                Ok(true) => return Some(regex),
                Ok(false) => {}
                Err(err) => tracing::debug!("candidate {regex} not checked: {err}"),
            }
        }
        None
    }

    /// Whether `regex` describes exactly the language of `target`, whose
    /// complement is `complement` once computed.
    ///
    /// Its strings must all be in the language, which a product with the
    /// complement tells without determinizing `regex`; then its minimal DFA
    /// must be `target`'s, as their canonical forms tell.
    fn describes(
        regex: &RegularExpression,
        target: &Target,
        complement: &mut Option<FastAutomaton>,
    ) -> Result<bool, EngineError> {
        let automaton = regex.to_automaton()?;
        let complement = match complement {
            Some(complement) => complement,
            None => {
                let mut dfa = target.dfa.clone();
                dfa.complement()?;
                complement.insert(dfa)
            }
        };
        if automaton.has_intersection(complement)? {
            return Ok(false);
        }
        let dfa = minimal_dfa(&automaton)?;
        Ok(dfa.number_of_states() == target.dfa.number_of_states()
            && canonical_form(&dfa)? == *target.form)
    }
}

impl RegexSynthesizer for NeuralSynthesizer {
    /// The model first proposes a regex for the whole automaton, unless its
    /// success rate for it is below the threshold
    /// ([`with_min_success`](NeuralSynthesizer::with_min_success)). When it has
    /// no correct one, or the automaton is beyond its limits or below the
    /// threshold, the automaton's
    /// minimal DFA is split into independent pieces (unless
    /// [`with_decomposition`](NeuralSynthesizer::with_decomposition) disables
    /// it), each converted by the model or by state elimination, whichever is
    /// simpler, and their regexes combined. Returns the simpler of that result
    /// and [`FastAutomaton::to_regex`]'s state elimination; state elimination
    /// alone if the model fails (for instance on the execution profile's
    /// deadline). The result always describes exactly the automaton's
    /// language.
    fn synthesize(&self, automaton: &FastAutomaton) -> Result<RegularExpression, EngineError> {
        let (proposed, dfa_eliminated) = match self.propose_or_decompose(automaton) {
            Ok(proposed) => proposed,
            Err(err) => {
                tracing::debug!("neural synthesis skipped: {err}");
                (None, None)
            }
        };
        // A minimal automaton is its own minimal DFA, whose state
        // elimination regex may be known already.
        let eliminated = match dfa_eliminated {
            Some(eliminated) if automaton.is_minimal() => Ok(eliminated),
            _ => automaton.to_regex(),
        };
        let Some(proposed) = proposed else {
            return eliminated;
        };
        match eliminated {
            Ok(eliminated) if eliminated.evaluate_complexity() < proposed.evaluate_complexity() => {
                Ok(eliminated)
            }
            // The proposal stands on its own if state elimination runs out
            // of budget.
            _ => Ok(proposed),
        }
    }
}

/// A minimal DFA's states numbered by BFS from the start, following
/// transitions in the order of their characters: equal for isomorphic DFAs,
/// hence for minimal DFAs of the same language.
type CanonicalForm = Vec<(bool, Vec<(CharRange, usize)>)>;

/// `dfa`'s canonical form (see [`CanonicalForm`]); `dfa` must be minimal.
fn canonical_form(dfa: &FastAutomaton) -> Result<CanonicalForm, EngineError> {
    let spanning_set = dfa.spanning_set();
    let mut order = vec![dfa.start_state()];
    let mut numbers: IntMap<State, usize> = IntMap::default();
    numbers.insert(dfa.start_state(), 0);
    let mut form = Vec::with_capacity(dfa.number_of_states());
    let mut next = 0;
    while let Some(&state) = order.get(next) {
        next += 1;
        let mut transitions = Vec::with_capacity(dfa.out_degree(state));
        for (condition, to_state) in dfa.transitions_from(state) {
            transitions.push((condition.to_range(spanning_set)?, *to_state));
        }
        transitions.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let transitions = transitions
            .into_iter()
            .map(|(range, to_state)| {
                let number = *numbers.entry(to_state).or_insert_with(|| {
                    order.push(to_state);
                    order.len() - 1
                });
                (range, number)
            })
            .collect();
        form.push((dfa.is_accepted(state), transitions));
    }
    Ok(form)
}

/// Determinizes, minimizes, and shrinks the spanning set to the coarsest one
/// the transitions allow, so the number of bases is a property of the
/// language rather than of the construction.
fn minimal_dfa(automaton: &FastAutomaton) -> Result<FastAutomaton, EngineError> {
    let mut dfa = automaton.determinize()?.into_owned();
    dfa.minimize()?;
    Ok(dfa)
}

/// The patterns of the parity tests (`tests/data/kleene_patterns.txt`),
/// which the benchmarks also use.
#[cfg(test)]
fn test_patterns() -> Vec<RegularExpression> {
    include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/kleene_patterns.txt"
    ))
    .lines()
    .map(|pattern| RegularExpression::new(pattern).unwrap())
    .collect()
}

/// The device of the tests that run a model with weights: `KLEENE_DEVICE`,
/// `cpu` (the default), `cuda` or `metal`, with an ordinal after a colon
/// (`cuda:1`). Its backend's feature must be enabled.
#[cfg(test)]
fn test_device() -> Device {
    let device = std::env::var("KLEENE_DEVICE").unwrap_or_default();
    let (name, ordinal) = device.split_once(':').unwrap_or((&device, "0"));
    let ordinal = ordinal
        .parse()
        .unwrap_or_else(|_| panic!("KLEENE_DEVICE={device}: invalid ordinal"));
    match name {
        "" | "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(ordinal),
        "metal" => Device::Metal(ordinal),
        _ => panic!("KLEENE_DEVICE={device}: cpu, cuda or metal"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model under `KLEENE_MODEL_DIR`, for the tests that need weights,
    /// on [`test_device`]:
    /// `KLEENE_MODEL_DIR=<release dir> cargo test --release --features neural-synthesis neural_synthesis -- --ignored`
    fn kleene() -> NeuralSynthesizer {
        let dir = std::env::var("KLEENE_MODEL_DIR").expect("KLEENE_MODEL_DIR");
        NeuralSynthesizer::from_dir(dir, test_device()).unwrap()
    }

    #[test]
    #[ignore = "needs the model weights (KLEENE_MODEL_DIR)"]
    fn to_regex_with_keeps_the_language() {
        let synthesizer = kleene();
        assert!(synthesizer.model_name().starts_with("kleene"));
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
            (".*", ".*test.*"),
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
        // The greedy decoding, as on the model card.
        let regex = kleene()
            .with_candidates(1)
            .with_beam(0)
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

    fn gate(json: serde_json::Value) -> Gate {
        let gate = Gate::deserialize(json).unwrap();
        gate.validate().unwrap();
        gate
    }

    fn dfa(pattern: &str) -> FastAutomaton {
        minimal_dfa(
            &RegularExpression::new(pattern)
                .unwrap()
                .to_automaton()
                .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn gate_looks_up_the_cell_of_the_dfa() {
        // States in bins [1, 3), [3, ...); complexity in [0, 5), [5, ...).
        let gate = gate(serde_json::json!({
            "axes": [
                {"feature": "states", "bins": [1, 3]},
                {"feature": "complexity", "bins": [0, 5]},
            ],
            "rates": [0.9, 0.5, null, 0.01],
        }));
        // 2 states, complexity 1.
        assert_eq!(gate.rate(&dfa("a"), None).unwrap(), Some(0.9));
        // 2 states, complexity at least 5.
        let ab = dfa("(a|bc)*");
        assert_eq!(ab.number_of_states(), 2);
        assert!(ab.to_regex().unwrap().evaluate_complexity() >= 5.0);
        assert_eq!(gate.rate(&ab, None).unwrap(), Some(0.5));
        // 3 states or more: no data below complexity 5, 0.01 above.
        assert_eq!(gate.rate(&dfa("abc"), None).unwrap(), None);
        assert_eq!(gate.rate(&dfa("a(bc|de)*f"), None).unwrap(), Some(0.01));
        // A state elimination regex already computed gives the complexity.
        let complex = RegularExpression::new("(a|bc)*").unwrap();
        assert_eq!(gate.rate(&dfa("a"), Some(&complex)).unwrap(), Some(0.5));
        // Bases: the spanning ranges, without the rest.
        let bases = gate_bases();
        assert_eq!(bases.rate(&dfa("a*"), None).unwrap(), Some(1.0));
        assert_eq!(bases.rate(&dfa("[ab]c*"), None).unwrap(), Some(2.0));
    }

    fn gate_bases() -> Gate {
        gate(serde_json::json!({
            "axes": [{"feature": "bases", "bins": [1, 2]}],
            "rates": [1.0, 2.0],
        }))
    }

    #[test]
    fn invalid_gates_are_config_errors() {
        for gate in [
            serde_json::json!({"axes": [{"feature": "states", "bins": [1, 3]}], "rates": [0.5]}),
            serde_json::json!({"axes": [{"feature": "states", "bins": [3, 1]}], "rates": [0.5, 0.5]}),
            serde_json::json!({"axes": [{"feature": "depth", "bins": [1]}], "rates": [0.5]}),
        ] {
            let config = serde_json::json!({
                "name": "x", "format": "kleene", "format_version": 2, "gate": gate,
            });
            assert!(matches!(
                NeuralSynthesizer::from_bytes(config.to_string().as_bytes(), vec![], Device::Cpu),
                Err(NeuralSynthesisError::Config(err)) if err.starts_with("gate")
            ));
        }
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
