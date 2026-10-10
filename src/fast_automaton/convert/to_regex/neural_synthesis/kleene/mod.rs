//! The Kleene models (format `kleene` v1): a graph transformer
//! encoder over the canonical minimal DFA, and a transformer decoder writing
//! the regex in prefix order, constrained to well-formed trees.
//!
//! A port of the PyTorch implementation the models are trained with; the
//! input encoding is documented in each release's `config.json` (`inputs`).

use serde::Deserialize;

use super::{Model, NeuralSynthesisError, Sampling};
use crate::{
    CharRange,
    fast_automaton::{FastAutomaton, spanning_set::SpanningSet},
    regex::RegularExpression,
};

mod canonical;
mod grammar;
mod graph;
mod network;
mod tokens;

use network::{ModelConfig, Network, Rng};

pub(super) const FORMAT: &str = "kleene";
pub(super) const FORMAT_VERSION: u32 = 1;

/// `config.json`, the parts used here.
#[derive(Deserialize)]
struct Config {
    limits: Limits,
    trained_on: TrainedOn,
    model_config: ModelConfig,
}

#[derive(Deserialize)]
struct Limits {
    max_bases: usize,
}

/// The largest DFAs seen in training: larger ones are left to state
/// elimination, the model's quality on them being unknown.
#[derive(Deserialize)]
struct TrainedOn {
    max_states: usize,
}

pub(super) struct Kleene {
    network: Network,
    max_states: usize,
    max_bases: usize,
}

impl Kleene {
    /// Loads a model from its `config.json` and its weights.
    pub(super) fn new(
        config: &serde_json::Value,
        weights: Vec<u8>,
        device: &candle_core::Device,
    ) -> Result<Self, NeuralSynthesisError> {
        let config = Config::deserialize(config)
            .map_err(|err| NeuralSynthesisError::Config(err.to_string()))?;
        let model_config = &config.model_config;
        let vocab: std::collections::BTreeMap<String, u32> = tokens::vocab().into_iter().collect();
        if model_config.vocab != vocab {
            return Err(NeuralSynthesisError::Config(
                "unknown vocabulary".to_string(),
            ));
        }
        if config.limits.max_bases > model_config.max_bases || model_config.max_bases > 64 {
            return Err(NeuralSynthesisError::Config(format!(
                "max_bases {} (model {}) is not supported",
                config.limits.max_bases, model_config.max_bases
            )));
        }
        // The shapes the weights cannot check, which the network divides by
        // or counts down from.
        if model_config.n_heads == 0 || model_config.d_model % model_config.n_heads != 0 {
            return Err(NeuralSynthesisError::Config(format!(
                "d_model {} is not divisible into {} heads",
                model_config.d_model, model_config.n_heads
            )));
        }
        if model_config.max_len < 2 {
            return Err(NeuralSynthesisError::Config(format!(
                "max_len {} leaves no room between BOS and EOS",
                model_config.max_len
            )));
        }
        let network = Network::new(model_config, weights, device)
            .map_err(|err| NeuralSynthesisError::Weights(err.to_string()))?;
        Ok(Self {
            network,
            max_states: config.trained_on.max_states,
            max_bases: config.limits.max_bases,
        })
    }
}

impl Model for Kleene {
    fn max_states(&self) -> usize {
        self.max_states
    }

    fn max_bases(&self) -> usize {
        self.max_bases
    }

    fn candidates(
        &self,
        dfa: &FastAutomaton,
        sampling: &Sampling,
    ) -> Result<Vec<RegularExpression>, String> {
        if dfa.number_of_states() > self.max_states {
            return Ok(vec![]);
        }
        let canonical = canonical::canonicalize(dfa);
        if canonical.k == 0 || canonical.k > self.max_bases {
            return Ok(vec![]);
        }
        let graph = graph::Graph::new(
            &canonical,
            self.network.max_bases(),
            self.network.max_depth(),
        );
        let candidates = self
            .network
            .generate(
                &graph,
                canonical.k,
                sampling.candidates,
                sampling.temperature,
                &mut Rng::new(sampling.seed),
            )
            .map_err(|err| err.to_string())?;
        Ok(candidates
            .into_iter()
            .filter_map(|(tokens, masks)| tokens::decode(&tokens, &masks, &canonical.bases))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use regex_charclass::CharacterClass;

    use crate::neural_synthesis::minimal_dfa;

    use super::*;

    /// Compares with the outputs of the PyTorch model the weights were trained
    /// with (`tests/data/kleene_parity.json`, from the dfa2regex project): the
    /// canonical bases, the input graph, the first logits and the greedy
    /// decoding of each pattern's minimal DFA.
    ///
    /// `KLEENE_MODEL_DIR=<release dir> cargo test --release --features neural-synthesis neural_synthesis -- --ignored`
    /// (`KLEENE_PARITY=<json>` for other reference outputs).
    #[test]
    #[ignore = "needs the model weights (KLEENE_MODEL_DIR)"]
    fn parity() {
        let dir = std::path::PathBuf::from(std::env::var("KLEENE_MODEL_DIR").unwrap());
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let weights = std::fs::read(dir.join("model.safetensors")).unwrap();
        let kleene = Kleene::new(&config, weights, &candle_core::Device::Cpu).unwrap();

        let path = std::env::var("KLEENE_PARITY").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/kleene_parity.json").to_string()
        });
        let reference: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let ints = |v: &serde_json::Value| -> Vec<u64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect()
        };
        let floats = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let mut max_diff = 0f32;
        for case in &reference {
            let pattern = case["pattern"].as_str().unwrap();
            let dfa = minimal_dfa(
                &RegularExpression::new(pattern)
                    .unwrap()
                    .to_automaton()
                    .unwrap(),
            )
            .unwrap();
            let canonical = canonical::canonicalize(&dfa);
            let bases: Vec<String> = canonical.bases.iter().map(|b| b.to_regex()).collect();
            let expected: Vec<&str> = case["base_ranges"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_str().unwrap())
                .collect();
            assert_eq!(bases, expected, "{pattern}: bases");

            let g = graph::Graph::new(&canonical, 16, 64);
            let r = &case["graph"];
            let as_u64 = |v: &[u32]| v.iter().map(|x| *x as u64).collect::<Vec<_>>();
            assert_eq!(as_u64(&g.node_type), ints(&r["node_type"]), "{pattern}");
            assert_eq!(as_u64(&g.is_accept), ints(&r["is_accept"]), "{pattern}");
            assert_eq!(as_u64(&g.fwd_depth), ints(&r["fwd_depth"]), "{pattern}");
            assert_eq!(as_u64(&g.bwd_depth), ints(&r["bwd_depth"]), "{pattern}");
            assert_eq!(as_u64(&g.to_reject), ints(&r["to_reject"]), "{pattern}");
            assert_eq!(as_u64(&g.edge_src), ints(&r["edge_src"]), "{pattern}");
            assert_eq!(as_u64(&g.edge_dst), ints(&r["edge_dst"]), "{pattern}");
            assert_eq!(as_u64(&g.edge_type), ints(&r["edge_type"]), "{pattern}");

            let (tok, mask) = kleene.network.first_logits(&g).unwrap();
            let expected: Vec<f32> = floats(&case["first_tok_logits"])
                .into_iter()
                .chain(floats(&case["first_mask_logits"]))
                .collect();
            for (a, b) in tok.iter().chain(&mask).zip(&expected) {
                max_diff = max_diff.max((a - b).abs());
            }

            let greedy = kleene
                .network
                .generate(&g, canonical.k, 1, 1.0, &mut Rng::new(0))
                .unwrap();
            let (tokens, masks) = &greedy[0];
            assert_eq!(
                tokens.iter().map(|t| *t as u64).collect::<Vec<_>>(),
                ints(&case["tokens"]),
                "{pattern}: tokens"
            );
            assert_eq!(masks, &ints(&case["masks"]), "{pattern}: masks");
        }
        println!("{} cases, max logit difference {max_diff}", reference.len());
        assert!(max_diff < 1e-3, "max logit difference {max_diff}");
    }

    #[test]
    fn rejects_configs_the_network_cannot_run() {
        let config = |n_heads: usize, max_len: usize| {
            serde_json::json!({
                "limits": {"max_bases": 16},
                "trained_on": {"max_states": 16},
                "model_config": {
                    "vocab": tokens::vocab().into_iter().collect::<std::collections::BTreeMap<_, _>>(),
                    "max_bases": 16,
                    "max_len": max_len,
                    "d_model": 256,
                    "n_heads": n_heads,
                    "enc_layers": 4,
                    "enc_global_layers": 2,
                    "dec_layers": 4,
                    "ffn": null,
                    "edge_dim": 32,
                    "max_depth": 64
                }
            })
        };
        for (n_heads, max_len) in [(0, 128), (3, 128), (8, 1)] {
            assert!(
                matches!(
                    Kleene::new(&config(n_heads, max_len), vec![], &candle_core::Device::Cpu),
                    Err(NeuralSynthesisError::Config(_))
                ),
                "n_heads {n_heads}, max_len {max_len}"
            );
        }
    }

    /// Times [`Network::generate`] over the parity patterns' DFAs, and prints
    /// a checksum of the candidates so an optimization can be checked not to
    /// change them:
    /// `KLEENE_MODEL_DIR=<release dir> cargo test --release --features neural-synthesis kleene::tests::bench_generate -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark; needs the model weights (KLEENE_MODEL_DIR)"]
    fn bench_generate() {
        use std::{
            hash::{DefaultHasher, Hash, Hasher},
            time::Instant,
        };

        let dir = std::path::PathBuf::from(std::env::var("KLEENE_MODEL_DIR").unwrap());
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let weights = std::fs::read(dir.join("model.safetensors")).unwrap();
        let kleene = Kleene::new(&config, weights, &candle_core::Device::Cpu).unwrap();

        let reference: Vec<serde_json::Value> = serde_json::from_slice(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/kleene_parity.json"
        )))
        .unwrap();
        let inputs: Vec<(graph::Graph, usize)> = reference
            .iter()
            .map(|case| {
                let dfa = minimal_dfa(
                    &RegularExpression::new(case["pattern"].as_str().unwrap())
                        .unwrap()
                        .to_automaton()
                        .unwrap(),
                )
                .unwrap();
                let canonical = canonical::canonicalize(&dfa);
                let g = graph::Graph::new(
                    &canonical,
                    kleene.network.max_bases(),
                    kleene.network.max_depth(),
                );
                (g, canonical.k)
            })
            .collect();

        const RUNS: usize = 10;
        for count in [1, 8] {
            let mut checksum = DefaultHasher::new();
            let mut pass = || {
                let start = Instant::now();
                for (g, k) in &inputs {
                    let candidates = kleene
                        .network
                        .generate(g, *k, count, 1.0, &mut Rng::new(0))
                        .unwrap();
                    candidates.hash(&mut checksum);
                }
                start.elapsed()
            };
            pass(); // warm-up
            let mut times: Vec<_> = (0..RUNS).map(|_| pass()).collect();
            times.sort();
            let per_dfa = |t: std::time::Duration| t / inputs.len() as u32;
            println!(
                "{count} candidate(s): {:?} per DFA (median of {RUNS}; min {:?}, max {:?}), checksum {:016x}",
                per_dfa(times[RUNS / 2]),
                per_dfa(times[0]),
                per_dfa(times[RUNS - 1]),
                checksum.finish()
            );
        }
    }
}
