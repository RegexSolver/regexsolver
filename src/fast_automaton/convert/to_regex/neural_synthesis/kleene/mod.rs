//! The Kleene models (format `kleene` v1 and v2): a graph transformer
//! encoder over the canonical minimal DFA, and a transformer decoder writing
//! the regex in prefix order, constrained to well-formed trees. The two
//! versions differ only in the input: v1 (Kleene 1) counts the rest of the
//! spanning set as a base, v2 (Kleene 2) does not.
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

mod attention;
mod canonical;
mod edge_attention;
mod grammar;
mod graph;
mod layers;
mod network;
mod tokens;

use network::{ModelConfig, Network};

pub(super) const FORMAT: &str = "kleene";
/// The versions of the format this module runs.
pub(super) const FORMAT_VERSIONS: [u32; 2] = [1, 2];

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
    /// Format v1: the rest of the spanning set is a base.
    rest_is_base: bool,
}

impl Kleene {
    /// Loads a model of format version `version` from its `config.json` and
    /// its weights.
    pub(super) fn new(
        config: &serde_json::Value,
        version: u32,
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
            rest_is_base: version == 1,
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
        dfas: &[&FastAutomaton],
        sampling: &Sampling,
    ) -> Result<Vec<Vec<RegularExpression>>, String> {
        let mut inputs = Vec::with_capacity(dfas.len());
        let mut bases = Vec::with_capacity(dfas.len());
        let mut decoded = vec![vec![]; dfas.len()];
        for (i, dfa) in dfas.iter().enumerate() {
            if dfa.number_of_states() > self.max_states {
                continue;
            }
            let canonical = canonical::canonicalize(dfa, self.rest_is_base);
            if canonical.k == 0 || canonical.k > self.max_bases {
                continue;
            }
            let graph = graph::Graph::new(
                &canonical,
                self.network.max_bases(),
                self.network.max_depth(),
            );
            inputs.push((i, graph, canonical.k));
            bases.push(canonical.bases);
        }
        let graphs: Vec<(&graph::Graph, usize)> = inputs.iter().map(|(_, g, k)| (g, *k)).collect();
        let candidates = self
            .network
            .generate_batch(
                &graphs,
                sampling.candidates,
                sampling.temperature,
                sampling.seed,
                sampling.beam,
            )
            .map_err(|err| err.to_string())?;
        for (((i, _, _), candidates), bases) in inputs.iter().zip(candidates).zip(&bases) {
            decoded[*i] = candidates
                .into_iter()
                .filter_map(|(tokens, masks)| tokens::decode(&tokens, &masks, bases))
                .collect();
        }
        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use regex_charclass::CharacterClass;

    use crate::neural_synthesis::{minimal_dfa, test_device, test_patterns};

    use super::*;

    /// Compares with the outputs of the PyTorch model the weights were trained
    /// with (`tests/data/kleene_parity-<name>-<release>.json`, from the
    /// dfa2regex project's release check, for the patterns of
    /// `tests/data/kleene_patterns.txt`): the canonical bases, the input
    /// graph, the first logits and the greedy decoding of each pattern's
    /// minimal DFA.
    ///
    /// `KLEENE_MODEL_DIR=<release dir> cargo test --release --features neural-synthesis neural_synthesis -- --ignored`
    /// (`KLEENE_PARITY=<json>` for other reference outputs, `KLEENE_DEVICE`
    /// for another device than the CPU: see [`test_device`]).
    #[test]
    #[ignore = "needs the model weights (KLEENE_MODEL_DIR)"]
    fn parity() {
        let dir = std::path::PathBuf::from(std::env::var("KLEENE_MODEL_DIR").unwrap());
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let weights = std::fs::read(dir.join("model.safetensors")).unwrap();
        let version = config["format_version"].as_u64().unwrap() as u32;
        let device = test_device().to_candle().unwrap();
        let kleene = Kleene::new(&config, version, weights, &device).unwrap();

        // The reference outputs depend on the weights: one file per release.
        let path = std::env::var("KLEENE_PARITY").unwrap_or_else(|_| {
            format!(
                "{}/tests/data/kleene_parity-{}-{}.json",
                env!("CARGO_MANIFEST_DIR"),
                config["name"].as_str().unwrap(),
                config["release"].as_str().unwrap_or("unreleased"),
            )
        });
        let reference = std::fs::read(&path).unwrap_or_else(|err| {
            panic!(
                "{path}: {err}; write the reference outputs of these weights with \
                 dfa2regex's release check, then pass them as KLEENE_PARITY"
            )
        });
        let reference: Vec<serde_json::Value> = serde_json::from_slice(&reference).unwrap();
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
            let canonical = canonical::canonicalize(&dfa, kleene.rest_is_base);
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

            let (tok, mask, locations) = kleene.network.first_logits(&g).unwrap();
            // On the device asked for: candle does not fall back to the CPU,
            // but the network has CPU-only branches.
            assert_eq!(locations, [device.location(); 2], "{pattern}: device");
            let expected: Vec<f32> = floats(&case["first_tok_logits"])
                .into_iter()
                .chain(floats(&case["first_mask_logits"]))
                .collect();
            for (a, b) in tok.iter().chain(&mask).zip(&expected) {
                max_diff = max_diff.max((a - b).abs());
            }

            let greedy = kleene
                .network
                .generate(&g, canonical.k, 1, 1.0, 0, 0)
                .unwrap();
            let (tokens, masks) = &greedy[0];
            assert_eq!(
                tokens.iter().map(|t| *t as u64).collect::<Vec<_>>(),
                ints(&case["tokens"]),
                "{pattern}: tokens"
            );
            assert_eq!(masks, &ints(&case["masks"]), "{pattern}: masks");
        }
        println!(
            "{} cases on {:?}, max logit difference {max_diff}",
            reference.len(),
            device.location()
        );
        assert!(max_diff < 1e-3, "max logit difference {max_diff}");
    }

    /// Compares the beam search with the PyTorch model's
    /// (`Graph2Seq.beam_search`): for each pattern of
    /// `tests/data/kleene_beam-<name>-<release>.json` and each width, the
    /// same hypotheses in the same order. Without that file, nothing is
    /// checked.
    ///
    /// `KLEENE_MODEL_DIR=<release dir> cargo test --release --features neural-synthesis kleene::tests::beam_parity -- --ignored`
    /// (`KLEENE_BEAM_PARITY=<json>` for other reference outputs).
    #[test]
    #[ignore = "needs the model weights (KLEENE_MODEL_DIR)"]
    fn beam_parity() {
        let dir = std::path::PathBuf::from(std::env::var("KLEENE_MODEL_DIR").unwrap());
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let weights = std::fs::read(dir.join("model.safetensors")).unwrap();
        let version = config["format_version"].as_u64().unwrap() as u32;
        let device = test_device().to_candle().unwrap();
        let kleene = Kleene::new(&config, version, weights, &device).unwrap();

        // A release may have no reference beams (dfa2regex writes them only
        // for format v2): the test then checks nothing.
        let path = match std::env::var("KLEENE_BEAM_PARITY") {
            Ok(path) => path,
            Err(_) => {
                let path = format!(
                    "{}/tests/data/kleene_beam-{}-{}.json",
                    env!("CARGO_MANIFEST_DIR"),
                    config["name"].as_str().unwrap(),
                    config["release"].as_str().unwrap_or("unreleased"),
                );
                if !std::path::Path::new(&path).exists() {
                    println!("{path}: no reference beams, nothing checked");
                    return;
                }
                path
            }
        };
        let reference = std::fs::read(&path).unwrap();
        let reference: Vec<serde_json::Value> = serde_json::from_slice(&reference).unwrap();
        let (mut hypotheses, mut different) = (0, vec![]);
        for case in &reference {
            let pattern = case["pattern"].as_str().unwrap();
            let dfa = minimal_dfa(
                &RegularExpression::new(pattern)
                    .unwrap()
                    .to_automaton()
                    .unwrap(),
            )
            .unwrap();
            let canonical = canonical::canonicalize(&dfa, kleene.rest_is_base);
            let g = graph::Graph::new(
                &canonical,
                kleene.network.max_bases(),
                kleene.network.max_depth(),
            );
            for (key, expected) in case.as_object().unwrap() {
                let Some(width) = key.strip_prefix("beam") else {
                    continue;
                };
                let width: usize = width.parse().unwrap();
                let expected: Vec<(Vec<u32>, Vec<u64>)> = (expected.as_array().unwrap().iter())
                    .map(|h| {
                        let ints = |v: &serde_json::Value| -> Vec<u64> {
                            v.as_array()
                                .unwrap()
                                .iter()
                                .map(|x| x.as_u64().unwrap())
                                .collect()
                        };
                        let tokens = ints(&h[0]).into_iter().map(|t| t as u32).collect();
                        (tokens, ints(&h[1]))
                    })
                    .collect();
                let beam = kleene
                    .network
                    .generate(&g, canonical.k, 1, 1.0, 0, width)
                    .unwrap();
                hypotheses += expected.len();
                if beam != expected {
                    different.push(format!("{pattern} (width {width})"));
                }
            }
        }
        println!(
            "{} patterns, {hypotheses} hypotheses; different beams: {different:?}",
            reference.len()
        );
        assert!(different.is_empty(), "{} different beams", different.len());
    }

    /// With the tiny model: a beam gives at most its width of distinct,
    /// well-formed hypotheses, before the samples, which are those decoded
    /// without a beam; decoding DFAs together gives what decoding them one
    /// by one does.
    #[test]
    fn beam_search_with_samples() {
        let dir = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/kleene_tiny"
        ));
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let weights = std::fs::read(dir.join("model.safetensors")).unwrap();
        let kleene = Kleene::new(&config, 1, weights, &candle_core::Device::Cpu).unwrap();
        let canonicals: Vec<canonical::CanonicalDfa> = [
            "abc",
            "(ab|cd)*e",
            "[a-c]*d[e-g]*",
            "(a|b)*abb",
            "x|yz",
            "a*",
        ]
        .iter()
        .map(|pattern| {
            let automaton = RegularExpression::new(pattern).unwrap().to_automaton();
            canonical::canonicalize(&minimal_dfa(&automaton.unwrap()).unwrap(), true)
        })
        .collect();
        let graphs: Vec<graph::Graph> = (canonicals.iter())
            .map(|c| graph::Graph::new(c, kleene.network.max_bases(), kleene.network.max_depth()))
            .collect();
        let inputs: Vec<(&graph::Graph, usize)> = graphs
            .iter()
            .zip(&canonicals)
            .map(|(g, c)| (g, c.k))
            .collect();
        let (count, seed) = (4, 7);
        let drawn = kleene
            .network
            .generate_batch(&inputs, count, 1.0, seed, 0)
            .unwrap();
        for width in [2, 3, 8] {
            let batch = (kleene.network)
                .generate_batch(&inputs, count, 1.0, seed, width)
                .unwrap();
            for (i, candidates) in batch.iter().enumerate() {
                let (beam, samples) = candidates.split_at(candidates.len() - (count - 1));
                assert_eq!(samples, &drawn[i][1..], "width {width}, DFA {i}");
                assert!((1..=width).contains(&beam.len()), "width {width}, DFA {i}");
                for (j, (tokens, masks)) in beam.iter().enumerate() {
                    assert!(!beam[..j].contains(&(tokens.clone(), masks.clone())));
                    assert!(tokens::decode(tokens, masks, &canonicals[i].bases).is_some());
                }
                let (graph, bases) = inputs[i];
                let alone = (kleene.network)
                    .generate(graph, bases, count, 1.0, seed, width)
                    .unwrap();
                assert_eq!(&alone, candidates, "width {width}, DFA {i}");
            }
        }
    }

    /// Writes `tests/data/kleene_tiny`, a model of the format with tiny
    /// dimensions and untrained weights, for the tests that need a model to
    /// run but not good proposals. Its regexes are at most 6 tokens long:
    /// untrained, longer ones nest counted repetitions too costly to check.
    /// The weights are drawn from generators seeded by the parameters' names,
    /// so writing it again gives the same files.
    /// `cargo test --features neural-synthesis kleene::tests::write_tiny_model -- --ignored`
    #[test]
    #[ignore = "writes the tiny model fixture"]
    fn write_tiny_model() {
        let config = serde_json::json!({
            "name": "kleene-tiny",
            "format": FORMAT,
            "format_version": 1,
            "limits": {"max_bases": 16},
            "trained_on": {"max_states": 16},
            "model_config": {
                "vocab": tokens::vocab().into_iter().collect::<std::collections::BTreeMap<_, _>>(),
                "max_bases": 16,
                "max_len": 8,
                "d_model": 16,
                "n_heads": 2,
                "enc_layers": 1,
                "enc_global_layers": 1,
                "dec_layers": 1,
                "ffn": 32,
                "edge_dim": 4,
                "max_depth": 64
            }
        });
        let model_config = ModelConfig::deserialize(&config["model_config"]).unwrap();
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(
            &varmap,
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        );
        Network::from_var_builder(&model_config, vb, &candle_core::Device::Cpu).unwrap();
        for (name, var) in varmap.data().lock().unwrap().iter() {
            // FNV-1a of the name, then SplitMix64: uniform in [-0.5, 0.5).
            let mut state = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
                (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3)
            });
            let values: Vec<f32> = (0..var.elem_count())
                .map(|_| {
                    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                })
                .collect();
            var.set(
                &candle_core::Tensor::from_vec(values, var.shape(), &candle_core::Device::Cpu)
                    .unwrap(),
            )
            .unwrap();
        }

        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/kleene_tiny");
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            format!("{dir}/config.json"),
            serde_json::to_string_pretty(&config).unwrap() + "\n",
        )
        .unwrap();
        varmap.save(format!("{dir}/model.safetensors")).unwrap();
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
                    Kleene::new(
                        &config(n_heads, max_len),
                        1,
                        vec![],
                        &candle_core::Device::Cpu
                    ),
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
        let version = config["format_version"].as_u64().unwrap() as u32;
        let device = test_device().to_candle().unwrap();
        let kleene = Kleene::new(&config, version, weights, &device).unwrap();

        let inputs: Vec<(graph::Graph, usize)> = test_patterns()
            .iter()
            .map(|pattern| {
                let dfa = minimal_dfa(&pattern.to_automaton().unwrap()).unwrap();
                let canonical = canonical::canonicalize(&dfa, kleene.rest_is_base);
                let g = graph::Graph::new(
                    &canonical,
                    kleene.network.max_bases(),
                    kleene.network.max_depth(),
                );
                (g, canonical.k)
            })
            .collect();

        const RUNS: usize = 10;
        // Candidates, and beam widths (0: greedy decoding).
        for (count, beam) in [(1, 0), (8, 0), (1, 4), (1, 8), (4, 4)] {
            let setting = if beam == 0 {
                format!("{count} candidate(s)")
            } else {
                format!("beam {beam} + {} sample(s)", count - 1)
            };
            let mut checksum = DefaultHasher::new();
            let mut pass = || {
                let start = Instant::now();
                for (g, k) in &inputs {
                    let candidates = kleene.network.generate(g, *k, count, 1.0, 0, beam).unwrap();
                    candidates.hash(&mut checksum);
                }
                start.elapsed()
            };
            pass(); // warm-up
            let mut times: Vec<_> = (0..RUNS).map(|_| pass()).collect();
            times.sort();
            let per_dfa = |t: std::time::Duration| t / inputs.len() as u32;
            println!(
                "{setting}: {:?} per DFA (median of {RUNS}; min {:?}, max {:?}), checksum {:016x}",
                per_dfa(times[RUNS / 2]),
                per_dfa(times[0]),
                per_dfa(times[RUNS - 1]),
                checksum.finish()
            );

            // All the DFAs in one batch: the same candidates, up to float
            // rounding (counted), in less time.
            let one_by_one: Vec<_> = inputs
                .iter()
                .map(|(g, k)| kleene.network.generate(g, *k, count, 1.0, 0, beam).unwrap())
                .collect();
            let batch: Vec<(&graph::Graph, usize)> = inputs.iter().map(|(g, k)| (g, *k)).collect();
            let batched = kleene
                .network
                .generate_batch(&batch, count, 1.0, 0, beam)
                .unwrap();
            let same = one_by_one
                .iter()
                .zip(&batched)
                .flat_map(|(a, b)| a.iter().zip(b))
                .filter(|(a, b)| a == b)
                .count();
            let mut times: Vec<_> = (0..RUNS)
                .map(|_| {
                    let start = Instant::now();
                    kleene
                        .network
                        .generate_batch(&batch, count, 1.0, 0, beam)
                        .unwrap();
                    start.elapsed()
                })
                .collect();
            times.sort();
            println!(
                "{setting}, batched: {:?} per DFA (median of {RUNS}); {same}/{} candidates as one by one",
                per_dfa(times[RUNS / 2]),
                batched.iter().map(Vec::len).sum::<usize>()
            );
        }
    }
}
