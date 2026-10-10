//! The Kleene graph-to-sequence network, on candle: a graph transformer
//! encoder over the DFA, a transformer decoder over the regex tokens. A port
//! of the PyTorch model it was trained as; parameter names match its
//! `model.safetensors`.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear, linear_no_bias,
};
use serde::Deserialize;

use super::grammar::PrefixGrammar;
use super::graph::Graph;
use super::tokens::{BOS, CHAR, EOS, VOCAB_SIZE};

const LAYER_NORM_EPS: f64 = 1e-5;
/// Added to the attention denominators of the graph layers, as PyG does.
const SOFTMAX_EPS: f64 = 1e-16;

/// `model_config` in `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub(super) struct ModelConfig {
    pub vocab: std::collections::BTreeMap<String, u32>,
    pub max_bases: usize,
    /// Decoder positions, BOS and EOS included.
    pub max_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub enc_layers: usize,
    pub enc_global_layers: usize,
    pub dec_layers: usize,
    /// Defaults to `4 * d_model`.
    pub ffn: Option<usize>,
    pub edge_dim: usize,
    pub max_depth: usize,
}

/// A multi-head attention with PyTorch's packed input projection.
struct Attention {
    /// Queries, keys and values at once.
    qkv: Linear,
    q: Linear,
    /// Keys and values at once.
    kv: Linear,
    out: Linear,
    heads: usize,
}

impl Attention {
    fn new(d: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        let w = vb.get((3 * d, d), "in_proj_weight")?;
        let b = vb.get(3 * d, "in_proj_bias")?;
        let rows = |from: usize, to: usize| -> Result<Linear> {
            Ok(Linear::new(
                w.narrow(0, from * d, (to - from) * d)?,
                Some(b.narrow(0, from * d, (to - from) * d)?),
            ))
        };
        Ok(Self {
            qkv: rows(0, 3)?,
            q: rows(0, 1)?,
            kv: rows(1, 3)?,
            out: linear(d, d, vb.pp("out_proj"))?,
            heads,
        })
    }

    /// `[T, d]` → `[heads, T, d / heads]`.
    fn split(&self, x: &Tensor) -> Result<Tensor> {
        let (t, d) = x.dims2()?;
        x.reshape((t, self.heads, d / self.heads))?
            .transpose(0, 1)?
            .contiguous()
    }

    /// Unmasked self-attention over `x` `[N, d]`.
    fn self_attend(&self, x: &Tensor) -> Result<Tensor> {
        let (n, d) = x.dims2()?;
        let qkv = self.qkv.forward(x)?;
        let [q, k, v] = [0, 1, 2].map(|i| qkv.narrow(1, i * d, d).and_then(|x| self.split(&x)));
        let (q, k, v) = (q?, k?, v?);
        let scores = (q.matmul(&k.t()?)? / ((d / self.heads) as f64).sqrt())?;
        let attended = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        self.out
            .forward(&attended.transpose(0, 1)?.reshape((n, d))?)
    }

    /// One query per row: `q` `[B, d]` over `keys` and `values` `[B or 1,
    /// heads, T, d / heads]`. Returns `[B, d]` after the output projection.
    ///
    /// Multiply-and-sum rather than matmul: candle runs a batched matmul as
    /// one gemm call per matrix, far slower for `B × heads` single rows.
    fn attend_one(&self, q: &Tensor, keys: &Tensor, values: &Tensor) -> Result<Tensor> {
        let (b, d) = q.dims2()?;
        let hd = d / self.heads;
        let q = q.reshape((b, self.heads, 1, hd))?;
        let scores = (keys.broadcast_mul(&q)?.sum(D::Minus1)? / (hd as f64).sqrt())?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?.unsqueeze(D::Minus1)?;
        let attended = values.broadcast_mul(&weights)?.sum(2)?;
        self.out.forward(&attended.reshape((b, d))?)
    }
}

/// `linear2(gelu(linear1(x)))`.
struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn new(d: usize, ffn: usize, up: VarBuilder, down: VarBuilder) -> Result<Self> {
        Ok(Self {
            up: linear(d, ffn, up)?,
            down: linear(ffn, d, down)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down.forward(&self.up.forward(x)?.gelu_erf()?)
    }
}

/// The edges of a graph, laid out for [`TransformerConv`].
struct Edges {
    src: Tensor,
    dst: Tensor,
    /// `[N, E]`: 1 where edge `e` ends at node `n`.
    incoming: Tensor,
    /// `[N, E, 1]`: 0 where edge `e` ends at node `n`, -∞ elsewhere.
    incoming_bias: Tensor,
    /// `[E, edge_dim]` edge-type embeddings.
    embedded: Tensor,
}

/// PyG's `TransformerConv` (`concat`, `beta`, `root_weight`, edge features),
/// evaluated with dense `[nodes, edges]` matrices: the graphs are tiny.
struct TransformerConv {
    key: Linear,
    query: Linear,
    value: Linear,
    edge: Linear,
    skip: Linear,
    beta: Linear,
    heads: usize,
}

impl TransformerConv {
    fn new(d: usize, heads: usize, edge_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            key: linear(d, d, vb.pp("lin_key"))?,
            query: linear(d, d, vb.pp("lin_query"))?,
            value: linear(d, d, vb.pp("lin_value"))?,
            edge: linear_no_bias(edge_dim, d, vb.pp("lin_edge"))?,
            skip: linear(d, d, vb.pp("lin_skip"))?,
            beta: linear_no_bias(3 * d, 1, vb.pp("lin_beta"))?,
            heads,
        })
    }

    fn forward(&self, x: &Tensor, edges: &Edges) -> Result<Tensor> {
        let (n, d) = x.dims2()?;
        let (h, c) = (self.heads, d / self.heads);
        let x_r = self.skip.forward(x)?;
        let e = edges.src.dim(0)?;
        let out = if e == 0 {
            Tensor::zeros((n, d), x.dtype(), x.device())?
        } else {
            let edge = self.edge.forward(&edges.embedded)?;
            let query = self.query.forward(x)?.index_select(&edges.dst, 0)?;
            let key = (self.key.forward(x)?.index_select(&edges.src, 0)? + &edge)?;
            let value = (self.value.forward(x)?.index_select(&edges.src, 0)? + &edge)?;
            // [E, H]
            let alpha = ((query * key)?.reshape((e, h, c))?.sum(D::Minus1)? / (c as f64).sqrt())?;
            // Softmax over the edges into each node.
            let max = alpha
                .unsqueeze(0)?
                .broadcast_add(&edges.incoming_bias)?
                .max(1)?
                .index_select(&edges.dst, 0)?;
            let exp = (alpha - max)?.exp()?;
            let sum = (edges.incoming.matmul(&exp)? + SOFTMAX_EPS)?.index_select(&edges.dst, 0)?;
            let alpha = (exp / sum)?;
            let messages = value
                .reshape((e, h, c))?
                .broadcast_mul(&alpha.unsqueeze(D::Minus1)?)?
                .reshape((e, d))?;
            edges.incoming.matmul(&messages)?
        };
        let beta = candle_nn::ops::sigmoid(
            &self
                .beta
                .forward(&Tensor::cat(&[&out, &x_r, &(&out - &x_r)?], D::Minus1)?)?,
        )?;
        x_r.broadcast_mul(&beta)? + out.broadcast_mul(&(1.0 - beta)?)?
    }
}

/// A pre-norm graph layer: `TransformerConv`, then a feed-forward.
struct ConvBlock {
    norm1: LayerNorm,
    conv: TransformerConv,
    norm2: LayerNorm,
    ffn: FeedForward,
}

impl ConvBlock {
    fn forward(&self, x: &Tensor, edges: &Edges) -> Result<Tensor> {
        let x = (x + self.conv.forward(&self.norm1.forward(x)?, edges)?)?;
        &x + self.ffn.forward(&self.norm2.forward(&x)?)?
    }
}

/// A pre-norm `nn.TransformerEncoderLayer`.
struct EncoderLayer {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    ffn: FeedForward,
}

impl EncoderLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.self_attend(&self.norm1.forward(x)?)?)?;
        &x + self.ffn.forward(&self.norm2.forward(&x)?)?
    }
}

struct Encoder {
    type_emb: Embedding,
    accept_emb: Embedding,
    fwd_depth_emb: Embedding,
    to_accept_emb: Embedding,
    to_reject_emb: Embedding,
    edge_emb: Embedding,
    layers: Vec<ConvBlock>,
    global_layers: Vec<EncoderLayer>,
    out_norm: LayerNorm,
}

impl Encoder {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let (d, depth) = (cfg.d_model, cfg.max_depth);
        let ffn = cfg.ffn.unwrap_or(4 * d);
        let layers = (0..cfg.enc_layers)
            .map(|i| {
                let vb = vb.pp("layers").pp(i);
                Ok(ConvBlock {
                    norm1: layer_norm(d, LAYER_NORM_EPS, vb.pp("norm1"))?,
                    conv: TransformerConv::new(d, cfg.n_heads, cfg.edge_dim, vb.pp("conv"))?,
                    norm2: layer_norm(d, LAYER_NORM_EPS, vb.pp("norm2"))?,
                    ffn: FeedForward::new(d, ffn, vb.pp("ffn.0"), vb.pp("ffn.3"))?,
                })
            })
            .collect::<Result<_>>()?;
        let global_layers = (0..cfg.enc_global_layers)
            .map(|i| {
                let vb = vb.pp("global_layers.layers").pp(i);
                Ok(EncoderLayer {
                    norm1: layer_norm(d, LAYER_NORM_EPS, vb.pp("norm1"))?,
                    attn: Attention::new(d, cfg.n_heads, vb.pp("self_attn"))?,
                    norm2: layer_norm(d, LAYER_NORM_EPS, vb.pp("norm2"))?,
                    ffn: FeedForward::new(d, ffn, vb.pp("linear1"), vb.pp("linear2"))?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            type_emb: embedding(3, d, vb.pp("type_emb"))?,
            accept_emb: embedding(2, d, vb.pp("accept_emb"))?,
            fwd_depth_emb: embedding(depth + 1, d, vb.pp("fwd_depth_emb"))?,
            to_accept_emb: embedding(depth + 2, d, vb.pp("to_accept_emb"))?,
            to_reject_emb: embedding(depth + 2, d, vb.pp("to_reject_emb"))?,
            edge_emb: embedding(2 * cfg.max_bases, cfg.edge_dim, vb.pp("edge_emb"))?,
            layers,
            global_layers,
            out_norm: layer_norm(d, LAYER_NORM_EPS, vb.pp("out_norm"))?,
        })
    }

    /// Node embeddings `[N, d]`.
    fn forward(&self, graph: &Graph, device: &Device) -> Result<Tensor> {
        let ids = |v: &[u32]| Tensor::from_slice(v, v.len(), device);
        let mut h = (self.type_emb.forward(&ids(&graph.node_type)?)?
            + self.accept_emb.forward(&ids(&graph.is_accept)?)?)?;
        h = (h + self.fwd_depth_emb.forward(&ids(&graph.fwd_depth)?)?)?;
        h = (h + self.to_accept_emb.forward(&ids(&graph.bwd_depth)?)?)?;
        h = (h + self.to_reject_emb.forward(&ids(&graph.to_reject)?)?)?;

        let (n, e) = (graph.nodes, graph.edge_dst.len());
        let incoming: Vec<f32> = (0..n)
            .flat_map(|node| {
                graph
                    .edge_dst
                    .iter()
                    .map(move |dst| (*dst as usize == node) as u8 as f32)
            })
            .collect();
        let incoming_bias: Vec<f32> = incoming
            .iter()
            .map(|i| if *i > 0.0 { 0.0 } else { f32::NEG_INFINITY })
            .collect();
        let edges = Edges {
            src: ids(&graph.edge_src)?,
            dst: ids(&graph.edge_dst)?,
            incoming: Tensor::from_vec(incoming, (n, e), device)?,
            incoming_bias: Tensor::from_vec(incoming_bias, (n, e, 1), device)?,
            embedded: self.edge_emb.forward(&ids(&graph.edge_type)?)?,
        };
        for layer in &self.layers {
            h = layer.forward(&h, &edges)?;
        }
        for layer in &self.global_layers {
            h = layer.forward(&h)?;
        }
        self.out_norm.forward(&h)
    }
}

/// A pre-norm `nn.TransformerDecoderLayer`.
struct DecoderLayer {
    norm1: LayerNorm,
    self_attn: Attention,
    norm2: LayerNorm,
    cross_attn: Attention,
    norm3: LayerNorm,
    ffn: FeedForward,
}

struct Decoder {
    tok_emb: Embedding,
    mask_emb: Linear,
    pos_emb: Tensor,
    start_proj: Linear,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    tok_head: Linear,
    mask_head: Linear,
}

/// The state of an incremental decoding of `B` sequences.
struct Cache {
    /// `[1, d]`, added at position 0.
    start: Tensor,
    /// Per layer: memory keys and values, `[1, heads, N, d / heads]`.
    cross: Vec<(Tensor, Tensor)>,
    /// Per layer: self-attention keys and values of the positions so far,
    /// `[B, heads, T, d / heads]`.
    past: Vec<Option<(Tensor, Tensor)>>,
    /// Positions fed so far.
    t: usize,
}

impl Decoder {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let ffn = cfg.ffn.unwrap_or(4 * d);
        let layers = (0..cfg.dec_layers)
            .map(|i| {
                let vb = vb.pp("layers.layers").pp(i);
                Ok(DecoderLayer {
                    norm1: layer_norm(d, LAYER_NORM_EPS, vb.pp("norm1"))?,
                    self_attn: Attention::new(d, cfg.n_heads, vb.pp("self_attn"))?,
                    norm2: layer_norm(d, LAYER_NORM_EPS, vb.pp("norm2"))?,
                    cross_attn: Attention::new(d, cfg.n_heads, vb.pp("multihead_attn"))?,
                    norm3: layer_norm(d, LAYER_NORM_EPS, vb.pp("norm3"))?,
                    ffn: FeedForward::new(d, ffn, vb.pp("linear1"), vb.pp("linear2"))?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            tok_emb: embedding(VOCAB_SIZE, d, vb.pp("tok_emb"))?,
            mask_emb: linear_no_bias(cfg.max_bases, d, vb.pp("mask_emb"))?,
            pos_emb: vb.get((cfg.max_len, d), "pos_emb.weight")?,
            start_proj: linear(d, d, vb.pp("start_proj"))?,
            layers,
            norm: layer_norm(d, LAYER_NORM_EPS, vb.pp("layers.norm"))?,
            tok_head: linear(d, VOCAB_SIZE, vb.pp("tok_head"))?,
            mask_head: linear(d, cfg.max_bases, vb.pp("mask_head"))?,
        })
    }

    /// Prepares the decoding over the node embeddings `memory` `[N, d]`; the
    /// start state is node 0.
    fn start_cache(&self, memory: &Tensor) -> Result<Cache> {
        let d = memory.dim(1)?;
        let cross = self
            .layers
            .iter()
            .map(|layer| {
                let attn = &layer.cross_attn;
                let kv = attn.kv.forward(memory)?;
                Ok((
                    attn.split(&kv.narrow(1, 0, d)?)?.unsqueeze(0)?,
                    attn.split(&kv.narrow(1, d, d)?)?.unsqueeze(0)?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Cache {
            start: self.start_proj.forward(&memory.narrow(0, 0, 1)?)?,
            cross,
            past: vec![None; self.layers.len()],
            t: 0,
        })
    }

    /// Feeds position `cache.t`: `tokens` `[B]` and their base masks `[B,
    /// max_bases]`. Returns the token logits `[B, vocab]` and base-mask
    /// logits `[B, max_bases]` of the next position.
    fn step(&self, cache: &mut Cache, tokens: &Tensor, masks: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut x = (self.tok_emb.forward(tokens)? + self.mask_emb.forward(masks)?)?
            .broadcast_add(&self.pos_emb.narrow(0, cache.t, 1)?)?;
        if cache.t == 0 {
            x = x.broadcast_add(&cache.start)?;
        }
        let (b, d) = x.dims2()?;
        for (i, layer) in self.layers.iter().enumerate() {
            let attn = &layer.self_attn;
            let (heads, hd) = (attn.heads, d / attn.heads);
            let qkv = attn.qkv.forward(&layer.norm1.forward(&x)?)?;
            let head = |part: usize| qkv.narrow(1, part * d, d)?.reshape((b, heads, 1, hd));
            let (k, v) = match cache.past[i].take() {
                Some((pk, pv)) => (
                    Tensor::cat(&[&pk, &head(1)?], 2)?,
                    Tensor::cat(&[&pv, &head(2)?], 2)?,
                ),
                None => (head(1)?, head(2)?),
            };
            x = (x + attn.attend_one(&qkv.narrow(1, 0, d)?, &k, &v)?)?;
            cache.past[i] = Some((k, v));

            let attn = &layer.cross_attn;
            let q = attn.q.forward(&layer.norm2.forward(&x)?)?;
            let (k, v) = &cache.cross[i];
            x = (&x + attn.attend_one(&q, k, v)?)?;

            x = (&x + layer.ffn.forward(&layer.norm3.forward(&x)?)?)?;
        }
        let x = self.norm.forward(&x)?;
        cache.t += 1;
        Ok((self.tok_head.forward(&x)?, self.mask_head.forward(&x)?))
    }
}

/// A small deterministic generator (SplitMix64) for sampling candidates.
pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The network, ready for inference.
pub(super) struct Network {
    encoder: Encoder,
    decoder: Decoder,
    max_bases: usize,
    max_len: usize,
    max_depth: usize,
    device: Device,
}

/// A decoded candidate: tokens between BOS and EOS, and one base mask per
/// token (bit `i` = base `i`).
pub(super) type Candidate = (Vec<u32>, Vec<u64>);

impl Network {
    pub(super) fn new(cfg: &ModelConfig, weights: Vec<u8>, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, device)?;
        Ok(Self {
            encoder: Encoder::new(cfg, vb.pp("encoder"))?,
            decoder: Decoder::new(cfg, vb.pp("decoder"))?,
            max_bases: cfg.max_bases,
            max_len: cfg.max_len,
            max_depth: cfg.max_depth,
            device: device.clone(),
        })
    }

    /// Edge types are `base` and `max_bases + base`.
    pub(super) fn max_bases(&self) -> usize {
        self.max_bases
    }

    /// Distances in the input graph are clipped to this.
    pub(super) fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// The logits after BOS, for comparison with the PyTorch model.
    #[cfg(test)]
    pub(super) fn first_logits(&self, graph: &Graph) -> Result<(Vec<f32>, Vec<f32>)> {
        let memory = self.encoder.forward(graph, &self.device)?;
        let mut cache = self.decoder.start_cache(&memory)?;
        let (tok, mask) = self.decoder.step(
            &mut cache,
            &Tensor::new(&[BOS], &self.device)?,
            &Tensor::zeros((1, self.max_bases), DType::F32, &self.device)?,
        )?;
        Ok((tok.squeeze(0)?.to_vec1()?, mask.squeeze(0)?.to_vec1()?))
    }

    /// `count` candidates for `graph`, a DFA with `bases` bases: the first is
    /// the greedy decoding, the others are sampled at `temperature` (greedy
    /// too when it is not positive; tokens
    /// from the softmax, each base-mask bit from its sigmoid). Only
    /// well-formed trees within `max_len` are produced, masks only use the
    /// DFA's bases, and the empty class `[]` appears only as the whole regex.
    /// Candidates may repeat.
    pub(super) fn generate(
        &self,
        graph: &Graph,
        bases: usize,
        count: usize,
        temperature: f64,
        rng: &mut Rng,
    ) -> Result<Vec<Candidate>> {
        let nb = self.max_bases;
        let memory = self.encoder.forward(graph, &self.device)?;
        let mut cache = self.decoder.start_cache(&memory)?;
        let present: u64 = if bases >= 64 {
            u64::MAX
        } else {
            (1 << bases) - 1
        };

        let mut grammars = vec![PrefixGrammar::new(self.max_len); count];
        let mut tokens: Vec<Vec<u32>> = vec![vec![BOS]; count];
        let mut masks: Vec<Vec<u64>> = vec![vec![0]; count];
        let mut done = vec![false; count];
        for _ in 0..self.max_len - 1 {
            let last_tokens: Vec<u32> = tokens.iter().map(|t| *t.last().unwrap()).collect();
            let last_masks: Vec<f32> = masks
                .iter()
                .flat_map(|m| {
                    let m = *m.last().unwrap();
                    (0..nb).map(move |i| (m >> i & 1) as f32)
                })
                .collect();
            let (tok_logits, mask_logits) = self.decoder.step(
                &mut cache,
                &Tensor::from_vec(last_tokens, count, &self.device)?,
                &Tensor::from_vec(last_masks, (count, nb), &self.device)?,
            )?;
            let tok_logits = tok_logits.to_vec2::<f32>()?;
            let mask_logits = mask_logits.to_vec2::<f32>()?;

            for row in 0..count {
                let greedy = row == 0 || temperature.is_nan() || temperature <= 0.0;
                let logits: Vec<f64> = tok_logits[row]
                    .iter()
                    .enumerate()
                    .map(|(t, l)| {
                        if grammars[row].allows(t as u32) {
                            *l as f64
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                    .collect();
                let mut next = if greedy {
                    argmax(&logits)
                } else {
                    sample(&logits, temperature, rng)
                } as u32;
                if done[row] {
                    next = EOS;
                }

                let ml = &mask_logits[row];
                let mut mask = 0u64;
                if next == CHAR {
                    for (i, l) in ml.iter().enumerate() {
                        let set = if greedy {
                            *l > 0.0
                        } else {
                            rng.next_f64() < sigmoid(*l as f64 / temperature)
                        };
                        mask |= (set as u64) << i;
                    }
                    mask &= present;
                    // No empty class [] except as the whole regex (the empty
                    // language): an empty mask gets its most likely base.
                    if mask == 0 && tokens[row].len() > 1 {
                        let likely = (0..nb.min(bases))
                            .max_by(|a, b| ml[*a].total_cmp(&ml[*b]).then(b.cmp(a)))
                            .unwrap_or(0);
                        mask = 1 << likely;
                    }
                }

                grammars[row].step(next);
                tokens[row].push(next);
                masks[row].push(mask);
                done[row] |= next == EOS;
            }
            if done.iter().all(|d| *d) {
                break;
            }
        }

        Ok(tokens
            .into_iter()
            .zip(masks)
            .map(|(tokens, masks)| {
                let end = tokens
                    .iter()
                    .position(|t| *t == EOS)
                    .unwrap_or(tokens.len());
                (tokens[1..end].to_vec(), masks[1..end].to_vec())
            })
            .collect())
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Index of the largest value, the first on ties.
fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// Samples an index from the softmax of `logits / temperature`.
fn sample(logits: &[f64], temperature: f64, rng: &mut Rng) -> usize {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits
        .iter()
        .map(|l| ((l - max) / temperature).exp())
        .collect();
    let mut target = rng.next_f64() * weights.iter().sum::<f64>();
    for (i, w) in weights.iter().enumerate() {
        if *w > 0.0 {
            if target < *w {
                return i;
            }
            target -= w;
        }
    }
    // Rounding left `target` past the last weight.
    weights.iter().rposition(|w| *w > 0.0).unwrap_or(0)
}
