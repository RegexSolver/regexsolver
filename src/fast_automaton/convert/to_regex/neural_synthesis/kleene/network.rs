//! The Kleene graph-to-sequence network, on candle: a graph transformer
//! encoder over the DFA, a transformer decoder over the regex tokens. A port
//! of the PyTorch model it was trained as; parameter names match its
//! `model.safetensors`.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear, linear_no_bias,
};
use serde::Deserialize;

use super::attention::{AttendOne, attend_one_unfused};
use super::grammar::PrefixGrammar;
use super::graph::Graph;
use super::tokens::{BOS, CHAR, EOS, VOCAB_SIZE};

const LAYER_NORM_EPS: f64 = 1e-5;
/// Most nodes encoded together.
const ENCODER_NODES: usize = 512;
/// Decoder positions the self-attention cache first has room for.
const INITIAL_CAPACITY: usize = 16;
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

    /// Self-attention over `x` `[N, d]`, the pairs where `mask` `[N, N]` is
    /// -∞ left out.
    fn self_attend(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (n, d) = x.dims2()?;
        let qkv = self.qkv.forward(x)?;
        let [q, k, v] = [0, 1, 2].map(|i| qkv.narrow(1, i * d, d).and_then(|x| self.split(&x)));
        let (q, k, v) = (q?, k?, v?);
        let mut scores = (q.matmul(&k.t()?)? / ((d / self.heads) as f64).sqrt())?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }
        let attended = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        self.out
            .forward(&attended.transpose(0, 1)?.reshape((n, d))?)
    }

    /// One query per row: `q` `[B, d]` over `keys` and `values` `[B or 1,
    /// heads, T, d / heads]`, each row attending to its first
    /// `key_lengths[row]` keys, all without them. `key_mask` `[B, 1, T]`, -∞
    /// past each row's keys, says the same: the CPU's [`AttendOne`] reads the
    /// lengths, the other devices' tensor operations the mask. Returns
    /// `[B, d]` after the output projection.
    fn attend_one(
        &self,
        q: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        key_lengths: Option<&[u32]>,
        key_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attended = if q.device().is_cpu() {
            q.apply_op3_no_bwd(
                keys,
                values,
                &AttendOne {
                    heads: self.heads,
                    lengths: key_lengths,
                },
            )?
        } else {
            attend_one_unfused(q, keys, values, key_mask, self.heads)?
        };
        self.out.forward(&attended)
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

/// The edges of graphs, laid out for [`TransformerConv`].
struct Edges {
    src: Tensor,
    dst: Tensor,
    /// `dst` on the host, for the softmax's per-node maxima.
    dst_nodes: Vec<u32>,
    /// `[E, edge_dim]` edge-type embeddings.
    embedded: Tensor,
}

impl Edges {
    /// For each edge, the largest of `scores` `[E, H]` over the edges into
    /// the same node, per head.
    fn incoming_max(&self, scores: &Tensor, nodes: usize) -> Result<Tensor> {
        let (e, h) = scores.dims2()?;
        let mut max = vec![f32::NEG_INFINITY; nodes * h];
        for (row, node) in scores.to_vec2::<f32>()?.iter().zip(&self.dst_nodes) {
            let node = *node as usize;
            for (max, score) in max[node * h..(node + 1) * h].iter_mut().zip(row) {
                *max = max.max(*score);
            }
        }
        let per_edge: Vec<f32> = self
            .dst_nodes
            .iter()
            .flat_map(|node| {
                max[*node as usize * h..(*node as usize + 1) * h]
                    .iter()
                    .copied()
            })
            .collect();
        Tensor::from_vec(per_edge, (e, h), scores.device())
    }
}

/// PyG's `TransformerConv` (`concat`, `beta`, `root_weight`, edge features),
/// aggregating over the edges into each node with scatter-adds.
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
            let max = edges.incoming_max(&alpha, n)?;
            let exp = (alpha - max)?.exp()?;
            let sum =
                Tensor::zeros((n, h), exp.dtype(), exp.device())?.index_add(&edges.dst, &exp, 0)?;
            let sum = (sum + SOFTMAX_EPS)?.index_select(&edges.dst, 0)?;
            let alpha = (exp / sum)?;
            let messages = value
                .reshape((e, h, c))?
                .broadcast_mul(&alpha.unsqueeze(D::Minus1)?)?
                .reshape((e, d))?;
            Tensor::zeros((n, d), messages.dtype(), messages.device())?
                .index_add(&edges.dst, &messages, 0)?
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
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let x = (x + self.attn.self_attend(&self.norm1.forward(x)?, mask)?)?;
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

    /// Node embeddings of `graphs`, one after the other: `[ΣN, d]`. The
    /// graphs are encoded together, as one graph whose nodes attend only
    /// within their own.
    fn forward(&self, graphs: &[&Graph], device: &Device) -> Result<Tensor> {
        let ids = |v: Vec<u32>| {
            let len = v.len();
            Tensor::from_vec(v, len, device)
        };
        let concat = |field: fn(&Graph) -> &Vec<u32>| -> Vec<u32> {
            graphs
                .iter()
                .flat_map(|graph| field(graph).iter().copied())
                .collect()
        };
        let mut h = (self.type_emb.forward(&ids(concat(|g| &g.node_type))?)?
            + self.accept_emb.forward(&ids(concat(|g| &g.is_accept))?)?)?;
        h = (h + self
            .fwd_depth_emb
            .forward(&ids(concat(|g| &g.fwd_depth))?)?)?;
        h = (h + self
            .to_accept_emb
            .forward(&ids(concat(|g| &g.bwd_depth))?)?)?;
        h = (h + self
            .to_reject_emb
            .forward(&ids(concat(|g| &g.to_reject))?)?)?;

        // Each graph's edges, renumbered to its place among the nodes.
        let mut offset = 0;
        let (mut src, mut dst) = (vec![], vec![]);
        for graph in graphs {
            src.extend(graph.edge_src.iter().map(|node| node + offset));
            dst.extend(graph.edge_dst.iter().map(|node| node + offset));
            offset += graph.nodes as u32;
        }
        let edges = Edges {
            src: ids(src)?,
            dst: ids(dst.clone())?,
            dst_nodes: dst,
            embedded: self.edge_emb.forward(&ids(concat(|g| &g.edge_type))?)?,
        };
        for layer in &self.layers {
            h = layer.forward(&h, &edges)?;
        }

        let mask = if graphs.len() > 1 {
            let n = offset as usize;
            let mut mask = vec![f32::NEG_INFINITY; n * n];
            let mut start = 0;
            for graph in graphs {
                for row in start..start + graph.nodes {
                    mask[row * n + start..row * n + start + graph.nodes].fill(0.0);
                }
                start += graph.nodes;
            }
            Some(Tensor::from_vec(mask, (n, n), device)?)
        } else {
            None
        };
        for layer in &self.global_layers {
            h = layer.forward(&h, mask.as_ref())?;
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

/// The state of an incremental decoding of `B` sequences, over one DFA or
/// several.
struct Cache {
    /// `[1 or B, d]`, added at position 0.
    start: Tensor,
    /// Per layer: memory keys and values, `[1, heads, N, d / heads]` for one
    /// DFA, `[B, heads, N, d / heads]` (padded to the largest DFA) for
    /// several.
    cross: Vec<(Tensor, Tensor)>,
    /// For several DFAs: each row's number of nodes, and `[B, 1, N]`, -∞ on
    /// the padding nodes.
    key_lengths: Option<Vec<u32>>,
    key_mask: Option<Tensor>,
    /// Per layer: self-attention keys and values of the positions so far,
    /// in buffers with room for more, `[B, heads, capacity, d / heads]`.
    past: Vec<Option<(Tensor, Tensor)>>,
    /// Positions fed so far.
    t: usize,
}

impl Cache {
    /// Writes layer `layer`'s self-attention keys and values of position
    /// `self.t`, `[B, heads, 1, d / heads]`, in place (doubling the buffers
    /// when full), and returns those of the positions so far.
    fn append(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = self.t;
        let (keys, values) = match self.past[layer].take() {
            None => {
                let (rows, heads, _, hd) = key.dims4()?;
                let buffer = || {
                    Tensor::zeros(
                        (rows, heads, INITIAL_CAPACITY, hd),
                        key.dtype(),
                        key.device(),
                    )
                };
                (buffer()?, buffer()?)
            }
            Some((keys, values)) if keys.dim(2)? == t => (
                keys.pad_with_zeros(2, 0, t)?,
                values.pad_with_zeros(2, 0, t)?,
            ),
            Some(past) => past,
        };
        keys.slice_set(&key.contiguous()?, 2, t)?;
        values.slice_set(&value.contiguous()?, 2, t)?;
        let so_far = (keys.narrow(2, 0, t + 1)?, values.narrow(2, 0, t + 1)?);
        self.past[layer] = Some((keys, values));
        Ok(so_far)
    }

    /// Keeps only the sequences `keep` (indices into the current rows).
    fn retain(&mut self, keep: &[u32]) -> Result<()> {
        if let Some(key_lengths) = &mut self.key_lengths {
            *key_lengths = keep.iter().map(|row| key_lengths[*row as usize]).collect();
        }
        let keep = &Tensor::new(keep, self.start.device())?;
        for past in self.past.iter_mut().flatten() {
            *past = (past.0.index_select(keep, 0)?, past.1.index_select(keep, 0)?);
        }
        if let Some(key_mask) = &self.key_mask {
            self.key_mask = Some(key_mask.index_select(keep, 0)?);
            for cross in &mut self.cross {
                *cross = (
                    cross.0.index_select(keep, 0)?,
                    cross.1.index_select(keep, 0)?,
                );
            }
        }
        Ok(())
    }
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

    /// Prepares the decoding of `rows` sequences per DFA over `memory`
    /// `[ΣN, d]`, the node embeddings of DFAs of `nodes` nodes one after the
    /// other; each DFA's start state is its first node.
    fn start_cache(&self, memory: &Tensor, nodes: &[usize], rows: usize) -> Result<Cache> {
        let d = memory.dim(1)?;
        let device = memory.device();
        let dfas = nodes.len();
        let firsts: Vec<u32> = nodes
            .iter()
            .scan(0, |first, n| {
                let this = *first;
                *first += *n as u32;
                Some(this)
            })
            .collect();
        let start = self
            .start_proj
            .forward(&memory.index_select(&Tensor::new(firsts.as_slice(), device)?, 0)?)?;
        if dfas == 1 {
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
            return Ok(Cache {
                start,
                cross,
                key_lengths: None,
                key_mask: None,
                past: vec![None; self.layers.len()],
                t: 0,
            });
        }

        // Each DFA's nodes in `longest` slots, the padding at a zero row
        // appended after the nodes.
        let longest = nodes.iter().copied().max().unwrap_or(0);
        let padding = firsts[dfas - 1] + nodes[dfas - 1] as u32;
        let mut slots = Vec::with_capacity(dfas * longest);
        let mut key_mask = Vec::with_capacity(dfas * longest);
        for (first, n) in firsts.iter().zip(nodes) {
            for slot in 0..longest {
                let real = slot < *n;
                slots.push(if real { first + slot as u32 } else { padding });
                key_mask.push(if real { 0.0 } else { f32::NEG_INFINITY });
            }
        }
        let slots = Tensor::from_vec(slots, dfas * longest, device)?;
        // Row `r` decodes DFA `r / rows`.
        let owner: Vec<u32> = (0..dfas * rows).map(|r| (r / rows) as u32).collect();
        let owner = Tensor::from_vec(owner, dfas * rows, device)?;
        let cross = self
            .layers
            .iter()
            .map(|layer| {
                let attn = &layer.cross_attn;
                let (heads, hd) = (attn.heads, d / attn.heads);
                let kv = attn
                    .kv
                    .forward(memory)?
                    .pad_with_zeros(0, 0, 1)?
                    .index_select(&slots, 0)?
                    .reshape((dfas, longest, 2 * d))?;
                // `[DFAs, longest, d]` → `[B, heads, longest, d / heads]`.
                let per_row = |part: usize| -> Result<Tensor> {
                    kv.narrow(2, part * d, d)?
                        .reshape((dfas, longest, heads, hd))?
                        .transpose(1, 2)?
                        .contiguous()?
                        .index_select(&owner, 0)
                };
                Ok((per_row(0)?, per_row(1)?))
            })
            .collect::<Result<_>>()?;
        let key_mask = Tensor::from_vec(key_mask, (dfas, 1, longest), device)?;
        let key_lengths = (0..dfas * rows).map(|r| nodes[r / rows] as u32).collect();
        Ok(Cache {
            start: start.index_select(&owner, 0)?,
            cross,
            key_lengths: Some(key_lengths),
            key_mask: Some(key_mask.index_select(&owner, 0)?),
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
            let (k, v) = cache.append(i, &head(1)?, &head(2)?)?;
            x = (x + attn.attend_one(&qkv.narrow(1, 0, d)?, &k, &v, None, None)?)?;

            let attn = &layer.cross_attn;
            let q = attn.q.forward(&layer.norm2.forward(&x)?)?;
            let (k, v) = &cache.cross[i];
            x = (&x
                + attn.attend_one(
                    &q,
                    k,
                    v,
                    cache.key_lengths.as_deref(),
                    cache.key_mask.as_ref(),
                )?)?;

            x = (&x + layer.ffn.forward(&layer.norm3.forward(&x)?)?)?;
        }
        let x = self.norm.forward(&x)?;
        cache.t += 1;
        Ok((self.tok_head.forward(&x)?, self.mask_head.forward(&x)?))
    }
}

/// A small deterministic generator (SplitMix64) for sampling candidates.
struct Rng(u64);

impl Rng {
    /// The generator of candidate `index` for `seed`: each candidate has its
    /// own, so a candidate does not depend on what is decoded with it.
    fn for_candidate(seed: u64, index: usize) -> Self {
        let mut rng = Self(seed ^ (index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03));
        rng.next_u64();
        rng
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
        Self::from_var_builder(cfg, vb, device)
    }

    /// The network whose parameters `vb` provides.
    pub(super) fn from_var_builder(
        cfg: &ModelConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
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

    /// Node embeddings of `graphs`, one after the other: `[ΣN, d]`. The
    /// graphs are encoded in groups of at most [`ENCODER_NODES`] nodes, which
    /// bounds the global attention's `[N, N]` scores.
    fn encode<'g>(&self, graphs: impl Iterator<Item = &'g Graph>) -> Result<Tensor> {
        let mut encoded = vec![];
        let mut group: Vec<&Graph> = vec![];
        let mut group_nodes = 0;
        for graph in graphs {
            if !group.is_empty() && group_nodes + graph.nodes > ENCODER_NODES {
                encoded.push(self.encoder.forward(&group, &self.device)?);
                group.clear();
                group_nodes = 0;
            }
            group.push(graph);
            group_nodes += graph.nodes;
        }
        encoded.push(self.encoder.forward(&group, &self.device)?);
        Tensor::cat(&encoded, 0)
    }

    /// The logits after BOS, for comparison with the PyTorch model.
    #[cfg(test)]
    pub(super) fn first_logits(&self, graph: &Graph) -> Result<(Vec<f32>, Vec<f32>)> {
        let memory = self.encoder.forward(&[graph], &self.device)?;
        let mut cache = self.decoder.start_cache(&memory, &[graph.nodes], 1)?;
        let (tok, mask) = self.decoder.step(
            &mut cache,
            &Tensor::new(&[BOS], &self.device)?,
            &Tensor::zeros((1, self.max_bases), DType::F32, &self.device)?,
        )?;
        Ok((tok.squeeze(0)?.to_vec1()?, mask.squeeze(0)?.to_vec1()?))
    }

    /// `count` candidates for `graph`, a DFA with `bases` bases: see
    /// [`generate_batch`](Self::generate_batch).
    #[cfg(test)]
    pub(super) fn generate(
        &self,
        graph: &Graph,
        bases: usize,
        count: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<Vec<Candidate>> {
        Ok(self
            .generate_batch(&[(graph, bases)], count, temperature, seed)?
            .pop()
            .unwrap())
    }

    /// `count` candidates for each of `inputs`, DFAs as graphs with their
    /// number of bases, decoded together. For each DFA, the first candidate
    /// is the greedy decoding, the others are sampled at `temperature`
    /// (greedy too when it is not positive; tokens from the softmax, each
    /// base-mask bit from its sigmoid), each from its own generator seeded
    /// from `seed` and its index, so a DFA's candidates do not depend on the
    /// other DFAs. Only well-formed trees within `max_len` are produced,
    /// masks only use the DFA's bases, and the empty class `[]` appears only
    /// as the whole regex. Candidates may repeat.
    pub(super) fn generate_batch(
        &self,
        inputs: &[(&Graph, usize)],
        count: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<Vec<Vec<Candidate>>> {
        if inputs.is_empty() || count == 0 {
            return Ok(vec![vec![]; inputs.len()]);
        }
        let nb = self.max_bases;
        let memory = self.encode(inputs.iter().map(|(graph, _)| *graph))?;
        let nodes: Vec<usize> = inputs.iter().map(|(graph, _)| graph.nodes).collect();
        let mut cache = self.decoder.start_cache(&memory, &nodes, count)?;
        let present: Vec<u64> = inputs
            .iter()
            .map(|(_, bases)| {
                if *bases >= 64 {
                    u64::MAX
                } else {
                    (1 << bases) - 1
                }
            })
            .collect();

        let rows = inputs.len() * count;
        let mut grammars = vec![PrefixGrammar::new(self.max_len); rows];
        let mut rngs: Vec<Rng> = (0..rows)
            .map(|r| Rng::for_candidate(seed, r % count))
            .collect();
        let mut tokens: Vec<Vec<u32>> = vec![vec![BOS]; rows];
        let mut masks: Vec<Vec<u64>> = vec![vec![0]; rows];
        // The rows still decoding, in the order of the cache's rows.
        let mut active: Vec<usize> = (0..rows).collect();
        for _ in 0..self.max_len - 1 {
            let last_tokens: Vec<u32> =
                active.iter().map(|r| *tokens[*r].last().unwrap()).collect();
            let last_masks: Vec<f32> = active
                .iter()
                .flat_map(|r| {
                    let m = *masks[*r].last().unwrap();
                    (0..nb).map(move |i| (m >> i & 1) as f32)
                })
                .collect();
            let (tok_logits, mask_logits) = self.decoder.step(
                &mut cache,
                &Tensor::from_vec(last_tokens, active.len(), &self.device)?,
                &Tensor::from_vec(last_masks, (active.len(), nb), &self.device)?,
            )?;
            let tok_logits = tok_logits.to_vec2::<f32>()?;
            let mask_logits = mask_logits.to_vec2::<f32>()?;

            for (i, &row) in active.iter().enumerate() {
                let (input, index) = (row / count, row % count);
                let bases = inputs[input].1;
                let greedy = index == 0 || temperature.is_nan() || temperature <= 0.0;
                let rng = &mut rngs[row];
                let logits: Vec<f64> = tok_logits[i]
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
                let next = if greedy {
                    argmax(&logits)
                } else {
                    sample(&logits, temperature, rng)
                } as u32;

                let ml = &mask_logits[i];
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
                    mask &= present[input];
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
            }

            let keep: Vec<u32> = (0..active.len() as u32)
                .filter(|i| *tokens[active[*i as usize]].last().unwrap() != EOS)
                .collect();
            if keep.is_empty() {
                break;
            }
            if keep.len() < active.len() {
                active = keep.iter().map(|i| active[*i as usize]).collect();
                cache.retain(&keep)?;
            }
        }

        let mut candidates: Vec<Vec<Candidate>> = vec![Vec::with_capacity(count); inputs.len()];
        for (row, (tokens, masks)) in tokens.into_iter().zip(masks).enumerate() {
            let end = tokens
                .iter()
                .position(|t| *t == EOS)
                .unwrap_or(tokens.len());
            candidates[row / count].push((tokens[1..end].to_vec(), masks[1..end].to_vec()));
        }
        Ok(candidates)
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
