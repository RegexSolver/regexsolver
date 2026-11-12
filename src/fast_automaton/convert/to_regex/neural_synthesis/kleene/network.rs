//! The Kleene graph-to-sequence network, on candle: a graph transformer
//! encoder over the DFA, a transformer decoder over the regex tokens. A port
//! of the PyTorch model it was trained as; parameter names match its
//! `model.safetensors`.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder, embedding};
use serde::Deserialize;

use super::attention::{
    AttendOne, GraphAttention, PastCpu, attend_one_unfused, graph_attention_padded,
};
use super::edge_attention::{EdgeAttention, Incoming, edge_attention_unfused};
use super::grammar::PrefixGrammar;
use super::graph::Graph;
use super::layers::{LayerNorm, Linear, layer_norm, linear, linear_no_bias};
use super::tokens::{BOS, CHAR, EOS, VOCAB_SIZE};

const LAYER_NORM_EPS: f64 = 1e-5;
/// Most nodes encoded together.
const ENCODER_NODES: usize = 512;
/// Decoder positions the self-attention cache first has room for.
const INITIAL_CAPACITY: usize = 16;

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

    /// Self-attention over `x` `[N, d]`, each node attending to the nodes
    /// of its graph.
    fn self_attend(&self, x: &Tensor, blocks: &Blocks) -> Result<Tensor> {
        let qkv = self.qkv.forward(x)?;
        let attended = if x.device().is_cpu() {
            qkv.apply_op1_no_bwd(&GraphAttention {
                heads: self.heads,
                starts: &blocks.starts,
                lengths: &blocks.lengths,
            })?
        } else {
            graph_attention_padded(&qkv, &blocks.graphs, self.heads)?
        };
        self.out.forward(&attended)
    }

    /// One query per row: `q` `[B, d]` over `keys` and `values` `[1 or
    /// DFAs, heads, N, d / heads]`, each row attending to the nodes of its
    /// DFA (`owners`, for several). Returns `[B, d]` after the output
    /// projection.
    fn cross_attend(
        &self,
        q: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        owners: Option<&Owners>,
    ) -> Result<Tensor> {
        let attended = if q.device().is_cpu() {
            q.apply_op3_no_bwd(
                keys,
                values,
                &AttendOne {
                    heads: self.heads,
                    lengths: owners.map(|owners| owners.lengths.as_slice()),
                    key_rows: owners.map(|owners| owners.dfa.as_slice()),
                },
            )?
        } else if let Some(owners) = owners {
            let expand = Tensor::new(owners.dfa.as_slice(), q.device())?;
            attend_one_unfused(
                q,
                &keys.index_select(&expand, 0)?,
                &values.index_select(&expand, 0)?,
                Some(&owners.mask.index_select(&expand, 0)?),
                self.heads,
            )?
        } else {
            attend_one_unfused(q, keys, values, None, self.heads)?
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
        self.down.forward(&self.up.forward_gelu(x)?)
    }
}

/// The edges of graphs, laid out for [`TransformerConv`].
struct Edges {
    src: Tensor,
    dst: Tensor,
    /// `dst` on the host, for the softmax's per-node maxima.
    dst_nodes: Vec<u32>,
    /// The edges into each node, for [`EdgeAttention`].
    incoming: Incoming,
    /// `[E]` edge types.
    types: Tensor,
}

/// The graphs encoded together, for the global attention.
struct Blocks {
    /// Per node: the first node of its graph, and its graph's number of
    /// nodes.
    starts: Vec<u32>,
    lengths: Vec<u32>,
    /// Per graph: its first node and number of nodes.
    graphs: Vec<(usize, usize)>,
}

/// PyG's `TransformerConv` (`concat`, `beta`, `root_weight`, edge features).
struct TransformerConv {
    /// `lin_query`, `lin_key`, `lin_value` and `lin_skip` at once.
    qkvs: Linear,
    /// `lin_edge` of each edge type's embedding, `[types, d]`.
    edge_table: Tensor,
    /// `lin_beta` of `[out, x_r, out - x_r]`, as weights of `[out, x_r]`.
    beta: Linear,
    heads: usize,
}

impl TransformerConv {
    /// `edge_emb`: the edge types' embeddings `[types, edge_dim]`.
    fn new(d: usize, heads: usize, edge_emb: &Tensor, vb: VarBuilder) -> Result<Self> {
        let edge_dim = edge_emb.dim(1)?;
        let parts = ["lin_query", "lin_key", "lin_value", "lin_skip"];
        let weights = (parts.iter())
            .map(|part| vb.pp(part).get((d, d), "weight"))
            .collect::<Result<Vec<_>>>()?;
        let biases = (parts.iter())
            .map(|part| vb.pp(part).get(d, "bias"))
            .collect::<Result<Vec<_>>>()?;
        let beta = vb.pp("lin_beta").get((1, 3 * d), "weight")?;
        let third = |i: usize| beta.narrow(1, i * d, d);
        Ok(Self {
            qkvs: Linear::new(Tensor::cat(&weights, 0)?, Some(Tensor::cat(&biases, 0)?)),
            edge_table: linear_no_bias(edge_dim, d, vb.pp("lin_edge"))?.forward(edge_emb)?,
            beta: Linear::new(
                Tensor::cat(&[(third(0)? + third(2)?)?, (third(1)? - third(2)?)?], 1)?,
                None,
            ),
            heads,
        })
    }

    fn forward(&self, x: &Tensor, edges: &Edges) -> Result<Tensor> {
        let (n, d) = x.dims2()?;
        let qkvs = self.qkvs.forward(x)?;
        let x_r = qkvs.narrow(1, 3 * d, d)?;
        let out = if edges.dst_nodes.is_empty() {
            Tensor::zeros((n, d), x.dtype(), x.device())?
        } else {
            let qkv = qkvs.narrow(1, 0, 3 * d)?;
            if x.device().is_cpu() {
                qkv.apply_op2_no_bwd(
                    &self.edge_table,
                    &EdgeAttention {
                        heads: self.heads,
                        incoming: &edges.incoming,
                    },
                )?
            } else {
                let (src, dst) = (&edges.src, &edges.dst);
                let edge = self.edge_table.index_select(&edges.types, 0)?;
                edge_attention_unfused(&qkv, &edge, src, dst, &edges.dst_nodes, self.heads)?
            }
        };
        let beta =
            candle_nn::ops::sigmoid(&self.beta.forward(&Tensor::cat(&[&out, &x_r], D::Minus1)?)?)?;
        &out + (x_r - &out)?.broadcast_mul(&beta)?
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
    fn forward(&self, x: &Tensor, blocks: &Blocks) -> Result<Tensor> {
        let x = (x + self.attn.self_attend(&self.norm1.forward(x)?, blocks)?)?;
        &x + self.ffn.forward(&self.norm2.forward(&x)?)?
    }
}

struct Encoder {
    type_emb: Embedding,
    accept_emb: Embedding,
    fwd_depth_emb: Embedding,
    to_accept_emb: Embedding,
    to_reject_emb: Embedding,
    layers: Vec<ConvBlock>,
    global_layers: Vec<EncoderLayer>,
    out_norm: LayerNorm,
}

impl Encoder {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let (d, depth) = (cfg.d_model, cfg.max_depth);
        let ffn = cfg.ffn.unwrap_or(4 * d);
        let edge_emb = embedding(2 * cfg.max_bases, cfg.edge_dim, vb.pp("edge_emb"))?;
        let layers = (0..cfg.enc_layers)
            .map(|i| {
                let vb = vb.pp("layers").pp(i);
                Ok(ConvBlock {
                    norm1: layer_norm(d, LAYER_NORM_EPS, vb.pp("norm1"))?,
                    conv: TransformerConv::new(
                        d,
                        cfg.n_heads,
                        edge_emb.embeddings(),
                        vb.pp("conv"),
                    )?,
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
        let n = offset as usize;
        let types = concat(|g| &g.edge_type);
        let edges = Edges {
            src: ids(src.clone())?,
            dst: ids(dst.clone())?,
            incoming: Incoming::new(n, &src, &dst, &types),
            dst_nodes: dst,
            types: ids(types)?,
        };
        for layer in &self.layers {
            h = layer.forward(&h, &edges)?;
        }

        let mut blocks = Blocks {
            starts: Vec::with_capacity(n),
            lengths: Vec::with_capacity(n),
            graphs: Vec::with_capacity(graphs.len()),
        };
        for graph in graphs {
            let first = blocks.starts.len();
            blocks
                .starts
                .extend(std::iter::repeat_n(first as u32, graph.nodes));
            blocks
                .lengths
                .extend(std::iter::repeat_n(graph.nodes as u32, graph.nodes));
            blocks.graphs.push((first, graph.nodes));
        }
        for layer in &self.global_layers {
            h = layer.forward(&h, &blocks)?;
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
    /// `tok_head` and `mask_head` at once: the token logits, then the
    /// base-mask logits.
    head: Linear,
}

/// The rows of an incremental decoding over several DFAs.
struct Owners {
    /// Each row's DFA, and its number of nodes.
    dfa: Vec<u32>,
    lengths: Vec<u32>,
    /// `[DFAs, 1, N]`, -∞ on each DFA's padding nodes.
    mask: Tensor,
}

/// The self-attention keys and values of the positions so far.
enum Past {
    /// On the CPU: buffers written in place.
    Cpu(PastCpu),
    /// On the other devices, per layer: buffers with room for more, `[B,
    /// heads, capacity, d / heads]`.
    Tensors(Vec<Option<(Tensor, Tensor)>>),
}

/// The state of an incremental decoding of `B` sequences, over one DFA or
/// several.
struct Cache {
    /// `[1 or B, d]`, added at position 0.
    start: Tensor,
    /// Per layer: memory keys and values, `[1, heads, N, d / heads]` for one
    /// DFA, `[DFAs, heads, N, d / heads]` (padded to the largest DFA) for
    /// several.
    cross: Vec<(Tensor, Tensor)>,
    /// For several DFAs.
    owners: Option<Owners>,
    past: Past,
    /// Positions fed so far.
    t: usize,
}

/// Writes the self-attention keys and values of position `t`, `[B, heads,
/// 1, d / heads]`, in place in `past` (doubling the buffers when full), and
/// returns those of the positions so far.
fn append(
    past: &mut Option<(Tensor, Tensor)>,
    t: usize,
    key: &Tensor,
    value: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (keys, values) = match past.take() {
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
        Some(buffers) => buffers,
    };
    keys.slice_set(&key.contiguous()?, 2, t)?;
    values.slice_set(&value.contiguous()?, 2, t)?;
    let so_far = (keys.narrow(2, 0, t + 1)?, values.narrow(2, 0, t + 1)?);
    *past = Some((keys, values));
    Ok(so_far)
}

impl Cache {
    /// Row `i` becomes a copy of row `parents[i]` (indices into the current
    /// rows, in any order): sequences are dropped, kept or forked.
    fn reorder(&mut self, parents: &[u32]) -> Result<()> {
        if let Some(owners) = &mut self.owners {
            owners.dfa = (parents.iter())
                .map(|row| owners.dfa[*row as usize])
                .collect();
            owners.lengths = (parents.iter())
                .map(|row| owners.lengths[*row as usize])
                .collect();
        }
        match &mut self.past {
            Past::Cpu(past) => past.reorder(parents),
            Past::Tensors(past) => {
                let parents = &Tensor::new(parents, self.start.device())?;
                for past in past.iter_mut().flatten() {
                    *past = (
                        past.0.index_select(parents, 0)?,
                        past.1.index_select(parents, 0)?,
                    );
                }
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
        let head = |name: &str, outputs: usize| -> Result<(Tensor, Tensor)> {
            let vb = vb.pp(name);
            Ok((vb.get((outputs, d), "weight")?, vb.get(outputs, "bias")?))
        };
        let (tok_w, tok_b) = head("tok_head", VOCAB_SIZE)?;
        let (mask_w, mask_b) = head("mask_head", cfg.max_bases)?;
        Ok(Self {
            tok_emb: embedding(VOCAB_SIZE, d, vb.pp("tok_emb"))?,
            mask_emb: linear_no_bias(cfg.max_bases, d, vb.pp("mask_emb"))?,
            pos_emb: vb.get((cfg.max_len, d), "pos_emb.weight")?,
            start_proj: linear(d, d, vb.pp("start_proj"))?,
            layers,
            norm: layer_norm(d, LAYER_NORM_EPS, vb.pp("layers.norm"))?,
            head: Linear::new(
                Tensor::cat(&[tok_w, mask_w], 0)?,
                Some(Tensor::cat(&[tok_b, mask_b], 0)?),
            ),
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
        let past = match self.layers.first() {
            Some(layer) if device.is_cpu() => {
                let heads = layer.self_attn.heads;
                Past::Cpu(PastCpu::new(
                    self.layers.len(),
                    dfas * rows,
                    heads,
                    d / heads,
                    INITIAL_CAPACITY,
                ))
            }
            _ => Past::Tensors(vec![None; self.layers.len()]),
        };
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
                owners: None,
                past,
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
                // `[DFAs, longest, d]` → `[DFAs, heads, longest, d / heads]`.
                let per_dfa = |part: usize| -> Result<Tensor> {
                    kv.narrow(2, part * d, d)?
                        .reshape((dfas, longest, heads, hd))?
                        .transpose(1, 2)?
                        .contiguous()
                };
                Ok((per_dfa(0)?, per_dfa(1)?))
            })
            .collect::<Result<_>>()?;
        // Row `r` decodes DFA `r / rows`.
        let dfa: Vec<u32> = (0..dfas * rows).map(|r| (r / rows) as u32).collect();
        let owner = Tensor::new(dfa.as_slice(), device)?;
        Ok(Cache {
            start: start.index_select(&owner, 0)?,
            cross,
            owners: Some(Owners {
                lengths: dfa.iter().map(|dfa| nodes[*dfa as usize] as u32).collect(),
                dfa,
                mask: Tensor::from_vec(key_mask, (dfas, 1, longest), device)?,
            }),
            past,
            t: 0,
        })
    }

    /// Feeds position `cache.t`: `tokens` `[B]` and their base masks `[B,
    /// max_bases]`. Returns the next position's token logits and base-mask
    /// logits, `[B, vocab + max_bases]`.
    fn step(&self, cache: &mut Cache, tokens: &Tensor, masks: &Tensor) -> Result<Tensor> {
        let mut x = (self.tok_emb.forward(tokens)? + self.mask_emb.forward(masks)?)?
            .broadcast_add(&self.pos_emb.narrow(0, cache.t, 1)?)?;
        if cache.t == 0 {
            x = x.broadcast_add(&cache.start)?;
        }
        let (b, d) = x.dims2()?;
        let t = cache.t;
        for (i, layer) in self.layers.iter().enumerate() {
            let attn = &layer.self_attn;
            let (heads, hd) = (attn.heads, d / attn.heads);
            let qkv = attn.qkv.forward(&layer.norm1.forward(&x)?)?;
            let attended = match &mut cache.past {
                Past::Cpu(past) => past.attend(i, t, &qkv)?,
                Past::Tensors(past) => {
                    let head = |part: usize| qkv.narrow(1, part * d, d)?.reshape((b, heads, 1, hd));
                    let (k, v) = append(&mut past[i], t, &head(1)?, &head(2)?)?;
                    attend_one_unfused(&qkv.narrow(1, 0, d)?, &k, &v, None, heads)?
                }
            };
            x = (x + attn.out.forward(&attended)?)?;

            let attn = &layer.cross_attn;
            let q = attn.q.forward(&layer.norm2.forward(&x)?)?;
            let (k, v) = &cache.cross[i];
            x = (&x + attn.cross_attend(&q, k, v, cache.owners.as_ref())?)?;

            x = (&x + layer.ffn.forward(&layer.norm3.forward(&x)?)?)?;
        }
        cache.t += 1;
        self.head.forward(&self.norm.forward(&x)?)
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
    /// bounds the temporaries.
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

    /// The logits after BOS, for comparison with the PyTorch model, and where
    /// the encoder's output and the logits were computed.
    #[cfg(test)]
    pub(super) fn first_logits(
        &self,
        graph: &Graph,
    ) -> Result<(Vec<f32>, Vec<f32>, [candle_core::DeviceLocation; 2])> {
        let memory = self.encoder.forward(&[graph], &self.device)?;
        let mut cache = self.decoder.start_cache(&memory, &[graph.nodes], 1)?;
        let logits = self.decoder.step(
            &mut cache,
            &Tensor::new(&[BOS], &self.device)?,
            &Tensor::zeros((1, self.max_bases), DType::F32, &self.device)?,
        )?;
        let locations = [memory.device().location(), logits.device().location()];
        let mut tok = logits.squeeze(0)?.to_vec1()?;
        let mask = tok.split_off(VOCAB_SIZE);
        Ok((tok, mask, locations))
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
        beam: usize,
    ) -> Result<Vec<Candidate>> {
        Ok(self
            .generate_batch(&[(graph, bases)], count, temperature, seed, beam)?
            .pop()
            .unwrap())
    }

    /// Candidates for each of `inputs`, DFAs as graphs with their number of
    /// bases, decoded together. For each DFA, first the greedy decoding or,
    /// with a `beam` width above 1, the hypotheses of a beam search of that
    /// width, most likely first; then `count - 1` samples at `temperature`
    /// (greedy too when it is not positive; tokens from the softmax, each
    /// base-mask bit from its sigmoid), each from its own generator seeded
    /// from `seed` and its index, so a DFA's candidates do not depend on the
    /// other DFAs. Only well-formed trees within `max_len` are produced,
    /// masks only use the DFA's bases, and the empty class `[]` appears only
    /// as the whole regex. Candidates may repeat.
    ///
    /// The beam search is the PyTorch model's: a hypothesis scores the sum of
    /// its tokens' log-probabilities, with no length normalization (shorter
    /// regexes are not penalized); a `CHAR`'s base mask is the greedy one,
    /// and its bits' log-probabilities count in the `CHAR`'s. Every row, the
    /// beams' and the samples', is decoded in the same steps; a finished
    /// hypothesis leaves the decoding, and a hypothesis extended several ways
    /// is forked in the cache.
    pub(super) fn generate_batch(
        &self,
        inputs: &[(&Graph, usize)],
        count: usize,
        temperature: f64,
        seed: u64,
        beam: usize,
    ) -> Result<Vec<Vec<Candidate>>> {
        if inputs.is_empty() || count == 0 {
            return Ok(vec![vec![]; inputs.len()]);
        }
        let nb = self.max_bases;
        let beam = (beam > 1).then_some(beam);
        // Without sampling, the samples are all the greedy decoding: it is
        // decoded once.
        let greedy_only = temperature.is_nan() || temperature <= 0.0;
        // Per DFA, the candidates the beam does not give, by index: the
        // greedy decoding (without a beam) and the samples.
        let first = usize::from(beam.is_some());
        let drawn = if greedy_only {
            (count - first).min(1)
        } else {
            count - first
        };
        let memory = self.encode(inputs.iter().map(|(graph, _)| *graph))?;
        let nodes: Vec<usize> = inputs.iter().map(|(graph, _)| graph.nodes).collect();
        let mut cache = self.decoder.start_cache(&memory, &nodes, first + drawn)?;
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

        // The rows being decoded, in the order of the cache's rows: per DFA,
        // its beam's first hypothesis, then its drawn candidates.
        let mut tree = Tree::new();
        let mut rows: Vec<Row> = Vec::with_capacity(inputs.len() * (first + drawn));
        let mut beams: Vec<Vec<Hypothesis>> = vec![vec![]; inputs.len()];
        for (input, hypotheses) in beams.iter_mut().enumerate() {
            if beam.is_some() {
                hypotheses.push(Hypothesis {
                    score: 0.0,
                    node: Tree::ROOT,
                    row: Some(rows.len() as u32),
                });
                rows.push(Row::new(input, Draw::Beam, self.max_len));
            }
            for index in first..first + drawn {
                let draw = if index == 0 || greedy_only {
                    Draw::Greedy(index)
                } else {
                    Draw::Sample(index, Rng::for_candidate(seed, index))
                };
                rows.push(Row::new(input, draw, self.max_len));
            }
        }
        // The last node of each drawn candidate, by DFA and index.
        let mut ends: Vec<Vec<u32>> = vec![vec![Tree::ROOT; first + drawn]; inputs.len()];

        let width = VOCAB_SIZE + nb;
        let mut last_tokens: Vec<u32> = Vec::with_capacity(rows.len());
        let mut last_masks: Vec<f32> = Vec::with_capacity(rows.len() * nb);
        let mut weights: Vec<f64> = Vec::with_capacity(VOCAB_SIZE);
        // Per beam row: each token's log-probability, the mask of a `CHAR`.
        let mut log_probs: Vec<f64> = vec![];
        let mut char_masks: Vec<u64> = vec![];
        let mut keep: Vec<bool> = vec![];
        let mut expansions: Vec<Expansion> = vec![];
        for t in 0..self.max_len - 1 {
            if rows.is_empty() {
                break;
            }
            last_tokens.clear();
            last_tokens.extend(rows.iter().map(|row| tree.token(row.node)));
            last_masks.clear();
            last_masks.extend(rows.iter().flat_map(|row| {
                let m = tree.mask(row.node);
                (0..nb).map(move |i| (m >> i & 1) as f32)
            }));
            let mut logits = self
                .decoder
                .step(
                    &mut cache,
                    &Tensor::from_slice(&last_tokens, rows.len(), &self.device)?,
                    &Tensor::from_slice(&last_masks, (rows.len(), nb), &self.device)?,
                )?
                .flatten_all()?
                .to_vec1::<f32>()?;

            log_probs.resize(rows.len() * VOCAB_SIZE, f64::NEG_INFINITY);
            char_masks.resize(rows.len(), 0);
            keep.clear();
            keep.resize(rows.len(), false);
            for (i, (row, logits)) in rows
                .iter_mut()
                .zip(logits.chunks_exact_mut(width))
                .enumerate()
            {
                let (tok, ml) = logits.split_at_mut(VOCAB_SIZE);
                for (token, l) in tok.iter_mut().enumerate() {
                    if !row.grammar.allows(token as u32) {
                        *l = f32::NEG_INFINITY;
                    }
                }
                let bases = inputs[row.input].1.min(nb);
                // A `CHAR`'s mask: its bits drawn, then only the DFA's bases,
                // and empty only as the whole regex (the empty language): an
                // empty mask gets its most likely base.
                let char_mask = |rng: Option<&mut Rng>| {
                    let mut mask = 0u64;
                    match rng {
                        Some(rng) => {
                            for (bit, l) in ml.iter().enumerate() {
                                let set = rng.next_f64() < sigmoid(*l as f64 / temperature);
                                mask |= (set as u64) << bit;
                            }
                        }
                        None => {
                            for (bit, l) in ml.iter().enumerate() {
                                mask |= ((*l > 0.0) as u64) << bit;
                            }
                        }
                    }
                    mask &= present[row.input];
                    if mask == 0 && t > 0 {
                        let likely = (0..bases)
                            .max_by(|a, b| ml[*a].total_cmp(&ml[*b]).then(b.cmp(a)))
                            .unwrap_or(0);
                        mask = 1 << likely;
                    }
                    mask
                };
                let (index, next, mask) = match &mut row.draw {
                    Draw::Beam => {
                        let mask = char_mask(None);
                        char_masks[i] = mask;
                        let log_probs = &mut log_probs[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE];
                        log_softmax(tok, log_probs);
                        let char_log_prob = &mut log_probs[CHAR as usize];
                        if char_log_prob.is_finite() {
                            for (bit, l) in ml[..bases].iter().enumerate() {
                                let l = *l as f64;
                                *char_log_prob +=
                                    log_sigmoid(if mask >> bit & 1 == 1 { l } else { -l });
                            }
                        }
                        continue;
                    }
                    Draw::Greedy(index) => {
                        let next = argmax(tok) as u32;
                        (*index, next, if next == CHAR { char_mask(None) } else { 0 })
                    }
                    Draw::Sample(index, rng) => {
                        let next = sample(tok, temperature, rng, &mut weights) as u32;
                        let mask = if next == CHAR {
                            char_mask(Some(rng))
                        } else {
                            0
                        };
                        (*index, next, mask)
                    }
                };
                if next == EOS {
                    ends[row.input][index] = row.node;
                } else {
                    row.grammar.step(next);
                    row.node = tree.push(row.node, next, mask);
                    keep[i] = true;
                }
            }

            // The next rows: the drawn ones still decoding, then the beams'.
            let mut parents: Vec<u32> = Vec::with_capacity(rows.len());
            let mut children: Vec<Row> = vec![];
            let continuing = keep.iter().filter(|keep| **keep).count();
            if let Some(width) = beam {
                for hypotheses in &mut beams {
                    // The most likely of the finished hypotheses and of the
                    // extensions of the others; ties keep the earlier, as the
                    // PyTorch model's top-k over (rank, token).
                    expansions.clear();
                    for (rank, hypothesis) in hypotheses.iter().enumerate() {
                        match hypothesis.row {
                            None => expansions.push(Expansion {
                                score: hypothesis.score,
                                rank,
                                token: EOS,
                            }),
                            Some(row) => {
                                let row = row as usize;
                                let log_probs =
                                    &log_probs[row * VOCAB_SIZE..(row + 1) * VOCAB_SIZE];
                                expansions.extend(
                                    (log_probs.iter().enumerate())
                                        .filter(|(_, log_prob)| log_prob.is_finite())
                                        .map(|(token, log_prob)| Expansion {
                                            score: hypothesis.score + log_prob,
                                            rank,
                                            token: token as u32,
                                        }),
                                );
                            }
                        }
                    }
                    let order = |a: &Expansion, b: &Expansion| {
                        (b.score.total_cmp(&a.score))
                            .then(a.rank.cmp(&b.rank))
                            .then(a.token.cmp(&b.token))
                    };
                    if expansions.len() > width {
                        expansions.select_nth_unstable_by(width - 1, order);
                        expansions.truncate(width);
                    }
                    expansions.sort_unstable_by(order);

                    let previous = std::mem::take(hypotheses);
                    for expansion in &expansions {
                        let parent = &previous[expansion.rank];
                        let Some(parent_row) = parent.row else {
                            hypotheses.push(parent.clone());
                            continue;
                        };
                        let row = &rows[parent_row as usize];
                        if expansion.token == EOS {
                            hypotheses.push(Hypothesis {
                                score: expansion.score,
                                node: row.node,
                                row: None,
                            });
                            continue;
                        }
                        let mask = if expansion.token == CHAR {
                            char_masks[parent_row as usize]
                        } else {
                            0
                        };
                        let node = tree.push(row.node, expansion.token, mask);
                        hypotheses.push(Hypothesis {
                            score: expansion.score,
                            node,
                            row: Some((continuing + children.len()) as u32),
                        });
                        let mut grammar = row.grammar.clone();
                        grammar.step(expansion.token);
                        children.push(Row {
                            input: row.input,
                            draw: Draw::Beam,
                            grammar,
                            node,
                        });
                        parents.push(parent_row);
                    }
                }
            }
            // The drawn rows' parents go first.
            let beam_parents = std::mem::take(&mut parents);
            parents.extend((0..rows.len() as u32).filter(|i| keep[*i as usize]));
            parents.extend(beam_parents);
            let mut kept = keep.iter();
            rows.retain(|_| *kept.next().unwrap());
            rows.extend(children);
            if rows.is_empty() {
                break;
            }
            cache.reorder(&parents)?;
        }
        // Grammar-constrained sequences end within `max_len`; any row left
        // is cut there.
        for row in &rows {
            match row.draw {
                Draw::Greedy(index) | Draw::Sample(index, _) => ends[row.input][index] = row.node,
                Draw::Beam => {}
            }
        }

        let copies = if greedy_only { count - first } else { 1 };
        Ok(beams
            .into_iter()
            .zip(ends)
            .map(|(hypotheses, ends)| {
                let mut candidates = Vec::with_capacity(hypotheses.len() + count);
                candidates.extend(
                    hypotheses
                        .iter()
                        .map(|hypothesis| tree.sequence(hypothesis.node)),
                );
                for end in &ends[first..] {
                    let candidate = tree.sequence(*end);
                    candidates.extend(std::iter::repeat_n(candidate, copies));
                }
                candidates
            })
            .collect())
    }
}

/// How a row draws its tokens; drawn candidates have an index (0, the
/// greedy decoding, then the samples).
enum Draw {
    Greedy(usize),
    Sample(usize, Rng),
    Beam,
}

/// A sequence being decoded.
struct Row {
    /// Its DFA.
    input: usize,
    draw: Draw,
    grammar: PrefixGrammar,
    /// Its last token, in the [`Tree`].
    node: u32,
}

impl Row {
    fn new(input: usize, draw: Draw, max_len: usize) -> Self {
        Self {
            input,
            draw,
            grammar: PrefixGrammar::new(max_len),
            node: Tree::ROOT,
        }
    }
}

/// A hypothesis of a beam: its score, its last token in the [`Tree`], and
/// its row while it is decoded.
#[derive(Clone, Debug)]
struct Hypothesis {
    score: f64,
    node: u32,
    row: Option<u32>,
}

/// A way to extend the hypothesis of rank `rank` of a beam (`EOS` for a
/// finished one, at no cost).
struct Expansion {
    score: f64,
    rank: usize,
    token: u32,
}

/// The tokens of all sequences, each with its base mask and its parent:
/// sequences share their prefixes.
struct Tree(Vec<(u32, u64, u32)>);

impl Tree {
    /// BOS, the first token of every sequence.
    const ROOT: u32 = 0;

    fn new() -> Self {
        Self(vec![(BOS, 0, u32::MAX)])
    }

    fn push(&mut self, parent: u32, token: u32, mask: u64) -> u32 {
        self.0.push((token, mask, parent));
        (self.0.len() - 1) as u32
    }

    fn token(&self, node: u32) -> u32 {
        self.0[node as usize].0
    }

    fn mask(&self, node: u32) -> u64 {
        self.0[node as usize].1
    }

    /// The sequence ending at `node`, BOS left out.
    fn sequence(&self, mut node: u32) -> Candidate {
        let (mut tokens, mut masks) = (vec![], vec![]);
        while node != Self::ROOT {
            let (token, mask, parent) = self.0[node as usize];
            tokens.push(token);
            masks.push(mask);
            node = parent;
        }
        tokens.reverse();
        masks.reverse();
        (tokens, masks)
    }
}

/// `log_probs` the log-softmax of `logits` (-∞ for -∞), in `f64`.
fn log_softmax(logits: &[f32], log_probs: &mut [f64]) {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, l| m.max(*l)) as f64;
    if max == f64::NEG_INFINITY {
        log_probs.fill(f64::NEG_INFINITY);
        return;
    }
    let sum: f64 = logits.iter().map(|l| (*l as f64 - max).exp()).sum();
    let log_sum = max + sum.ln();
    for (log_prob, l) in log_probs.iter_mut().zip(logits) {
        *log_prob = *l as f64 - log_sum;
    }
}

/// `ln(sigmoid(x))`, without overflow.
fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Index of the largest value, the first on ties.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// Samples an index from the softmax of `logits / temperature`, computed in
/// `f64` in `weights`.
fn sample(logits: &[f32], temperature: f64, rng: &mut Rng, weights: &mut Vec<f64>) -> usize {
    let max = logits
        .iter()
        .map(|l| *l as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    weights.clear();
    weights.extend(
        logits
            .iter()
            .map(|l| ((*l as f64 - max) / temperature).exp()),
    );
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
