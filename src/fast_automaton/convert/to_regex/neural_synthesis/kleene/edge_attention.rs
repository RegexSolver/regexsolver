//! The attention of PyG's `TransformerConv` over the edges into each node.
//!
//! On the CPU, [`EdgeAttention`] computes it in one pass per node and head
//! over the node's incoming edges: the scores, their softmax and the weighted
//! sum of the messages, with no `[edges, d]` temporary. Built from tensor
//! operations instead ([`edge_attention_unfused`], for the other devices),
//! it gathers per-edge queries, keys and values and scatter-adds the
//! messages.

use candle_core::{CpuStorage, CustomOp2, D, Layout, Result, Shape, Tensor, bail};

use super::attention::axpy;

/// Added to the softmax denominators, as PyG does.
const SOFTMAX_EPS: f64 = 1e-16;

/// The edges into each node: those of node `i` are at `offsets[i]..
/// offsets[i + 1]`, in their order among all the edges.
pub(super) struct Incoming {
    offsets: Vec<u32>,
    /// The edges' rows of the edge features.
    features: Vec<u32>,
    /// The edges' sources, in the same order.
    sources: Vec<u32>,
}

impl Incoming {
    /// The edges `src[e]` → `dst[e]`, whose features are row `features[e]`.
    pub(super) fn new(nodes: usize, src: &[u32], dst: &[u32], features: &[u32]) -> Self {
        let mut offsets = vec![0u32; nodes + 1];
        for node in dst {
            offsets[*node as usize + 1] += 1;
        }
        for i in 0..nodes {
            offsets[i + 1] += offsets[i];
        }
        let mut next = offsets.clone();
        let (mut sorted, mut sources) = (vec![0; dst.len()], vec![0; dst.len()]);
        for ((src, dst), feature) in src.iter().zip(dst).zip(features) {
            let at = &mut next[*dst as usize];
            sorted[*at as usize] = *feature;
            sources[*at as usize] = *src;
            *at += 1;
        }
        Self {
            offsets,
            features: sorted,
            sources,
        }
    }

    fn nodes(&self) -> usize {
        self.offsets.len() - 1
    }
}

/// The attention of each node over the edges into it: `qkv` `[N, 3 d]` holds
/// the nodes' queries, keys and values, `edge` `[F, d]` projected edge
/// features, each edge's row added to the keys and values of its source.
/// Returns `[N, d]`, zero for the nodes no edge goes into. Both tensors must
/// be `f32` and have unit stride on their last dimension.
pub(super) struct EdgeAttention<'a> {
    pub(super) heads: usize,
    pub(super) incoming: &'a Incoming,
}

impl CustomOp2 for EdgeAttention<'_> {
    fn name(&self) -> &'static str {
        "edge-attention"
    }

    fn cpu_fwd(
        &self,
        qkv: &CpuStorage,
        qkv_layout: &Layout,
        edge: &CpuStorage,
        edge_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (n, columns) = qkv_layout.shape().dims2()?;
        let (features, d) = edge_layout.shape().dims2()?;
        let incoming = self.incoming;
        let (&[qkv_row, 1], &[edge_row, 1]) = (qkv_layout.stride(), edge_layout.stride()) else {
            bail!("edge-attention: the last dimensions must be contiguous");
        };
        if self.heads == 0
            || columns != 3 * d
            || d % self.heads != 0
            || incoming.nodes() != n
            || incoming.sources.iter().any(|src| *src as usize >= n)
            || incoming
                .features
                .iter()
                .any(|row| *row as usize >= features)
        {
            bail!(
                "edge-attention: qkv {:?}, edge {:?}, {} heads, {} nodes",
                qkv_layout.shape(),
                edge_layout.shape(),
                self.heads,
                incoming.nodes(),
            );
        }
        let (qkv, edge) = (qkv.as_slice::<f32>()?, edge.as_slice::<f32>()?);
        let (qkv_start, edge_start) = (qkv_layout.start_offset(), edge_layout.start_offset());
        let (heads, c) = (self.heads, d / self.heads);
        let scale = 1.0 / (c as f32).sqrt();
        // Node `node`'s query (part 0), key (1) or value (2) of `head`.
        let node_at = |node: usize, part: usize, head: usize| {
            let at = qkv_start + node * qkv_row + part * d + head * c;
            &qkv[at..at + c]
        };
        let edge_at = |row: u32, head: usize| {
            let at = edge_start + row as usize * edge_row + head * c;
            &edge[at..at + c]
        };

        let attend_node = |node: usize, out: &mut [f32], scores: &mut Vec<f32>| {
            let range = incoming.offsets[node] as usize..incoming.offsets[node + 1] as usize;
            let (rows, sources) = (&incoming.features[range.clone()], &incoming.sources[range]);
            scores.resize(rows.len(), 0.0);
            for head in 0..heads {
                let query = node_at(node, 0, head);
                let mut max = f32::NEG_INFINITY;
                for ((score, row), src) in scores.iter_mut().zip(rows).zip(sources) {
                    let key = node_at(*src as usize, 1, head);
                    *score = dot_sum(query, key, edge_at(*row, head)) * scale;
                    max = max.max(*score);
                }
                let mut sum = 0.0;
                for score in scores.iter_mut() {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                let inverse = 1.0 / (sum + SOFTMAX_EPS as f32);
                let out = &mut out[head * c..(head + 1) * c];
                for ((score, row), src) in scores.iter().zip(rows).zip(sources) {
                    let weight = score * inverse;
                    axpy(out, weight, node_at(*src as usize, 2, head));
                    axpy(out, weight, edge_at(*row, head));
                }
            }
        };

        let mut attended = vec![0f32; n * d];
        #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
        if incoming.sources.len() * d >= super::attention::PARALLEL_WORK {
            use rayon::prelude::*;
            attended
                .par_chunks_mut(d)
                .enumerate()
                .for_each_init(Vec::new, |scores, (node, out)| {
                    attend_node(node, out, scores)
                });
            return Ok((CpuStorage::F32(attended), Shape::from((n, d))));
        }
        let mut scores = vec![];
        for (node, out) in attended.chunks_mut(d).enumerate() {
            attend_node(node, out, &mut scores);
        }
        Ok((CpuStorage::F32(attended), Shape::from((n, d))))
    }
}

/// `Σ q (k + e)`, in 8 lanes so that it vectorizes.
fn dot_sum(q: &[f32], k: &[f32], e: &[f32]) -> f32 {
    let len = q.len().min(k.len()).min(e.len());
    let (q_chunks, q_tail) = q[..len].as_chunks::<8>();
    let (k_chunks, k_tail) = k[..len].as_chunks::<8>();
    let (e_chunks, e_tail) = e[..len].as_chunks::<8>();
    let mut lanes = [0f32; 8];
    for ((q, k), e) in q_chunks.iter().zip(k_chunks).zip(e_chunks) {
        for i in 0..8 {
            lanes[i] += q[i] * (k[i] + e[i]);
        }
    }
    let tail: f32 = (q_tail.iter().zip(k_tail).zip(e_tail))
        .map(|((q, k), e)| q * (k + e))
        .sum();
    lanes.iter().sum::<f32>() + tail
}

/// [`EdgeAttention`] from tensor operations, over the edges `src` → `dst`
/// (`dst_nodes` on the host, for the softmax's per-node maxima), `edge` `[E,
/// d]` each edge's features.
pub(super) fn edge_attention_unfused(
    qkv: &Tensor,
    edge: &Tensor,
    src: &Tensor,
    dst: &Tensor,
    dst_nodes: &[u32],
    heads: usize,
) -> Result<Tensor> {
    let (n, columns) = qkv.dims2()?;
    let (e, d) = edge.dims2()?;
    let c = d / heads;
    if columns != 3 * d {
        bail!(
            "edge-attention: qkv {:?}, edge {:?}",
            qkv.shape(),
            edge.shape()
        );
    }
    let part = |i: usize| qkv.narrow(1, i * d, d)?.contiguous();
    let query = part(0)?.index_select(dst, 0)?;
    let key = (part(1)?.index_select(src, 0)? + edge)?;
    let value = (part(2)?.index_select(src, 0)? + edge)?;
    // [E, H]
    let alpha = ((query * key)?.reshape((e, heads, c))?.sum(D::Minus1)? / (c as f64).sqrt())?;
    // Softmax over the edges into each node.
    let max = incoming_max(&alpha, dst_nodes, n)?;
    let exp = (alpha - max)?.exp()?;
    let sum = Tensor::zeros((n, heads), exp.dtype(), exp.device())?.index_add(dst, &exp, 0)?;
    let sum = (sum + SOFTMAX_EPS)?.index_select(dst, 0)?;
    let alpha = (exp / sum)?;
    let messages = value
        .reshape((e, heads, c))?
        .broadcast_mul(&alpha.unsqueeze(D::Minus1)?)?
        .reshape((e, d))?;
    Tensor::zeros((n, d), messages.dtype(), messages.device())?.index_add(dst, &messages, 0)
}

/// For each edge, the largest of `scores` `[E, H]` over the edges into the
/// same node, per head.
fn incoming_max(scores: &Tensor, dst_nodes: &[u32], nodes: usize) -> Result<Tensor> {
    let (e, h) = scores.dims2()?;
    let mut max = vec![f32::NEG_INFINITY; nodes * h];
    for (row, node) in scores.to_vec2::<f32>()?.iter().zip(dst_nodes) {
        let node = *node as usize;
        for (max, score) in max[node * h..(node + 1) * h].iter_mut().zip(row) {
            *max = max.max(*score);
        }
    }
    let per_edge: Vec<f32> = dst_nodes
        .iter()
        .flat_map(|node| {
            max[*node as usize * h..(*node as usize + 1) * h]
                .iter()
                .copied()
        })
        .collect();
    Tensor::from_vec(per_edge, (e, h), scores.device())
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::super::attention::tests::{assert_close, random};
    use super::*;

    #[test]
    fn matches_the_tensor_operations() {
        let (n, heads, c) = (7, 4, 8);
        let d = heads * c;
        // Node 5 has no incoming edge, node 0 several, one a self-loop.
        let src: Vec<u32> = vec![1, 2, 0, 3, 4, 6, 0, 2, 1, 3, 6];
        let dst: Vec<u32> = vec![0, 0, 0, 1, 2, 2, 3, 4, 6, 6, 6];
        let ids = |v: &[u32]| Tensor::new(v, &Device::Cpu).unwrap();
        // Read through narrowed buffers, as the encoder's layout may be.
        let qkv = random(&[n, 4 * d]).narrow(1, d, 3 * d).unwrap();
        // Three edge types, their features in a table.
        let table = random(&[3, d]);
        let types: Vec<u32> = vec![0, 2, 1, 1, 0, 2, 2, 0, 1, 0, 2];
        let incoming = Incoming::new(n, &src, &dst, &types);
        let fused = qkv
            .apply_op2_no_bwd(
                &table,
                &EdgeAttention {
                    heads,
                    incoming: &incoming,
                },
            )
            .unwrap();
        let edge = table.index_select(&ids(&types), 0).unwrap();
        let unfused =
            edge_attention_unfused(&qkv, &edge, &ids(&src), &ids(&dst), &dst, heads).unwrap();
        assert_close(&fused, &unfused);
        assert_eq!(
            fused
                .get(5)
                .unwrap()
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn rejects_mismatched_shapes() {
        let incoming = Incoming::new(3, &[0, 1], &[1, 2], &[0, 1]);
        let attention = EdgeAttention {
            heads: 4,
            incoming: &incoming,
        };
        assert!(
            random(&[3, 96])
                .apply_op2_no_bwd(&random(&[2, 32]), &attention)
                .is_ok()
        );
        assert!(
            random(&[3, 96])
                .apply_op2_no_bwd(&random(&[1, 32]), &attention)
                .is_err()
        );
        assert!(
            random(&[3, 96])
                .apply_op2_no_bwd(&random(&[2, 16]), &attention)
                .is_err()
        );
        assert!(
            random(&[4, 96])
                .apply_op2_no_bwd(&random(&[2, 32]), &attention)
                .is_err()
        );
    }
}
