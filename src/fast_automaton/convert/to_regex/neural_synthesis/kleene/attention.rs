//! Attention computed in one pass per row and head on the CPU: the scores,
//! their softmax and the weighted sum of the values, with no intermediate
//! tensor.
//!
//! - [`AttendOne`]: the decoder's attention of one query per row over cached
//!   keys and values. Built from tensor operations instead
//!   ([`attend_one_unfused`], for the other devices), each step writes a
//!   `[rows, heads, keys, d / heads]` temporary, which dominates decoding.
//! - [`PastCpu`]: the decoder's self-attention cache, written in place.
//! - [`GraphAttention`]: the encoder's global attention, each node over the
//!   nodes of its own graph ([`graph_attention_padded`] for the other
//!   devices).

use candle_core::{
    CpuStorage, CustomOp1, CustomOp3, D, Device, Layout, Result, Shape, Storage, Tensor, bail,
};

/// Below this many multiply-adds, an attention runs on one thread: handing
/// it to the pool costs more.
pub(super) const PARALLEL_WORK: usize = 1 << 16;

/// A strided `f32` view of rows, heads and positions, each run of `d /
/// heads` values contiguous.
#[derive(Clone, Copy)]
pub(super) struct View<'a> {
    data: &'a [f32],
    offset: usize,
    /// The strides of a row, a head and a position.
    stride: [usize; 3],
}

impl<'a> View<'a> {
    /// The view of `[rows, heads, positions, hd]` with unit stride on its
    /// last dimension.
    fn of_4d(storage: &'a CpuStorage, layout: &Layout) -> Result<Self> {
        let &[row, head, position, 1] = layout.stride() else {
            bail!("attention: the last dimension must be contiguous");
        };
        Ok(Self {
            data: storage.as_slice::<f32>()?,
            offset: layout.start_offset(),
            stride: [row, head, position],
        })
    }

    /// The view of `[rows, columns]` with unit stride on its last dimension
    /// as `[rows, heads, 1, hd]` from its first column; as keys, a position
    /// is the next row.
    fn of_rows(data: &'a [f32], layout: &Layout, hd: usize) -> Result<Self> {
        let &[row, 1] = layout.stride() else {
            bail!("attention: the last dimension must be contiguous");
        };
        Ok(Self {
            data,
            offset: layout.start_offset(),
            stride: [row, hd, row],
        })
    }

    /// The same view `columns` further.
    fn shifted(self, columns: usize) -> Self {
        Self {
            offset: self.offset + columns,
            ..self
        }
    }

    /// The `hd` values of `row`, `head` and `position`.
    fn at(&self, row: usize, head: usize, position: usize, hd: usize) -> &'a [f32] {
        let start =
            self.offset + row * self.stride[0] + head * self.stride[1] + position * self.stride[2];
        &self.data[start..start + hd]
    }
}

/// Writes in `out` `[rows, heads * hd]` the attention of each row `r` of `q`
/// over the first `length(r)` positions of row `key_row(r)` of `keys` and
/// `values`.
#[allow(clippy::too_many_arguments)]
pub(super) fn attend(
    out: &mut [f32],
    heads: usize,
    hd: usize,
    q: View,
    keys: View,
    values: View,
    key_row: impl Fn(usize) -> usize + Sync,
    length: impl Fn(usize) -> usize + Sync,
) {
    let d = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let attend_row = |row: usize, out: &mut [f32], scores: &mut Vec<f32>| {
        let (key_row, length) = (key_row(row), length(row));
        scores.resize(length, 0.0);
        for head in 0..heads {
            let query = q.at(row, head, 0, hd);
            let mut max = f32::NEG_INFINITY;
            for (position, score) in scores.iter_mut().enumerate() {
                *score = dot(query, keys.at(key_row, head, position, hd)) * scale;
                max = max.max(*score);
            }
            let mut sum = 0.0;
            for score in scores.iter_mut() {
                *score = (*score - max).exp();
                sum += *score;
            }
            let inverse = 1.0 / sum;
            let out = &mut out[head * hd..(head + 1) * hd];
            for (position, score) in scores.iter().enumerate() {
                axpy(out, score * inverse, values.at(key_row, head, position, hd));
            }
        }
    };

    let rows = out.len() / d;
    let work: usize = (0..rows).map(&length).sum::<usize>() * d;
    #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
    if work >= PARALLEL_WORK {
        use rayon::prelude::*;
        out.par_chunks_mut(d)
            .enumerate()
            .for_each_init(Vec::new, |scores, (row, out)| attend_row(row, out, scores));
        return;
    }
    let _ = (work, PARALLEL_WORK);
    let mut scores = vec![];
    for (row, out) in out.chunks_mut(d).enumerate() {
        attend_row(row, out, &mut scores);
    }
}

/// `Σ a b`, in 8 lanes so that it vectorizes.
pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let ((a_chunks, a_tail), (b_chunks, b_tail)) =
        (a[..len].as_chunks::<8>(), b[..len].as_chunks::<8>());
    let tail: f32 = a_tail.iter().zip(b_tail).map(|(a, b)| a * b).sum();
    lanes(a_chunks, b_chunks).iter().sum::<f32>() + tail
}

/// [`dot`]'s lanes. On their own, without the tail: around it, the loop may
/// not vectorize.
#[inline(never)]
fn lanes(a: &[[f32; 8]], b: &[[f32; 8]]) -> [f32; 8] {
    let mut lanes = [0f32; 8];
    for (a, b) in a.iter().zip(b) {
        for i in 0..8 {
            lanes[i] += a[i] * b[i];
        }
    }
    lanes
}

/// `out += weight x`.
pub(super) fn axpy(out: &mut [f32], weight: f32, x: &[f32]) {
    for (out, x) in out.iter_mut().zip(x) {
        *out += weight * x;
    }
}

/// One query per row, `q` `[B, d]`, over `keys` and `values` `[K, heads, T,
/// d / heads]`: row `r` attends to the first `lengths[r]` keys (all without
/// `lengths`) of key row `key_rows[r]` (without `key_rows`, `r`, or 0 when
/// `K` is 1). Returns `[B, d]`, before the output projection. Every tensor
/// must be `f32` and have unit stride on its last dimension.
pub(super) struct AttendOne<'a> {
    pub(super) heads: usize,
    pub(super) lengths: Option<&'a [u32]>,
    pub(super) key_rows: Option<&'a [u32]>,
}

impl CustomOp3 for AttendOne<'_> {
    fn name(&self) -> &'static str {
        "attend-one"
    }

    fn cpu_fwd(
        &self,
        q: &CpuStorage,
        q_layout: &Layout,
        keys: &CpuStorage,
        keys_layout: &Layout,
        values: &CpuStorage,
        values_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (rows, d) = q_layout.shape().dims2()?;
        let (key_rows, heads, length, hd) = keys_layout.shape().dims4()?;
        let rows_ok = match self.key_rows {
            Some(map) => map.len() == rows && map.iter().all(|r| (*r as usize) < key_rows),
            None => key_rows == 1 || key_rows == rows,
        };
        if values_layout.shape() != keys_layout.shape()
            || heads != self.heads
            || heads * hd != d
            || !rows_ok
            || self.lengths.is_some_and(|lengths| lengths.len() != rows)
        {
            bail!(
                "attend-one: q {:?}, keys {:?}, values {:?}, {} heads",
                q_layout.shape(),
                keys_layout.shape(),
                values_layout.shape(),
                self.heads
            );
        }
        let q = View::of_rows(q.as_slice::<f32>()?, q_layout, hd)?;
        let keys = View::of_4d(keys, keys_layout)?;
        let values = View::of_4d(values, values_layout)?;

        let mut attended = vec![0f32; rows * d];
        attend(
            &mut attended,
            heads,
            hd,
            q,
            keys,
            values,
            |row| match self.key_rows {
                Some(map) => map[row] as usize,
                None if key_rows == 1 => 0,
                None => row,
            },
            |row| {
                self.lengths
                    .map_or(length, |lengths| (lengths[row] as usize).min(length))
            },
        );
        Ok((CpuStorage::F32(attended), Shape::from((rows, d))))
    }
}

/// [`AttendOne`] from tensor operations, `keys` where `key_mask` `[B, 1, T]`
/// is -∞ left out; `keys` and `values` are `[B or 1, heads, T, d / heads]`.
///
/// Multiply-and-sum rather than matmul: candle runs a batched matmul as one
/// gemm call per matrix, far slower for `B × heads` single rows.
pub(super) fn attend_one_unfused(
    q: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    key_mask: Option<&Tensor>,
    heads: usize,
) -> Result<Tensor> {
    let (rows, d) = q.dims2()?;
    let hd = d / heads;
    let q = q.reshape((rows, heads, 1, hd))?;
    let mut scores = (keys.broadcast_mul(&q)?.sum(D::Minus1)? / (hd as f64).sqrt())?;
    if let Some(key_mask) = key_mask {
        scores = scores.broadcast_add(key_mask)?;
    }
    let weights = candle_nn::ops::softmax_last_dim(&scores)?.unsqueeze(D::Minus1)?;
    values.broadcast_mul(&weights)?.sum(2)?.reshape((rows, d))
}

/// Runs `f` on the `f32` values and layout of `tensor`, on the CPU.
fn with_cpu_values<R>(tensor: &Tensor, f: impl FnOnce(&[f32], &Layout) -> Result<R>) -> Result<R> {
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cpu(storage) = &*storage else {
        bail!("not on the CPU");
    };
    f(storage.as_slice::<f32>()?, layout)
}

/// The decoder's self-attention keys and values on the CPU, written in place
/// at each step. A row that finished leaves its slot free, until fewer than
/// half the slots are in use and the buffers are compacted: the cost of
/// dropping rows is amortized. A row forked into several (beam search)
/// keeps its slot for its first copy; the others take free slots.
pub(super) struct PastCpu {
    heads: usize,
    hd: usize,
    /// Positions the buffers first have room for.
    initial_capacity: usize,
    /// Slots in each buffer.
    slots: usize,
    /// Each row's slot.
    slot_of_row: Vec<u32>,
    /// The slots of no row.
    free: Vec<u32>,
    layers: Vec<PastLayer>,
}

#[derive(Default)]
struct PastLayer {
    /// Positions each buffer has room for.
    capacity: usize,
    /// `[slots, heads, capacity, hd]`.
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl PastCpu {
    pub(super) fn new(
        layers: usize,
        rows: usize,
        heads: usize,
        hd: usize,
        initial_capacity: usize,
    ) -> Self {
        Self {
            heads,
            hd,
            initial_capacity,
            slots: rows,
            slot_of_row: (0..rows as u32).collect(),
            free: vec![],
            layers: (0..layers).map(|_| PastLayer::default()).collect(),
        }
    }

    /// Writes `layer`'s keys and values at position `t`, from `qkv` `[B, 3
    /// d]` (queries, keys, values), and returns the attention of the queries
    /// over the positions so far, `[B, d]`, before the output projection.
    pub(super) fn attend(&mut self, layer: usize, t: usize, qkv: &Tensor) -> Result<Tensor> {
        let (heads, hd) = (self.heads, self.hd);
        let d = heads * hd;
        let rows = self.slot_of_row.len();
        if qkv.dims2()? != (rows, 3 * d) {
            bail!("past: qkv {:?}, {rows} rows", qkv.shape());
        }
        let past = &mut self.layers[layer];
        if t >= past.capacity {
            let capacity = (2 * past.capacity).max(self.initial_capacity).max(t + 1);
            past.grow(self.slots * heads, hd, capacity);
        }
        let capacity = past.capacity;
        let attended = with_cpu_values(qkv, |data, layout| {
            let q = View::of_rows(data, layout, hd)?;
            let (k, v) = (q.shifted(d), q.shifted(2 * d));
            for (row, slot) in self.slot_of_row.iter().enumerate() {
                for head in 0..heads {
                    let at = ((*slot as usize * heads + head) * capacity + t) * hd;
                    past.keys[at..at + hd].copy_from_slice(k.at(row, head, 0, hd));
                    past.values[at..at + hd].copy_from_slice(v.at(row, head, 0, hd));
                }
            }
            let stride = [heads * capacity * hd, capacity * hd, hd];
            let view = |data| View {
                data,
                offset: 0,
                stride,
            };
            let mut attended = vec![0f32; rows * d];
            attend(
                &mut attended,
                heads,
                hd,
                q,
                view(&past.keys),
                view(&past.values),
                |row| self.slot_of_row[row] as usize,
                |_| t + 1,
            );
            Ok(attended)
        })?;
        Tensor::from_vec(attended, (rows, d), &Device::Cpu)
    }

    /// Row `i` becomes a copy of row `parents[i]` (indices into the current
    /// rows, in any order): rows are dropped, kept or forked. A row's first
    /// copy keeps its slot; each other copy gets a free slot (or a new one)
    /// and its keys and values so far.
    pub(super) fn reorder(&mut self, parents: &[u32]) {
        let old = std::mem::take(&mut self.slot_of_row);
        let mut copies = vec![0u32; old.len()];
        for parent in parents {
            copies[*parent as usize] += 1;
        }
        self.free.extend(
            old.iter()
                .zip(&copies)
                .filter(|(_, copies)| **copies == 0)
                .map(|(slot, _)| *slot),
        );
        let mut kept = vec![false; old.len()];
        self.slot_of_row = Vec::with_capacity(parents.len());
        for parent in parents {
            let (parent, from) = (*parent as usize, old[*parent as usize] as usize);
            if !std::mem::replace(&mut kept[parent], true) {
                self.slot_of_row.push(from as u32);
                continue;
            }
            let to = match self.free.pop() {
                Some(slot) => slot as usize,
                None => {
                    self.slots += 1;
                    for past in &mut self.layers {
                        let block = self.heads * past.capacity * self.hd;
                        past.keys.resize(self.slots * block, 0.0);
                        past.values.resize(self.slots * block, 0.0);
                    }
                    self.slots - 1
                }
            };
            for past in &mut self.layers {
                let block = self.heads * past.capacity * self.hd;
                past.keys
                    .copy_within(from * block..(from + 1) * block, to * block);
                past.values
                    .copy_within(from * block..(from + 1) * block, to * block);
            }
            self.slot_of_row.push(to as u32);
        }
        if 2 * self.slot_of_row.len() < self.slots {
            self.compact();
        }
    }

    /// Moves the rows to the first slots, in the order of their slots, so
    /// that each block moves towards the front, over blocks already moved or
    /// unused.
    fn compact(&mut self) {
        let mut by_slot: Vec<usize> = (0..self.slot_of_row.len()).collect();
        by_slot.sort_unstable_by_key(|row| self.slot_of_row[*row]);
        for past in &mut self.layers {
            let block = self.heads * past.capacity * self.hd;
            for buffer in [&mut past.keys, &mut past.values] {
                for (to, row) in by_slot.iter().enumerate() {
                    let from = self.slot_of_row[*row] as usize;
                    if from != to && block > 0 {
                        buffer.copy_within(from * block..(from + 1) * block, to * block);
                    }
                }
                buffer.truncate(by_slot.len() * block);
            }
        }
        for (to, row) in by_slot.into_iter().enumerate() {
            self.slot_of_row[row] = to as u32;
        }
        self.slots = self.slot_of_row.len();
        self.free.clear();
    }
}

impl PastLayer {
    /// Makes room for `capacity` positions in each of `blocks` (slot, head)
    /// blocks.
    fn grow(&mut self, blocks: usize, hd: usize, capacity: usize) {
        let (old, new) = (self.capacity * hd, capacity * hd);
        for buffer in [&mut self.keys, &mut self.values] {
            let mut grown = vec![0f32; blocks * new];
            if old > 0 {
                for (from, to) in buffer.chunks_exact(old).zip(grown.chunks_exact_mut(new)) {
                    to[..old].copy_from_slice(from);
                }
            }
            *buffer = grown;
        }
        self.capacity = capacity;
    }
}

/// The encoder's global attention of nodes, laid out as `qkv` `[N, 3 d]`
/// (queries, keys, values), each node `i` over the `lengths[i]` nodes from
/// `starts[i]`, those of its graph. Returns `[N, d]`, before the output
/// projection.
pub(super) struct GraphAttention<'a> {
    pub(super) heads: usize,
    pub(super) starts: &'a [u32],
    pub(super) lengths: &'a [u32],
}

impl CustomOp1 for GraphAttention<'_> {
    fn name(&self) -> &'static str {
        "graph-attention"
    }

    fn cpu_fwd(&self, qkv: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (n, columns) = layout.shape().dims2()?;
        let d = columns / 3;
        let hd = d / self.heads.max(1);
        if self.heads == 0
            || 3 * d != columns
            || self.heads * hd != d
            || self.starts.len() != n
            || self.lengths.len() != n
            || (self.starts.iter())
                .zip(self.lengths)
                .any(|(start, length)| (start + length) as usize > n)
        {
            bail!(
                "graph-attention: qkv {:?}, {} heads",
                layout.shape(),
                self.heads
            );
        }
        let q = View::of_rows(qkv.as_slice::<f32>()?, layout, hd)?;
        let mut attended = vec![0f32; n * d];
        attend(
            &mut attended,
            self.heads,
            hd,
            q,
            q.shifted(d),
            q.shifted(2 * d),
            |node| self.starts[node] as usize,
            |node| self.lengths[node] as usize,
        );
        Ok((CpuStorage::F32(attended), Shape::from((n, d))))
    }
}

/// [`GraphAttention`] from tensor operations: the graphs, `(first node,
/// nodes)`, padded to the largest, attend in one batch `[graphs, heads,
/// largest, largest]`.
pub(super) fn graph_attention_padded(
    qkv: &Tensor,
    graphs: &[(usize, usize)],
    heads: usize,
) -> Result<Tensor> {
    let (n, columns) = qkv.dims2()?;
    let (d, device) = (columns / 3, qkv.device());
    let hd = d / heads;
    let largest = graphs.iter().map(|(_, nodes)| *nodes).max().unwrap_or(0);
    let g = graphs.len();
    // Each graph's nodes in `largest` slots, the padding at a zero row
    // appended after the nodes.
    let mut slots = Vec::with_capacity(g * largest);
    let mut key_mask = Vec::with_capacity(g * largest);
    let mut real = Vec::with_capacity(n);
    for (i, (first, nodes)) in graphs.iter().enumerate() {
        for slot in 0..largest {
            let is_real = slot < *nodes;
            slots.push(if is_real { first + slot } else { n } as u32);
            key_mask.push(if is_real { 0f32 } else { f32::NEG_INFINITY });
            if is_real {
                real.push((i * largest + slot) as u32);
            }
        }
    }
    let padded = qkv
        .pad_with_zeros(0, 0, 1)?
        .index_select(&Tensor::new(slots.as_slice(), device)?, 0)?
        .reshape((g, largest, 3, heads, hd))?;
    // `[graphs, heads, largest, hd]` each.
    let part = |i: usize| -> Result<Tensor> {
        padded
            .narrow(2, i, 1)?
            .squeeze(2)?
            .transpose(1, 2)?
            .contiguous()
    };
    let (q, k, v) = (part(0)?, part(1)?, part(2)?);
    let key_mask = Tensor::from_vec(key_mask, (g, 1, 1, largest), device)?;
    let scores = (q.matmul(&k.t()?)? / (hd as f64).sqrt())?.broadcast_add(&key_mask)?;
    candle_nn::ops::softmax_last_dim(&scores)?
        .matmul(&v)?
        .transpose(1, 2)?
        .reshape((g * largest, d))?
        .index_select(&Tensor::new(real.as_slice(), device)?, 0)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in super::super) fn random(shape: &[usize]) -> Tensor {
        Tensor::randn(0f32, 1f32, shape, &Device::Cpu).unwrap()
    }

    pub(in super::super) fn assert_close(a: &Tensor, b: &Tensor) {
        let difference = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(difference < 1e-5, "difference {difference}");
    }

    fn mask(lengths: &[u32], keys: usize) -> Tensor {
        let key_mask: Vec<f32> = lengths
            .iter()
            .flat_map(|length| {
                (0..keys).map(move |key| {
                    if key < *length as usize {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        Tensor::from_vec(key_mask, (lengths.len(), 1, keys), &Device::Cpu).unwrap()
    }

    #[test]
    fn matches_the_tensor_operations() {
        let (rows, heads, hd) = (5, 4, 8);
        let q = random(&[rows, heads * hd]);
        let attend_one = |keys, values, lengths, key_rows| {
            q.apply_op3_no_bwd(
                keys,
                values,
                &AttendOne {
                    heads,
                    lengths,
                    key_rows,
                },
            )
            .unwrap()
        };

        // Per-row keys, read through a narrowed buffer.
        let keys = random(&[rows, heads, 12, hd]).narrow(2, 0, 7).unwrap();
        let values = random(&[rows, heads, 12, hd]).narrow(2, 0, 7).unwrap();
        let fused = attend_one(&keys, &values, None, None);
        let unfused = attend_one_unfused(&q, &keys, &values, None, heads).unwrap();
        assert_close(&fused, &unfused);

        // Keys shared by every row, with each row's own number of them.
        let keys = random(&[1, heads, 6, hd]);
        let values = random(&[1, heads, 6, hd]);
        let lengths = [6, 2, 5, 1, 3];
        let fused = attend_one(&keys, &values, Some(&lengths), None);
        let unfused =
            attend_one_unfused(&q, &keys, &values, Some(&mask(&lengths, 6)), heads).unwrap();
        assert_close(&fused, &unfused);

        // Rows reading the keys of their DFA, as the cross-attention over
        // several DFAs does.
        let keys = random(&[3, heads, 6, hd]);
        let values = random(&[3, heads, 6, hd]);
        let key_rows = [2, 0, 0, 1, 2];
        let fused = attend_one(&keys, &values, Some(&lengths), Some(&key_rows));
        let expand = &Tensor::new(&key_rows, &Device::Cpu).unwrap();
        let unfused = attend_one_unfused(
            &q,
            &keys.index_select(expand, 0).unwrap(),
            &values.index_select(expand, 0).unwrap(),
            Some(&mask(&lengths, 6)),
            heads,
        )
        .unwrap();
        assert_close(&fused, &unfused);
    }

    #[test]
    fn rejects_mismatched_shapes() {
        let q = random(&[3, 32]);
        let keys = random(&[2, 4, 5, 8]);
        let attend = |key_rows| AttendOne {
            heads: 4,
            lengths: None,
            key_rows,
        };
        assert!(q.apply_op3_no_bwd(&keys, &keys, &attend(None)).is_err());
        assert!(
            q.apply_op3_no_bwd(&keys, &keys, &attend(Some(&[0, 1, 2])))
                .is_err()
        );
        assert!(
            q.apply_op3_no_bwd(&keys, &keys, &attend(Some(&[0, 1, 1])))
                .is_ok()
        );
    }

    /// The in-place cache against tensors of all the keys and values so far,
    /// through growths and compactions.
    #[test]
    fn past_matches_the_tensor_operations() {
        let (rows, heads, hd, layers) = (6, 2, 4, 2);
        let d = heads * hd;
        let mut past = PastCpu::new(layers, rows, heads, hd, 2);
        // Per layer, per original row: the keys and values so far.
        let mut history: Vec<Vec<(Vec<Tensor>, Vec<Tensor>)>> =
            vec![vec![(vec![], vec![]); rows]; layers];
        let mut alive: Vec<usize> = (0..rows).collect();
        for t in 0..9 {
            for (layer, history) in history.iter_mut().enumerate() {
                let qkv = random(&[alive.len(), 3 * d]);
                let fused = past.attend(layer, t, &qkv).unwrap();
                for (i, row) in alive.iter().enumerate() {
                    let split = |part: usize| {
                        qkv.narrow(0, i, 1)
                            .unwrap()
                            .narrow(1, part * d, d)
                            .unwrap()
                            .reshape((1, heads, 1, hd))
                            .unwrap()
                    };
                    history[*row].0.push(split(1));
                    history[*row].1.push(split(2));
                    let keys = Tensor::cat(&history[*row].0, 2).unwrap();
                    let values = Tensor::cat(&history[*row].1, 2).unwrap();
                    let q = qkv.narrow(0, i, 1).unwrap().narrow(1, 0, d).unwrap();
                    let unfused = attend_one_unfused(&q, &keys, &values, None, heads).unwrap();
                    assert_close(&fused.narrow(0, i, 1).unwrap(), &unfused);
                }
            }
            // Fork, reorder and drop rows (8, 5, 6, 4, 4, 6, 4, 3, 2 rows):
            // a forked row's copy gets the history so far, rows that finished
            // free their slots, and the last drops compact.
            let parents: Vec<u32> = match t {
                0 => vec![0, 0, 1, 2, 3, 4, 5, 5],
                2 => vec![4, 3, 3, 3, 0, 1],
                4 => vec![3, 1, 0, 2],
                5 => vec![3, 3, 1, 0, 2, 2],
                _ => (0..alive.len() as u32).filter(|i| i % 3 != 1).collect(),
            };
            for history in &mut history {
                *history = (parents.iter())
                    .map(|parent| history[alive[*parent as usize]].clone())
                    .collect();
            }
            alive = (0..parents.len()).collect();
            past.reorder(&parents);
        }
        assert!(past.slots < rows);
    }

    #[test]
    fn graph_attention_matches_the_tensor_operations() {
        let (heads, hd) = (4, 8);
        let d = heads * hd;
        let graphs = [(0, 3), (3, 5), (8, 1), (9, 4)];
        let n = 13;
        // Read through a narrowed buffer, as the encoder's layout may be.
        let qkv = random(&[n, 3 * d + 5]).narrow(1, 2, 3 * d).unwrap();
        let (mut starts, mut lengths) = (vec![], vec![]);
        for (first, nodes) in graphs {
            starts.extend(std::iter::repeat_n(first as u32, nodes));
            lengths.extend(std::iter::repeat_n(nodes as u32, nodes));
        }
        let fused = qkv
            .apply_op1_no_bwd(&GraphAttention {
                heads,
                starts: &starts,
                lengths: &lengths,
            })
            .unwrap();
        let padded = graph_attention_padded(&qkv.contiguous().unwrap(), &graphs, heads).unwrap();
        assert_close(&fused, &padded);

        // Each graph on its own, without padding.
        for (first, nodes) in graphs {
            let qkv = qkv.narrow(0, first, nodes).unwrap().contiguous().unwrap();
            let alone = graph_attention_padded(&qkv, &[(0, nodes)], heads).unwrap();
            assert_close(&fused.narrow(0, first, nodes).unwrap(), &alone);
        }
    }
}
