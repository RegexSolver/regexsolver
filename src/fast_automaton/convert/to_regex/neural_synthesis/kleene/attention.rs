//! The decoder's attention of one query per row over cached keys and values.
//!
//! On the CPU, [`AttendOne`] computes it in one pass per row and head: the
//! scores, their softmax and the weighted sum of the values, with no
//! intermediate tensor. Built from tensor operations instead
//! ([`attend_one_unfused`], for the other devices), each step writes a
//! `[rows, heads, keys, d / heads]` temporary, which dominates decoding.

use candle_core::{CpuStorage, CustomOp3, D, Layout, Result, Shape, Tensor, bail};

/// One query per row, `q` `[B, d]`, over `keys` and `values` `[B or 1,
/// heads, T, d / heads]`, each row attending to its first `lengths[row]` keys (all
/// without `lengths`): `[B, d]`, before the output projection. Every tensor
/// must be `f32` and have unit stride on its last dimension.
pub(super) struct AttendOne<'a> {
    pub(super) heads: usize,
    pub(super) lengths: Option<&'a [u32]>,
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
        if values_layout.shape() != keys_layout.shape()
            || heads != self.heads
            || heads * hd != d
            || (key_rows != 1 && key_rows != rows)
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
        let layouts = [q_layout, keys_layout, values_layout];
        if layouts
            .iter()
            .any(|layout| layout.stride().last() != Some(&1))
        {
            bail!("attend-one: the last dimensions must be contiguous");
        }

        let q = Strided::new(q, q_layout)?;
        let keys = Strided::new(keys, keys_layout)?;
        let values = Strided::new(values, values_layout)?;
        let sqrt_hd = (hd as f32).sqrt();

        let attend_row = |row: usize, out: &mut [f32]| {
            let key_row = if key_rows == 1 { 0 } else { row };
            let length = self
                .lengths
                .map_or(length, |lengths| (lengths[row] as usize).min(length));
            let mut scores = vec![0f32; length];
            for head in 0..heads {
                let query = q.slice(&[row, head * hd], hd);
                let mut max = f32::NEG_INFINITY;
                for (position, score) in scores.iter_mut().enumerate() {
                    let key = keys.slice(&[key_row, head, position, 0], hd);
                    *score = dot(query, key) / sqrt_hd;
                    max = max.max(*score);
                }
                let mut sum = 0.0;
                for score in &mut scores {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                let out = &mut out[head * hd..(head + 1) * hd];
                for (position, score) in scores.iter().enumerate() {
                    let weight = score / sum;
                    let value = values.slice(&[key_row, head, position, 0], hd);
                    for (out, value) in out.iter_mut().zip(value) {
                        *out += value * weight;
                    }
                }
            }
        };

        let mut attended = vec![0f32; rows * d];
        #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
        {
            use rayon::prelude::*;
            attended
                .par_chunks_mut(d)
                .enumerate()
                .for_each(|(row, out)| attend_row(row, out));
        }
        #[cfg(any(not(feature = "parallel"), target_family = "wasm"))]
        for (row, out) in attended.chunks_mut(d).enumerate() {
            attend_row(row, out);
        }
        Ok((CpuStorage::F32(attended), Shape::from((rows, d))))
    }
}

/// A strided view of an `f32` storage.
struct Strided<'a> {
    data: &'a [f32],
    offset: usize,
    stride: Vec<usize>,
}

impl<'a> Strided<'a> {
    fn new(storage: &'a CpuStorage, layout: &Layout) -> Result<Self> {
        Ok(Self {
            data: storage.as_slice::<f32>()?,
            offset: layout.start_offset(),
            stride: layout.stride().to_vec(),
        })
    }

    /// The `len` contiguous values from the element at `index`.
    fn slice(&self, index: &[usize], len: usize) -> &[f32] {
        let start = self.offset
            + index
                .iter()
                .zip(&self.stride)
                .map(|(i, stride)| i * stride)
                .sum::<usize>();
        &self.data[start..start + len]
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// [`AttendOne`] from tensor operations, `keys` where `key_mask` `[B, 1, T]`
/// is -∞ left out.
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

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    fn random(shape: &[usize]) -> Tensor {
        Tensor::randn(0f32, 1f32, shape, &Device::Cpu).unwrap()
    }

    fn assert_close(a: &Tensor, b: &Tensor) {
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

    #[test]
    fn matches_the_tensor_operations() {
        let (rows, heads, hd) = (5, 4, 8);
        let q = random(&[rows, heads * hd]);

        // Per-row keys, read through a narrowed buffer as the self-attention
        // cache does.
        let keys = random(&[rows, heads, 12, hd]).narrow(2, 0, 7).unwrap();
        let values = random(&[rows, heads, 12, hd]).narrow(2, 0, 7).unwrap();
        let fused = q
            .apply_op3_no_bwd(
                &keys,
                &values,
                &AttendOne {
                    heads,
                    lengths: None,
                },
            )
            .unwrap();
        let unfused = attend_one_unfused(&q, &keys, &values, None, heads).unwrap();
        assert_close(&fused, &unfused);

        // Keys shared by every row, with each row's own number of them, as
        // the cross-attention does.
        let keys = random(&[1, heads, 6, hd]);
        let values = random(&[1, heads, 6, hd]);
        let lengths = [6, 2, 5, 1, 3];
        let key_mask: Vec<f32> = lengths
            .iter()
            .flat_map(|length| {
                (0..6).map(move |key| {
                    if key < *length {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        let key_mask = Tensor::from_vec(key_mask, (rows, 1, 6), &Device::Cpu).unwrap();
        let fused = q
            .apply_op3_no_bwd(
                &keys,
                &values,
                &AttendOne {
                    heads,
                    lengths: Some(&lengths),
                },
            )
            .unwrap();
        let unfused = attend_one_unfused(&q, &keys, &values, Some(&key_mask), heads).unwrap();
        assert_close(&fused, &unfused);
    }

    #[test]
    fn rejects_mismatched_shapes() {
        let q = random(&[3, 32]);
        let keys = random(&[2, 4, 5, 8]);
        let attend = AttendOne {
            heads: 4,
            lengths: None,
        };
        assert!(q.apply_op3_no_bwd(&keys, &keys, &attend).is_err());
    }
}
