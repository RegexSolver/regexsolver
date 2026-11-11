//! Linear layers and layer norms that run few rows directly on the CPU.
//!
//! candle runs every matmul through gemm, which costs tens of microseconds a
//! call whatever its size (candle counts the physical cores, reading
//! `/proc/cpuinfo`, on each call unless `RAYON_NUM_THREADS` is set): for the
//! decoder's few rows per step, most of the step. [`Linear`] computes up to
//! [`FEW_ROWS`] rows itself instead, one dot product per row and output,
//! each weight row read once for the rows (two at a time), in parallel over
//! the outputs. It also applies the feed-forwards' GELU, cheaper than
//! candle's. candle's layer norm hands every row to the thread pool;
//! [`LayerNorm`] keeps few rows on the calling thread.

use candle_core::{
    CpuStorage, CustomOp1, CustomOp3, Layout, Module, Result, Shape, Storage, Tensor, bail,
};
use candle_nn::VarBuilder;

use super::attention::dot;

/// Most rows [`Linear`] computes itself; more go to candle's gemm.
const FEW_ROWS: usize = 32;
/// Outputs per parallel task.
const OUTPUTS_PER_TASK: usize = 32;
/// Below this many multiply-adds, a product runs on one thread. The
/// decoder's products are bound by reading the weights from memory, faster
/// from several cores.
const PARALLEL_WORK: usize = 1 << 15;

/// `x Wᵀ + b`, as `candle_nn::Linear`.
#[derive(Clone, Debug)]
pub(super) struct Linear(candle_nn::Linear);

impl Linear {
    pub(super) fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self(candle_nn::Linear::new(weight, bias))
    }

    fn apply(&self, x: &Tensor, gelu: bool) -> Result<Tensor> {
        if !x.device().is_cpu() {
            let y = self.0.forward(x)?;
            return if gelu { y.gelu_erf() } else { Ok(y) };
        }
        match x.dims() {
            [rows, _] if *rows <= FEW_ROWS => x.apply_op1_no_bwd(&FewRows {
                weight: self.0.weight(),
                bias: self.0.bias(),
                gelu,
            }),
            _ if gelu => self.0.forward(x)?.apply_op1_no_bwd(&Gelu),
            _ => self.0.forward(x),
        }
    }

    /// `gelu(x Wᵀ + b)`, GELU with erf.
    pub(super) fn forward_gelu(&self, x: &Tensor) -> Result<Tensor> {
        self.apply(x, true)
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.apply(x, false)
    }
}

pub(super) fn linear(inputs: usize, outputs: usize, vb: VarBuilder) -> Result<Linear> {
    candle_nn::linear(inputs, outputs, vb).map(Linear)
}

pub(super) fn linear_no_bias(inputs: usize, outputs: usize, vb: VarBuilder) -> Result<Linear> {
    candle_nn::linear_no_bias(inputs, outputs, vb).map(Linear)
}

/// [`Linear`]'s product on the CPU, for `x` `[rows, inputs]`.
struct FewRows<'a> {
    /// `[outputs, inputs]`.
    weight: &'a Tensor,
    /// `[outputs]`.
    bias: Option<&'a Tensor>,
    /// Whether to apply [`gelu`].
    gelu: bool,
}

impl CustomOp1 for FewRows<'_> {
    fn name(&self) -> &'static str {
        "few-rows-linear"
    }

    fn cpu_fwd(&self, x: &CpuStorage, x_layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (rows, inputs) = x_layout.shape().dims2()?;
        let &[x_row, 1] = x_layout.stride() else {
            bail!("few-rows-linear: x must be contiguous on its last dimension");
        };
        let x = &x.as_slice::<f32>()?[x_layout.start_offset()..];
        let (weight, weight_layout) = self.weight.storage_and_layout();
        let (outputs, weight_inputs) = weight_layout.shape().dims2()?;
        let (Storage::Cpu(weight), &[weight_row, 1]) = (&*weight, weight_layout.stride()) else {
            bail!(
                "few-rows-linear: the weights must be on the CPU, contiguous on their last dimension"
            );
        };
        if weight_inputs != inputs {
            bail!(
                "few-rows-linear: x {:?}, weights {:?}",
                x_layout.shape(),
                weight_layout.shape()
            );
        }
        let weight = &weight.as_slice::<f32>()?[weight_layout.start_offset()..];
        let bias_guard = self.bias.map(|bias| bias.storage_and_layout());
        let bias = match &bias_guard {
            Some((bias, layout)) => {
                let (Storage::Cpu(bias), &[1]) = (&**bias, layout.stride()) else {
                    bail!("few-rows-linear: the bias must be on the CPU and contiguous");
                };
                if layout.shape().dims1()? != outputs {
                    bail!(
                        "few-rows-linear: bias {:?}, {outputs} outputs",
                        layout.shape()
                    );
                }
                Some(&bias.as_slice::<f32>()?[layout.start_offset()..][..outputs])
            }
            None => None,
        };

        // `[outputs, rows]`, transposed after when there are several rows.
        let mut transposed = vec![0f32; outputs * rows];
        let x_at = |row: usize| &x[row * x_row..][..inputs];
        let w_at = |output: usize| &weight[output * weight_row..][..inputs];
        let b_at = |output: usize| bias.map_or(0.0, |bias| bias[output]);
        let activate = |y: f32| if self.gelu { gelu(y) } else { y };
        // Writes the outputs from `first`, `[outputs, rows]`, into `out`.
        let task = |first: usize, out: &mut [f32]| {
            if rows == 1 {
                // The outputs two by two, with the same products.
                let x = x_at(0);
                let (pairs, rest) = out.as_chunks_mut::<2>();
                for (pair, out) in pairs.iter_mut().enumerate() {
                    let output = first + 2 * pair;
                    let dots = dot2([w_at(output), w_at(output + 1)], x);
                    *out = [
                        activate(dots[0] + b_at(output)),
                        activate(dots[1] + b_at(output + 1)),
                    ];
                }
                if let [out] = rest {
                    let output = first + 2 * pairs.len();
                    *out = activate(dot(x, w_at(output)) + b_at(output));
                }
                return;
            }
            for (i, out) in out.chunks_exact_mut(rows).enumerate() {
                let output = first + i;
                let (w, b) = (w_at(output), b_at(output));
                let (pairs, rest) = out.as_chunks_mut::<2>();
                for (pair, out) in pairs.iter_mut().enumerate() {
                    let row = 2 * pair;
                    let dots = dot2([x_at(row), x_at(row + 1)], w);
                    *out = dots.map(|dot| activate(dot + b));
                }
                if let [out] = rest {
                    *out = activate(dot(x_at(rows - 1), w) + b);
                }
            }
        };
        let chunk = OUTPUTS_PER_TASK * rows;
        #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
        if rows * inputs * outputs >= PARALLEL_WORK {
            use rayon::prelude::*;
            transposed
                .par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(i, out)| task(i * OUTPUTS_PER_TASK, out));
        } else {
            task(0, &mut transposed);
        }
        #[cfg(any(not(feature = "parallel"), target_family = "wasm"))]
        {
            let _ = (chunk, PARALLEL_WORK);
            task(0, &mut transposed);
        }

        let out = if rows == 1 {
            transposed
        } else {
            let mut out = vec![0f32; rows * outputs];
            for (output, column) in transposed.chunks_exact(rows).enumerate() {
                for (row, value) in column.iter().enumerate() {
                    out[row * outputs + output] = *value;
                }
            }
            out
        };
        Ok((CpuStorage::F32(out), Shape::from((rows, outputs))))
    }
}

/// GELU with erf, `x Φ(x)`, erf from Abramowitz and Stegun's 7.1.26
/// (absolute error below 1.5e-7).
fn gelu(x: f32) -> f32 {
    let z = (x * std::f32::consts::FRAC_1_SQRT_2).abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152 + t * 1.061_405_4))));
    // `1 - erf(|x| / √2)`.
    let erfc = poly * (-z * z).exp();
    0.5 * x * if x >= 0.0 { 2.0 - erfc } else { erfc }
}

/// `nn.LayerNorm`, as `candle_nn::LayerNorm` with a bias.
#[derive(Clone, Debug)]
pub(super) struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    inner: candle_nn::LayerNorm,
}

pub(super) fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let inner = candle_nn::layer_norm(size, eps, vb)?;
    let (weight, bias) = (inner.weight().clone(), inner.bias().cloned());
    let bias = bias.ok_or_else(|| candle_core::Error::Msg("layer norm without bias".into()))?;
    Ok(LayerNorm {
        weight,
        bias,
        eps: eps as f32,
        inner,
    })
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.device().is_cpu() && x.rank() == 2 {
            x.apply_op3_no_bwd(&self.weight, &self.bias, &Normalize { eps: self.eps })
        } else {
            self.inner.forward(x)
        }
    }
}

/// [`LayerNorm`] on the CPU, `x` `[rows, d]`, with candle's arithmetic.
struct Normalize {
    eps: f32,
}

/// Below this many values, a layer norm runs on one thread.
const NORMALIZE_PARALLEL: usize = 1 << 16;

impl CustomOp3 for Normalize {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(
        &self,
        x: &CpuStorage,
        x_layout: &Layout,
        weight: &CpuStorage,
        weight_layout: &Layout,
        bias: &CpuStorage,
        bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (_, d) = x_layout.shape().dims2()?;
        let contiguous = |layout: &Layout| layout.contiguous_offsets();
        let (Some(x_range), Some(weight_range), Some(bias_range)) = (
            contiguous(x_layout),
            contiguous(weight_layout),
            contiguous(bias_layout),
        ) else {
            bail!("layer-norm: every tensor must be contiguous");
        };
        let x = &x.as_slice::<f32>()?[x_range.0..x_range.1];
        let weight = &weight.as_slice::<f32>()?[weight_range.0..weight_range.1];
        let bias = &bias.as_slice::<f32>()?[bias_range.0..bias_range.1];
        if weight.len() != d || bias.len() != d {
            bail!(
                "layer-norm: x {:?}, {} weights, {} biases",
                x_layout.shape(),
                weight.len(),
                bias.len()
            );
        }
        let eps = self.eps;
        let normalize = |(x, y): (&[f32], &mut [f32])| {
            let (mut sum, mut sum2) = (0f32, 0f32);
            for v in x {
                sum += v;
                sum2 += v * v;
            }
            let mean = sum / d as f32;
            let var = sum2 / d as f32 - mean * mean;
            let inv_std = (var + eps).sqrt().recip();
            for ((y, x), (weight, bias)) in y.iter_mut().zip(x).zip(weight.iter().zip(bias)) {
                *y = (x - mean) * inv_std * weight + bias;
            }
        };
        let mut y = vec![0f32; x.len()];
        #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
        if x.len() >= NORMALIZE_PARALLEL {
            use rayon::prelude::*;
            x.par_chunks(d).zip(y.par_chunks_mut(d)).for_each(normalize);
            return Ok((CpuStorage::F32(y), x_layout.shape().clone()));
        }
        let _ = NORMALIZE_PARALLEL;
        x.chunks(d).zip(y.chunks_mut(d)).for_each(normalize);
        Ok((CpuStorage::F32(y), x_layout.shape().clone()))
    }
}

/// [`gelu`] of each value, on the CPU.
struct Gelu;

/// Values per parallel task of [`Gelu`].
const GELU_CHUNK: usize = 4096;

impl CustomOp1 for Gelu {
    fn name(&self) -> &'static str {
        "gelu"
    }

    fn cpu_fwd(&self, x: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let Some((start, end)) = layout.contiguous_offsets() else {
            bail!("gelu: x must be contiguous");
        };
        let mut y = x.as_slice::<f32>()?[start..end].to_vec();
        let apply = |y: &mut [f32]| y.iter_mut().for_each(|y| *y = gelu(*y));
        #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
        {
            use rayon::prelude::*;
            y.par_chunks_mut(GELU_CHUNK).for_each(apply);
        }
        #[cfg(any(not(feature = "parallel"), target_family = "wasm"))]
        y.chunks_mut(GELU_CHUNK).for_each(apply);
        Ok((CpuStorage::F32(y), layout.shape().clone()))
    }
}

/// `[dot(x[0], w), dot(x[1], w)]`, each `w` value loaded once for both,
/// with [`dot`]'s rounding. (Four rows at once do not vectorize.)
fn dot2(x: [&[f32]; 2], w: &[f32]) -> [f32; 2] {
    let (w_chunks, w_tail) = w.as_chunks::<8>();
    let [(x0, x0_tail), (x1, x1_tail)] = x.map(|x| x[..w.len()].as_chunks::<8>());
    let [lanes0, lanes1] = lanes2(w_chunks, x0, x1);
    let tail = |x: &[f32]| -> f32 { x.iter().zip(w_tail).map(|(x, w)| x * w).sum() };
    [
        lanes0.iter().sum::<f32>() + tail(x0_tail),
        lanes1.iter().sum::<f32>() + tail(x1_tail),
    ]
}

/// [`dot2`]'s lanes. On its own, without the tails: around them, the loop
/// does not vectorize.
#[inline(never)]
fn lanes2(w: &[[f32; 8]], x0: &[[f32; 8]], x1: &[[f32; 8]]) -> [[f32; 8]; 2] {
    let (mut lanes0, mut lanes1) = ([0f32; 8], [0f32; 8]);
    for ((w, x0), x1) in w.iter().zip(x0).zip(x1) {
        for i in 0..8 {
            lanes0[i] += x0[i] * w[i];
        }
        for i in 0..8 {
            lanes1[i] += x1[i] * w[i];
        }
    }
    [lanes0, lanes1]
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::super::attention::tests::{assert_close, random};
    use super::*;

    #[test]
    fn layer_norm_matches_candle() {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let ours = layer_norm(24, 1e-5, vb.pp("norm")).unwrap();
        for var in varmap.all_vars() {
            var.set(&random(var.dims())).unwrap();
        }
        for rows in [1, 5, 3000] {
            let x = random(&[rows, 24]);
            let candle = ours.inner.forward(&x).unwrap();
            assert_close(&ours.forward(&x).unwrap(), &candle);
        }
    }

    #[test]
    fn gelu_matches_candle() {
        let x = (random(&[3000]) * 3.0).unwrap();
        let candle = x.gelu_erf().unwrap();
        assert_close(&x.apply_op1_no_bwd(&Gelu).unwrap(), &candle);
        let x = x.reshape((3, 1000)).unwrap();
        let identity = Linear::new(Tensor::eye(1000, DType::F32, &Device::Cpu).unwrap(), None);
        assert_close(
            &identity.forward_gelu(&x).unwrap(),
            &candle.reshape((3, 1000)).unwrap(),
        );
    }

    #[test]
    fn matches_candle() {
        // A view into larger weights, as the attentions' projections are.
        let weight = random(&[90, 24]).narrow(0, 10, 70).unwrap();
        let bias = random(&[70]);
        for bias in [Some(bias), None] {
            let ours = Linear::new(weight.clone(), bias.clone());
            let candle = candle_nn::Linear::new(weight.clone(), bias);
            for rows in [1, 3, FEW_ROWS, FEW_ROWS + 1] {
                let x = random(&[rows, 30]).narrow(1, 4, 24).unwrap();
                assert_close(&ours.forward(&x).unwrap(), &candle.forward(&x).unwrap());
            }
        }
    }
}
