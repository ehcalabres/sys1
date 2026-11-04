// LFM3 / LFM3-VL inference, following Transformers PR #50036 (Apache-2.0).
use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::{
    Embedding, Linear, VarBuilder, embedding, linear_b,
    ops::{sigmoid, silu, softmax_last_dim},
};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Clone, Deserialize)]
pub struct Rope {
    pub rope_theta: f64,
    pub rope_type: String,
    #[serde(default = "one")]
    pub partial_rotary_factor: f64,
}
fn one() -> f64 {
    1.0
}
#[derive(Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub layer_types: Vec<String>,
    pub norm_eps: f64,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub conv_bias: bool,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub use_sandwich_norm: bool,
    pub rope_parameters: HashMap<String, Rope>,
}
#[derive(Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub norm_eps: f64,
    pub layer_types: Vec<String>,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub in_channels: usize,
    pub conv_kernel_size: usize,
    pub rope_parameters: Rope,
}
#[derive(Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub projector_hidden_size: usize,
    pub projector_bias: bool,
    pub projector_hidden_act: String,
    pub projector_layer_norm_eps: f64,
    pub image_token_id: u32,
    pub video_token_id: u32,
    #[serde(default = "default_vision_start")]
    pub vision_start_token_id: u32,
    #[serde(default = "default_vision_end")]
    pub vision_end_token_id: u32,
    #[serde(default = "default_bos")]
    pub bos_token_id: u32,
    pub model_type: String,
    pub audio_encoder_config: Option<AudioConfig>,
    pub audio_token_id: Option<u32>,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub d1_numbers: bool,
    #[serde(default = "prompt_one")]
    pub d1_prompt: usize,
    #[serde(default = "median_edges")]
    pub d1_box_edges: String,
    pub d1_codes: usize,
    pub d1_attention: String,
    pub d1_set_size: usize,
    pub d1_vision_max_pixels: usize,
}

fn default_vision_start() -> u32 {
    125009
}
fn default_vision_end() -> u32 {
    125010
}
fn default_bos() -> u32 {
    124894
}

pub struct Norm {
    weight: Tensor,
    eps: f64,
}
impl Norm {
    pub fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let f = x.to_dtype(DType::F32)?;
        let y = f.broadcast_div(&(f.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?)?;
        y.to_dtype(x.dtype())?.broadcast_mul(&self.weight)
    }
}
struct Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}
impl Mlp {
    fn load(h: usize, i: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear_b(h, i, false, vb.pp("w1"))?,
            w2: linear_b(i, h, false, vb.pp("w2"))?,
            w3: linear_b(h, i, false, vb.pp("w3"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.w2
            .forward(&(silu(&self.w1.forward(x)?)? * self.w3.forward(x)?)?)
    }
}

/// RoPE uses the full-head frequency progression with a zero tail for partial rotation.
fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let f = x.to_dtype(DType::F32)?;
    let rotated = Tensor::cat(
        &[
            &f.narrow(D::Minus1, half, half)?.neg()?,
            &f.narrow(D::Minus1, 0, half)?,
        ],
        D::Minus1,
    )?;
    (f.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?.to_dtype(x.dtype())
}
fn rope(
    positions: &[[usize; 2]],
    dim: usize,
    theta: f64,
    axial: bool,
    fraction: f64,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let axis_dim = if axial { dim / 2 } else { dim };
    let mut phases = Vec::with_capacity(positions.len() * dim);
    for p in positions {
        let row: Vec<f32> = (0..half)
            .map(|j| {
                let axis = if axial { j / (dim / 4) } else { 0 };
                let index = if axial { j % (dim / 4) } else { j };
                if !axial && j >= (fraction * dim as f64) as usize / 2 {
                    0.
                } else {
                    p[axis] as f32 / theta.powf((2 * index) as f64 / axis_dim as f64) as f32
                }
            })
            .collect();
        phases.extend_from_slice(&row);
        phases.extend_from_slice(&row);
    }
    let phase = Tensor::from_vec(phases, (1, positions.len(), dim), device)?;
    Ok((phase.cos()?, phase.sin()?))
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    qnorm: Norm,
    knorm: Norm,
    gate: Option<Linear>,
    heads: usize,
    kv: usize,
    dim: usize,
}
impl Attention {
    fn load(
        h: usize,
        heads: usize,
        kv: usize,
        dim: usize,
        eps: f64,
        vision: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            q: linear_b(h, heads * dim, false, vb.pp("q_proj"))?,
            k: linear_b(h, kv * dim, false, vb.pp("k_proj"))?,
            v: linear_b(h, kv * dim, false, vb.pp("v_proj"))?,
            out: linear_b(
                heads * dim,
                h,
                false,
                vb.pp(if vision { "out_proj" } else { "o_proj" }),
            )?,
            qnorm: Norm::load(
                dim,
                eps,
                vb.pp(if vision { "q_layernorm" } else { "q_norm" }),
            )?,
            knorm: Norm::load(
                dim,
                eps,
                vb.pp(if vision { "k_layernorm" } else { "k_norm" }),
            )?,
            gate: if vision {
                None
            } else {
                Some(linear_b(h, heads, false, vb.pp("gate_proj"))?)
            },
            heads,
            kv,
            dim,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        pos: &(Tensor, Tensor),
        groups: Option<&[usize]>,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let n = x.dim(0)?;
        let q = rotate(
            &self
                .qnorm
                .forward(&self.q.forward(x)?.reshape((n, self.heads, self.dim))?)?
                .transpose(0, 1)?,
            &pos.0,
            &pos.1,
        )?;
        let k = rotate(
            &self
                .knorm
                .forward(&self.k.forward(x)?.reshape((n, self.kv, self.dim))?)?
                .transpose(0, 1)?,
            &pos.0,
            &pos.1,
        )?;
        let v = self
            .v
            .forward(x)?
            .reshape((n, self.kv, self.dim))?
            .transpose(0, 1)?;
        let repeat = |t: Tensor| -> Result<Tensor> {
            t.unsqueeze(1)?
                .broadcast_as((self.kv, self.heads / self.kv, n, self.dim))?
                .contiguous()?
                .reshape((self.heads, n, self.dim))
        };
        let k = repeat(k)?;
        let v = repeat(v)?;
        let mut pieces = Vec::new();
        // Bound the score allocation for long contexts. Local attention reads only
        // the preceding window rather than allocating a dense sequence mask.
        for start in (0..n).step_by(128) {
            let len = (n - start).min(128);
            let left = window.map_or(0, |w| (start + 1).saturating_sub(w));
            let right = if window.is_some() {
                (start + len).min(n)
            } else {
                n
            };
            let keys = right - left;
            let mut scores = (q
                .narrow(1, start, len)?
                .contiguous()?
                .matmul(&k.narrow(1, left, keys)?.transpose(1, 2)?.contiguous()?)?
                / (self.dim as f64).sqrt())?
            .to_dtype(DType::F32)?;
            if groups.is_some() || window.is_some() {
                let mask: Vec<f32> = (start..start + len)
                    .flat_map(|i| {
                        (left..right).map(move |j| {
                            let allowed = if let Some(w) = window {
                                j <= i && i - j < w
                            } else if let Some(g) = groups {
                                g[i] == 0 || g[j] == 0 || g[i] == g[j]
                            } else {
                                true
                            };
                            if allowed { 0. } else { f32::NEG_INFINITY }
                        })
                    })
                    .collect();
                scores =
                    scores.broadcast_add(&Tensor::from_vec(mask, (1, len, keys), x.device())?)?;
            }
            pieces.push(
                softmax_last_dim(&scores)?
                    .to_dtype(x.dtype())?
                    .matmul(&v.narrow(1, left, keys)?.contiguous()?)?,
            );
        }
        let mut y = Tensor::cat(&pieces, 1)?.transpose(0, 1)?.contiguous()?;
        if let Some(gate) = &self.gate {
            y = y.broadcast_mul(&sigmoid(&gate.forward(x)?)?.unsqueeze(2)?)?;
        }
        self.out.forward(&y.reshape((n, self.heads * self.dim))?)
    }
}

struct ShortConv {
    input: Linear,
    output: Linear,
    weight: Tensor,
    bias: Option<Tensor>,
    width: usize,
    h: usize,
}
impl ShortConv {
    fn load(c: &TextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input: linear_b(
                c.hidden_size,
                3 * c.hidden_size,
                c.conv_bias,
                vb.pp("in_proj"),
            )?,
            output: linear_b(c.hidden_size, c.hidden_size, c.conv_bias, vb.pp("out_proj"))?,
            weight: vb
                .pp("conv")
                .get((c.hidden_size, 1, c.conv_l_cache), "weight")?
                .squeeze(1)?
                .t()?
                .contiguous()?,
            bias: if c.conv_bias {
                Some(vb.pp("conv").get(c.hidden_size, "bias")?)
            } else {
                None
            },
            width: c.conv_l_cache,
            h: c.hidden_size,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let z = self.input.forward(x)?;
        let bx = (z.narrow(1, 0, self.h)? * z.narrow(1, 2 * self.h, self.h)?)?;
        let mut y = Tensor::zeros((n, self.h), x.dtype(), x.device())?;
        for j in 0..self.width.min(n) {
            let part = bx
                .narrow(0, 0, n - j)?
                .broadcast_mul(&self.weight.get(self.width - 1 - j)?)?;
            y = (y + part.pad_with_zeros(0, j, 0)?)?;
        }
        if let Some(b) = &self.bias {
            y = y.broadcast_add(b)?;
        }
        self.output.forward(&(y * z.narrow(1, self.h, self.h)?)?)
    }
}
enum TextOperator {
    Attention(Box<Attention>, bool),
    Conv(ShortConv),
}
struct TextLayer {
    op: TextOperator,
    pre: Norm,
    post: Option<Norm>,
    ffn_norm: Norm,
    ffn_post: Option<Norm>,
    mlp: Mlp,
}
pub struct TextModel {
    pub embedding: Embedding,
    layers: Vec<TextLayer>,
    norm: Norm,
    config: TextConfig,
}
impl TextModel {
    pub fn load(c: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for (i, kind) in c.layer_types.iter().enumerate() {
            let v = vb.pp("layers").pp(i);
            let op = if kind == "conv" {
                TextOperator::Conv(ShortConv::load(c, v.pp("conv"))?)
            } else {
                TextOperator::Attention(
                    Box::new(Attention::load(
                        c.hidden_size,
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        c.head_dim,
                        c.norm_eps,
                        false,
                        v.pp("self_attn"),
                    )?),
                    kind == "sliding_attention",
                )
            };
            layers.push(TextLayer {
                op,
                pre: Norm::load(c.hidden_size, c.norm_eps, v.pp("operator_norm"))?,
                post: if c.use_sandwich_norm {
                    Some(Norm::load(
                        c.hidden_size,
                        c.norm_eps,
                        v.pp("post_operator_norm"),
                    )?)
                } else {
                    None
                },
                ffn_norm: Norm::load(c.hidden_size, c.norm_eps, v.pp("ffn_norm"))?,
                ffn_post: if c.use_sandwich_norm {
                    Some(Norm::load(
                        c.hidden_size,
                        c.norm_eps,
                        v.pp("post_ffn_norm"),
                    )?)
                } else {
                    None
                },
                mlp: Mlp::load(c.hidden_size, c.intermediate_size, v.pp("feed_forward"))?,
            });
        }
        Ok(Self {
            embedding: embedding(c.vocab_size, c.hidden_size, vb.pp("embed_tokens"))?,
            layers,
            norm: Norm::load(c.hidden_size, c.norm_eps, vb.pp("embedding_norm"))?,
            config: c.clone(),
        })
    }
    pub fn forward(&self, mut x: Tensor, groups: &[usize]) -> Result<Tensor> {
        let positions: Vec<_> = (0..x.dim(0)?).map(|i| [i, i]).collect();
        let full = rope(
            &positions,
            self.config.head_dim,
            self.config.rope_parameters["full_attention"].rope_theta,
            false,
            self.config.rope_parameters["full_attention"].partial_rotary_factor,
            x.device(),
        )?;
        let sliding = rope(
            &positions,
            self.config.head_dim,
            self.config.rope_parameters["sliding_attention"].rope_theta,
            false,
            self.config.rope_parameters["sliding_attention"].partial_rotary_factor,
            x.device(),
        )?;
        for layer in &self.layers {
            let normalized = layer.pre.forward(&x)?;
            let mut y = match &layer.op {
                TextOperator::Conv(c) => c.forward(&normalized)?,
                TextOperator::Attention(a, local) => a.forward(
                    &normalized,
                    if *local { &sliding } else { &full },
                    Some(groups),
                    if *local {
                        Some(self.config.sliding_window)
                    } else {
                        None
                    },
                )?,
            };
            if let Some(norm) = &layer.post {
                y = norm.forward(&y)?;
            }
            x = (x + y)?;
            let mut y = layer.mlp.forward(&layer.ffn_norm.forward(&x)?)?;
            if let Some(norm) = &layer.ffn_post {
                y = norm.forward(&y)?;
            }
            x = (x + y)?;
        }
        self.norm.forward(&x)
    }
}

struct VisionConv {
    input: Linear,
    output: Linear,
    weight: Tensor,
    h: usize,
    kernel: usize,
    merge: usize,
}
impl VisionConv {
    fn load(c: &VisionConfig, v: VarBuilder) -> Result<Self> {
        Ok(Self {
            input: linear_b(c.hidden_size, 3 * c.hidden_size, false, v.pp("in_proj"))?,
            output: linear_b(c.hidden_size, c.hidden_size, false, v.pp("out_proj"))?,
            weight: v.pp("spatial_conv").get(
                (c.hidden_size, 1, c.conv_kernel_size, c.conv_kernel_size),
                "weight",
            )?,
            h: c.hidden_size,
            kernel: c.conv_kernel_size,
            merge: c.spatial_merge_size,
        })
    }
    fn forward(&self, x: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        let z = self.input.forward(x)?;
        let bx = (z.narrow(1, 0, self.h)? * z.narrow(1, 2 * self.h, self.h)?)?;
        let m = self.merge;
        let raster = bx
            .reshape((height / m, width / m, m, m, self.h))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((height, width, self.h))?;
        let pad = self.kernel / 2;
        let padded = raster
            .pad_with_zeros(0, pad, pad)?
            .pad_with_zeros(1, pad, pad)?;
        let mut y = Tensor::zeros((height, width, self.h), x.dtype(), x.device())?;
        for row in 0..self.kernel {
            for col in 0..self.kernel {
                let weight = self
                    .weight
                    .narrow(2, row, 1)?
                    .narrow(3, col, 1)?
                    .reshape(self.h)?;
                y = (y + padded
                    .narrow(0, row, height)?
                    .narrow(1, col, width)?
                    .broadcast_mul(&weight)?)?;
            }
        }
        let y = y
            .reshape((height / m, m, width / m, m, self.h))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((height * width, self.h))?;
        self.output.forward(&(y * z.narrow(1, self.h, self.h)?)?)
    }
}
enum VisionOperator {
    Attention(Box<Attention>),
    Conv(VisionConv),
}
struct VisionLayer {
    op: VisionOperator,
    pre: Norm,
    ffn_norm: Norm,
    mlp: Mlp,
}
pub struct VisionModel {
    patch: Linear,
    norm: Norm,
    layers: Vec<VisionLayer>,
    config: VisionConfig,
    projector_norm: candle_nn::LayerNorm,
    p1: Linear,
    p2: Linear,
}
impl VisionModel {
    pub fn load(c: &Config, vb: VarBuilder) -> Result<Self> {
        let v = vb.pp("vision_tower");
        let vc = &c.vision_config;
        let mut layers = Vec::new();
        for (i, kind) in vc.layer_types.iter().enumerate() {
            let w = v.pp("layers").pp(i);
            layers.push(VisionLayer {
                op: if kind == "conv" {
                    VisionOperator::Conv(VisionConv::load(vc, w.pp("conv"))?)
                } else {
                    VisionOperator::Attention(Box::new(Attention::load(
                        vc.hidden_size,
                        vc.num_attention_heads,
                        vc.num_key_value_heads,
                        vc.hidden_size / vc.num_attention_heads,
                        vc.norm_eps,
                        true,
                        w.pp("self_attn"),
                    )?))
                },
                pre: Norm::load(vc.hidden_size, vc.norm_eps, w.pp("operator_norm"))?,
                ffn_norm: Norm::load(vc.hidden_size, vc.norm_eps, w.pp("ffn_norm"))?,
                mlp: Mlp::load(vc.hidden_size, vc.intermediate_size, w.pp("feed_forward"))?,
            });
        }
        let p = vb.pp("multi_modal_projector");
        Ok(Self {
            patch: linear_b(
                vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size,
                vc.hidden_size,
                false,
                v.pp("patchifier"),
            )?,
            norm: Norm::load(vc.hidden_size, vc.norm_eps, v.pp("embedding_norm"))?,
            layers,
            config: vc.clone(),
            projector_norm: candle_nn::layer_norm(
                vc.hidden_size,
                c.projector_layer_norm_eps,
                p.pp("layer_norm"),
            )?,
            p1: linear_b(
                vc.hidden_size * vc.spatial_merge_size.pow(2),
                c.projector_hidden_size,
                c.projector_bias,
                p.pp("linear_1"),
            )?,
            p2: linear_b(
                c.projector_hidden_size,
                c.text_config.hidden_size,
                c.projector_bias,
                p.pp("linear_2"),
            )?,
        })
    }
    pub fn forward(&self, patches: &Tensor, grids: &[[usize; 3]]) -> Result<Tensor> {
        let mut frames = Vec::new();
        let mut offset = 0;
        let c = &self.config;
        let merge = c.spatial_merge_size;
        for &[t, h, w] in grids {
            for _ in 0..t {
                let n = h * w;
                let mut positions = Vec::with_capacity(n);
                for bh in 0..h / merge {
                    for bw in 0..w / merge {
                        for ih in 0..merge {
                            for iw in 0..merge {
                                positions.push([bh * merge + ih, bw * merge + iw]);
                            }
                        }
                    }
                }
                let pos = rope(
                    &positions,
                    c.hidden_size / c.num_attention_heads,
                    c.rope_parameters.rope_theta,
                    true,
                    1.0,
                    patches.device(),
                )?;
                let mut x = self
                    .norm
                    .forward(&self.patch.forward(&patches.narrow(0, offset, n)?)?)?;
                for layer in &self.layers {
                    let z = layer.pre.forward(&x)?;
                    let y = match &layer.op {
                        VisionOperator::Attention(a) => a.forward(&z, &pos, None, None)?,
                        VisionOperator::Conv(v) => v.forward(&z, h, w)?,
                    };
                    x = (x + y)?;
                    x = (&x + layer.mlp.forward(&layer.ffn_norm.forward(&x)?)?)?;
                }
                let y = self
                    .projector_norm
                    .forward(&x)?
                    .reshape((n / (merge * merge), c.hidden_size * merge * merge))?;
                frames.push(self.p2.forward(&self.p1.forward(&y)?.gelu_erf()?)?);
                offset += n;
            }
        }
        Tensor::cat(&frames, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    fn linear(weight: &[f32], rows: usize, cols: usize) -> Linear {
        Linear::new(
            Tensor::from_slice(weight, (rows, cols), &Device::Cpu).unwrap(),
            None,
        )
    }
    #[test]
    fn partial_rope_leaves_zero_frequency_pairs_unchanged() -> Result<()> {
        let x = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6., 7., 8.], (1, 1, 8), &Device::Cpu)?;
        let (cos, sin) = rope(&[[3, 3]], 8, 10000., false, 0.5, &Device::Cpu)?;
        let y = rotate(&x, &cos, &sin)?.flatten_all()?.to_vec1::<f32>()?;
        for i in [2, 3, 6, 7] {
            assert!((y[i] - (i + 1) as f32).abs() < 1e-6);
        }
        assert!((y[0] - 1.).abs() > 0.1);
        Ok(())
    }
    #[test]
    fn short_convolution_is_causal_and_uses_unactivated_gates() -> Result<()> {
        let conv = ShortConv {
            input: linear(&[1., 1., 1.], 3, 1),
            output: linear(&[1.], 1, 1),
            weight: Tensor::from_slice(&[1f32, 2., 3.], (3, 1), &Device::Cpu)?,
            bias: None,
            width: 3,
            h: 1,
        };
        let input = Tensor::from_slice(&[1f32, 2., 3.], (3, 1), &Device::Cpu)?;
        assert_eq!(
            conv.forward(&input)?.flatten_all()?.to_vec1::<f32>()?,
            vec![3., 28., 108.]
        );
        Ok(())
    }
    #[test]
    fn set_attention_shares_state_and_isolates_question_keys() -> Result<()> {
        let norm = || Norm {
            weight: Tensor::ones(2, DType::F32, &Device::Cpu).unwrap(),
            eps: 1e-5,
        };
        let attention = Attention {
            q: linear(&[0.; 4], 2, 2),
            k: linear(&[0.; 4], 2, 2),
            v: linear(&[1., 0., 0., 1.], 2, 2),
            out: linear(&[1., 0., 0., 1.], 2, 2),
            qnorm: norm(),
            knorm: norm(),
            gate: None,
            heads: 1,
            kv: 1,
            dim: 2,
        };
        let x = Tensor::from_slice(&[1f32, 1., 3., 3., 5., 5., 9., 9.], (4, 2), &Device::Cpu)?;
        let pos = (
            Tensor::ones((1, 4, 2), DType::F32, &Device::Cpu)?,
            Tensor::zeros((1, 4, 2), DType::F32, &Device::Cpu)?,
        );
        let y = attention
            .forward(&x, &pos, Some(&[0, 1, 1, 2]), None)?
            .to_vec2::<f32>()?;
        for (row, expected) in y.iter().zip([4.5, 3., 3., 5.]) {
            assert!((row[0] - expected).abs() < 1e-6);
        }
        let local = attention
            .forward(&x, &pos, Some(&[0, 1, 1, 2]), Some(2))?
            .to_vec2::<f32>()?;
        for (row, expected) in local.iter().zip([1., 2., 4., 7.]) {
            assert!((row[0] - expected).abs() < 1e-6);
        }
        Ok(())
    }
    #[test]
    fn vision_depthwise_convolution_restores_merge_block_order() -> Result<()> {
        let conv = VisionConv {
            input: linear(&[1., 1., 1.], 3, 1),
            output: linear(&[1.], 1, 1),
            weight: Tensor::ones((1, 1, 3, 3), DType::F32, &Device::Cpu)?,
            h: 1,
            kernel: 3,
            merge: 2,
        };
        // 2x4 raster is [1,2,5,6; 3,4,7,8] in 2x2 merge-block order.
        let x = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6., 7., 8.], (8, 1), &Device::Cpu)?;
        let actual = conv.forward(&x, 2, 4)?.flatten_all()?.to_vec1::<f32>()?;
        let coords = [
            (0usize, 0usize),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
        ];
        for (i, &(row, col)) in coords.iter().enumerate() {
            let sum: f32 = coords
                .iter()
                .enumerate()
                .filter(|(_, (r, c))| r.abs_diff(row) <= 1 && c.abs_diff(col) <= 1)
                .map(|(j, _)| ((j + 1) * (j + 1)) as f32)
                .sum();
            assert_eq!(actual[i], (i + 1) as f32 * sum);
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize)]
pub struct AudioConfig {
    pub conv_channels: usize,
    pub hidden_size: usize,
    pub num_mel_bins: usize,
    pub sliding_window: usize,
    pub text_config: AudioTextConfig,
}
#[derive(Clone, Deserialize)]
pub struct AudioTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub layer_types: Vec<String>,
    pub norm_eps: f64,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub conv_bias: bool,
    pub rope_parameters: Rope,
    pub block_auto_adjust_ff_dim: bool,
    pub block_ffn_dim_multiplier: Option<f64>,
    pub block_multiple_of: usize,
}
/// The audio encoder uses LFM2's ungated attention and unsandwiched residuals.
/// Its norms, SwiGLU, causal convolution and GQA kernels are shared with LFM3.
pub struct AudioModel {
    conv_in: candle_nn::Conv2d,
    conv_mid: candle_nn::Conv2d,
    lin_mid: candle_nn::Conv2d,
    conv_end: candle_nn::Conv2d,
    lin_end: candle_nn::Conv2d,
    lin_out: Linear,
    layers: Vec<TextLayer>,
    norm: Norm,
    projector_norm: candle_nn::LayerNorm,
    p1: Linear,
    p2: Linear,
    config: AudioConfig,
}
impl AudioModel {
    pub fn load(c: &AudioConfig, out_hidden: usize, vb: VarBuilder) -> Result<Self> {
        let a = &c.text_config;
        let mut intermediate = a.intermediate_size;
        if a.block_auto_adjust_ff_dim {
            intermediate = 2 * intermediate / 3;
            if let Some(factor) = a.block_ffn_dim_multiplier {
                intermediate = ((factor * intermediate as f64) as usize)
                    .div_ceil(a.block_multiple_of)
                    * a.block_multiple_of;
            }
        }
        let tc = TextConfig {
            hidden_size: a.hidden_size,
            vocab_size: 2,
            intermediate_size: intermediate,
            num_hidden_layers: a.num_hidden_layers,
            num_attention_heads: a.num_attention_heads,
            num_key_value_heads: a.num_key_value_heads,
            head_dim: a.hidden_size / a.num_attention_heads,
            layer_types: a.layer_types.clone(),
            norm_eps: a.norm_eps,
            conv_l_cache: a.conv_l_cache,
            conv_bias: a.conv_bias,
            sliding_window: c.sliding_window,
            max_position_embeddings: 128000,
            use_sandwich_norm: false,
            rope_parameters: HashMap::new(),
        };
        let encoder = vb.pp("audio_encoder");
        let v = encoder.pp("downsample");
        let conv = |input, output, kernel, stride, padding, groups, name| {
            candle_nn::conv2d(
                input,
                output,
                kernel,
                candle_nn::Conv2dConfig {
                    stride,
                    padding,
                    groups,
                    ..Default::default()
                },
                v.pp(name),
            )
        };
        let channels = c.conv_channels;
        let raw = v
            .pp("lin_out")
            .get((c.hidden_size, c.num_mel_bins * channels / 8, 1), "weight")?
            .squeeze(2)?;
        let lin_out = Linear::new(raw, Some(v.pp("lin_out").get(c.hidden_size, "bias")?));
        let mut layers = Vec::new();
        for (i, kind) in a.layer_types.iter().enumerate() {
            let w = encoder.pp("lfm.layers").pp(i);
            layers.push(TextLayer {
                op: if kind == "conv" {
                    TextOperator::Conv(ShortConv::load(&tc, w.pp("conv"))?)
                } else {
                    TextOperator::Attention(
                        Box::new(Attention::load(
                            a.hidden_size,
                            a.num_attention_heads,
                            a.num_key_value_heads,
                            tc.head_dim,
                            a.norm_eps,
                            true,
                            w.pp("self_attn"),
                        )?),
                        true,
                    )
                },
                pre: Norm::load(a.hidden_size, a.norm_eps, w.pp("operator_norm"))?,
                post: None,
                ffn_norm: Norm::load(a.hidden_size, a.norm_eps, w.pp("ffn_norm"))?,
                ffn_post: None,
                mlp: Mlp::load(a.hidden_size, intermediate, w.pp("feed_forward"))?,
            });
        }
        let p = vb.pp("audio_projector");
        Ok(Self {
            conv_in: conv(1, channels, 3, 2, 1, 1, "conv_in")?,
            conv_mid: conv(channels, channels, 3, 2, 1, channels, "conv_mid")?,
            lin_mid: conv(channels, channels, 1, 1, 0, 1, "lin_mid")?,
            conv_end: conv(channels, channels, 3, 2, 1, channels, "conv_end")?,
            lin_end: conv(channels, channels, 1, 1, 0, 1, "lin_end")?,
            lin_out,
            layers,
            norm: Norm::load(a.hidden_size, a.norm_eps, encoder.pp("lfm.embedding_norm"))?,
            projector_norm: candle_nn::layer_norm(c.hidden_size, 1e-5, p.pp("layer_norm"))?,
            p1: linear_b(c.hidden_size, out_hidden, true, p.pp("linear_1"))?,
            p2: linear_b(out_hidden, out_hidden, true, p.pp("linear_2"))?,
            config: c.clone(),
        })
    }
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Clips are encoded separately: there are no padded feature frames to
        // mask between the stride-2 stages, and each clip keeps its own context.
        let x = silu(&self.conv_in.forward(&mel.unsqueeze(0)?.unsqueeze(0)?)?)?;
        let x = silu(&self.lin_mid.forward(&self.conv_mid.forward(&x)?)?)?;
        let x = silu(&self.lin_end.forward(&self.conv_end.forward(&x)?)?)?;
        let (_, channels, bins, frames) = x.dims4()?;
        let mut x = self
            .lin_out
            .forward(&x.reshape((channels * bins, frames))?.t()?.contiguous()?)?;
        let positions: Vec<_> = (0..frames).map(|i| [i, i]).collect();
        let a = &self.config.text_config;
        let pos = rope(
            &positions,
            a.hidden_size / a.num_attention_heads,
            a.rope_parameters.rope_theta,
            false,
            1.,
            x.device(),
        )?;
        for layer in &self.layers {
            let z = layer.pre.forward(&x)?;
            let y = match &layer.op {
                TextOperator::Conv(c) => c.forward(&z)?,
                TextOperator::Attention(attn, _) => {
                    attn.forward(&z, &pos, None, Some(self.config.sliding_window))?
                }
            };
            x = (x + y)?;
            x = (&x + layer.mlp.forward(&layer.ffn_norm.forward(&x)?)?)?;
        }
        let x = self.projector_norm.forward(&self.norm.forward(&x)?)?;
        self.p2.forward(&self.p1.forward(&x)?.gelu_erf()?)
    }
}

fn prompt_one() -> usize {
    1
}
fn median_edges() -> String {
    "median".into()
}
