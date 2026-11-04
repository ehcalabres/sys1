#[cfg(feature = "metal")]
#[path = "qwen35_main_metal.rs"]
pub(crate) mod main_metal;

#[cfg(feature = "cuda")]
#[path = "../kernels/qwen35_aot_v91.rs"]
mod native_aot_v91;
#[cfg(feature = "cuda")]
#[path = "qwen35_prepared_only_fork.rs"]
mod prepared_only_fork;

use crate::kernels::activation::sigmoid;
use crate::models::AttentionImplementation;
use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Embedding, Linear as CandleLinear, VarBuilder, embedding, ops::softmax_last_dim};
use serde::Deserialize;
use std::{fs, path::Path, sync::OnceLock, time::Instant};

#[cfg(feature = "cuda")]
#[path = "qwen35_event_profile.rs"]
mod event_profile;

#[cfg(feature = "cuda")]
use candle_core::{
    CudaStorage, Storage,
    backend::BackendStorage,
    cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    },
    op::BackpropOp,
};
#[cfg(feature = "cuda")]
use half::bf16;

#[cfg(feature = "cuda")]
const CUDA_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx"));

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub dtype: Option<String>,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub dtype: Option<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeConfig {
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_section: [usize; 3],
}

#[derive(Clone, Debug, Deserialize)]
pub struct VisionConfig {
    pub dtype: Option<String>,
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let config: Self = serde_json::from_slice(&fs::read(path)?)?;
        anyhow::ensure!(
            config.text_config.layer_types.len() == config.text_config.num_hidden_layers,
            "Qwen3.5 layer_types count does not match num_hidden_layers"
        );
        Ok(config)
    }
}

fn linear(input: usize, output: usize, vb: VarBuilder) -> Result<Linear> {
    let result = Linear::new(vb.get((output, input), "weight")?, None);
    Ok(result)
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(width: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(width, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() && x.dtype() == DType::BF16 {
            return cuda_rms_norm_delta(x, &self.weight, self.eps);
        }
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let scale = (variance + self.eps)?.sqrt()?.recip()?;
        x.broadcast_mul(&scale)?
            .broadcast_mul(&(&self.weight.to_dtype(DType::F32)? + 1.)?)?
            .to_dtype(dtype)
    }

    fn forward_residual(&self, x: &Tensor, mixed: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if x.device().is_cuda()
            && x.dtype() == DType::BF16
            && mixed.dtype() == DType::BF16
            && self.weight.dtype() == DType::BF16
            && x.dims() == mixed.dims()
        {
            return cuda_residual_rms_norm_delta(x, mixed, &self.weight, self.eps);
        }
        let residual = (x + mixed)?;
        let normalized = self.forward(&residual)?;
        Ok((residual, normalized))
    }
}

#[cfg(feature = "cuda")]
fn cuda_l2_128(input: &Tensor, scale: f32) -> Result<Tensor> {
    if input.dtype() != DType::F32 || input.dim(D::Minus1)? != 128 {
        candle_core::bail!("Qwen L2 requires CUDA FP32 width128");
    }
    let input = input.contiguous()?;
    let (storage, layout) = input.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Qwen L2 requires CUDA storage");
    };
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("Qwen L2 noncontiguous input".into()))?;
    let source = storage.as_cuda_slice::<f32>()?.slice(start..end);
    let device = storage.device().clone();
    let count = input.elem_count();
    let rows = u32::try_from(count / 128)
        .map_err(|_| candle_core::Error::Msg("Qwen L2 row count overflow".into()))?;
    if rows == 0 {
        candle_core::bail!("Qwen L2 empty input");
    }
    let output = unsafe { device.alloc::<f32>(count)? };
    let function = device.get_or_load_custom_func("qwen35_l2_128_f32", "sys1_qwen35", CUDA_PTX)?;
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&output);
    launch.arg(&rows);
    launch.arg(&scale);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (rows.div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn cuda_rms_norm_delta(input: &Tensor, delta_weight: &Tensor, epsilon: f64) -> Result<Tensor> {
    let columns = input.dim(D::Minus1)?;
    cuda_rms_norm_delta_with_block_size(
        input,
        delta_weight,
        epsilon,
        columns.next_power_of_two().clamp(32, 1024) as u32,
    )
}

#[cfg(feature = "cuda")]
fn cuda_rms_norm_delta_with_block_size(
    input: &Tensor,
    delta_weight: &Tensor,
    epsilon: f64,
    block_size: u32,
) -> Result<Tensor> {
    let columns = input.dim(D::Minus1)?;
    let vector = columns == 4096 && block_size == 1024;
    if columns == 0
        || delta_weight.dims() != [columns]
        || delta_weight.dtype() != DType::BF16
        || !(32..=1024).contains(&block_size)
        || !block_size.is_power_of_two()
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA RMSNorm shape or dtype")
    }
    let input = input.contiguous()?;
    let delta_weight = delta_weight.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (weight_storage, weight_layout) = delta_weight.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(weight_storage)) =
        (&*input_storage, &*weight_storage)
    else {
        candle_core::bail!("Qwen3.5 RMSNorm requires CUDA storage")
    };
    let (input_start, input_end) = input_layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 RMSNorm input is not contiguous".into()))?;
    let (weight_start, weight_end) = weight_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 RMSNorm weight is not contiguous".into())
    })?;
    let source = input_storage
        .as_cuda_slice::<bf16>()?
        .slice(input_start..input_end);
    let weight = weight_storage
        .as_cuda_slice::<bf16>()?
        .slice(weight_start..weight_end);
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let output = unsafe { device.alloc::<bf16>(count)? };
    let function = device.get_or_load_custom_func(
        if vector {
            "sys1_hidden_rms_vec4_bf16"
        } else {
            "qwen35_rms_norm_delta_bf16"
        },
        "sys1_qwen35",
        CUDA_PTX,
    )?;
    let config = LaunchConfig {
        grid_dim: ((count / columns) as u32, 1, 1),
        block_dim: (if vector { 256 } else { block_size }, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&weight);
    launch.arg(&output);
    let columns = columns as i32;
    let epsilon = epsilon as f32;
    launch.arg(&columns);
    launch.arg(&epsilon);
    unsafe { launch.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn cuda_residual_rms_norm_delta(
    input: &Tensor,
    mixed: &Tensor,
    delta_weight: &Tensor,
    epsilon: f64,
) -> Result<(Tensor, Tensor)> {
    let columns = input.dim(D::Minus1)?;
    let vector = columns == 4096;
    if columns == 0
        || input.dims() != mixed.dims()
        || input.dtype() != DType::BF16
        || mixed.dtype() != DType::BF16
        || delta_weight.dims() != [columns]
        || delta_weight.dtype() != DType::BF16
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA fused residual/RMSNorm shape or dtype")
    }
    let input = input.contiguous()?;
    let mixed = mixed.contiguous()?;
    let delta_weight = delta_weight.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (mixed_storage, mixed_layout) = mixed.storage_and_layout();
    let (weight_storage, weight_layout) = delta_weight.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(mixed_storage), Storage::Cuda(weight_storage)) =
        (&*input_storage, &*mixed_storage, &*weight_storage)
    else {
        candle_core::bail!("Qwen3.5 fused residual/RMSNorm requires CUDA storage")
    };
    let slice = |layout: &candle_core::Layout| {
        layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("Qwen3.5 fused residual/RMSNorm input is not contiguous".into())
        })
    };
    let (input_start, input_end) = slice(input_layout)?;
    let (mixed_start, mixed_end) = slice(mixed_layout)?;
    let (weight_start, weight_end) = slice(weight_layout)?;
    let source = input_storage
        .as_cuda_slice::<bf16>()?
        .slice(input_start..input_end);
    let mixed_source = mixed_storage
        .as_cuda_slice::<bf16>()?
        .slice(mixed_start..mixed_end);
    let weight = weight_storage
        .as_cuda_slice::<bf16>()?
        .slice(weight_start..weight_end);
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let rows = u32::try_from(count / columns).map_err(|_| {
        candle_core::Error::Msg("Qwen3.5 fused residual/RMSNorm row count overflow".into())
    })?;
    let residual = unsafe { device.alloc::<bf16>(count)? };
    let normalized = unsafe { device.alloc::<bf16>(count)? };
    let function = device.get_or_load_custom_func(
        if vector {
            "sys1_hidden_residual_rms_vec4_bf16"
        } else {
            "qwen35_residual_rms_norm_delta_bf16"
        },
        "sys1_qwen35",
        CUDA_PTX,
    )?;
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&mixed_source);
    launch.arg(&weight);
    launch.arg(&residual);
    launch.arg(&normalized);
    let columns = i32::try_from(columns).map_err(|_| {
        candle_core::Error::Msg("Qwen3.5 fused residual/RMSNorm width overflow".into())
    })?;
    let epsilon = epsilon as f32;
    launch.arg(&columns);
    launch.arg(&epsilon);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (
                if vector {
                    256
                } else {
                    (columns as u32).next_power_of_two().clamp(32, 1024)
                },
                1,
                1,
            ),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    let wrap = |data| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(data, device.clone())),
            input.shape().clone(),
            BackpropOp::none(),
            false,
        )
    };
    Ok((wrap(residual), wrap(normalized)))
}

struct Mlp {
    #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
    direct_weight: Option<crate::kernels::fused_gate_up_direct::Prepared>,
    #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
    fused_weight: std::sync::Arc<std::sync::Mutex<Option<Tensor>>>,
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(config: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let gate = linear(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up = linear(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let down = linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;
        #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
        let direct_weight =
            if crate::kernels::fused_gate_up_direct::available(gate.weight(), up.weight()) {
                Some(crate::kernels::fused_gate_up_direct::prepare(
                    gate.weight(),
                    up.weight(),
                )?)
            } else {
                None
            };
        Ok(Self {
            #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
            direct_weight,
            #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
            fused_weight: Default::default(),
            gate,
            up,
            down,
        })
    }

    #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
    fn direct_product(&self, x: &Tensor, capture_index: usize) -> Result<Option<Tensor>> {
        let rows = x.elem_count() / x.dim(D::Minus1)?.max(1);
        if !x.device().is_cuda() || x.dtype() != DType::BF16 || !(2048..=32768).contains(&rows) {
            return Ok(None);
        }
        let Some(weight) = self.direct_weight.as_ref() else {
            return Ok(None);
        };
        let result =
            crate::kernels::fused_gate_up_direct::forward(x, weight, crate::numerics::enabled())?;
        #[cfg(test)]
        DIRECT_MLP_CALLS.with(|slot| slot.set(slot.get() + 1));
        if let Some((gate, up)) = result.projections {
            crate::numerics::layer_capture(capture_index, "mlp-gate-projected", &gate)?;
            crate::numerics::layer_capture(capture_index, "mlp-up-projected", &up)?;
            crate::numerics::layer_capture(
                capture_index,
                "mlp-gate-activated",
                &crate::kernels::activation::silu(&gate)?,
            )?;
        }
        event_profile::mark(x.device(), capture_index, "mlp_direct_gate_up_product")?;
        Ok(Some(result.product))
    }

    fn forward(&self, x: &Tensor, capture_index: usize) -> Result<Tensor> {
        let fused_product: Option<Tensor> = {
            #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
            {
                let rows = x.elem_count() / x.dim(D::Minus1)?.max(1);
                if x.device().is_cuda() && x.dtype() == DType::BF16 && (512..=32768).contains(&rows)
                {
                    if let Some(weight) = self.cached_fused()? {
                        let result = crate::kernels::fused_gate_up::forward(
                            x,
                            &weight,
                            crate::numerics::enabled(),
                        )?;
                        #[cfg(test)]
                        FUSED_MLP_CALLS.with(|slot| slot.set(slot.get() + 1));
                        if let Some((gate, up)) = result.projections {
                            crate::numerics::layer_capture(
                                capture_index,
                                "mlp-gate-projected",
                                &gate,
                            )?;
                            crate::numerics::layer_capture(capture_index, "mlp-up-projected", &up)?;
                            crate::numerics::layer_capture(
                                capture_index,
                                "mlp-gate-activated",
                                &crate::kernels::activation::silu(&gate)?,
                            )?;
                        }
                        event_profile::mark(
                            x.device(),
                            capture_index,
                            "mlp_fused_gate_up_product",
                        )?;
                        Some(result.product)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            #[cfg(not(all(feature = "cuda", sys1_cuda_fused_ffn)))]
            {
                None
            }
        };
        #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
        let fused_product = match fused_product {
            Some(product) => Some(product),
            None => self.direct_product(x, capture_index)?,
        };
        let product = if let Some(product) = fused_product {
            product
        } else if x.device().is_cuda() && x.dtype() == DType::BF16 {
            let gate = linear_attention_projection_forward(x, &self.gate)?;
            #[cfg(feature = "cuda")]
            event_profile::mark(x.device(), capture_index, "mlp_gate_projection")?;
            crate::numerics::layer_capture(capture_index, "mlp-gate-projected", &gate)?;

            if crate::numerics::enabled() {
                let activated = crate::kernels::activation::silu(&gate)?;
                crate::numerics::layer_capture(capture_index, "mlp-gate-activated", &activated)?;
            }
            let up = linear_attention_projection_forward(x, &self.up)?;
            #[cfg(feature = "cuda")]
            event_profile::mark(x.device(), capture_index, "mlp_up_projection")?;
            crate::numerics::layer_capture(capture_index, "mlp-up-projected", &up)?;
            crate::kernels::activation::mul_silu(&gate, &up)?
        } else {
            let activated = {
                let gate = linear_attention_projection_forward(x, &self.gate)?;
                crate::numerics::layer_capture(capture_index, "mlp-gate-projected", &gate)?;
                let activated = crate::kernels::activation::silu(&gate)?;
                crate::numerics::layer_capture(capture_index, "mlp-gate-activated", &activated)?;
                activated
            };
            let up = linear_attention_projection_forward(x, &self.up)?;
            crate::numerics::layer_capture(capture_index, "mlp-up-projected", &up)?;
            (activated * up)?
        };
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), capture_index, "mlp_activation_product")?;
        crate::numerics::layer_capture(capture_index, "mlp-product", &product)?;

        #[cfg(feature = "cuda")]
        if product.device().is_cuda() && product.dtype() == DType::BF16 {
            #[cfg(feature = "cuda")]
            if let Some(output) = try_nn_projection(&product, &self.down)? {
                event_profile::mark(x.device(), capture_index, "mlp_down_projection")?;
                return Ok(output);
            }
            let output = cuda_bf16_projection(&product, &self.down, false)?;
            event_profile::mark(x.device(), capture_index, "mlp_down_projection")?;
            return Ok(output);
        }
        crate::kernels::linear::forward_compatible(&product, &self.down)
    }
}

struct Rotary {
    dimension: usize,
    divisors: Vec<f32>,
    sections: [usize; 3],
}

struct RotaryTensors {
    dimension: usize,
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(dimension: usize, theta: f64, sections: [usize; 3]) -> Result<Self> {
        if !dimension.is_multiple_of(2) || sections.iter().sum::<usize>() != dimension / 2 {
            candle_core::bail!("Qwen3.5 mRoPE sections do not match an even rotary dimension")
        }
        Ok(Self {
            dimension,
            divisors: (0..dimension / 2)
                .map(|index| theta.powf(2. * index as f64 / dimension as f64) as f32)
                .collect(),
            sections,
        })
    }

    fn prepare(
        &self,
        positions: &[[u32; 3]],
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<RotaryTensors> {
        let frequencies = positions
            .iter()
            .flat_map(|position| {
                self.divisors
                    .iter()
                    .enumerate()
                    .map(move |(index, divisor)| {
                        let axis = if index < self.sections[1] * 3 && index % 3 == 1 {
                            1
                        } else if index < self.sections[2] * 3 && index % 3 == 2 {
                            2
                        } else {
                            0
                        };
                        position[axis] as f32 / divisor
                    })
            })
            .collect::<Vec<_>>();
        let phase = Tensor::from_vec(
            frequencies,
            (1, positions.len(), 1, self.dimension / 2),
            device,
        )?;
        let cos = phase.cos()?;
        let sin = phase.sin()?;
        Ok(RotaryTensors {
            dimension: self.dimension,
            cos: Tensor::cat(&[&cos, &cos], D::Minus1)?.to_dtype(dtype)?,
            sin: Tensor::cat(&[&sin, &sin], D::Minus1)?.to_dtype(dtype)?,
        })
    }
}

impl RotaryTensors {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_, length, _, width) = x.dims4()?;
        if length != self.cos.dim(1)? || width < self.dimension {
            candle_core::bail!("Qwen3.5 rotary tensor shape does not match attention input")
        }
        let active = x.narrow(D::Minus1, 0, self.dimension)?;
        let first = active.narrow(D::Minus1, 0, self.dimension / 2)?;
        let second = active.narrow(D::Minus1, self.dimension / 2, self.dimension / 2)?;
        let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
        let active = (active.broadcast_mul(&self.cos)? + rotated.broadcast_mul(&self.sin)?)?;
        Tensor::cat(
            &[
                &active,
                &x.narrow(D::Minus1, self.dimension, width - self.dimension)?,
            ],
            D::Minus1,
        )
    }
}

struct FullAttention {
    #[cfg(feature = "cuda")]
    merged_qkv: Option<Linear>,
    q: Linear,
    k: Linear,
    v: Linear,
    output: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary: Rotary,
    implementation: AttentionImplementation,
}

fn full_attention_kv_forward(input: &Tensor, linear: &Linear) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(output) = try_nn_projection(input, linear)? {
        return Ok(output);
    }

    #[cfg(feature = "cuda")]
    if input.device().is_cuda() && input.dtype() == DType::BF16 {
        return cuda_bf16_projection(input, linear, false);
    }
    crate::kernels::linear::forward_compatible(input, linear)
}

fn linear_attention_projection_forward(input: &Tensor, linear: &Linear) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(output) = try_nn_projection(input, linear)? {
        return Ok(output);
    }

    #[cfg(feature = "cuda")]
    if input.device().is_cuda() && input.dtype() == DType::BF16 {
        return cuda_bf16_projection(input, linear, false);
    }
    crate::kernels::linear::forward_compatible(input, linear)
}

#[cfg(all(test, feature = "cuda"))]
static FULL_QKV_MERGED_SELECTIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

impl FullAttention {
    fn project_finish_fork(
        &self,
        x: &Tensor,
        rotary: &RotaryTensors,
        capture_index: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (batch, length, _) = x.dims3()?;
        #[cfg(feature = "cuda")]
        if matches!(
            option_env!("SYS1_CUDA_COMPUTE_CAP"),
            Some("80" | "90" | "120")
        ) && x.device().is_cuda()
            && x.dtype() == DType::BF16
            && self.q_norm.weight.dtype() == DType::BF16
            && self.k_norm.weight.dtype() == DType::BF16
            && (32..=1024).contains(&self.head_dim)
            && rotary.dimension <= self.head_dim
            && rotary.dimension.is_multiple_of(2)
            && rotary.cos.dtype() == DType::BF16
            && rotary.sin.dtype() == DType::BF16
        {
            if let Some(merged) = &self.merged_qkv {
                if (16_384..=65_536).contains(&x.elem_count().checked_div(x.dim(2)?).unwrap_or(0)) {
                    let projected = crate::kernels::blaslt::project_unblocked(x, merged)?;
                    #[cfg(test)]
                    FULL_QKV_MERGED_SELECTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    if crate::numerics::enabled()
                        && capture_index < 4
                        && std::env::var_os("SYS1_NUMERICS_INPUTS_ONLY").is_none()
                    {
                        let q_columns = 2 * self.heads * self.head_dim;
                        crate::numerics::layer_capture(
                            capture_index,
                            "full-qg-projected",
                            &projected.narrow(2, 0, q_columns)?,
                        )?;
                        crate::numerics::layer_capture(
                            capture_index,
                            "full-k-projected",
                            &projected.narrow(2, q_columns, self.kv_heads * self.head_dim)?,
                        )?;
                    }
                    let (q, k, v, gate) = cuda_full_qkv_finish_fork(
                        &projected,
                        None,
                        &self.q_norm.weight,
                        &self.k_norm.weight,
                        rotary,
                        self.heads,
                        self.kv_heads,
                        self.q_norm.eps,
                    )?;
                    crate::numerics::layer_capture(capture_index, "full-q", &q)?;
                    crate::numerics::layer_capture(capture_index, "full-k", &k)?;
                    crate::numerics::layer_capture(capture_index, "full-v", &v)?;
                    crate::numerics::layer_capture(capture_index, "full-gate", &gate)?;
                    return Ok((q, k, v, gate));
                }
            }

            let qg = linear_attention_projection_forward(x, &self.q)?;
            let k = full_attention_kv_forward(x, &self.k)?;
            let v = full_attention_kv_forward(x, &self.v)?;
            crate::numerics::layer_capture(
                capture_index,
                "full-qg-projected",
                &qg.reshape((batch, length, self.heads, 2, self.head_dim))?,
            )?;
            crate::numerics::layer_capture(
                capture_index,
                "full-k-projected",
                &k.reshape((batch, length, self.kv_heads, self.head_dim))?,
            )?;
            let (q, k, v, gate) = cuda_full_qkv_finish_fork(
                &qg,
                Some((&k, &v)),
                &self.q_norm.weight,
                &self.k_norm.weight,
                rotary,
                self.heads,
                self.kv_heads,
                self.q_norm.eps,
            )?;
            crate::numerics::layer_capture(capture_index, "full-q", &q)?;
            crate::numerics::layer_capture(capture_index, "full-k", &k)?;
            crate::numerics::layer_capture(capture_index, "full-v", &v)?;
            crate::numerics::layer_capture(capture_index, "full-gate", &gate)?;
            return Ok((q, k, v, gate));
        }
        let qg = linear_attention_projection_forward(x, &self.q)?.reshape((
            batch,
            length,
            self.heads,
            2,
            self.head_dim,
        ))?;
        crate::numerics::layer_capture(capture_index, "full-qg-projected", &qg)?;
        let q = full_attention_norm_rotary_fork(
            &qg.narrow(3, 0, 1)?.squeeze(3)?,
            &self.q_norm,
            rotary,
        )?;
        let gate = sigmoid(&qg.narrow(3, 1, 1)?.squeeze(3)?.reshape((
            batch,
            length,
            self.heads * self.head_dim,
        ))?)?;
        let k_projected = full_attention_kv_forward(x, &self.k)?.reshape((
            batch,
            length,
            self.kv_heads,
            self.head_dim,
        ))?;
        crate::numerics::layer_capture(capture_index, "full-k-projected", &k_projected)?;
        let k = full_attention_norm_rotary_fork(&k_projected, &self.k_norm, rotary)?;
        let v = full_attention_kv_forward(x, &self.v)?.reshape((
            batch,
            length,
            self.kv_heads,
            self.head_dim,
        ))?;
        crate::numerics::layer_capture(capture_index, "full-q", &q)?;
        crate::numerics::layer_capture(capture_index, "full-k", &k)?;
        crate::numerics::layer_capture(capture_index, "full-v", &v)?;
        crate::numerics::layer_capture(capture_index, "full-gate", &gate)?;
        Ok((q, k, v, gate))
    }

    fn load(
        config: &TextConfig,
        vb: VarBuilder,
        implementation: AttentionImplementation,
    ) -> Result<Self> {
        let rotary_dim =
            (config.head_dim as f64 * config.rope_parameters.partial_rotary_factor) as usize;
        let rotary = Rotary::new(
            rotary_dim,
            config.rope_parameters.rope_theta,
            config.rope_parameters.mrope_section,
        )?;
        let q = linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim * 2,
            vb.pp("q_proj"),
        )?;
        let k = linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            vb.pp("k_proj"),
        )?;
        let v = linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            vb.pp("v_proj"),
        )?;
        #[cfg(feature = "cuda")]
        let (q, k, v, merged_qkv) = {
            let geometry = config.hidden_size == 4096
                && config.num_attention_heads == 16
                && config.num_key_value_heads == 4
                && config.head_dim == 256
                && rotary_dim == 64
                && q.weight().dtype() == DType::BF16
                && q.bias().is_none()
                && k.bias().is_none()
                && v.bias().is_none();
            let hopper = match q.weight().device() {
                candle_core::Device::Cuda(device) => {
                    geometry && device.cuda_stream().context().compute_capability().w()? == (9, 0)
                }
                _ => false,
            };
            if hopper {
                let q_columns = q.weight().dim(0)?;
                let k_columns = k.weight().dim(0)?;
                let v_columns = v.weight().dim(0)?;
                let packed = Tensor::cat(&[q.weight(), k.weight(), v.weight()], 0)?;
                let q = Linear::new(packed.narrow(0, 0, q_columns)?, None);
                let k = Linear::new(packed.narrow(0, q_columns, k_columns)?, None);
                let v = Linear::new(packed.narrow(0, q_columns + k_columns, v_columns)?, None);
                (q, k, v, Some(Linear::new(packed, None)))
            } else {
                (q, k, v, None)
            }
        };
        Ok(Self {
            #[cfg(feature = "cuda")]
            merged_qkv,
            q,
            k,
            v,
            output: linear(
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
                vb.pp("o_proj"),
            )?,
            q_norm: RmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("k_norm"))?,
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rotary,
            implementation,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &RotaryTensors, capture_index: usize) -> Result<Tensor> {
        let (batch, length, _) = x.dims3()?;
        let (q, k, v, gate) = self.project_finish_fork(x, rotary, capture_index)?;
        let scale = (self.head_dim as f32).powf(-0.5);
        #[cfg(feature = "metal")]
        if x.device().is_metal()
            && x.dtype() != DType::BF16
            && self.implementation != AttentionImplementation::Eager
        {
            let attention = candle_nn::ops::sdpa(
                &q.transpose(1, 2)?.contiguous()?,
                &k.transpose(1, 2)?.contiguous()?,
                &v.transpose(1, 2)?.contiguous()?,
                None,
                true,
                scale,
                1.0,
            )?
            .transpose(1, 2)?;
            crate::numerics::layer_capture(capture_index, "full-attention", &attention)?;
            let gated = full_attention_gate_mul(
                &attention,
                &gate,
                batch,
                length,
                self.heads * self.head_dim,
            )?;
            crate::numerics::layer_capture(capture_index, "full-gated", &gated)?;
            return gated.apply(&self.output);
        }
        let attention = attention_forward(&q, &k, &v, scale, true, self.implementation)?;
        crate::numerics::layer_capture(capture_index, "full-attention", &attention)?;
        #[cfg(feature = "cuda")]
        if crate::numerics::full_attention_reference_enabled(capture_index) {
            let reference = crate::attention::ordinary_reference(
                &q.to_dtype(DType::F32)?,
                &k.to_dtype(DType::F32)?,
                &v.to_dtype(DType::F32)?,
                scale,
                true,
            )?;
            let ordinary = crate::attention::ordinary_reference(&q, &k, &v, scale, true)?;
            crate::numerics::layer_capture(capture_index, "full-reference-f32", &reference)?;
            crate::numerics::layer_capture(capture_index, "full-reference-ordinary", &ordinary)?;
        }
        let gated =
            full_attention_gate_mul(&attention, &gate, batch, length, self.heads * self.head_dim)?;
        crate::numerics::layer_capture(capture_index, "full-gated", &gated)?;

        linear_attention_projection_forward(&gated, &self.output)
    }

    #[cfg(feature = "cuda")]
    fn forward_packed(
        &self,
        x: &Tensor,
        rotary: &RotaryTensors,
        capture_index: usize,
        sequences: &PackedSequences,
    ) -> Result<Tensor> {
        if crate::numerics::enabled() {
            candle_core::bail!("packed inference is not a singleton capture path")
        }
        let (batch, length, _) = x.dims3()?;
        let (q, k, v, gate) = self.project_finish_fork(x, rotary, capture_index)?;
        let scale = (self.head_dim as f32).powf(-0.5);
        #[cfg(feature = "metal")]
        if x.device().is_metal()
            && x.dtype() != DType::BF16
            && self.implementation != AttentionImplementation::Eager
        {
            let attention = candle_nn::ops::sdpa(
                &q.transpose(1, 2)?.contiguous()?,
                &k.transpose(1, 2)?.contiguous()?,
                &v.transpose(1, 2)?.contiguous()?,
                None,
                true,
                scale,
                1.0,
            )?
            .transpose(1, 2)?;
            crate::numerics::layer_capture(capture_index, "full-attention", &attention)?;
            let gated = full_attention_gate_mul(
                &attention,
                &gate,
                batch,
                length,
                self.heads * self.head_dim,
            )?;
            crate::numerics::layer_capture(capture_index, "full-gated", &gated)?;
            return gated.apply(&self.output);
        }
        let attention = crate::attention::flash_attn_varlen(
            &q.squeeze(0)?.contiguous()?,
            &k.squeeze(0)?.contiguous()?,
            &v.squeeze(0)?.contiguous()?,
            &sequences.cumulative,
            &sequences.cumulative,
            *sequences.lengths.iter().max().unwrap(),
            *sequences.lengths.iter().max().unwrap(),
            scale,
            true,
        )?
        .unsqueeze(0)?;
        crate::numerics::layer_capture(capture_index, "full-attention", &attention)?;
        #[cfg(feature = "cuda")]
        if crate::numerics::full_attention_reference_enabled(capture_index) {
            let reference = crate::attention::ordinary_reference(
                &q.to_dtype(DType::F32)?,
                &k.to_dtype(DType::F32)?,
                &v.to_dtype(DType::F32)?,
                scale,
                true,
            )?;
            let ordinary = crate::attention::ordinary_reference(&q, &k, &v, scale, true)?;
            crate::numerics::layer_capture(capture_index, "full-reference-f32", &reference)?;
            crate::numerics::layer_capture(capture_index, "full-reference-ordinary", &ordinary)?;
        }
        let gated =
            full_attention_gate_mul(&attention, &gate, batch, length, self.heads * self.head_dim)?;
        crate::numerics::layer_capture(capture_index, "full-gated", &gated)?;

        linear_attention_projection_forward(&gated, &self.output)
    }
}

pub(crate) fn attention_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
    implementation: AttentionImplementation,
) -> Result<Tensor> {
    let implementation = if implementation == AttentionImplementation::Auto {
        crate::models::attention_for_device(q.dtype(), q.device())
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?
    } else {
        implementation
    };
    match implementation {
        AttentionImplementation::Auto | AttentionImplementation::Eager => {
            eager_attention(q, k, v, scale, causal)
        }
        #[cfg(feature = "cuda")]
        AttentionImplementation::FlashAttention2
        | AttentionImplementation::FlashAttentionBlackwell => crate::attention::flash_attn(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            scale,
            causal,
        ),
        #[allow(unreachable_patterns)]
        other => candle_core::bail!("{} support is not compiled in", other.cli_name()),
    }
}

pub(crate) fn eager_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let (batch, q_length, heads, width) = q.dims4()?;
    let (k_batch, k_length, kv_heads, k_width) = k.dims4()?;
    if kv_heads == 0
        || heads == 0
        || q_length == 0
        || k_length == 0
        || k_batch != batch
        || k_width != width
        || heads % kv_heads != 0
        || k.dims() != v.dims()
        || (causal && q_length != k_length)
    {
        candle_core::bail!("Unsupported eager attention dimensions")
    }
    let expand = |tensor: &Tensor| -> Result<Tensor> {
        tensor
            .to_dtype(DType::F32)?
            .unsqueeze(3)?
            .broadcast_as((batch, k_length, kv_heads, heads / kv_heads, width))?
            .reshape((batch, k_length, heads, width))?
            .transpose(1, 2)
    };
    let query = q.to_dtype(DType::F32)?.transpose(1, 2)?;
    let key = expand(k)?.transpose(D::Minus2, D::Minus1)?;
    let value = expand(v)?;

    const QUERY_CHUNK: usize = 256;
    let mut chunks = Vec::with_capacity(q_length.div_ceil(QUERY_CHUNK));
    for start in (0..q_length).step_by(QUERY_CHUNK) {
        let count = (q_length - start).min(QUERY_CHUNK);
        let mut scores =
            (contiguous_matmul(&query.narrow(2, start, count)?, &key)? * scale as f64)?;
        if causal {
            let mask = Tensor::from_vec(
                (start..start + count)
                    .flat_map(|row| {
                        (0..k_length).map(move |col| if col > row { f32::NEG_INFINITY } else { 0. })
                    })
                    .collect::<Vec<_>>(),
                (1, 1, count, k_length),
                q.device(),
            )?;
            scores = scores.broadcast_add(&mask)?;
        }
        let probabilities = softmax_last_dim(&scores.contiguous()?)?;
        chunks.push(contiguous_matmul(&probabilities, &value)?);
    }
    Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 2)?
        .transpose(1, 2)?
        .to_dtype(q.dtype())
}

#[cfg(feature = "cuda")]
#[path = "qwen35_hopper_qkvz.rs"]
mod hopper_qkvz;

struct LinearAttention {
    #[cfg(feature = "cuda")]
    merged_hopper_qkvz: Option<Linear>,
    #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
    qkvz_weight: std::sync::Mutex<Option<Tensor>>,
    qkv: Linear,
    z: Linear,
    b: Linear,
    a: Linear,
    #[cfg(feature = "cuda")]
    ba: Option<Linear>,
    conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    output: Linear,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel: usize,
    eps: f64,
}

impl LinearAttention {
    fn load(config: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let key_width = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_width = config.linear_num_value_heads * config.linear_value_head_dim;
        let b = vb.pp("in_proj_b").get(
            (config.linear_num_value_heads, config.hidden_size),
            "weight",
        )?;
        let a = vb.pp("in_proj_a").get(
            (config.linear_num_value_heads, config.hidden_size),
            "weight",
        )?;
        #[cfg(feature = "cuda")]
        let ba = if vb.device().is_cuda() && vb.dtype() == DType::BF16 {
            Some(Linear::new(Tensor::cat(&[&b, &a], 0)?, None))
        } else {
            None
        };
        let qkv = linear(
            config.hidden_size,
            key_width * 2 + value_width,
            vb.pp("in_proj_qkv"),
        )?;
        let z = linear(config.hidden_size, value_width, vb.pp("in_proj_z"))?;
        #[cfg(feature = "cuda")]
        let (qkv, z, merged_hopper_qkvz) = hopper_qkvz::prepare(qkv, z)?;
        Ok(Self {
            #[cfg(feature = "cuda")]
            merged_hopper_qkvz,
            #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
            qkvz_weight: std::sync::Mutex::new(None),
            qkv,
            z,
            b: Linear::new(b, None),
            a: Linear::new(a, None),
            #[cfg(feature = "cuda")]
            ba,
            conv: vb.get(
                (
                    key_width * 2 + value_width,
                    1,
                    config.linear_conv_kernel_dim,
                ),
                "conv1d.weight",
            )?,
            a_log: vb
                .get(config.linear_num_value_heads, "A_log")?
                .to_dtype(DType::F32)?
                .exp()?
                .neg()?,
            dt_bias: vb
                .get(config.linear_num_value_heads, "dt_bias")?
                .to_dtype(DType::F32)?,
            norm: vb.get(config.linear_value_head_dim, "norm.weight")?,
            output: linear(value_width, config.hidden_size, vb.pp("out_proj"))?,
            key_heads: config.linear_num_key_heads,
            value_heads: config.linear_num_value_heads,
            key_dim: config.linear_key_head_dim,
            value_dim: config.linear_value_head_dim,
            kernel: config.linear_conv_kernel_dim,
            eps: config.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, capture_index: usize) -> Result<Tensor> {
        let mut profile = if layer_profiling_enabled() {
            x.device().synchronize()?;
            Some(Instant::now())
        } else {
            None
        };
        let (batch, length, _) = x.dims3()?;
        let key_width = self.key_heads * self.key_dim;
        let value_width = self.value_heads * self.value_dim;
        let (projected, early_z) = self.project_qkvz(x)?;
        crate::numerics::layer_capture(capture_index, "linear-projected", &projected)?;
        linear_profile_checkpoint(x, capture_index, "qkv_projection", &mut profile)?;

        let (q, k, v) = {
            #[cfg(feature = "cuda")]
            if x.device().is_cuda()
                && x.dtype() == DType::BF16
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && batch.saturating_mul(length) >= 512
                && self.key_dim == 128
                && self.value_dim == 128
            {
                #[cfg(test)]
                let vstorage_enabled = VSTORAGE_MODEL_REPLAY.with(std::cell::Cell::get);
                #[cfg(not(test))]
                let vstorage_enabled = true;
                let values = if vstorage_enabled && length >= 512 {
                    #[cfg(test)]
                    VSTORAGE_MODEL_CALLS.with(|slot| slot.set(slot.get() + 1));
                    cuda_conv_vstore_fork(
                        &projected,
                        &self.conv,
                        self.key_heads,
                        self.value_heads,
                        None,
                    )?
                } else {
                    cuda_conv_qkv_fork(
                        &projected,
                        &self.conv,
                        self.key_heads,
                        self.value_heads,
                        None,
                    )?
                };

                if crate::numerics::enabled() {
                    let mixed = cuda_conv_silu(&projected, &self.conv)?;
                    crate::numerics::layer_capture(capture_index, "linear-conv", &mixed)?;
                }
                linear_profile_checkpoint(x, capture_index, "convolution_qkv_fused", &mut profile)?;
                values
            } else {
                #[cfg(feature = "cuda")]
                let mixed = if x.device().is_cuda() && matches!(x.dtype(), DType::BF16 | DType::F32)
                {
                    cuda_conv_silu(&projected, &self.conv)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(feature = "metal")]
                let mixed = if x.device().is_metal()
                    && matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32)
                {
                    crate::kernels::metal::conv_silu(&projected, &self.conv)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                let mixed = causal_conv_silu(&projected, &self.conv, self.kernel)?;
                crate::numerics::layer_capture(capture_index, "linear-conv", &mixed)?;
                linear_profile_checkpoint(x, capture_index, "convolution", &mut profile)?;
                let (q, k, v) = {
                    #[cfg(feature = "cuda")]
                    if x.device().is_cuda()
                        && x.dtype() == DType::BF16
                        && self.key_dim == 128
                        && self.value_dim == 128
                    {
                        cuda_fused_qkv_prepare(&mixed, self.key_heads, self.value_heads)?
                    } else {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            #[cfg(feature = "cuda")]
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            #[cfg(feature = "cuda")]
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                };
                (q, k, v)
            }
            #[cfg(not(feature = "cuda"))]
            {
                #[cfg(feature = "cuda")]
                let mixed = if x.device().is_cuda() && matches!(x.dtype(), DType::BF16 | DType::F32)
                {
                    cuda_conv_silu(&projected, &self.conv)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(feature = "metal")]
                let mixed = if x.device().is_metal()
                    && matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32)
                {
                    crate::kernels::metal::conv_silu(&projected, &self.conv)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                let mixed = causal_conv_silu(&projected, &self.conv, self.kernel)?;
                crate::numerics::layer_capture(capture_index, "linear-conv", &mixed)?;
                linear_profile_checkpoint(x, capture_index, "convolution", &mut profile)?;
                let (q, k, v) = {
                    #[cfg(feature = "cuda")]
                    if x.device().is_cuda()
                        && x.dtype() == DType::BF16
                        && self.key_dim == 128
                        && self.value_dim == 128
                    {
                        cuda_fused_qkv_prepare(&mixed, self.key_heads, self.value_heads)?
                    } else {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            #[cfg(feature = "cuda")]
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            #[cfg(feature = "cuda")]
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                };
                (q, k, v)
            }
        };
        let (g, beta) = self.reference_gates(x, capture_index)?;
        crate::numerics::layer_capture(capture_index, "linear-q", &q)?;
        crate::numerics::layer_capture(capture_index, "linear-k", &k)?;
        crate::numerics::layer_capture(capture_index, "linear-v", &v)?;
        crate::numerics::layer_capture(capture_index, "linear-g", &g)?;
        crate::numerics::layer_capture(capture_index, "linear-beta", &beta)?;
        linear_profile_checkpoint(x, capture_index, "qk_and_gates", &mut profile)?;
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() && self.key_dim == 128 {
            let output = if x.dtype() == DType::BF16
                && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
                && self.value_dim == 128
                && length >= 128
            {
                cuda_c64bfout_uniform_fork(&q, &k, &v, &g, &beta, 32, false)?
            } else if x.dtype() == DType::BF16
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && self.value_dim == 128
            {
                if length >= 512 {
                    if v.dtype() == DType::BF16 {
                        cuda_vstore_gdn_uniform_fork(&q, &k, &v, &g, &beta, 32, false)?
                    } else {
                        cuda_c64bfout_uniform_fork(&q, &k, &v, &g, &beta, 32, false)?
                    }
                } else {
                    cuda_delta_rule_token_major_bfout_fork(&q, &k, &v, &g, &beta, true)?
                }
            } else if x.dtype() == DType::BF16
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && length >= 512
                && self.value_dim <= 128
            {
                cuda_chunk64_direct_fork(&q, &k, &v, &g, &beta, 32, false)?
            } else {
                cuda_delta_rule_token_major(&q, &k, &v, &g, &beta)?
            };
            crate::numerics::layer_capture(capture_index, "linear-delta", &output)?;
            if crate::numerics::enabled() {
                crate::numerics::layer_capture(
                    capture_index,
                    "linear-delta-rounded",
                    &output.to_dtype(x.dtype())?,
                )?;
            }
            linear_profile_checkpoint(x, capture_index, "delta_rule", &mut profile)?;
            let result = self.finish_delta(
                x,
                output,
                batch,
                length,
                value_width,
                capture_index,
                early_z,
            )?;
            linear_profile_checkpoint(x, capture_index, "finalize_and_projection", &mut profile)?;
            return Ok(result);
        }
        #[cfg(feature = "metal")]
        let grouped_qk = (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
            || (cfg!(feature = "metal")
                && x.device().is_metal()
                && self.key_dim == 128
                && self.value_dim == 128);
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let beta = beta.transpose(1, 2)?;
        let g = g.transpose(1, 2)?;
        #[cfg(feature = "metal")]
        let output = if grouped_qk {
            let packed = crate::kernels::metal::pack_delta_f32(&q, &k, &v, &g, &beta)?;
            crate::kernels::metal::delta_rule_packed(&packed, self.value_dim)?
        } else {
            chunked_delta_rule(&q, &k, &v, &g, &beta)?
        };
        #[cfg(not(feature = "metal"))]
        let output = chunked_delta_rule(&q, &k, &v, &g, &beta)?;
        if crate::numerics::enabled() {
            crate::numerics::layer_capture(
                capture_index,
                "linear-delta-rounded",
                &output.to_dtype(x.dtype())?,
            )?;
        }
        linear_profile_checkpoint(x, capture_index, "delta_rule", &mut profile)?;
        let result = self.finish_delta(
            x,
            output,
            batch,
            length,
            value_width,
            capture_index,
            early_z,
        )?;
        linear_profile_checkpoint(x, capture_index, "finalize_and_projection", &mut profile)?;
        Ok(result)
    }

    #[cfg(feature = "cuda")]
    fn forward_packed(
        &self,
        x: &Tensor,
        capture_index: usize,
        sequences: &PackedSequences,
    ) -> Result<Tensor> {
        let mut profile = if layer_profiling_enabled() {
            x.device().synchronize()?;
            Some(Instant::now())
        } else {
            None
        };
        if !x.device().is_cuda()
            || x.dtype() != DType::BF16
            || self.key_dim != 128
            || crate::numerics::enabled()
        {
            candle_core::bail!("unsupported packed recurrent architecture/capture mode")
        }
        let (batch, length, _) = x.dims3()?;
        let key_width = self.key_heads * self.key_dim;
        let value_width = self.value_heads * self.value_dim;
        let (projected, early_z) = self.project_qkvz(x)?;
        let mut prepared_qv = None;
        crate::numerics::layer_capture(capture_index, "linear-projected", &projected)?;
        linear_profile_checkpoint(x, capture_index, "qkv_projection", &mut profile)?;

        if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
            && self.value_dim == 128
        {
            if let Some(ticket) = prepared_only_fork::prepare(
                &projected,
                &self.conv,
                self.key_heads,
                self.value_heads,
                sequences,
            )? {
                linear_profile_checkpoint(x, capture_index, "convolution_qkv_fused", &mut profile)?;
                let (g, beta) =
                    self.reference_gates_packed_tensor16(x, capture_index, sequences)?;
                linear_profile_checkpoint(x, capture_index, "qk_and_gates", &mut profile)?;
                if let Some(output) = native_aot_v91::try_forward_prepared_source(
                    &projected,
                    &self.conv,
                    &g,
                    &beta,
                    &sequences.lengths,
                    ticket,
                )? {
                    crate::numerics::layer_capture(capture_index, "linear-delta", &output)?;
                    linear_profile_checkpoint(x, capture_index, "delta_rule", &mut profile)?;
                    let result = self.finish_delta(
                        x,
                        output,
                        batch,
                        length,
                        value_width,
                        capture_index,
                        early_z,
                    )?;
                    linear_profile_checkpoint(
                        x,
                        capture_index,
                        "finalize_and_projection",
                        &mut profile,
                    )?;
                    return Ok(result);
                }
            }
        }

        let (q, k, v) = {
            #[cfg(feature = "cuda")]
            if x.device().is_cuda()
                && x.dtype() == DType::BF16
                && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("90" | "120"))
                && batch.saturating_mul(length) >= 512
                && self.key_dim == 128
                && self.value_dim == 128
            {
                #[cfg(test)]
                let vstorage_enabled = VSTORAGE_MODEL_REPLAY.with(std::cell::Cell::get);
                #[cfg(not(test))]
                let vstorage_enabled = true;
                let values = if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                    let (values, prepared) = cuda_conv_qkv_prepared_fork(
                        &projected,
                        &self.conv,
                        self.key_heads,
                        self.value_heads,
                        sequences,
                    )?;
                    prepared_qv = prepared;
                    values
                } else if vstorage_enabled {
                    #[cfg(test)]
                    VSTORAGE_MODEL_CALLS.with(|slot| slot.set(slot.get() + 1));
                    cuda_conv_vstore_fork(
                        &projected,
                        &self.conv,
                        self.key_heads,
                        self.value_heads,
                        Some(sequences),
                    )?
                } else {
                    cuda_conv_qkv_fork(
                        &projected,
                        &self.conv,
                        self.key_heads,
                        self.value_heads,
                        Some(sequences),
                    )?
                };

                if crate::numerics::enabled() {
                    let mixed = cuda_conv_silu_packed_sequences(&projected, &self.conv, sequences)?;
                    crate::numerics::layer_capture(capture_index, "linear-conv", &mixed)?;
                }
                linear_profile_checkpoint(x, capture_index, "convolution_qkv_fused", &mut profile)?;
                values
            } else {
                #[cfg(feature = "cuda")]
                let mixed = if x.device().is_cuda() && matches!(x.dtype(), DType::BF16 | DType::F32)
                {
                    cuda_conv_silu_packed_sequences(&projected, &self.conv, sequences)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(feature = "metal")]
                let mixed = if x.device().is_metal()
                    && matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32)
                {
                    crate::kernels::metal::conv_silu(&projected, &self.conv)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                let mixed = causal_conv_silu(&projected, &self.conv, self.kernel)?;
                crate::numerics::layer_capture(capture_index, "linear-conv", &mixed)?;
                linear_profile_checkpoint(x, capture_index, "convolution", &mut profile)?;
                let (q, k, v) = {
                    #[cfg(feature = "cuda")]
                    if x.device().is_cuda()
                        && x.dtype() == DType::BF16
                        && self.key_dim == 128
                        && self.value_dim == 128
                    {
                        cuda_fused_qkv_prepare(&mixed, self.key_heads, self.value_heads)?
                    } else {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            #[cfg(feature = "cuda")]
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            #[cfg(feature = "cuda")]
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                };
                (q, k, v)
            }
            #[cfg(not(feature = "cuda"))]
            {
                #[cfg(feature = "cuda")]
                let mixed = if x.device().is_cuda() && matches!(x.dtype(), DType::BF16 | DType::F32)
                {
                    cuda_conv_silu_packed_sequences(&projected, &self.conv, sequences)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(feature = "metal")]
                let mixed = if x.device().is_metal()
                    && matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32)
                {
                    crate::kernels::metal::conv_silu(&projected, &self.conv)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                let mixed = causal_conv_silu(&projected, &self.conv, self.kernel)?;
                crate::numerics::layer_capture(capture_index, "linear-conv", &mixed)?;
                linear_profile_checkpoint(x, capture_index, "convolution", &mut profile)?;
                let (q, k, v) = {
                    #[cfg(feature = "cuda")]
                    if x.device().is_cuda()
                        && x.dtype() == DType::BF16
                        && self.key_dim == 128
                        && self.value_dim == 128
                    {
                        cuda_fused_qkv_prepare(&mixed, self.key_heads, self.value_heads)?
                    } else {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            #[cfg(feature = "cuda")]
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            #[cfg(feature = "cuda")]
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                };
                (q, k, v)
            }
        };
        let (g, beta) = self.reference_gates_packed_tensor16(x, capture_index, sequences)?;
        crate::numerics::layer_capture(capture_index, "linear-q", &q)?;
        crate::numerics::layer_capture(capture_index, "linear-k", &k)?;
        crate::numerics::layer_capture(capture_index, "linear-v", &v)?;
        crate::numerics::layer_capture(capture_index, "linear-g", &g)?;
        crate::numerics::layer_capture(capture_index, "linear-beta", &beta)?;
        linear_profile_checkpoint(x, capture_index, "qk_and_gates", &mut profile)?;
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() && self.key_dim == 128 {
            let output = if x.dtype() == DType::BF16
                && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
                && self.value_dim == 128
            {
                cuda_c64bfout_varlen_prepared_fork(
                    &q,
                    &k,
                    &v,
                    &g,
                    &beta,
                    sequences,
                    false,
                    prepared_qv.take(),
                )?
            } else if x.dtype() == DType::BF16
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && self.value_dim == 128
            {
                let output = if v.dtype() == DType::BF16 {
                    cuda_vstore_gdn_varlen_fork(&q, &k, &v, &g, &beta, sequences, false)?
                } else {
                    cuda_c64bfout_varlen_fork(&q, &k, &v, &g, &beta, sequences, false)?
                };
                output
            } else if x.dtype() == DType::BF16
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && self.value_dim <= 128
            {
                cuda_c64_varlen_fork(&q, &k, &v, &g, &beta, sequences, false)?
            } else {
                cuda_delta_rule_packed_sequences(&q, &k, &v, &g, &beta, sequences, true)?
            };
            crate::numerics::layer_capture(capture_index, "linear-delta", &output)?;
            if crate::numerics::enabled() {
                crate::numerics::layer_capture(
                    capture_index,
                    "linear-delta-rounded",
                    &output.to_dtype(x.dtype())?,
                )?;
            }
            linear_profile_checkpoint(x, capture_index, "delta_rule", &mut profile)?;
            let result = self.finish_delta(
                x,
                output,
                batch,
                length,
                value_width,
                capture_index,
                early_z,
            )?;
            linear_profile_checkpoint(x, capture_index, "finalize_and_projection", &mut profile)?;
            return Ok(result);
        }
        let grouped_qk = (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
            || (cfg!(feature = "metal")
                && x.device().is_metal()
                && self.key_dim == 128
                && self.value_dim == 128);
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let beta = beta.transpose(1, 2)?;
        let g = g.transpose(1, 2)?;
        #[cfg(feature = "metal")]
        let output = if grouped_qk {
            let packed = crate::kernels::metal::pack_delta_f32(&q, &k, &v, &g, &beta)?;
            crate::kernels::metal::delta_rule_packed(&packed, self.value_dim)?
        } else {
            chunked_delta_rule(&q, &k, &v, &g, &beta)?
        };
        #[cfg(not(feature = "metal"))]
        let output = chunked_delta_rule(&q, &k, &v, &g, &beta)?;
        if crate::numerics::enabled() {
            crate::numerics::layer_capture(
                capture_index,
                "linear-delta-rounded",
                &output.to_dtype(x.dtype())?,
            )?;
        }
        linear_profile_checkpoint(x, capture_index, "delta_rule", &mut profile)?;
        let result = self.finish_delta(
            x,
            output,
            batch,
            length,
            value_width,
            capture_index,
            early_z,
        )?;
        linear_profile_checkpoint(x, capture_index, "finalize_and_projection", &mut profile)?;
        Ok(result)
    }

    #[cfg(feature = "cuda")]
    fn reference_gates_packed_tensor16(
        &self,
        x: &Tensor,
        capture_index: usize,
        sequences: &PackedSequences,
    ) -> Result<(Tensor, Tensor)> {
        if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") && x.dtype() == DType::BF16 {
            let ba = self
                .ba
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("missing packed BF16 gates".into()))?;
            if ba.weight().dims() == [64, 4096] {
                let projected = cuda_gate_tensor16_packed_fork(
                    x,
                    ba,
                    true,
                    SimtProjection::Compensated,
                    sequences,
                )?;
                return cuda_packed_ba_nonlinearity(
                    &projected,
                    &projected,
                    &self.dt_bias,
                    &self.a_log,
                );
            }
        }
        self.reference_gates(x, capture_index)
    }

    fn reference_gates(&self, x: &Tensor, capture_index: usize) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() && x.dtype() == DType::BF16 {
            let ba = self
                .ba
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("missing packed BF16 gates".into()))?;
            let projected = if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && x.dim(1)? >= 1024
                && ba.weight().dims() == [64, 4096]
            {
                cuda_gate_tensor16_rows32_fork(x, ba, true, SimtProjection::Compensated)?
            } else {
                cuda_gate_projection_bf16(x, ba)?
            };
            if crate::numerics::enabled() {
                let b = projected.narrow(D::Minus1, 0, self.value_heads)?;
                let a = projected.narrow(D::Minus1, self.value_heads, self.value_heads)?;
                crate::numerics::layer_capture(capture_index, "linear-b-projection", &b)?;
                crate::numerics::layer_capture(
                    capture_index,
                    "linear-a-projection",
                    &a.to_dtype(DType::F32)?,
                )?;
            }

            return cuda_packed_ba_nonlinearity(&projected, &projected, &self.dt_bias, &self.a_log);
        }
        let project = |linear: &Linear| {
            #[cfg(feature = "cuda")]
            if x.device().is_cuda() && x.dtype() == DType::BF16 {
                return cuda_gate_projection_bf16(x, linear);
            }
            crate::kernels::linear::forward_compatible(x, linear)
        };
        let b = project(&self.b)?;
        crate::numerics::layer_capture(capture_index, "linear-b-projection", &b)?;
        let beta = rounded_beta(&b)?;
        drop(b);
        let a = project(&self.a)?.to_dtype(DType::F32)?;
        crate::numerics::layer_capture(capture_index, "linear-a-projection", &a)?;
        let gate_input = a.broadcast_add(&self.dt_bias)?;
        let softplus = (gate_input.relu()? + (gate_input.abs()?.neg()?.exp()? + 1.)?.log()?)?;
        Ok((softplus.broadcast_mul(&self.a_log)?, beta))
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Keep the validated decoder geometry and optional early-Z cast boundary explicit"
    )]
    fn finish_delta(
        &self,
        x: &Tensor,
        output: Tensor,
        batch: usize,
        length: usize,
        value_width: usize,
        capture_index: usize,
        early_z: Option<Tensor>,
    ) -> Result<Tensor> {
        let gate = match early_z {
            Some(z) => z,
            None => linear_attention_projection_forward(x, &self.z)?,
        };

        #[cfg(feature = "cuda")]
        let merged_gate = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
            && x.device().is_cuda()
            && output.dtype() == DType::BF16
            && self.value_heads == 32
            && self.value_dim == 128
            && gate.dims() == [batch, length, 4096]
            && gate.stride() == [length * 12288, 12288, 1];
        #[cfg(not(feature = "cuda"))]
        let merged_gate = false;
        let gate = if merged_gate {
            gate
        } else {
            gate.reshape((batch, length, self.value_heads, self.value_dim))?
        };
        if crate::numerics::enabled() && merged_gate {
            let captured_gate = gate.reshape((batch, length, self.value_heads, self.value_dim))?;
            crate::numerics::layer_capture(capture_index, "linear-z-projected", &captured_gate)?;
        } else {
            crate::numerics::layer_capture(capture_index, "linear-z-projected", &gate)?;
        }
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() && x.dtype() == DType::BF16 && self.value_dim == 128 {
            let finalized = cuda_post_finalize_bf16(&output, &gate, &self.norm, self.eps)?
                .reshape((batch, length, value_width))?;
            crate::numerics::layer_capture(capture_index, "linear-finalized", &finalized)?;
            return linear_attention_projection_forward(&finalized, &self.output);
        }
        #[cfg(feature = "metal")]
        if x.device().is_metal() && x.dtype() == DType::BF16 && self.value_dim == 128 {
            let output = output.to_dtype(x.dtype())?.to_dtype(DType::F32)?;
            let variance = output.sqr()?.mean_keepdim(D::Minus1)?;
            let denominator = (variance + self.eps)?.sqrt()?;
            let gate_silu = gate.to_dtype(DType::F32)?.silu()?;
            let norm = self.norm.to_dtype(DType::F32)?;
            return crate::kernels::metal::post_finalize_bf16(
                &output,
                &denominator,
                &gate_silu,
                &norm,
            )?
            .reshape((batch, length, value_width))?
            .apply(&self.output);
        }
        let finalized = gated_norm_reference(&output, &gate, &self.norm, self.eps, x.dtype())?
            .reshape((batch, length, value_width))?;
        crate::numerics::layer_capture(capture_index, "linear-finalized", &finalized)?;
        crate::kernels::linear::forward_compatible(&finalized, &self.output)
    }
}

#[cfg(feature = "cuda")]
fn cuda_gate_projection_bf16(input: &Tensor, linear: &Linear) -> Result<Tensor> {
    cuda_bf16_projection(input, linear, false)
}

#[cfg(feature = "cuda")]
pub(crate) fn fixed_bf16_projection(input: &Tensor, linear: &Linear) -> Result<Tensor> {
    head_bf16_projection(input, linear, true)
}

#[cfg(feature = "cuda")]
pub(crate) fn compensated_bf16_projection(input: &Tensor, linear: &CandleLinear) -> Result<Tensor> {
    head_bf16_projection(input, linear, false)
}

#[cfg(feature = "cuda")]
fn head_bf16_projection(
    input: &Tensor,
    linear: &CandleLinear,
    tensor_core: bool,
) -> Result<Tensor> {
    let inner = input.dim(D::Minus1)?;
    if inner == 0 || input.elem_count() == 0 {
        return input.apply(linear);
    }
    let rows = input.elem_count() / inner;
    let mut shape = input.dims().to_vec();
    *shape.last_mut().unwrap() = linear.weight().dim(0)?;
    cuda_bf16_projection(
        &input.contiguous()?.reshape((1, rows, inner))?,
        linear,
        tensor_core,
    )?
    .reshape(shape)
}

#[cfg(feature = "cuda")]
fn cuda_bf16_projection(
    input: &Tensor,
    linear: &CandleLinear,
    tensor_core: bool,
) -> Result<Tensor> {
    let inner = input.dim(D::Minus1)?;
    let columns = linear.weight().dim(0)?;
    let rows = input.elem_count() / inner.max(1);

    if !tensor_core
        && input.device().is_cuda()
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && inner == 12288
        && columns == 4096
        && (32768..=65536).contains(&rows)
    {
        return crate::kernels::blaslt::project_hopper_down_cooperative(input, linear);
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
        && inner == 4096
        && matches!(columns, 32 | 64)
        && (1..=65536).contains(&rows)
    {
        return crate::kernels::head_blaslt::project_gdn_ba(input, linear);
    }

    #[cfg(sys1_cuda_fp64_emulation)]
    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("86")
        && (16385..=32768).contains(&rows)
        && inner == 1024
        && columns == 2048
    {
        let (batch, length, _) = input.dims3()?;
        let flattened = input.contiguous()?.reshape((1, rows, inner))?;
        let chunk_rows = rows.div_ceil(rows.div_ceil(16384));
        let mut parts = Vec::with_capacity(rows.div_ceil(chunk_rows));
        for start in (0..rows).step_by(chunk_rows) {
            let count = chunk_rows.min(rows - start);
            parts.push(cuda_bf16_projection(
                &flattened.narrow(1, start, count)?,
                linear,
                false,
            )?);
        }
        return Tensor::cat(&parts, 1)?.reshape((batch, length, columns));
    }

    #[cfg(sys1_cuda_fp64_emulation)]
    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("86")
        && (8192..=16384).contains(&rows)
        && inner == 1024
        && columns == 2048
    {
        return crate::kernels::guarded_dgemm::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
        && rows >= 128
        && inner == 1024
        && columns == 2048
    {
        return cuda_bf16_projection_impl(input, linear, true, SimtProjection::Compensated);
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80")
        && (64..=8192).contains(&rows)
        && inner == 1024
        && matches!(columns, 1024 | 3072 | 4096)
    {
        #[cfg(test)]
        crate::kernels::guarded_dgemm::record_head_route163();
        return crate::kernels::guarded_dgemm::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && (64..=65536).contains(&rows)
        && ((inner == 1024 && matches!(columns, 1024 | 2048 | 3072 | 4096))
            || (inner == 4096 && columns == 1024))
    {
        #[cfg(test)]
        if !(inner == 1024 && columns == 2048 && rows >= 128) {
            crate::kernels::guarded_dgemm::record_head_route163();
        }
        return crate::kernels::guarded_dgemm::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && rows >= 128
        && inner == 1024
        && columns == 2048
    {
        return crate::kernels::guarded_dgemm::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80")
        && rows >= 128
        && inner == 1024
        && columns == 2048
    {
        return crate::kernels::guarded_dgemm::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
        && rows >= 128
        && inner == 1024
        && columns == 2048
    {
        return crate::kernels::biased_projection::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && rows >= 1536
        && inner == 4096
        && columns == 64
    {
        return crate::kernels::guarded_dgemm::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
        && rows >= 1536
        && inner == 4096
        && columns == 64
    {
        return cuda_bf16_projection_impl(
            input,
            linear,
            false,
            SimtProjection::GuardedTensorDoubleRows16,
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
        && rows >= 2048
        && inner == 4096
        && columns == 64
    {
        return cuda_bf16_projection_impl(input, linear, false, SimtProjection::TiledCompensated);
    }
    if !tensor_core
        && linear.bias().is_none()
        && long_down_actual_m(
            rows,
            inner,
            columns,
            option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120"),
        )
    {
        return crate::kernels::blaslt::project_unblocked(input, linear);
    }

    if !tensor_core
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && rows >= 513
        && inner == 4096
        && columns == 12288
    {
        return crate::kernels::blaslt::project_unblocked(input, linear);
    }
    if !tensor_core
        && linear.bias().is_none()
        && columns >= 2048
        && columns >= inner.saturating_mul(2)
        && (rows <= 512 || columns == inner.saturating_mul(2))
    {
        return crate::kernels::blaslt::project_unblocked(input, linear);
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "120"))
        && rows >= 8193
        && columns >= 256
    {
        return crate::kernels::blaslt::project_unblocked(input, linear);
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("89")
        && (513..=1024).contains(&rows)
        && inner == 4096
        && columns == 12288
    {
        return crate::kernels::blaslt::project_pinned2048(input, linear);
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("86")
        && (513..=4096).contains(&rows)
        && inner == 12288
        && columns == 4096
    {
        return crate::kernels::blaslt::project_pinned2048(input, linear);
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80")
        && inner == 12288
        && columns == 4096
        && (3730..=16384).contains(&rows)
    {
        return crate::kernels::blaslt::project_unblocked(input, linear);
    }

    if !tensor_core
        && linear.bias().is_none()
        && linear.weight().dims().first().copied().unwrap_or(0) >= 256
    {
        return crate::kernels::blaslt::project_pinned512(input, linear);
    }
    cuda_bf16_projection_impl(input, linear, tensor_core, SimtProjection::Compensated)
}

#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
fn blocked_projection(input: &Tensor, linear: &Linear, block_rows: usize) -> Result<Tensor> {
    if block_rows == 0 {
        candle_core::bail!("Blocked projection requires a nonzero row block")
    }
    let (batch, length, inner) = input.dims3()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| candle_core::Error::Msg("Blocked projection row count overflow".into()))?;
    if rows == 0 {
        return input.apply(linear);
    }
    let columns = linear.weight().dim(0)?;
    let matrix = input.contiguous()?.reshape((rows, inner))?;
    let mut chunks = Vec::new();
    for start in (0..rows).step_by(block_rows) {
        let count = (rows - start).min(block_rows);
        let piece = matrix.narrow(0, start, count)?;
        let padded = if count == block_rows {
            piece
        } else {
            let zeros = Tensor::zeros((block_rows - count, inner), input.dtype(), input.device())?;
            Tensor::cat(&[&piece, &zeros], 0)?
        };
        chunks.push(padded.apply(linear)?.narrow(0, 0, count)?);
    }
    let output = if chunks.len() == 1 {
        chunks.pop().unwrap()
    } else {
        Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 0)?
    };
    output.reshape((batch, length, columns))
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum SimtProjection {
    #[cfg(test)]
    TensorRows32Probe,
    #[cfg(test)]
    TensorRows128Probe,
    Compensated,
    TiledCompensated,

    #[allow(dead_code)]
    GuardedDouble,
    #[allow(dead_code)]
    GuardedTensorDouble,
    #[allow(dead_code)]
    GuardedTensorDoubleRows16,
}

#[cfg(feature = "cuda")]
impl SimtProjection {
    fn is_tensor_double(self) -> bool {
        matches!(
            self,
            Self::GuardedTensorDouble | Self::GuardedTensorDoubleRows16
        )
    }
}

#[cfg(any(feature = "cuda", test))]
fn guarded_double_radius_coefficient(inner: usize) -> f64 {
    guarded_double_radius_coefficient_impl(inner, false)
}

#[cfg(any(feature = "cuda", test))]
fn guarded_double_radius_coefficient_impl(inner: usize, tensor_dot: bool) -> f64 {
    let count = inner.div_ceil(32);
    let unit32 = f64::from(f32::EPSILON) / 2.;
    let unit64 = f64::EPSILON / 2.;
    if count == 0 || count as f64 * unit32 >= 0.25 {
        return f64::INFINITY;
    }
    let gamma = |n: usize, unit: f64| {
        let nu = (n as f64 * unit).next_up();
        (nu / (1. - nu).next_down()).next_up()
    };
    let g32 = gamma(count, unit32);
    let amplified = (1. + g32).next_up();
    let final_error = (gamma(6, unit64) * (amplified * amplified).next_up()).next_up();
    let old_error = ((g32 * g32).next_up() + final_error).next_up();

    let candidate_operations = if tensor_dot { inner } else { count + 5 };
    (old_error + gamma(candidate_operations, unit64)).next_up()
}

#[cfg(feature = "cuda")]
fn cuda_bf16_projection_impl(
    input: &Tensor,
    linear: &CandleLinear,
    tensor_core: bool,
    simt: SimtProjection,
) -> Result<Tensor> {
    let (batch, length, inner) = input.dims3()?;
    let (columns, weight_inner) = linear.weight().dims2()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| candle_core::Error::Msg("Gate projection row count overflow".into()))?;
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| candle_core::Error::Msg("Gate projection output size overflow".into()))?;
    if input.dtype() != DType::BF16
        || linear.weight().dtype() != DType::BF16
        || (linear.bias().is_some()
            && !tensor_core
            && !matches!(
                simt,
                SimtProjection::Compensated | SimtProjection::TiledCompensated
            ))
        || inner != weight_inner
        || rows == 0
        || columns == 0
        || inner == 0
        || !input.device().same_device(linear.weight().device())
        || input.elem_count() > i32::MAX as usize
        || linear.weight().elem_count() > i32::MAX as usize
        || count > i32::MAX as usize
    {
        candle_core::bail!("Unsupported stable BF16 gate projection shape, dtype or device")
    }
    if let Some(bias) = linear.bias() {
        if bias.dtype() != DType::BF16
            || bias.dims() != [columns]
            || !input.device().same_device(bias.device())
        {
            candle_core::bail!(
                "Stable BF16 projection requires matching bias dtype, shape and device"
            )
        }
    }
    let input = input.contiguous()?;
    let weight = linear.weight().contiguous()?;
    let bias = linear.bias().map(Tensor::contiguous).transpose()?;
    let bias_storage = bias.as_ref().map(Tensor::storage_and_layout);
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("Stable gate projection requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("Gate projection storage is not contiguous".into())
        })
    };
    let (xstart, xend) = offsets(xl)?;
    let (wstart, wend) = offsets(wl)?;
    let x = xs.as_cuda_slice::<bf16>()?.slice(xstart..xend);
    let w = ws.as_cuda_slice::<bf16>()?.slice(wstart..wend);
    let b = if let Some((storage, layout)) = &bias_storage {
        let Storage::Cuda(storage) = &**storage else {
            candle_core::bail!("Stable BF16 projection requires CUDA bias storage")
        };
        let (start, end) = offsets(layout)?;
        Some(storage.as_cuda_slice::<bf16>()?.slice(start..end))
    } else {
        None
    };
    let device = xs.device().clone();
    let mut output = unsafe { device.alloc::<bf16>(count)? };

    let tile_rows = if tensor_core {
        projection_tile_rows(
            rows,
            columns,
            option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120"),
        )
    } else {
        16
    };

    #[cfg(test)]
    let tile_rows = if tensor_core && matches!(simt, SimtProjection::TensorRows32Probe) {
        32
    } else if tensor_core && matches!(simt, SimtProjection::TensorRows128Probe) {
        128
    } else {
        tile_rows
    };
    let kernel = if !tensor_core && matches!(simt, SimtProjection::TiledCompensated) {
        if b.is_some() {
            "qwen35_biased_tiled_compensated_projection_bf16"
        } else {
            "qwen35_tiled_compensated_projection_bf16"
        }
    } else if b.is_some() && !tensor_core {
        "qwen35_biased_compensated_projection_bf16"
    } else if b.is_some() && tile_rows == 128 {
        "qwen35_biased_projection_rows128_bf16"
    } else if b.is_some() && tile_rows == 32 {
        "qwen35_biased_projection_rows32_bf16"
    } else if b.is_some() {
        "qwen35_biased_projection_bf16"
    } else if tensor_core && tile_rows == 128 {
        "qwen35_kv_projection_rows128_bf16"
    } else if tensor_core && tile_rows == 32 {
        "qwen35_kv_projection_rows32_bf16"
    } else if tensor_core {
        "qwen35_kv_projection_bf16"
    } else {
        match simt {
            #[cfg(test)]
            SimtProjection::TensorRows32Probe | SimtProjection::TensorRows128Probe => {
                "qwen35_gate_projection_bf16"
            }
            SimtProjection::Compensated => "qwen35_gate_projection_bf16",
            SimtProjection::TiledCompensated => "qwen35_tiled_compensated_projection_bf16",
            SimtProjection::GuardedDouble => "qwen35_guarded_double_projection_bf16",
            SimtProjection::GuardedTensorDouble => "qwen35_guarded_tensor_double_projection_bf16",
            SimtProjection::GuardedTensorDoubleRows16 => {
                "qwen35_guarded_tensor_double_rows16_projection_bf16"
            }
        }
    };
    let function = device.get_or_load_custom_func(kernel, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: if tensor_core {
            (
                rows.div_ceil(tile_rows) as u32,
                columns.div_ceil(64) as u32,
                1,
            )
        } else if matches!(simt, SimtProjection::TiledCompensated) {
            (rows.div_ceil(4) as u32, columns.div_ceil(4) as u32, 1)
        } else if simt.is_tensor_double() {
            let row_tile = if matches!(simt, SimtProjection::GuardedTensorDoubleRows16) {
                16
            } else {
                8
            };
            (
                rows.div_ceil(row_tile) as u32,
                columns.div_ceil(32) as u32,
                1,
            )
        } else {
            (rows as u32, columns.div_ceil(8) as u32, 1)
        },
        block_dim: (
            if tensor_core {
                tile_rows.min(32) as u32 * 8
            } else if simt.is_tensor_double() || matches!(simt, SimtProjection::TiledCompensated) {
                128
            } else {
                256
            },
            1,
            1,
        ),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&x);
    launch.arg(&w);
    if let Some(bias) = &b {
        launch.arg(bias);
    }
    launch.arg(&mut output);
    let rows = rows as i32;
    if tensor_core || simt.is_tensor_double() || matches!(simt, SimtProjection::TiledCompensated) {
        launch.arg(&rows);
    }
    let inner = inner as i32;
    let columns_arg = columns as i32;
    let radius_coefficient = if !tensor_core
        && !matches!(
            simt,
            SimtProjection::Compensated | SimtProjection::TiledCompensated
        ) {
        Some(guarded_double_radius_coefficient_impl(
            inner as usize,
            simt.is_tensor_double(),
        ))
    } else {
        None
    };
    launch.arg(&inner);
    launch.arg(&columns_arg);
    if let Some(coefficient) = &radius_coefficient {
        launch.arg(coefficient);
    }
    unsafe { launch.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length, columns),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(any(feature = "cuda", test))]
fn projection_tile_rows(rows: usize, columns: usize, sm120: bool) -> usize {
    if columns < 2048 {
        16
    } else if sm120 && rows >= 128 {
        128
    } else if rows >= 64 {
        32
    } else {
        16
    }
}

fn rounded_beta(input: &Tensor) -> Result<Tensor> {
    sigmoid(&input.to_dtype(DType::F32)?)?
        .to_dtype(input.dtype())?
        .to_dtype(DType::F32)
}

fn gated_norm_reference(
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    epsilon: f64,
    dtype: DType,
) -> Result<Tensor> {
    let input = input.to_dtype(dtype)?.to_dtype(DType::F32)?;
    let variance = input.sqr()?.mean_keepdim(D::Minus1)?;
    let normalized = input
        .broadcast_div(&(variance + epsilon)?.sqrt()?)?
        .to_dtype(dtype)?;

    normalized
        .broadcast_mul(&weight.to_dtype(dtype)?)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&gate.to_dtype(DType::F32)?.silu()?)?
        .to_dtype(dtype)
}

#[cfg(feature = "cuda")]
fn cuda_post_finalize_original_fork(
    input: &Tensor,
    gate: &Tensor,
    norm: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    let columns = input.dim(D::Minus1)?;
    if input.dtype() != DType::F32
        || gate.dtype() != DType::BF16
        || norm.dtype() != DType::BF16
        || input.dims() != gate.dims()
        || norm.dims() != [columns]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA post-finalize shape or dtype")
    }
    let input = input.contiguous()?;
    let gate = gate.contiguous()?;
    let norm = norm.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (norm_storage, norm_layout) = norm.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(gate_storage), Storage::Cuda(norm_storage)) =
        (&*input_storage, &*gate_storage, &*norm_storage)
    else {
        candle_core::bail!("Qwen3.5 post-finalize requires CUDA storage")
    };
    let (input_start, input_end) = input_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize input is not contiguous".into())
    })?;
    let (gate_start, gate_end) = gate_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize gate is not contiguous".into())
    })?;
    let (norm_start, norm_end) = norm_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize norm is not contiguous".into())
    })?;
    let source = input_storage
        .as_cuda_slice::<f32>()?
        .slice(input_start..input_end);
    let gate = gate_storage
        .as_cuda_slice::<bf16>()?
        .slice(gate_start..gate_end);
    let norm = norm_storage
        .as_cuda_slice::<bf16>()?
        .slice(norm_start..norm_end);
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let output = unsafe { device.alloc::<bf16>(count)? };
    let function =
        device.get_or_load_custom_func("qwen35_post_finalize_bf16", "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: ((count / columns) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&gate);
    launch.arg(&norm);
    launch.arg(&output);
    let columns = columns as i32;
    let epsilon = epsilon as f32;
    launch.arg(&columns);
    launch.arg(&epsilon);
    unsafe { launch.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

fn causal_conv_silu(projected: &Tensor, conv: &Tensor, kernel: usize) -> Result<Tensor> {
    crate::kernels::activation::silu(&causal_conv(projected, conv, kernel)?)
}

fn causal_conv(projected: &Tensor, conv: &Tensor, kernel: usize) -> Result<Tensor> {
    let (batch, length, channels) = projected.dims3()?;
    let accumulation_dtype = match projected.dtype() {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    };
    let padded = projected
        .to_dtype(accumulation_dtype)?
        .pad_with_zeros(1, kernel - 1, 0)?;
    let weights = conv
        .to_dtype(accumulation_dtype)?
        .reshape((channels, kernel))?;
    let mut mixed = Tensor::zeros(
        (batch, length, channels),
        accumulation_dtype,
        projected.device(),
    )?;
    for tap in 0..kernel {
        let source = padded.narrow(1, tap, length)?;
        let weight = weights
            .narrow(1, tap, 1)?
            .squeeze(1)?
            .reshape((1, 1, channels))?;
        mixed = (mixed + source.broadcast_mul(&weight)?)?;
    }
    mixed.to_dtype(projected.dtype())
}

#[cfg(feature = "cuda")]
fn cuda_conv_silu(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (batch, length, channels) = input.dims3()?;
    let (weight_channels, groups, kernel) = weights.dims3()?;
    if weight_channels != channels || groups != 1 || kernel == 0 || input.dtype() != weights.dtype()
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA convolution shape or dtype")
    }
    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (weight_storage, weight_layout) = weights.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(weight_storage)) =
        (&*input_storage, &*weight_storage)
    else {
        candle_core::bail!("Qwen3.5 convolution requires CUDA storage")
    };
    let (input_start, input_end) = input_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 convolution input is not contiguous".into())
    })?;
    let (weight_start, weight_end) = weight_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 convolution weights are not contiguous".into())
    })?;
    let device = input_storage.device().clone();
    let count = batch * length * channels;
    let config = LaunchConfig {
        grid_dim: (count.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let count_arg = count as i32;
    let length_arg = length as i32;
    let channels_arg = channels as i32;
    let kernel_arg = kernel as i32;
    let output = match input.dtype() {
        DType::BF16 => {
            let source = input_storage
                .as_cuda_slice::<bf16>()?
                .slice(input_start..input_end);
            let filters = weight_storage
                .as_cuda_slice::<bf16>()?
                .slice(weight_start..weight_end);
            let output = unsafe { device.alloc::<bf16>(count)? };
            let function =
                device.get_or_load_custom_func("qwen35_conv_bf16", "sys1_qwen35", CUDA_PTX)?;
            let mut launch = function.builder();
            launch.arg(&source);
            launch.arg(&filters);
            launch.arg(&output);
            launch.arg(&count_arg);
            launch.arg(&length_arg);
            launch.arg(&channels_arg);
            launch.arg(&kernel_arg);
            unsafe { launch.launch(config) }.w()?;
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device))
        }
        DType::F32 => {
            let source = input_storage
                .as_cuda_slice::<f32>()?
                .slice(input_start..input_end);
            let filters = weight_storage
                .as_cuda_slice::<f32>()?
                .slice(weight_start..weight_end);
            let output = unsafe { device.alloc::<f32>(count)? };
            let function =
                device.get_or_load_custom_func("qwen35_conv_f32", "sys1_qwen35", CUDA_PTX)?;
            let mut launch = function.builder();
            launch.arg(&source);
            launch.arg(&filters);
            launch.arg(&output);
            launch.arg(&count_arg);
            launch.arg(&length_arg);
            launch.arg(&channels_arg);
            launch.arg(&kernel_arg);
            unsafe { launch.launch(config) }.w()?;
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device))
        }
        dtype => candle_core::bail!("Unsupported Qwen3.5 CUDA convolution dtype: {dtype:?}"),
    };
    Ok(Tensor::from_storage(
        output,
        (batch, length, channels),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn cuda_delta_rule_token_major(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    if matches!(
        option_env!("SYS1_CUDA_COMPUTE_CAP"),
        Some("90" | "80" | "86" | "89")
    ) && q.dim(1)? >= 512
    {
        return cuda_delta_rule_precomputed_h200_impl(q, k, v, g, beta, true);
    }
    cuda_delta_rule_token_major_impl(q, k, v, g, beta, true)
}

#[cfg(feature = "cuda")]
fn cuda_delta_rule_token_major_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    subwarp16: bool,
) -> Result<Tensor> {
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let device = q_storage.device().clone();
    let output = unsafe { device.alloc::<f32>(batch * heads * length * value_dim)? };
    let name = if subwarp16 {
        "qwen35_delta_subwarp16_f32"
    } else {
        "qwen35_delta_f32"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (
            (batch * heads) as u32,
            value_dim.div_ceil(if subwarp16 { 8 } else { 4 }) as u32,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&query);
    launch.arg(&key);
    launch.arg(&value);
    launch.arg(&gate);
    launch.arg(&beta);
    launch.arg(&output);
    let length = length as i32;
    let heads_arg = heads as i32;
    let key_heads_arg = key_heads as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&heads_arg);
    launch.arg(&key_heads_arg);
    launch.arg(&value_dim);
    unsafe { launch.launch(config) }.w()?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, heads, length as usize, value_dim as usize),
        BackpropOp::none(),
        false,
    );
    output.transpose(1, 2)
}

fn chunked_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    let key_dim = q.dim(D::Minus1)?;
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() && key_dim == 128 {
        candle_core::bail!("CUDA delta-rule inputs must remain token-major")
    }
    #[cfg(feature = "metal")]
    if q.device().is_metal() && key_dim == 128 {
        return crate::kernels::metal::delta_rule(q, k, v, g, beta);
    }
    chunked_delta_rule_fallback(q, k, v, g, beta)
}

fn chunked_delta_rule_fallback(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let (batch, heads, length, key_dim) = q.dims4()?;
    let value_dim = v.dim(D::Minus1)?;
    let mut state = Tensor::zeros((batch, heads, key_dim, value_dim), DType::F32, q.device())?;
    let mut outputs = Vec::with_capacity(length.div_ceil(32));
    let full_lower = Tensor::tril2(32, DType::F32, q.device())?;
    let full_identity = Tensor::eye(32, DType::F32, q.device())?;
    let full_strictly_lower = (&full_lower - &full_identity)?;
    for start in (0..length).step_by(32) {
        let count = (length - start).min(32);
        let qi = q.narrow(2, start, count)?;
        let ki = k.narrow(2, start, count)?;
        let vi = v.narrow(2, start, count)?;
        let bi = beta.narrow(2, start, count)?.unsqueeze(3)?;
        let gi = g.narrow(2, start, count)?.contiguous()?.cumsum(2)?;
        let (lower, identity, strictly_lower) = if count == 32 {
            (
                full_lower.clone(),
                full_identity.clone(),
                full_strictly_lower.clone(),
            )
        } else {
            let lower = Tensor::tril2(count, DType::F32, q.device())?;
            let identity = Tensor::eye(count, DType::F32, q.device())?;
            let strictly_lower = (&lower - &identity)?;
            (lower, identity, strictly_lower)
        };
        let decay = gi
            .unsqueeze(3)?
            .broadcast_sub(&gi.unsqueeze(2)?)?
            .broadcast_mul(&lower)?
            .exp()?
            .broadcast_mul(&lower)?;
        let key_beta = ki.broadcast_mul(&bi)?;
        let value_beta = vi.broadcast_mul(&bi)?;
        let gi_exp = gi.exp()?;
        let l = contiguous_matmul(&key_beta, &ki.transpose(D::Minus2, D::Minus1)?)?
            .broadcast_mul(&decay)?
            .broadcast_mul(&strictly_lower)?
            .neg()?;
        let mut inverse = l.broadcast_add(&identity)?;
        let mut power = l;
        let mut covered = 2;
        while covered < count {
            power = contiguous_matmul(&power, &power)?;
            inverse = (&inverse + contiguous_matmul(&power, &inverse)?)?;
            covered *= 2;
        }
        let value = contiguous_matmul(&inverse, &value_beta)?;
        let key_cumdecay =
            contiguous_matmul(&inverse, &key_beta.broadcast_mul(&gi_exp.unsqueeze(3)?)?)?;
        let value_new = (&value - contiguous_matmul(&key_cumdecay, &state)?)?;
        let attn =
            contiguous_matmul(&qi, &ki.transpose(D::Minus2, D::Minus1)?)?.broadcast_mul(&decay)?;
        let intermediate = contiguous_matmul(&qi.broadcast_mul(&gi_exp.unsqueeze(3)?)?, &state)?;
        outputs.push((intermediate + contiguous_matmul(&attn, &value_new)?)?);
        let last = gi.narrow(2, count - 1, 1)?;
        let updated = contiguous_matmul(
            &ki.broadcast_mul(&last.broadcast_sub(&gi)?.exp()?.unsqueeze(3)?)?
                .transpose(D::Minus2, D::Minus1)?,
            &value_new,
        )?;
        state = (state.broadcast_mul(&last.exp()?.unsqueeze(3)?)? + updated)?;
    }
    Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2)?.transpose(1, 2)
}

fn contiguous_matmul(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    crate::kernels::linear::matmul_compatible(&left.contiguous()?, &right.contiguous()?)
}

fn layer_profiling_enabled() -> bool {
    static PROFILE: OnceLock<bool> = OnceLock::new();
    *PROFILE.get_or_init(|| std::env::var_os("SYS1_PROFILE_LAYERS").is_some())
}

fn linear_profile_checkpoint(
    x: &Tensor,
    index: usize,
    stage: &'static str,
    started: &mut Option<Instant>,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    event_profile::mark(x.device(), index, stage)?;
    if let Some(started) = started.as_mut() {
        x.device().synchronize()?;
        tracing::info!(
            layer = index,
            stage,
            elapsed_ms = started.elapsed().as_secs_f64() * 1_000.,
            "Qwen3.5 linear attention profile"
        );
        *started = Instant::now();
    }
    Ok(())
}

enum Mixer {
    Full(FullAttention),
    Linear(LinearAttention),
}

struct Layer {
    index: usize,
    input_norm: RmsNorm,
    mixer: Mixer,
    post_norm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(
        config: &TextConfig,
        index: usize,
        vb: VarBuilder,
        implementation: AttentionImplementation,
    ) -> Result<Self> {
        let mixer = match config.layer_types[index].as_str() {
            "full_attention" => Mixer::Full(FullAttention::load(
                config,
                vb.pp("self_attn"),
                implementation,
            )?),
            "linear_attention" => {
                Mixer::Linear(LinearAttention::load(config, vb.pp("linear_attn"))?)
            }
            other => candle_core::bail!("unsupported Qwen3.5 layer type {other}"),
        };
        Ok(Self {
            index,
            input_norm: RmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            mixer,
            post_norm: RmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(config, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, rotary: Option<&RotaryTensors>) -> Result<Tensor> {
        let profile = layer_profiling_enabled();
        if profile {
            x.device().synchronize()?;
        }
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "start")?;
        let started = Instant::now();
        let normalized = self.input_norm.forward(x)?;
        crate::numerics::capture(&format!("layer-{:02}-normalized", self.index), &normalized)?;
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "norm")?;
        if profile {
            x.device().synchronize()?;
        }
        let norm_ms = started.elapsed().as_secs_f64() * 1_000.;
        let started = Instant::now();
        let mixed = match &self.mixer {
            Mixer::Full(layer) => layer.forward(
                &normalized,
                rotary.ok_or_else(|| {
                    candle_core::Error::Msg("Missing Qwen3.5 rotary tensors".into())
                })?,
                self.index,
            )?,
            Mixer::Linear(layer) => layer.forward(&normalized, self.index)?,
        };
        crate::numerics::capture(&format!("layer-{:02}-mixed", self.index), &mixed)?;
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "mixer_tail")?;
        if profile {
            x.device().synchronize()?;
        }
        let mixer_ms = started.elapsed().as_secs_f64() * 1_000.;
        let started = Instant::now();
        let (x, post_normalized) = self.post_norm.forward_residual(x, &mixed)?;
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "residual_and_post_norm")?;
        crate::numerics::capture(
            &format!("layer-{:02}-post-normalized", self.index),
            &post_normalized,
        )?;
        let mlp_output = self.mlp.forward(&post_normalized, self.index)?;
        crate::numerics::capture(&format!("layer-{:02}-mlp-output", self.index), &mlp_output)?;
        let output = (&x + mlp_output)?;
        crate::numerics::capture(&format!("layer-{:02}-output", self.index), &output)?;
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "output_residual")?;
        if profile {
            x.device().synchronize()?;
            tracing::info!(
                layer = self.index,
                mixer = match &self.mixer {
                    Mixer::Full(_) => "full",
                    Mixer::Linear(_) => "linear",
                },
                norm_ms,
                mixer_ms,
                mlp_ms = started.elapsed().as_secs_f64() * 1_000.,
                "Qwen3.5 layer profile"
            );
        }
        Ok(output)
    }

    #[cfg(feature = "cuda")]
    fn forward_packed(
        &self,
        x: &Tensor,
        rotary: Option<&RotaryTensors>,
        sequences: &PackedSequences,
    ) -> Result<Tensor> {
        let profile = layer_profiling_enabled();
        if profile {
            x.device().synchronize()?;
        }
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "start")?;
        let started = Instant::now();
        let normalized = self.input_norm.forward(x)?;
        crate::numerics::capture(&format!("layer-{:02}-normalized", self.index), &normalized)?;
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "norm")?;
        if profile {
            x.device().synchronize()?;
        }
        let norm_ms = started.elapsed().as_secs_f64() * 1_000.;
        let started = Instant::now();
        let mixed = match &self.mixer {
            Mixer::Full(layer) => layer.forward_packed(
                &normalized,
                rotary.ok_or_else(|| {
                    candle_core::Error::Msg("Missing Qwen3.5 rotary tensors".into())
                })?,
                self.index,
                sequences,
            )?,
            Mixer::Linear(layer) => layer.forward_packed(&normalized, self.index, sequences)?,
        };
        crate::numerics::capture(&format!("layer-{:02}-mixed", self.index), &mixed)?;
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "mixer_tail")?;
        if profile {
            x.device().synchronize()?;
        }
        let mixer_ms = started.elapsed().as_secs_f64() * 1_000.;
        let started = Instant::now();
        let (x, post_normalized) = self.post_norm.forward_residual(x, &mixed)?;
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "residual_and_post_norm")?;
        crate::numerics::capture(
            &format!("layer-{:02}-post-normalized", self.index),
            &post_normalized,
        )?;
        let mlp_output = self.mlp.forward(&post_normalized, self.index)?;
        crate::numerics::capture(&format!("layer-{:02}-mlp-output", self.index), &mlp_output)?;
        let output = (&x + mlp_output)?;
        crate::numerics::capture(&format!("layer-{:02}-output", self.index), &output)?;
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "output_residual")?;
        if profile {
            x.device().synchronize()?;
            tracing::info!(
                layer = self.index,
                mixer = match &self.mixer {
                    Mixer::Full(_) => "full",
                    Mixer::Linear(_) => "linear",
                },
                norm_ms,
                mixer_ms,
                mlp_ms = started.elapsed().as_secs_f64() * 1_000.,
                "Qwen3.5 layer profile"
            );
        }
        Ok(output)
    }
}

#[cfg(feature = "cuda")]
fn maybe_cpu_input_embedding(embedding: Embedding, hidden_size: usize) -> Result<Embedding> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("86")
        || embedding.embeddings().dtype() != DType::BF16
    {
        return Ok(embedding);
    }
    let candle_core::Device::Cuda(cuda) = embedding.embeddings().device() else {
        return Ok(embedding);
    };
    let (_, total) = cuda
        .cuda_stream()
        .context()
        .mem_get_info()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let host_limit = std::fs::read_to_string("/sys/fs/cgroup/memory.max")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok());
    if total <= 32usize * 1024 * 1024 * 1024
        && host_limit.is_some_and(|bytes| bytes >= 8usize * 1024 * 1024 * 1024)
    {
        let bytes = embedding.embeddings().elem_count() * 2;
        let weights = embedding
            .embeddings()
            .to_device(&candle_core::Device::Cpu)?;
        tracing::info!(
            CPU_embedding_bytes = bytes,
            GPU_total_bytes = total,
            "Qwen input embedding prepared on CPU for bounded SM86 capacity experiment"
        );
        return Ok(Embedding::new(weights, hidden_size));
    }
    Ok(embedding)
}

pub struct Decoder {
    embedding: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
}

impl Decoder {
    pub fn load(
        config: &TextConfig,
        vb: VarBuilder,
        implementation: AttentionImplementation,
    ) -> Result<Self> {
        let embedding = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        #[cfg(feature = "cuda")]
        let embedding = maybe_cpu_input_embedding(embedding, config.hidden_size)?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                Layer::load(
                    config,
                    index,
                    vb.pp(format!("layers.{index}")),
                    implementation,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embedding,
            layers,
            norm,
        })
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let length = ids.dim(1)?;
        let positions: Vec<_> = (0..length).map(|index| [index as u32; 3]).collect();
        self.forward_embeds(&self.embed(ids)?, &positions)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn supports_packed(&self) -> bool {
        self.layers.iter().all(|layer| match &layer.mixer {
            Mixer::Full(full) => matches!(
                full.implementation,
                AttentionImplementation::FlashAttention2
                    | AttentionImplementation::FlashAttentionBlackwell
            ),
            Mixer::Linear(linear) => linear.key_dim == 128 && linear.value_dim == 128,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn forward_packed(&self, ids: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        if !self.supports_packed() || crate::numerics::enabled() {
            candle_core::bail!(
                "packed decoder requires supported Flash attention outside capture mode"
            )
        }
        let sequences = PackedSequences::new(lengths, ids.device())?;
        if ids.dims() != [1, sequences.total] {
            candle_core::bail!("packed ID shape mismatch")
        }
        let positions = lengths
            .iter()
            .flat_map(|&length| (0..length).map(|i| [i as u32; 3]))
            .collect::<Vec<_>>();
        let embeds = self.embed(ids)?;
        if embeds.dtype() != DType::BF16 {
            candle_core::bail!("packed decoder requires BF16")
        }
        let rotary = self
            .layers
            .iter()
            .find_map(|layer| match &layer.mixer {
                Mixer::Full(full) => Some(&full.rotary),
                Mixer::Linear(_) => None,
            })
            .map(|rotary| rotary.prepare(&positions, embeds.dtype(), embeds.device()))
            .transpose()?;
        event_profile::begin(embeds.device())?;
        let output = self.forward_deferred_fork(&embeds, rotary.as_ref(), Some(&sequences))?;
        event_profile::finish(1, sequences.total)?;
        Ok(output)
    }

    pub fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        if self.embedding.embeddings().device().is_cpu() && ids.device().is_cuda() {
            let cpu_ids = ids.to_device(&candle_core::Device::Cpu)?;
            return cpu_ids.apply(&self.embedding)?.to_device(ids.device());
        }
        ids.apply(&self.embedding)
    }

    pub fn forward_embeds(&self, embeds: &Tensor, positions: &[[u32; 3]]) -> Result<Tensor> {
        if positions.len() != embeds.dim(1)? {
            candle_core::bail!("Qwen3.5 position count does not match sequence length")
        }
        let rotary = self
            .layers
            .iter()
            .find_map(|layer| match &layer.mixer {
                Mixer::Full(attention) => Some(&attention.rotary),
                Mixer::Linear(_) => None,
            })
            .map(|rotary| rotary.prepare(positions, embeds.dtype(), embeds.device()))
            .transpose()?;
        #[cfg(feature = "cuda")]
        event_profile::begin(embeds.device())?;
        crate::numerics::capture("embedding", embeds)?;
        let output = self.forward_layers_fork(embeds, rotary.as_ref())?;
        crate::numerics::capture("decoder-output", &output)?;
        #[cfg(feature = "cuda")]
        {
            let (batch, length, _) = embeds.dims3()?;
            event_profile::finish(batch, length)?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guarded_double_radius_coefficient_bounds_exact_rational_values() {
        for (count, nearest_exact) in [
            (1, 4.884981811276547e-15_f64),
            (80, 2.2747687422185558e-11),
            (288, 2.9471959623355573e-10),
            (576, 1.178851243061726e-9),
            (4_194_303, 0.11111104093424891),
        ] {
            let coefficient = guarded_double_radius_coefficient(count * 32);
            assert!(
                coefficient >= nearest_exact.next_up(),
                "must exceed the exact rational bound"
            );
            assert!(
                coefficient <= nearest_exact * (1. + 64. * f64::EPSILON),
                "rounding outward must not replace the derived bound with an arbitrary large radius"
            );
            assert_eq!(
                coefficient,
                guarded_double_radius_coefficient(count * 32 - 31),
                "same ceil(K/32) bound for the tail"
            );
        }
        assert!(guarded_double_radius_coefficient(0).is_infinite());
        assert!(guarded_double_radius_coefficient(4_194_304 * 32).is_infinite());
        assert!(guarded_double_radius_coefficient(usize::MAX).is_infinite());
    }

    #[test]
    fn guarded_tensor_double_radius_bounds_exact_rational_values() {
        for (count, nearest_exact) in [
            (1, 7.771561675301967e-15_f64),
            (80, 2.3022467620780365e-11),
            (288, 2.957102482384298e-10),
            (576, 1.1808331021829885e-9),
        ] {
            let coefficient = guarded_double_radius_coefficient_impl(count * 32, true);
            assert!(coefficient >= nearest_exact.next_up());
            assert!(coefficient <= nearest_exact * (1. + 64. * f64::EPSILON));
            assert!(coefficient >= guarded_double_radius_coefficient(count * 32));
        }
        assert!(guarded_double_radius_coefficient_impl(0, true).is_infinite());
        assert!(guarded_double_radius_coefficient_impl(usize::MAX, true).is_infinite());
    }

    #[test]
    fn projection_geometry_restricts_larger_tiles_to_measured_architecture() {
        for sm120 in [false, true] {
            assert_eq!(projection_tile_rows(4096, 2047, sm120), 16);
            assert_eq!(projection_tile_rows(63, 2048, sm120), 16);
            assert_eq!(projection_tile_rows(64, 2048, sm120), 32);
            assert_eq!(projection_tile_rows(127, 2048, sm120), 32);
        }
        assert_eq!(projection_tile_rows(128, 2048, true), 128);
        assert_eq!(projection_tile_rows(129, 2049, true), 128);
        assert_eq!(projection_tile_rows(128, 2048, false), 32);
        assert_eq!(projection_tile_rows(4096, 12288, false), 32);
    }

    #[test]
    fn blocked_projection_preserves_rows_storage_and_tail_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let weights = Tensor::from_vec(
            (0..15).map(|i| (i % 7) as f32 * 0.125).collect::<Vec<_>>(),
            (3, 5),
            &device,
        )?;
        let linear = Linear::new(weights, None);
        for (batch, length) in [(1, 1), (1, 16), (1, 17), (2, 19)] {
            let input = Tensor::from_vec(
                (0..batch * length * 5)
                    .map(|i| (i % 8) as f32 * 0.25)
                    .collect::<Vec<_>>(),
                (batch, length, 5),
                &device,
            )?;
            let actual = blocked_projection(&input, &linear, 16)?;
            let expected = input.apply(&linear)?;
            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(actual.dtype(), input.dtype());
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
            assert!(blocked_projection(&input, &linear, 0).is_err());
        }
        Ok(())
    }

    #[test]
    fn full_attention_kv_fixed_reduction_preserves_storage_and_cast_boundary() -> Result<()> {
        let devices = vec![candle_core::Device::Cpu];
        #[cfg(feature = "cuda")]
        let devices = {
            let mut devices = devices;
            devices.push(test_cuda_device()?);
            devices
        };
        for device in devices {
            for dtype in supported_rotary_test_dtypes(&device)? {
                for (batch, length, inner, columns) in [
                    (1, 2, 33, 9),
                    (2, 9, 129, 17),
                    (1, 33, 65, 65),
                    (2, 1, 4096, 1024),
                    (1, 2, 6144, 1024),
                    (1, 2, 12288, 65),
                    (1, 2, 18432, 65),
                    (1, 33, 257, 2049),
                    (1, 65, 65, 2049),
                    (2, 33, 129, 2049),
                    (1, 129, 65, 2049),
                    (1, 129, 257, 2049),
                ] {
                    let rows = batch * length;
                    let xs: Vec<f32> = (0..rows * inner)
                        .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                        .collect();
                    let ws: Vec<f32> = (0..columns * inner)
                        .map(|i| ((i * 29 % 127) as f32 - 63.) / 256.)
                        .collect();
                    let weights =
                        Tensor::from_vec(ws.clone(), (columns, inner), &device)?.to_dtype(dtype)?;
                    let original = Linear::new(weights.clone(), None);
                    let layer = Linear::new(weights, None);
                    assert_eq!(layer.weight().dtype(), dtype);
                    let x = Tensor::from_vec(xs.clone(), (batch, length, inner), &device)?
                        .to_dtype(dtype)?;
                    #[cfg(target_os = "macos")]
                    if device.is_cpu() && dtype == DType::F16 {
                        assert!(full_attention_kv_forward(&x, &layer).is_err());
                        assert!(linear_attention_projection_forward(&x, &layer).is_err());
                        continue;
                    }
                    let projected = full_attention_kv_forward(&x, &layer)?;
                    let linear_qkv = linear_attention_projection_forward(&x, &layer)?;
                    assert_eq!(
                        projected
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        linear_qkv
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        "QKV candidate preserves projection dtype and cast boundary"
                    );
                    assert_eq!(projected.dtype(), dtype, "round back BEFORE attention/norm");
                    let actual = projected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    if device.is_cuda() && dtype == DType::BF16 {
                        let repeated = full_attention_kv_forward(&x, &layer)?
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        assert_eq!(
                            actual, repeated,
                            "actual repeated fixed-reduction K/V projection"
                        );
                        #[cfg(feature = "cuda")]
                        {
                            for shaped in [x.clone(), x.reshape((rows, inner))?] {
                                let head = fixed_bf16_projection(&shaped, &layer)?;
                                assert_eq!(head.dtype(), dtype);
                                assert_eq!(head.dim(D::Minus1)?, columns);
                                assert_eq!(
                                    actual,
                                    head.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                                    "rank-two/three head adapter preserves validated projection"
                                );
                            }
                        }
                    }
                    if dtype != DType::F32 && !(device.is_cuda() && dtype == DType::BF16) {
                        let retained = if device.is_cpu() && dtype == DType::BF16 {
                            x.to_dtype(DType::F32)?
                                .apply(&Linear::new(original.weight().to_dtype(DType::F32)?, None))?
                                .to_dtype(dtype)?
                        } else {
                            x.apply(&original)?
                        }
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                        assert_eq!(actual, retained, "unchanged backend/dtype path");
                        continue;
                    }
                    let expected: Vec<f32> = (0..rows)
                        .flat_map(|row| {
                            let xs = &xs;
                            let ws = &ws;
                            (0..columns).map(move |column| {
                                (0..inner)
                                    .map(|k| {
                                        f64::from(xs[row * inner + k])
                                            * f64::from(ws[column * inner + k])
                                    })
                                    .sum::<f64>() as f32
                            })
                        })
                        .collect();
                    #[cfg(feature = "cuda")]
                    if device.is_cuda() && dtype == DType::BF16 {
                        let biases: Vec<f32> = (0..columns)
                            .map(|column| ((column % 17) as f32 - 8.) / 256.)
                            .collect();
                        let biased = Linear::new(
                            layer.weight().clone(),
                            Some(
                                Tensor::from_vec(biases.clone(), columns, &device)?
                                    .to_dtype(dtype)?,
                            ),
                        );
                        let expected_biased: Vec<f32> = expected
                            .iter()
                            .enumerate()
                            .map(|(i, value)| *value + biases[i % columns])
                            .collect();
                        let expected_biased = Tensor::from_vec(
                            expected_biased,
                            actual.len(),
                            &candle_core::Device::Cpu,
                        )?
                        .to_dtype(dtype)?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?;
                        for shaped in [x.clone(), x.reshape((rows, inner))?] {
                            let output = fixed_bf16_projection(&shaped, &biased)?;
                            assert_eq!(output.dim(D::Minus1)?, columns);
                            assert_eq!(
                                output
                                    .to_dtype(DType::F32)?
                                    .flatten_all()?
                                    .to_vec1::<f32>()?,
                                expected_biased,
                                "biased rank-two/three projection adds FP32 bias before sole BF16 cast"
                            );
                        }
                    }
                    let expected =
                        Tensor::from_vec(expected, actual.len(), &candle_core::Device::Cpu)?
                            .to_dtype(dtype)?
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?;
                    assert_eq!(
                        actual, expected,
                        "exact dyadic reference {dtype:?}/{inner}/{columns}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn mlp_diagnostic_boundaries_preserve_original_expression() -> Result<()> {
        let device = candle_core::Device::Cpu;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let weights = |output, input| -> Result<Linear> {
                let values: Vec<f32> = (0..output * input)
                    .map(|i| ((i * 13 % 31) as f32 - 15.) / 128.)
                    .collect();
                Ok(Linear::new(
                    Tensor::from_vec(values, (output, input), &device)?.to_dtype(dtype)?,
                    None,
                ))
            };
            let layer = Mlp {
                #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
                direct_weight: None,
                #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
                fused_weight: Default::default(),
                gate: weights(17, 33)?,
                up: weights(17, 33)?,
                down: weights(9, 17)?,
            };
            let values: Vec<f32> = (0..2 * 3 * 33)
                .map(|i| ((i * 7 % 29) as f32 - 14.) / 64.)
                .collect();
            let x = Tensor::from_vec(values, (2, 3, 33), &device)?.to_dtype(dtype)?;
            #[cfg(target_os = "macos")]
            if dtype == DType::F16 {
                assert!(layer.forward(&x, 0).is_err());
                continue;
            }
            let reference = |input: &Tensor, linear: &Linear| -> Result<Tensor> {
                if dtype == DType::BF16 {
                    input
                        .to_dtype(DType::F32)?
                        .apply(&Linear::new(linear.weight().to_dtype(DType::F32)?, None))?
                        .to_dtype(dtype)
                } else {
                    input.apply(linear)
                }
            };
            let original = reference(
                &(reference(&x, &layer.gate)?.silu()? * reference(&x, &layer.up)?)?,
                &layer.down,
            )?;
            let actual = layer.forward(&x, 0)?;
            assert_eq!(actual.dtype(), dtype);
            let values = |x: &Tensor| x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
            assert_eq!(
                values(&actual)?,
                values(&original)?,
                "diagnostic-only MLP ordering {dtype:?}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn test_cuda_device() -> Result<candle_core::Device> {
        crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn stable_gate_projection_cuda_matches_fp64_cast_bitwise() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        for (batch, length, inner, columns) in [
            (1, 3, 33, 9),
            (2, 5, 129, 17),
            (1, 17, 4096, 32),
            (1, 2, 6144, 64),
        ] {
            let xs: Vec<f32> = (0..batch * length * inner)
                .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                .collect();
            let ws: Vec<f32> = (0..columns * inner)
                .map(|i| ((i * 29 % 127) as f32 - 63.) / 512.)
                .collect();
            let x = Tensor::from_vec(xs.clone(), (batch, length, inner), &device)?
                .to_dtype(DType::BF16)?;
            let w =
                Tensor::from_vec(ws.clone(), (columns, inner), &device)?.to_dtype(DType::BF16)?;
            let linear = Linear::new(w, None);
            let actual = cuda_gate_projection_bf16(&x, &linear)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let repeated = cuda_gate_projection_bf16(&x, &linear)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut expected = Vec::with_capacity(actual.len());
            for row in 0..batch * length {
                for column in 0..columns {
                    let sum: f64 = (0..inner)
                        .map(|k| f64::from(xs[row * inner + k]) * f64::from(ws[column * inner + k]))
                        .sum();
                    expected.push(bf16::from_f32(sum as f32).to_f32());
                }
            }
            assert_eq!(
                actual, expected,
                "fixed dyadic FP64 reference {inner}/{columns}"
            );
            assert_eq!(actual, repeated, "actual repeated CUDA gate projection");
            let biased = Linear::new(
                linear.weight().clone(),
                Some(Tensor::zeros(columns, DType::BF16, &device)?),
            );
            let with_zero_bias = cuda_gate_projection_bf16(&x, &biased)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, with_zero_bias, "zero-bias compensated projection");
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn compensated_head_projection_cuda_matches_bias_cast_and_storage_offsets() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        let bits = |tensor: &Tensor| -> Result<Vec<f32>> {
            tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
        };
        for (rows, inner, columns) in [(3, 33, 9), (5, 129, 17), (2, 2048, 64)] {
            let xs: Vec<f32> = (0..(rows + 1) * inner)
                .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                .collect();
            let ws: Vec<f32> = (0..(columns + 1) * inner)
                .map(|i| ((i * 29 % 127) as f32 - 63.) / 512.)
                .collect();
            let bs: Vec<f32> = (0..columns + 1).map(|i| (i as f32 - 8.) / 128.).collect();
            let x = Tensor::from_vec(xs.clone(), (rows + 1, inner), &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, rows)?;
            let w = Tensor::from_vec(ws.clone(), (columns + 1, inner), &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, columns)?;
            let b = Tensor::from_vec(bs.clone(), columns + 1, &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, columns)?;
            let layer = Linear::new(w, Some(b));
            let actual = compensated_bf16_projection(&x, &layer)?;
            assert_eq!(actual.dims(), [rows, columns]);
            let mut expected = Vec::with_capacity(rows * columns);
            for row in 0..rows {
                for column in 0..columns {
                    let sum: f64 = (0..inner)
                        .map(|k| {
                            f64::from(xs[(row + 1) * inner + k])
                                * f64::from(ws[(column + 1) * inner + k])
                        })
                        .sum();
                    expected.push(bf16::from_f32(sum as f32 + bs[column + 1]).to_f32());
                }
            }
            assert_eq!(bits(&actual)?, expected, "FP64 dot / FP32 bias / BF16 cast");
            assert_eq!(bits(&compensated_bf16_projection(&x, &layer)?)?, expected);
            let batch = Tensor::stack(&[&x, &x], 0)?;
            let batch_actual = compensated_bf16_projection(&batch, &layer)?;
            assert_eq!(batch_actual.dims(), [2, rows, columns]);
            assert_eq!(
                bits(&batch_actual)?,
                [expected.clone(), expected.clone()].concat()
            );
            let padded = Tensor::cat(&[&x, &Tensor::zeros((19, inner), DType::BF16, &device)?], 0)?;
            assert_eq!(
                bits(&compensated_bf16_projection(&padded, &layer)?.narrow(0, 0, rows)?)?,
                expected
            );
        }

        let x = Tensor::ones((1, 1), DType::BF16, &device)?;
        let layer = Linear::new(
            Tensor::from_vec(vec![1_f32, -1_f32], (2, 1), &device)?.to_dtype(DType::BF16)?,
            Some(
                Tensor::from_vec(vec![1_f32 / 256., -1_f32 / 256.], 2, &device)?
                    .to_dtype(DType::BF16)?,
            ),
        );
        assert_eq!(
            bits(&compensated_bf16_projection(&x, &layer)?)?,
            vec![1., -1.]
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn stable_gate_projection_cuda_preserves_cancellation_and_exceptional_products() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        let mut weights = vec![2_f32.powi(-16); 129];
        weights[0] = 1024.;
        weights[128] = -1024.;
        let cases = [
            (vec![1.; 129], weights),
            (
                vec![2_f32.powi(100); 2],
                vec![2_f32.powi(100), -2_f32.powi(100)],
            ),
            (vec![2_f32.powi(-100); 2], vec![2_f32.powi(-100); 2]),
        ];
        for (xs, ws) in cases {
            let expected = bf16::from_f32(
                xs.iter()
                    .zip(&ws)
                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                    .sum::<f64>() as f32,
            )
            .to_f32();
            let x =
                Tensor::from_vec(xs.clone(), (1, 1, xs.len()), &device)?.to_dtype(DType::BF16)?;
            let linear = Linear::new(
                Tensor::from_vec(ws, (1, xs.len()), &device)?.to_dtype(DType::BF16)?,
                None,
            );
            let actual = cuda_gate_projection_bf16(&x, &linear)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(
                actual,
                vec![expected],
                "cancellation/exceptional product fixture"
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn attention_math_cuda_probe_records_exact_inputs_and_outputs() -> Result<()> {
        let device = test_cuda_device()?;

        let count = 4096usize;
        let values: Vec<f32> = (0..count)
            .map(|i| -80.0 + (i as f32) * 11.0 / 512.0 + ((i * 37 % 127) as f32) / 8192.0)
            .collect();
        let input = Tensor::from_vec(values.clone(), count, &device)?;
        let (storage, _) = input.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            candle_core::bail!("attention math probe requires CUDA")
        };
        let cuda = storage.device().clone();
        let x = storage.as_cuda_slice::<f32>()?;
        let output = unsafe { cuda.alloc::<f32>(count * 6)? };
        let function = cuda.get_or_load_custom_func(
            "qwen35_attention_math_probe_f32",
            "sys1_qwen35",
            CUDA_PTX,
        )?;
        let mut launch = function.builder();
        launch.arg(x);
        launch.arg(&output);
        let count_arg = count as i32;
        launch.arg(&count_arg);
        unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
        let result = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, cuda)),
            (count, 6),
            BackpropOp::none(),
            false,
        )
        .flatten_all()?
        .to_vec1::<f32>()?;
        assert_eq!(result.len(), count * 6);
        assert!(result.iter().all(|x| x.is_finite() && *x > 0.0));

        for (index, (&x, row)) in values.iter().zip(result.chunks_exact(6)).enumerate() {
            eprintln!(
                "ATTENTION_MATH_PROBE index={index} input={:08x} outputs={:08x},{:08x},{:08x},{:08x},{:08x},{:08x} scope=primitive_only_not_model_approval",
                x.to_bits(),
                row[0].to_bits(),
                row[1].to_bits(),
                row[2].to_bits(),
                row[3].to_bits(),
                row[4].to_bits(),
                row[5].to_bits()
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn attention_mma_cuda_probe_records_exact_bf16_operands_and_f32_outputs() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        let mut state = 0x39a71f05u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            bf16::from_bits(
                ((state >> 31) as u16) << 15
                    | ((121 + ((state >> 8) % 12)) as u16) << 7
                    | (state & 127) as u16,
            )
        };
        let xs: Vec<bf16> = (0..4096).map(|_| next()).collect();
        let ws: Vec<bf16> = (0..4096).map(|_| next()).collect();
        let input = Tensor::from_vec(xs.clone(), (16, 256), &device)?;
        let weight = Tensor::from_vec(ws.clone(), (16, 256), &device)?;
        let (x_storage, _) = input.storage_and_layout();
        let (w_storage, _) = weight.storage_and_layout();
        let (Storage::Cuda(x_storage), Storage::Cuda(w_storage)) = (&*x_storage, &*w_storage)
        else {
            candle_core::bail!("attention MMA probe requires CUDA")
        };
        let cuda = x_storage.device().clone();
        let x = x_storage.as_cuda_slice::<bf16>()?;
        let w = w_storage.as_cuda_slice::<bf16>()?;
        let output = unsafe { cuda.alloc::<f32>(768)? };
        let function = cuda.get_or_load_custom_func(
            "qwen35_attention_mma_probe_bf16",
            "sys1_qwen35",
            CUDA_PTX,
        )?;
        let mut launch = function.builder();
        launch.arg(x);
        launch.arg(w);
        launch.arg(&output);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        let result = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, cuda)),
            (3, 256),
            BackpropOp::none(),
            false,
        )
        .flatten_all()?
        .to_vec1::<f32>()?;
        assert!(result.iter().all(|x| x.is_finite()));
        for index in 0..256 {
            let row = index / 16;
            let column = index % 16;

            let reference: f64 = (0..256)
                .map(|k| {
                    f64::from(xs[row * 256 + k].to_f32()) * f64::from(ws[column * 256 + k].to_f32())
                })
                .sum();
            assert_eq!(result[512 + index].to_bits(), (reference as f32).to_bits());
            eprintln!(
                "ATTENTION_MMA_PROBE index={index} outputs={:08x},{:08x},{:08x} scope=primitive_only_not_model_approval",
                result[index].to_bits(),
                result[256 + index].to_bits(),
                result[512 + index].to_bits()
            );
        }
        for (index, (&x, &w)) in xs.iter().zip(&ws).enumerate() {
            eprintln!(
                "ATTENTION_MMA_INPUT index={index} x={:04x} w={:04x}",
                x.to_bits(),
                w.to_bits()
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn guarded_tensor_double_cuda_matches_compensated_for_normal_tile_tails() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        for (rows, inner, columns) in [(3, 7, 9), (17, 65, 33), (2, 513, 17)] {
            let x = Tensor::from_vec(
                (0..rows * inner)
                    .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                    .collect::<Vec<_>>(),
                (1, rows, inner),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let linear = Linear::new(
                Tensor::from_vec(
                    (0..columns * inner)
                        .map(|i| ((i * 29 % 127) as f32 - 63.) / 512.)
                        .collect::<Vec<_>>(),
                    (columns, inner),
                    &device,
                )?
                .to_dtype(DType::BF16)?,
                None,
            );
            let bits = |mode| -> Result<Vec<u32>> {
                Ok(cuda_bf16_projection_impl(&x, &linear, false, mode)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(
                bits(SimtProjection::Compensated)?,
                bits(SimtProjection::GuardedTensorDouble)?
            );
            assert_eq!(
                bits(SimtProjection::Compensated)?,
                bits(SimtProjection::GuardedTensorDoubleRows16)?
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn guarded_double_projection_cuda_matches_compensated_bits_across_tails() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        let mut state = 0x6a09_e667_u32;
        let mut values = |count| {
            (0..count)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let sign = (state >> 16) as u16 & 0x8000;
                    let exponent = ((state >> 24) % 128 + 63) as u16;
                    bf16::from_bits(sign | (exponent << 7) | ((state >> 17) as u16 & 127)).to_f32()
                })
                .collect::<Vec<_>>()
        };
        for (rows, inner, columns) in [
            (1, 1, 1),
            (3, 33, 9),
            (2, 65, 17),
            (17, 129, 65),
            (2, 4096, 65),
            (1, 18432, 9),
        ] {
            let x = Tensor::from_vec(values(rows * inner), (1, rows, inner), &device)?
                .to_dtype(DType::BF16)?;
            let linear = Linear::new(
                Tensor::from_vec(values(columns * inner), (columns, inner), &device)?
                    .to_dtype(DType::BF16)?,
                None,
            );
            let bits = |variant| -> Result<Vec<u32>> {
                Ok(cuda_bf16_projection_impl(&x, &linear, false, variant)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            for candidate in [
                SimtProjection::TiledCompensated,
                SimtProjection::GuardedDouble,
                SimtProjection::GuardedTensorDouble,
                SimtProjection::GuardedTensorDoubleRows16,
            ] {
                assert_eq!(
                    bits(SimtProjection::Compensated)?,
                    bits(candidate)?,
                    "guarded double must preserve exact original bits {rows}/{inner}/{columns}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn guarded_double_projection_cuda_preserves_midpoints_cancellation_and_exceptional_bits()
    -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        let mut cancellation = vec![0.; 65];
        cancellation[0] = 2_f32.powi(60);
        cancellation[32] = 1.;
        cancellation[64] = -2_f32.powi(60);
        let cases = [
            (vec![1.; 65], cancellation, Some(1_f32.to_bits())),
            (vec![1.; 3], vec![1., 2_f32.powi(-8), 2_f32.powi(-20)], None),
            (
                vec![1.; 3],
                vec![1., 2_f32.powi(-8), -2_f32.powi(-20)],
                None,
            ),
            (
                vec![2_f32.powi(100); 2],
                vec![2_f32.powi(100), -2_f32.powi(100)],
                None,
            ),
            (vec![2_f32.powi(-100); 2], vec![2_f32.powi(-100); 2], None),
            (vec![-0.; 33], vec![1.; 33], None),
            (vec![f32::INFINITY; 2], vec![1., -1.], None),
        ];
        for (xs, ws, exact) in cases {
            let inner = xs.len();
            let x = Tensor::from_vec(xs, (1, 1, inner), &device)?.to_dtype(DType::BF16)?;
            let linear = Linear::new(
                Tensor::from_vec(ws, (1, inner), &device)?.to_dtype(DType::BF16)?,
                None,
            );
            let bits = |variant| -> Result<u32> {
                Ok(cuda_bf16_projection_impl(&x, &linear, false, variant)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?[0]
                    .to_bits())
            };
            let retained = bits(SimtProjection::Compensated)?;
            assert_eq!(
                retained,
                bits(SimtProjection::TiledCompensated)?,
                "tiled compensated exceptional bits"
            );
            assert_eq!(
                retained,
                bits(SimtProjection::GuardedDouble)?,
                "including NaN/signed-zero bit patterns"
            );
            assert_eq!(
                retained,
                bits(SimtProjection::GuardedTensorDouble)?,
                "tensor-double exceptional bits"
            );
            assert_eq!(
                retained,
                bits(SimtProjection::GuardedTensorDoubleRows16)?,
                "rows16 tensor-double exceptional bits"
            );
            if let Some(expected) = exact {
                assert_eq!(
                    retained, expected,
                    "plain FP64 loses this cancellation residual; guard must fall back"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tiled_compensated_projection_cuda_preserves_bias_offsets_batch_and_padding() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        let bits = |tensor: &Tensor| -> Result<Vec<u32>> {
            Ok(tensor
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        for (rows, inner, columns) in [(1, 1, 2), (3, 33, 9), (5, 129, 17), (2, 2048, 64)] {
            let xs: Vec<f32> = (0..(rows + 1) * inner)
                .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                .collect();
            let ws: Vec<f32> = (0..(columns + 1) * inner)
                .map(|i| ((i * 29 % 127) as f32 - 63.) / 512.)
                .collect();
            let bs: Vec<f32> = (0..columns + 1).map(|i| (i as f32 - 8.) / 128.).collect();
            let x = Tensor::from_vec(xs, (rows + 1, inner), &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, rows)?
                .unsqueeze(0)?;
            let w = Tensor::from_vec(ws, (columns + 1, inner), &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, columns)?;
            let b = Tensor::from_vec(bs, columns + 1, &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, columns)?;
            let layer = Linear::new(w, Some(b));
            let run = |input, mode| cuda_bf16_projection_impl(input, &layer, false, mode);
            let expected = bits(&run(&x, SimtProjection::Compensated)?)?;
            assert_eq!(bits(&run(&x, SimtProjection::TiledCompensated)?)?, expected);
            assert_eq!(bits(&run(&x, SimtProjection::TiledCompensated)?)?, expected);
            let batch = Tensor::cat(&[&x, &x], 0)?;
            assert_eq!(
                bits(&run(&batch, SimtProjection::TiledCompensated)?)?,
                [expected.clone(), expected.clone()].concat()
            );
            let padded = Tensor::cat(
                &[&x, &Tensor::zeros((1, 19, inner), DType::BF16, &device)?],
                1,
            )?;
            assert_eq!(
                bits(&run(&padded, SimtProjection::TiledCompensated)?.narrow(1, 0, rows)?)?,
                expected
            );
        }

        let x = Tensor::ones((1, 1, 1), DType::BF16, &device)?;
        let layer = Linear::new(
            Tensor::from_vec(vec![1_f32, -1_f32], (2, 1), &device)?.to_dtype(DType::BF16)?,
            Some(
                Tensor::from_vec(vec![1_f32 / 256., -1_f32 / 256.], 2, &device)?
                    .to_dtype(DType::BF16)?,
            ),
        );
        assert_eq!(
            bits(&cuda_bf16_projection_impl(
                &x,
                &layer,
                false,
                SimtProjection::TiledCompensated
            )?)?,
            vec![1_f32.to_bits(), (-1_f32).to_bits()]
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn pinned_blaslt_vs_compensated_cuda_microprofile() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }

        for (rows, inner, columns) in [
            (128, 2560, 4096),
            (512, 2560, 4096),
            (128, 2560, 9216),
            (128, 9216, 2560),
        ] {
            let x = Tensor::from_vec(
                (0..rows * inner)
                    .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                    .collect::<Vec<_>>(),
                (1, rows, inner),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let linear = Linear::new(
                Tensor::from_vec(
                    (0..columns * inner)
                        .map(|i| ((i * 29 % 127) as f32 - 63.) / 512.)
                        .collect::<Vec<_>>(),
                    (columns, inner),
                    &device,
                )?
                .to_dtype(DType::BF16)?,
                None,
            );
            let run = |index| match index {
                0 => crate::kernels::blaslt::project_pinned512(&x, &linear),
                1 => cuda_bf16_projection_impl(&x, &linear, false, SimtProjection::Compensated),
                _ => {
                    cuda_bf16_projection_impl(&x, &linear, false, SimtProjection::TiledCompensated)
                }
            };
            let baseline = run(0)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for candidate in [1, 2] {
                let values = run(candidate)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let different = baseline
                    .iter()
                    .zip(&values)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let max_abs = baseline
                    .iter()
                    .zip(&values)
                    .map(|(a, b)| (*a - *b).abs())
                    .fold(0f32, f32::max);
                for _ in 0..2 {
                    let _ = run(0)?;
                    let _ = run(candidate)?;
                }
                let mut samples = [Vec::new(), Vec::new()];
                for iteration in 0..6 {
                    for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                        device.synchronize()?;
                        let started = Instant::now();
                        let _ = run(if index == 0 { 0 } else { candidate })?;
                        device.synchronize()?;
                        samples[index].push(started.elapsed().as_secs_f64() * 1000.);
                    }
                }
                let median = |values: &mut Vec<f64>| {
                    values.sort_by(f64::total_cmp);
                    (values[2] + values[3]) / 2.
                };
                let old_ms = median(&mut samples[0]);
                let candidate_ms = median(&mut samples[1]);
                println!(
                    "PINNED_BLASLT_PROJECTION_PROFILE rows={rows} inner={inner} columns={columns} candidate={candidate} old_ms={old_ms:.9} candidate_ms={candidate_ms:.9} ratio={:.9} different_elements={different} max_abs={max_abs} scope=seeded_operator_allocation_launch_and_barriers_not_http",
                    candidate_ms / old_ms
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn pinned_blaslt_combined_gate_up_cuda_microprofile() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }

        for (rows, inner, columns) in [
            (32, 4096, 12288),
            (128, 4096, 12288),
            (512, 4096, 12288),
            (128, 5120, 17408),
        ] {
            let tensor = |shape: &[usize], seed: u32| -> Result<Tensor> {
                let mut state = seed;
                let values = (0..shape.iter().product::<usize>())
                    .map(|_| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        ((state >> 16) as f32 / 65536. - 0.5) / 8.
                    })
                    .collect::<Vec<_>>();
                Tensor::from_vec(values, shape, &device)?.to_dtype(DType::BF16)
            };
            let x = tensor(&[1, rows, inner], 123)?;
            let gate = Linear::new(tensor(&[columns, inner], 456)?, None);
            let up = Linear::new(tensor(&[columns, inner], 789)?, None);
            let packed = Linear::new(Tensor::cat(&[gate.weight(), up.weight()], 0)?, None);
            let run = |combined: bool| -> Result<(Tensor, Tensor)> {
                if combined {
                    let output = crate::kernels::blaslt::project_pinned512(&x, &packed)?;
                    Ok((
                        output.narrow(2, 0, columns)?.contiguous()?,
                        output.narrow(2, columns, columns)?.contiguous()?,
                    ))
                } else {
                    Ok((
                        crate::kernels::blaslt::project_pinned512(&x, &gate)?,
                        crate::kernels::blaslt::project_pinned512(&x, &up)?,
                    ))
                }
            };
            let host = |pair: (Tensor, Tensor)| -> Result<Vec<f32>> {
                Tensor::cat(&[&pair.0, &pair.1], 2)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1()
            };
            let baseline = host(run(false)?)?;
            let candidate = host(run(true)?)?;
            let different = baseline
                .iter()
                .zip(&candidate)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let max_abs = baseline
                .iter()
                .zip(&candidate)
                .map(|(a, b)| (*a - *b).abs())
                .fold(0f32, f32::max);
            for _ in 0..2 {
                let _ = run(false)?;
                let _ = run(true)?;
            }
            let mut samples = [Vec::new(), Vec::new()];
            for iteration in 0..6 {
                for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                    device.synchronize()?;
                    let started = Instant::now();
                    let _ = run(index == 1)?;
                    device.synchronize()?;
                    samples[index].push(started.elapsed().as_secs_f64() * 1000.);
                }
            }
            let median = |values: &mut Vec<f64>| {
                values.sort_by(f64::total_cmp);
                (values[2] + values[3]) / 2.
            };
            let old_ms = median(&mut samples[0]);
            let candidate_ms = median(&mut samples[1]);
            println!(
                "PINNED_BLASLT_GATE_UP_PROFILE rows={rows} inner={inner} columns={columns} old_ms={old_ms:.9} candidate_ms={candidate_ms:.9} ratio={:.9} different_elements={different} max_abs={max_abs} scope=seeded_operator_including_output_copies_not_http",
                candidate_ms / old_ms
            );
            assert_eq!(
                different, 0,
                "packing changes seeded gate/up projection bits"
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn guarded_double_projection_cuda_paired_synchronized_microprofile() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        for (rows, inner, columns) in [
            (17, 4096, 32),
            (128, 2560, 4096),
            (512, 2560, 4096),
            (128, 2560, 9216),
            (128, 9216, 2560),
        ] {
            let xs = (0..rows * inner)
                .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                .collect::<Vec<_>>();
            let ws = (0..columns * inner)
                .map(|i| ((i * 29 % 127) as f32 - 63.) / 512.)
                .collect::<Vec<_>>();
            let x = Tensor::from_vec(xs, (1, rows, inner), &device)?.to_dtype(DType::BF16)?;
            let linear = Linear::new(
                Tensor::from_vec(ws, (columns, inner), &device)?.to_dtype(DType::BF16)?,
                None,
            );
            let run = |variant| cuda_bf16_projection_impl(&x, &linear, false, variant);
            for candidate in [
                SimtProjection::TiledCompensated,
                SimtProjection::GuardedDouble,
                SimtProjection::GuardedTensorDouble,
                SimtProjection::GuardedTensorDoubleRows16,
            ] {
                let retained = run(SimtProjection::Compensated)?;
                let guarded = run(candidate)?;
                assert_eq!(
                    retained
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    guarded
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                );
                for _ in 0..2 {
                    let _ = run(SimtProjection::Compensated)?;
                    let _ = run(candidate)?;
                }
                device.synchronize()?;
                let mut samples = [Vec::new(), Vec::new()];
                for iteration in 0..6 {
                    for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                        device.synchronize()?;
                        let started = Instant::now();
                        let _ = run(if index == 0 {
                            SimtProjection::Compensated
                        } else {
                            candidate
                        })?;
                        device.synchronize()?;
                        samples[index].push(started.elapsed().as_secs_f64() * 1000.);
                    }
                }
                let median = |values: &mut Vec<f64>| {
                    values.sort_by(f64::total_cmp);
                    (values[2] + values[3]) / 2.
                };
                let old_ms = median(&mut samples[0]);
                let candidate_ms = median(&mut samples[1]);
                let profile_kind = if matches!(candidate, SimtProjection::TiledCompensated) {
                    "TILED_COMPENSATED_PROFILE"
                } else if matches!(candidate, SimtProjection::GuardedTensorDoubleRows16) {
                    "GUARDED_TENSOR_DOUBLE_ROWS16_PROFILE"
                } else if matches!(candidate, SimtProjection::GuardedTensorDouble) {
                    "GUARDED_TENSOR_DOUBLE_PROFILE"
                } else {
                    "GUARDED_DOUBLE_PROFILE"
                };
                println!(
                    "{profile_kind} rows={rows} inner={inner} columns={columns} old_ms={old_ms:.9} candidate_ms={candidate_ms:.9} ratio={:.9} scope=allocation_launch_and_barriers_not_http",
                    candidate_ms / old_ms
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn full_attention_kv_cuda_preserves_exceptional_products() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        for (xs, ws) in [
            (
                vec![2_f32.powi(100); 2],
                vec![2_f32.powi(100), -2_f32.powi(100)],
            ),
            (vec![2_f32.powi(-70); 129], vec![2_f32.powi(-70); 129]),
        ] {
            let expected = bf16::from_f32(
                xs.iter()
                    .zip(&ws)
                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                    .sum::<f64>() as f32,
            )
            .to_f32();
            let x =
                Tensor::from_vec(xs.clone(), (1, 1, xs.len()), &device)?.to_dtype(DType::BF16)?;
            let layer = Linear::new(
                Tensor::from_vec(ws, (1, xs.len()), &device)?.to_dtype(DType::BF16)?,
                None,
            );
            let actual = full_attention_kv_forward(&x, &layer)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(
                actual,
                vec![expected],
                "full K/V exceptional-product recovery"
            );
        }
        Ok(())
    }

    fn supported_rotary_test_dtypes(_device: &candle_core::Device) -> Result<Vec<DType>> {
        let mut dtypes = vec![DType::F32, DType::F16];
        #[cfg(feature = "cuda")]
        if let candle_core::Device::Cuda(cuda) = _device {
            let (major, _) = cuda
                .cuda_stream()
                .context()
                .compute_capability()
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            if major >= 8 {
                dtypes.push(DType::BF16);
            }
            return Ok(dtypes);
        }
        dtypes.push(DType::BF16);
        Ok(dtypes)
    }

    #[test]
    fn eager_causal_gqa_matches_independent_f64_scalar_reference() -> Result<()> {
        let mut devices = vec![candle_core::Device::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(test_cuda_device()?);
        let (batch, length, heads, kv_heads, width) = (2, 129, 4, 2, 4);
        let make = |count, factor| {
            (0..count)
                .map(|i| ((i * 13 % 97) as f32 - 48.) * factor)
                .collect::<Vec<_>>()
        };
        let q = make(batch * length * heads * width, 0.013);
        let k = make(batch * length * kv_heads * width, 0.017);
        let v = make(batch * length * kv_heads * width, 0.019);
        let mut expected = Vec::new();
        for b in 0..batch {
            for row in 0..length {
                for h in 0..heads {
                    let kv = h / (heads / kv_heads);
                    let scores = (0..=row)
                        .map(|col| {
                            (0..width)
                                .map(|d| {
                                    q[((b * length + row) * heads + h) * width + d] as f64
                                        * k[((b * length + col) * kv_heads + kv) * width + d] as f64
                                })
                                .sum::<f64>()
                                * 0.5
                        })
                        .collect::<Vec<_>>();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights = scores
                        .iter()
                        .map(|score| (score - max).exp())
                        .collect::<Vec<_>>();
                    let total = weights.iter().sum::<f64>();
                    for d in 0..width {
                        expected.push(
                            weights
                                .iter()
                                .enumerate()
                                .map(|(col, weight)| {
                                    weight
                                        * v[((b * length + col) * kv_heads + kv) * width + d] as f64
                                })
                                .sum::<f64>()
                                / total,
                        );
                    }
                }
            }
        }
        for device in devices.drain(..) {
            let actual = eager_attention(
                &Tensor::from_vec(q.clone(), (batch, length, heads, width), &device)?,
                &Tensor::from_vec(k.clone(), (batch, length, kv_heads, width), &device)?,
                &Tensor::from_vec(v.clone(), (batch, length, kv_heads, width), &device)?,
                0.5,
                true,
            )?
            .flatten_all()?
            .to_vec1::<f32>()?;
            for (actual, expected) in actual.into_iter().zip(&expected) {
                assert!((actual as f64 - expected).abs() <= 1e-5 + 1.3e-6 * expected.abs());
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn right_sized_rmsnorm_launch_is_bitwise_equal_to_legacy() -> Result<()> {
        let device = test_cuda_device()?;
        for columns in [16, 32, 64, 128, 256, 512, 1024, 2560, 5120] {
            let input = Tensor::from_vec(
                (0..2 * 7 * columns)
                    .map(|i| ((i * 17 % 1009) as f32 - 504.) * 0.003)
                    .collect::<Vec<_>>(),
                (2, 7, columns),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let weight = Tensor::from_vec(
                (0..columns)
                    .map(|i| ((i % 37) as f32 - 18.) * 0.002)
                    .collect::<Vec<_>>(),
                columns,
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let bits = |tensor: Tensor| -> Result<Vec<u32>> {
                Ok(tensor
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(
                bits(cuda_rms_norm_delta(&input, &weight, 1e-6)?)?,
                bits(cuda_rms_norm_delta_with_block_size(
                    &input, &weight, 1e-6, 1024
                )?)?,
                "RMSNorm launch changes values for width {columns}",
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fused_residual_rmsnorm_is_bitwise_equal_to_separate_kernels() -> Result<()> {
        let device = test_cuda_device()?;

        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        for (rows, columns) in [(1, 64), (7, 4096), (33, 2560)] {
            let values = |offset: usize| {
                (0..rows * columns)
                    .map(|i| ((i * 17 + offset) % 1009) as f32 * 0.003 - 1.5)
                    .collect::<Vec<_>>()
            };
            let input =
                Tensor::from_vec(values(0), (1, rows, columns), &device)?.to_dtype(DType::BF16)?;
            let mixed = Tensor::from_vec(values(503), (1, rows, columns), &device)?
                .to_dtype(DType::BF16)?;
            let weight = Tensor::from_vec(
                (0..columns)
                    .map(|i| ((i % 37) as f32 - 18.) * 0.002)
                    .collect(),
                columns,
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let expected_residual = (&input + &mixed)?;
            let expected_normalized = cuda_rms_norm_delta(&expected_residual, &weight, 1e-6)?;
            let (actual_residual, actual_normalized) =
                cuda_residual_rms_norm_delta(&input, &mixed, &weight, 1e-6)?;
            let bits = |tensor: Tensor| -> Result<Vec<u32>> {
                Ok(tensor
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(
                bits(actual_residual)?,
                bits(expected_residual)?,
                "fused residual differs at rows={rows}, columns={columns}"
            );
            assert_eq!(
                bits(actual_normalized)?,
                bits(expected_normalized)?,
                "fused RMSNorm differs at rows={rows}, columns={columns}"
            );
        }
        Ok(())
    }

    #[test]
    fn shared_rotary_is_bitwise_equal_to_legacy_per_head_preparation() -> Result<()> {
        let mut devices = vec![candle_core::Device::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(test_cuda_device()?);
        for device in devices.drain(..) {
            for dtype in supported_rotary_test_dtypes(&device)? {
                for (dimension, sections, width) in [(64, [16, 8, 8], 64), (64, [16, 8, 8], 256)] {
                    let theta = 1_000_000f64;
                    let positions = (0..17)
                        .map(|i| [i * 1031, i * 7, i * 31])
                        .collect::<Vec<_>>();
                    let prepared = Rotary::new(dimension, theta, sections)?
                        .prepare(&positions, dtype, &device)?;
                    for heads in [4, 24] {
                        let x = Tensor::from_vec(
                            (0..2 * positions.len() * heads * width)
                                .map(|i| (i % 251) as f32 * 0.013 - 1.5)
                                .collect::<Vec<_>>(),
                            (2, positions.len(), heads, width),
                            &device,
                        )?
                        .to_dtype(dtype)?;

                        let phase = positions
                            .iter()
                            .flat_map(|position| {
                                (0..dimension / 2).map(move |i| {
                                    let axis = if i < sections[1] * 3 && i % 3 == 1 {
                                        1
                                    } else if i < sections[2] * 3 && i % 3 == 2 {
                                        2
                                    } else {
                                        0
                                    };
                                    position[axis] as f32
                                        / theta.powf(2. * i as f64 / dimension as f64) as f32
                                })
                            })
                            .collect::<Vec<_>>();
                        let phase = Tensor::from_vec(
                            phase,
                            (1, positions.len(), 1, dimension / 2),
                            &device,
                        )?;
                        let cos = Tensor::cat(&[&phase.cos()?, &phase.cos()?], D::Minus1)?
                            .to_dtype(dtype)?;
                        let sin = Tensor::cat(&[&phase.sin()?, &phase.sin()?], D::Minus1)?
                            .to_dtype(dtype)?;
                        let active = x.narrow(D::Minus1, 0, dimension)?;
                        let first = active.narrow(D::Minus1, 0, dimension / 2)?;
                        let second = active.narrow(D::Minus1, dimension / 2, dimension / 2)?;
                        let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
                        let legacy = Tensor::cat(
                            &[
                                &(active.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?,
                                &x.narrow(D::Minus1, dimension, width - dimension)?,
                            ],
                            D::Minus1,
                        )?;
                        let values = |tensor: Tensor| -> Result<Vec<u32>> {
                            Ok(tensor
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?
                                .into_iter()
                                .map(f32::to_bits)
                                .collect())
                        };
                        assert_eq!(
                            values(prepared.apply(&x)?)?,
                            values(legacy)?,
                            "{dtype:?}, heads={heads}, width={width}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn eager_attention_single_key_and_chunked_causal_are_finite() {
        let mut devices = vec![candle_core::Device::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(test_cuda_device().unwrap());
        #[cfg(feature = "metal")]
        devices.push(candle_core::Device::new_metal(0).unwrap());
        for device in devices.drain(..) {
            let q = Tensor::ones((1, 1, 2, 4), DType::F32, &device).unwrap();
            let k = Tensor::ones((1, 1, 1, 4), DType::F32, &device).unwrap();
            let v = Tensor::from_vec(vec![1f32, -2., 3., -4.], (1, 1, 1, 4), &device).unwrap();
            let actual = eager_attention(&q, &k, &v, 0.5, false)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(actual, vec![1., -2., 3., -4., 1., -2., 3., -4.]);

            let q = Tensor::zeros((1, 257, 2, 4), DType::F32, &device).unwrap();
            let k = Tensor::zeros((1, 257, 1, 4), DType::F32, &device).unwrap();
            let v = Tensor::from_vec(
                (0..257).flat_map(|i| [i as f32; 4]).collect::<Vec<_>>(),
                (1, 257, 1, 4),
                &device,
            )
            .unwrap();
            let actual = eager_attention(&q, &k, &v, 0.5, true)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (index, actual) in actual.into_iter().enumerate() {
                let expected = (index / 8) as f32 * 0.5;
                assert!(
                    (actual - expected).abs() <= 1e-5 + expected.abs() * 1.3e-6,
                    "{device:?}: index {index}, expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn convolution_matches_frozen_native_torch_gpu_fixtures() -> Result<()> {
        let directory = crate::assets::fixtures("qwen35_conv")?;
        let mut fixtures = std::fs::read_dir(directory)?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path
                    .extension()
                    .is_some_and(|extension| extension == "safetensors"))
                .then_some(path)
            })
            .collect::<Vec<_>>();
        fixtures.sort();
        assert_eq!(
            fixtures.len(),
            8,
            "all pinned native BF16/FP32 cases are required"
        );
        for path in fixtures {
            let tensors = candle_core::safetensors::load(&path, &candle_core::Device::Cpu)?;
            let input = &tensors["input"];
            let weight = &tensors["weight"];
            let kernel = weight.dim(2)?;
            let expected_conv = &tensors["expected_conv"];
            let expected = &tensors["expected"];
            let assert_close = |actual: &Tensor, reference: &Tensor, name: &str| -> Result<()> {
                let actual = actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let reference = reference
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(actual.len(), reference.len());

                let rtol = if input.dtype() == DType::BF16 {
                    1.6e-2
                } else {
                    1.3e-6
                };
                for (index, (actual, reference)) in actual.into_iter().zip(reference).enumerate() {
                    assert!(actual.is_finite() && reference.is_finite());
                    assert!(
                        (actual - reference).abs() <= 1e-5 + rtol * reference.abs(),
                        "{} {name} element {index}: {actual} vs {reference}",
                        path.display()
                    );
                }
                Ok(())
            };
            let convolution = causal_conv(input, weight, kernel)?;
            if input.dtype() == DType::BF16 {
                assert_eq!(
                    convolution
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    expected_conv
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                );
            }
            assert_close(&convolution, expected_conv, "CPU convolution")?;
            assert_close(
                &causal_conv_silu(input, weight, kernel)?,
                expected,
                "CPU SiLU",
            )?;
            #[cfg(feature = "cuda")]
            {
                let device = test_cuda_device()?;
                let capability = match &device {
                    candle_core::Device::Cuda(cuda) => {
                        cuda.cuda_stream()
                            .context()
                            .compute_capability()
                            .map_err(|error| candle_core::Error::Msg(error.to_string()))?
                            .0
                    }
                    _ => unreachable!(),
                };
                if input.dtype() != DType::BF16 || capability >= 8 {
                    let result =
                        cuda_conv_silu(&input.to_device(&device)?, &weight.to_device(&device)?)?;
                    assert_close(&result, expected, "CUDA SiLU")?;
                }
            }
            #[cfg(feature = "metal")]
            {
                let device = candle_core::Device::new_metal(0)?;
                let result = crate::kernels::metal::conv_silu(
                    &input.to_device(&device)?,
                    &weight.to_device(&device)?,
                )?;

                let round = |value: f32| {
                    if input.dtype() == DType::BF16 {
                        let bits = value.to_bits();
                        f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
                    } else {
                        value
                    }
                };
                let (batch, length, channels) = input.dims3()?;
                let xs = input
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let ws = weight
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let mut reference = Vec::with_capacity(xs.len());
                for b in 0..batch {
                    for token in 0..length {
                        for channel in 0..channels {
                            let mut sum = 0f32;
                            for tap in 0..kernel {
                                if token + tap + 1 >= kernel {
                                    let source = token + tap + 1 - kernel;
                                    let product = round(
                                        xs[(b * length + source) * channels + channel]
                                            * ws[channel * kernel + tap],
                                    );
                                    sum = round(sum + product);
                                }
                            }
                            reference.push(round(sum / (1. + (-sum).exp())));
                        }
                    }
                }
                let reference =
                    Tensor::from_vec(reference, input.shape().clone(), &candle_core::Device::Cpu)?
                        .to_dtype(input.dtype())?;
                assert_close(&result, &reference, "Metal retained per-tap SiLU")?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn conv_cuda_matches_candle_across_batches_and_padding() {
        let (batch, length, channels, kernel) = (2, 7, 31, 4);
        let cpu = candle_core::Device::Cpu;
        let projected = Tensor::from_vec(
            (0..batch * length * channels)
                .map(|index| ((index * 17 % 41) as f32 - 20.) * 0.04)
                .collect::<Vec<_>>(),
            (batch, length, channels),
            &cpu,
        )
        .unwrap();
        let weights = Tensor::from_vec(
            (0..channels * kernel)
                .map(|index| ((index * 11 % 23) as f32 - 11.) * 0.02)
                .collect::<Vec<_>>(),
            (channels, 1, kernel),
            &cpu,
        )
        .unwrap();
        let cuda = test_cuda_device().unwrap();
        for dtype in [DType::BF16, DType::F32] {
            let projected = projected.to_dtype(dtype).unwrap();
            let weights = weights.to_dtype(dtype).unwrap();
            let expected = causal_conv_silu(&projected, &weights, kernel)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let actual = cuda_conv_silu(
                &projected.to_device(&cuda).unwrap(),
                &weights.to_device(&cuda).unwrap(),
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            let max_error = expected
                .iter()
                .zip(actual)
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0f32, f32::max);
            assert!(
                max_error < 0.001,
                "{dtype:?} convolution error: {max_error}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn delta_cuda_matches_chunked_reference() {
        let batch = 2;
        let heads = 2;
        let length = 290;
        let key_dim = 128;
        let value_dim = 16;
        let values = |width: usize, factor: f32| {
            (0..batch * heads * length * width)
                .map(|index| ((index * 13 % 29) as f32 - 14.) * factor)
                .collect::<Vec<_>>()
        };
        let cpu = candle_core::Device::Cpu;
        let q = Tensor::from_vec(
            values(key_dim, 0.001),
            (batch, heads, length, key_dim),
            &cpu,
        )
        .unwrap();
        let k = Tensor::from_vec(
            values(key_dim, 0.006),
            (batch, heads, length, key_dim),
            &cpu,
        )
        .unwrap();
        let v = Tensor::from_vec(
            values(value_dim, 0.008),
            (batch, heads, length, value_dim),
            &cpu,
        )
        .unwrap();
        let g = Tensor::from_vec(
            vec![-0.002f32; batch * heads * length],
            (batch, heads, length),
            &cpu,
        )
        .unwrap();
        let beta = Tensor::from_vec(
            vec![0.5f32; batch * heads * length],
            (batch, heads, length),
            &cpu,
        )
        .unwrap();
        let reference = chunked_delta_rule(&q, &k, &v, &g, &beta)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cuda = test_cuda_device().unwrap();
        let actual = cuda_delta_rule_token_major(
            &q.transpose(1, 2).unwrap().to_device(&cuda).unwrap(),
            &k.transpose(1, 2).unwrap().to_device(&cuda).unwrap(),
            &v.transpose(1, 2).unwrap().to_device(&cuda).unwrap(),
            &g.transpose(1, 2).unwrap().to_device(&cuda).unwrap(),
            &beta.transpose(1, 2).unwrap().to_device(&cuda).unwrap(),
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        let rtol = 1.3e-6f32;
        let atol = 1e-5f32;
        for (index, (expected, actual)) in reference.iter().zip(actual).enumerate() {
            let error = (expected - actual).abs();
            let allowed = atol + rtol * expected.abs();
            assert!(
                error <= allowed,
                "Delta-rule mismatch at {index}: expected {expected}, got {actual}, error {error} > {allowed}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn subwarp16_cuda_is_bitwise_equal_to_uncached_recurrence() -> Result<()> {
        let device = test_cuda_device()?;
        for (batch, length, value_dim) in [
            (1usize, 17usize, 128usize),
            (2, 290, 128),
            (1, 1024, 128),
            (1, 17, 7),
        ] {
            let key_heads = 2usize;
            let heads = 4usize;
            let key_dim = 128usize;
            let data = |count: usize, factor: f32| {
                (0..count)
                    .map(|i| ((i * 13 % 29) as f32 - 14.) * factor)
                    .collect::<Vec<_>>()
            };
            let q = Tensor::from_vec(
                data(batch * length * key_heads * key_dim, 0.001),
                (batch, length, key_heads, key_dim),
                &device,
            )?;
            let k = Tensor::from_vec(
                data(batch * length * key_heads * key_dim, 0.006),
                (batch, length, key_heads, key_dim),
                &device,
            )?;
            let v = Tensor::from_vec(
                data(batch * length * heads * value_dim, 0.008),
                (batch, length, heads, value_dim),
                &device,
            )?;
            let gate = Tensor::from_vec(
                (0..batch * length * heads)
                    .map(|i| -0.001 * (1 + (i % 29)) as f32)
                    .collect::<Vec<_>>(),
                (batch, length, heads),
                &device,
            )?;
            let beta = Tensor::from_vec(
                vec![0.5f32; batch * length * heads],
                (batch, length, heads),
                &device,
            )?;
            let old = cuda_delta_rule_token_major_impl(&q, &k, &v, &gate, &beta, false)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let new = cuda_delta_rule_token_major(&q, &k, &v, &gate, &beta)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(old.len(), new.len());
            for (i, (a, b)) in old.iter().zip(&new).enumerate() {
                assert!(a.is_finite() && b.is_finite());
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "subwarp delta mismatch at {batch}/{length}/{i}"
                );
            }
        }
        println!("SUBWARP_DELTA_PARITY exact_recurrence=true grouped_heads=true batch_shapes=3");
        Ok(())
    }

    #[test]
    fn delta_rule_matches_pinned_native_chunk_and_fp64_fixtures() -> Result<()> {
        let directory = crate::assets::fixtures("qwen35_delta")?;
        let device =
            crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        for preparation in ["f32", "bf16"] {
            let tensors = candle_core::safetensors::load(
                directory.join(format!("b2-l65-kh2-h4-d16-{preparation}.safetensors")),
                &device,
            )?;
            let q = &tensors["query_scaled_f32"];
            let k = &tensors["key_f32"];
            let v = &tensors["value_f32"];
            let g = &tensors["gate_f32"];
            let beta = &tensors["beta_f32"];
            #[allow(unused_mut)]
            let mut actuals = vec![(
                "tensor_chunk32",
                chunked_delta_rule_fallback(
                    &q.transpose(1, 2)?,
                    &k.transpose(1, 2)?,
                    &v.transpose(1, 2)?,
                    &g.transpose(1, 2)?,
                    &beta.transpose(1, 2)?,
                )?,
            )];
            #[cfg(feature = "cuda")]
            if device.is_cuda() {
                actuals.push((
                    "cuda_expanded",
                    cuda_delta_rule_token_major(q, k, v, g, beta)?,
                ));
                let indices = Tensor::from_vec(vec![0u32, 2], 2, &device)?;
                actuals.push((
                    "cuda_grouped",
                    cuda_delta_rule_token_major(
                        &q.index_select(&indices, 2)?,
                        &k.index_select(&indices, 2)?,
                        v,
                        g,
                        beta,
                    )?,
                ));
            }
            for (implementation, actual) in actuals {
                assert_eq!(actual.dims(), [2, 65, 4, 16]);
                let actual = actual.flatten_all()?.to_vec1::<f32>()?;
                for oracle in [
                    "native_chunk_f32",
                    "native_sequential_f32",
                    "fp64_recurrence_rounded_f32",
                ] {
                    let expected = tensors[oracle].flatten_all()?.to_vec1::<f32>()?;
                    assert_eq!(actual.len(), expected.len());
                    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                        assert!(
                            actual.is_finite()
                                && expected.is_finite()
                                && (actual - expected).abs() <= 1e-5 + 1.3e-6 * expected.abs(),
                            "{preparation} {implementation} vs {oracle}, element{index}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn grouped_delta_cuda_is_bitwise_equal_to_expanded_preparation() {
        let device = test_cuda_device().unwrap();
        for (batch, length, key_heads, heads, value_dim) in
            [(1, 1, 1, 4, 5), (2, 33, 2, 8, 16), (2, 290, 4, 16, 128)]
        {
            for dtype in supported_rotary_test_dtypes(&device).unwrap() {
                let input = |width: usize, factor: f32, count_heads: usize| {
                    Tensor::from_vec(
                        (0..batch * length * count_heads * width)
                            .map(|i| ((i * 17 % 101) as f32 - 50.) * factor)
                            .collect::<Vec<_>>(),
                        (batch, length, count_heads, width),
                        &device,
                    )
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap()
                };
                let expand = |x: &Tensor| {
                    x.unsqueeze(3)
                        .unwrap()
                        .broadcast_as((batch, length, key_heads, heads / key_heads, 128))
                        .unwrap()
                        .reshape((batch, length, heads, 128))
                        .unwrap()
                };
                let normalize = |x: &Tensor, scale: f64| {
                    let x = x.to_dtype(DType::F32).unwrap();
                    let denominator = (x.sqr().unwrap().sum_keepdim(D::Minus1).unwrap() + 1e-6)
                        .unwrap()
                        .sqrt()
                        .unwrap();
                    (x.broadcast_div(&denominator).unwrap() * scale).unwrap()
                };
                let raw_q = input(128, 0.003, key_heads);
                let raw_k = input(128, 0.007, key_heads);
                let q = normalize(&raw_q, 128f64.powf(-0.5));
                let k = normalize(&raw_k, 1.);
                let old_q = normalize(&expand(&raw_q), 128f64.powf(-0.5));
                let old_k = normalize(&expand(&raw_k), 1.);
                let read = |x: &Tensor| x.flatten_all().unwrap().to_vec1::<f32>().unwrap();

                let assert_bits = |a: &Tensor, b: &Tensor| {
                    let a = read(a);
                    let b = read(b);
                    assert_eq!(a.len(), b.len());
                    for (i, (a, b)) in a.iter().zip(&b).enumerate() {
                        assert!(a.is_finite() && b.is_finite());
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{dtype:?} batch={batch} length={length} heads={heads} value_dim={value_dim} element={i}"
                        );
                    }
                };
                assert_bits(&expand(&q), &old_q);
                assert_bits(&expand(&k), &old_k);
                let v = input(value_dim, 0.008, heads).to_dtype(DType::F32).unwrap();
                let gate = Tensor::full(-0.002f32, (batch, length, heads), &device).unwrap();
                let beta = Tensor::full(0.5f32, (batch, length, heads), &device).unwrap();
                let grouped = cuda_delta_rule_token_major(&q, &k, &v, &gate, &beta).unwrap();
                let expanded =
                    cuda_delta_rule_token_major(&old_q, &old_k, &v, &gate, &beta).unwrap();
                assert_bits(&grouped, &expanded);
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn post_finalize_cuda_matches_tensor_reference() {
        let shape = (2, 3, 4, 128);
        let count = shape.0 * shape.1 * shape.2 * shape.3;
        let cpu = candle_core::Device::Cpu;
        let input = Tensor::from_vec(
            (0..count)
                .map(|index| ((index * 17 % 101) as f32 - 50.) * 0.003)
                .collect::<Vec<_>>(),
            shape,
            &cpu,
        )
        .unwrap();
        let gate = Tensor::from_vec(
            (0..count)
                .map(|index| ((index * 11 % 67) as f32 - 33.) * 0.02)
                .collect::<Vec<_>>(),
            shape,
            &cpu,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let norm = Tensor::from_vec(
            (0..128)
                .map(|index| 0.8 + (index % 13) as f32 * 0.01)
                .collect::<Vec<_>>(),
            128,
            &cpu,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let epsilon = 1e-6;
        let reference = gated_norm_reference(&input, &gate, &norm, epsilon, DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cuda = test_cuda_device().unwrap();
        let actual = cuda_post_finalize_bf16(
            &input.to_device(&cuda).unwrap(),
            &gate.to_device(&cuda).unwrap(),
            &norm.to_device(&cuda).unwrap(),
            epsilon,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        let rtol = 1.6e-2f32;
        let atol = 1e-5f32;
        for (index, (expected, actual)) in reference.iter().zip(actual).enumerate() {
            let error = (expected - actual).abs();
            let allowed = atol + rtol * expected.abs();
            assert!(
                error <= allowed,
                "Post-finalize mismatch at {index}: expected {expected}, got {actual}, error {error} > {allowed}"
            );
        }
    }

    #[test]
    fn recurrent_rounding_matches_native_fixtures() -> Result<()> {
        let root = crate::assets::fixtures("qwen35_recurrent")?;
        let device =
            crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        for (name, dtype, rtol) in [
            ("bf16", DType::BF16, 1.6e-2f32),
            ("f16", DType::F16, 1e-3),
            ("f32", DType::F32, 1.3e-6),
        ] {
            #[cfg(feature = "cuda")]
            if dtype == DType::BF16 {
                if let candle_core::Device::Cuda(cuda) = &device {
                    if cuda
                        .cuda_stream()
                        .context()
                        .compute_capability()
                        .map_err(|error| candle_core::Error::Msg(error.to_string()))?
                        .0
                        < 8
                    {
                        continue;
                    }
                }
            }
            let data = candle_core::safetensors::load(
                root.join(format!("post-finalize-{name}.safetensors")),
                &device,
            )?;
            let actual = gated_norm_reference(
                &data["recurrent_f32"],
                &data["gate"],
                &data["weight"],
                1e-6,
                dtype,
            )?;
            let values =
                |tensor: &Tensor| tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
            let check = |actual: &Tensor, expected: &Tensor| -> Result<()> {
                for (index, (actual, expected)) in
                    values(actual)?.iter().zip(values(expected)?).enumerate()
                {
                    assert!(actual.is_finite() && expected.is_finite());
                    assert!(
                        (actual - expected).abs() <= 1e-5 + rtol * expected.abs(),
                        "native {name} fixture index={index}: {actual} vs {expected}"
                    );
                }
                Ok(())
            };
            check(&actual, &data["expected"])?;
            check(&rounded_beta(&data["beta_input"])?, &data["expected_beta"])?;
            #[cfg(feature = "cuda")]
            if device.is_cuda() && dtype == DType::BF16 {
                check(
                    &cuda_post_finalize_bf16(
                        &data["recurrent_f32"],
                        &data["gate"],
                        &data["weight"],
                        1e-6,
                    )?,
                    &data["expected"],
                )?;
            }
        }
        Ok(())
    }

    #[test]
    fn chunked_delta_rule_matches_scalar_recurrence_across_chunk_boundary() {
        let length = 70;
        let key_dim = 4;
        let value_dim = 3;
        let make = |width: usize, factor: f32| {
            (0..length * width)
                .map(|index| ((index * 13 % 29) as f32 - 14.) * factor)
                .collect::<Vec<_>>()
        };
        let q = make(key_dim, 0.01);
        let k = make(key_dim, 0.012);
        let v = make(value_dim, 0.02);
        let beta = vec![0.45f32; length];
        let g = vec![-0.1f32; length];
        let device = candle_core::Device::Cpu;
        let output = chunked_delta_rule(
            &Tensor::from_vec(q.clone(), (1, 1, length, key_dim), &device).unwrap(),
            &Tensor::from_vec(k.clone(), (1, 1, length, key_dim), &device).unwrap(),
            &Tensor::from_vec(v.clone(), (1, 1, length, value_dim), &device).unwrap(),
            &Tensor::from_vec(g.clone(), (1, 1, length), &device).unwrap(),
            &Tensor::from_vec(beta.clone(), (1, 1, length), &device).unwrap(),
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let mut state = vec![0f32; key_dim * value_dim];
        for step in 0..length {
            let decay = g[step].exp();
            for value in &mut state {
                *value *= decay;
            }
            for column in 0..value_dim {
                let remembered = (0..key_dim)
                    .map(|row| k[step * key_dim + row] * state[row * value_dim + column])
                    .sum::<f32>();
                let delta = (v[step * value_dim + column] - remembered) * beta[step];
                for row in 0..key_dim {
                    state[row * value_dim + column] += k[step * key_dim + row] * delta;
                }
                let expected = (0..key_dim)
                    .map(|row| q[step * key_dim + row] * state[row * value_dim + column])
                    .sum::<f32>();
                assert!((output[step * value_dim + column] - expected).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn chunked_delta_rule_stays_finite_with_large_negative_gates() {
        let device = candle_core::Device::Cpu;
        let q = Tensor::ones((1, 1, 64, 4), DType::F32, &device).unwrap();
        let k = Tensor::ones((1, 1, 64, 4), DType::F32, &device).unwrap();
        let v = Tensor::ones((1, 1, 64, 3), DType::F32, &device).unwrap();
        let g = Tensor::new(-8f32, &device)
            .unwrap()
            .broadcast_as((1, 1, 64))
            .unwrap();
        let beta = Tensor::new(0.5f32, &device)
            .unwrap()
            .broadcast_as((1, 1, 64))
            .unwrap();
        let output = chunked_delta_rule(&q, &k, &v, &g, &beta)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn chunked_delta_rule_runs_with_broadcast_gates_on_metal() {
        let device = candle_core::Device::new_metal(0).unwrap();
        let q = Tensor::ones((1, 1, 64, 4), DType::F32, &device).unwrap();
        let k = Tensor::ones((1, 1, 64, 4), DType::F32, &device).unwrap();
        let v = Tensor::ones((1, 1, 64, 3), DType::F32, &device).unwrap();
        let g = Tensor::new(-8f32, &device)
            .unwrap()
            .broadcast_as((1, 1, 64))
            .unwrap();
        let beta = Tensor::new(0.5f32, &device)
            .unwrap()
            .broadcast_as((1, 1, 64))
            .unwrap();
        let output = chunked_delta_rule(&q, &k, &v, &g, &beta)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
    }
}

#[cfg(any(feature = "cuda", test))]
fn long_down_actual_m(rows: usize, inner: usize, columns: usize, sm120: bool) -> bool {
    sm120
        && (2048..=8192).contains(&rows)
        && columns >= 2048
        && columns.checked_mul(3) == Some(inner)
}

#[test]
fn long_down_geometry_policy_has_explicit_scope() {
    for (rows, inner, columns, sm120, expected) in [
        (2048, 12288, 4096, true, true),
        (2105, 12288, 4096, true, true),
        (8192, 12288, 4096, true, true),
        (2047, 12288, 4096, true, false),
        (8193, 12288, 4096, true, false),
        (2105, 12288, 4096, false, false),
        (2105, 4096, 12288, true, false),
        (2105, 6144, 2048, true, true),
        (2105, 192, 64, true, false),
    ] {
        assert_eq!(long_down_actual_m(rows, inner, columns, sm120), expected);
    }
}

#[cfg(feature = "cuda")]
fn cuda_fused_qkv_prepare(
    mixed: &Tensor,
    key_heads: usize,
    value_heads: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (batch, length, channels) = mixed.dims3()?;
    if batch == 0
        || length == 0
        || key_heads == 0
        || value_heads == 0
        || value_heads % key_heads != 0
        || mixed.dtype() != DType::BF16
        || mixed.elem_count() > i32::MAX as usize
        || key_heads
            .checked_mul(2)
            .and_then(|x| x.checked_add(value_heads))
            .and_then(|x| x.checked_mul(128))
            != Some(channels)
    {
        candle_core::bail!("unsupported packed QKV preparation");
    }
    let mixed = mixed.contiguous()?;
    let (storage, layout) = mixed.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("packed QKV requires CUDA");
    };
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("QKV layout".into()))?;
    let source = storage.as_cuda_slice::<bf16>()?.slice(start..end);
    let device = storage.device().clone();
    let q_count = batch * length * key_heads * 128;
    let v_count = batch * length * value_heads * 128;
    let mut q = unsafe { device.alloc::<f32>(q_count)? };
    let mut k = unsafe { device.alloc::<f32>(q_count)? };
    let mut v = unsafe { device.alloc::<f32>(v_count)? };
    let rows = (q_count / 128) as u32;
    let function = device.get_or_load_custom_func(
        "sys1_fused_qkv_prepare_bf16",
        "sys1_fused_qkv",
        CUDA_PTX,
    )?;
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&mut q);
    launch.arg(&mut k);
    launch.arg(&mut v);
    let kh = key_heads as i32;
    let vh = value_heads as i32;
    let scale = 128f64.powf(-0.5) as f32;
    launch.arg(&rows);
    launch.arg(&kh);
    launch.arg(&vh);
    launch.arg(&scale);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (rows.div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    let wrap = |data, heads| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(data, device.clone())),
            (batch, length, heads, 128),
            BackpropOp::none(),
            false,
        )
    };
    Ok((wrap(q, key_heads), wrap(k, key_heads), wrap(v, value_heads)))
}

#[cfg(feature = "cuda")]
fn cuda_packed_ba_nonlinearity(
    a: &Tensor,
    b: &Tensor,
    bias: &Tensor,
    a_log: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let heads = *a
        .dims()
        .last()
        .ok_or_else(|| candle_core::Error::Msg("gate rank".into()))?
        / 2;
    let count = a.elem_count() / 2;
    if a.elem_count() > i32::MAX as usize
        || a.dim(D::Minus1)? % 2 != 0
        || count == 0
        || count > i32::MAX as usize
        || heads == 0
        || a.dims() != b.dims()
        || a.dtype() != DType::BF16
        || b.dtype() != DType::BF16
        || bias.dtype() != DType::F32
        || a_log.dtype() != DType::F32
        || bias.dims() != [heads]
        || a_log.dims() != [heads]
        || !a.device().same_device(b.device())
        || !a.device().same_device(bias.device())
        || !a.device().same_device(a_log.device())
    {
        candle_core::bail!("unsupported fused gate tensors");
    }
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let bias = bias.contiguous()?;
    let a_log = a_log.contiguous()?;
    let (as_, al) = a.storage_and_layout();
    let (bs, bl) = b.storage_and_layout();
    let (ds, dl) = bias.storage_and_layout();
    let (ls, ll) = a_log.storage_and_layout();
    let (Storage::Cuda(as_), Storage::Cuda(bs), Storage::Cuda(ds), Storage::Cuda(ls)) =
        (&*as_, &*bs, &*ds, &*ls)
    else {
        candle_core::bail!("fused gates require CUDA");
    };
    let offsets = |l: &candle_core::Layout| {
        l.contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous gate storage".into()))
    };
    let (a0, a1) = offsets(&al)?;
    let (b0, b1) = offsets(&bl)?;
    let (d0, d1) = offsets(&dl)?;
    let (l0, l1) = offsets(&ll)?;
    let av = as_.as_cuda_slice::<bf16>()?.slice(a0..a1);
    let bv = bs.as_cuda_slice::<bf16>()?.slice(b0..b1);
    let dv = ds.as_cuda_slice::<f32>()?.slice(d0..d1);
    let lv = ls.as_cuda_slice::<f32>()?.slice(l0..l1);
    let device = as_.device().clone();
    let mut gate = unsafe { device.alloc::<f32>(count)? };
    let mut beta = unsafe { device.alloc::<f32>(count)? };
    let function = device.get_or_load_custom_func(
        "sys1_packed_ba_nonlinearity_bf16",
        "sys1_fused_gates",
        CUDA_PTX,
    )?;
    let mut launch = function.builder();
    launch.arg(&av);
    launch.arg(&bv);
    launch.arg(&dv);
    launch.arg(&lv);
    launch.arg(&mut gate);
    launch.arg(&mut beta);
    let count_arg = count as i32;
    let heads_arg = heads as i32;
    launch.arg(&count_arg);
    launch.arg(&heads_arg);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (count.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    let mut shape = a.dims().to_vec();
    *shape.last_mut().unwrap() = heads;
    let wrap = |data| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(data, device.clone())),
            shape.clone(),
            BackpropOp::none(),
            false,
        )
    };
    Ok((wrap(gate), wrap(beta)))
}

#[cfg(feature = "cuda")]
fn cuda_delta_rule_precomputed_h200_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    subwarp16: bool,
) -> Result<Tensor> {
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let device = q_storage.device().clone();
    let decay = unsafe { device.alloc::<f32>(batch * length * heads)? };
    let exp = device.get_or_load_custom_func("qwen35_preexp_gate_f32", "sys1_qwen35", CUDA_PTX)?;
    let count = (batch * length * heads) as u32;
    let mut exp_launch = exp.builder();
    exp_launch.arg(&gate);
    exp_launch.arg(&decay);
    exp_launch.arg(&count);
    unsafe { exp_launch.launch(LaunchConfig::for_num_elems(count)) }.w()?;
    let output = unsafe { device.alloc::<f32>(batch * heads * length * value_dim)? };
    let ampere_warp32 = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80");

    let hopper_uniform_prefetch8 = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && batch == 1
        && length >= 512
        && value_dim == 128
        && subwarp16;

    let ampere_uniform_prefetch8 =
        matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "86"))
            && batch == 1
            && length >= 512
            && heads == 32
            && key_heads == 16
            && value_dim == 128
            && subwarp16;
    let uniform_prefetch8 = hopper_uniform_prefetch8 || ampere_uniform_prefetch8;
    let name = if uniform_prefetch8 {
        "sys1_uniform_delta_prefetch8_preexp_f32"
    } else if ampere_warp32 {
        "sys1_delta_warp32_preexp_f32"
    } else {
        "qwen35_delta_subwarp16_preexp_f32"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (
            (batch * heads) as u32,
            value_dim.div_ceil(if uniform_prefetch8 {
                8
            } else if ampere_warp32 {
                4
            } else if subwarp16 {
                8
            } else {
                4
            }) as u32,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&query);
    launch.arg(&key);
    launch.arg(&value);
    launch.arg(&decay);
    launch.arg(&beta);
    launch.arg(&output);
    let length = length as i32;
    let heads_arg = heads as i32;
    let key_heads_arg = key_heads as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&heads_arg);
    launch.arg(&key_heads_arg);
    launch.arg(&value_dim);
    unsafe { launch.launch(config) }.w()?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, heads, length as usize, value_dim as usize),
        BackpropOp::none(),
        false,
    );
    output.transpose(1, 2)
}

#[cfg(feature = "cuda")]
fn cuda_post_finalize_bf16(
    input: &Tensor,
    gate: &Tensor,
    norm: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    if input.dtype() == DType::BF16 {
        cuda_post_finalize_bf16_warp_fork(input, gate, norm, epsilon)
    } else if input.rank() == 4 && input.stride()[3] == 1 && input.elem_count() > 0 {
        cuda_post_finalize_strided_fork(input, gate, norm, epsilon)
    } else {
        cuda_post_finalize_original_fork(input, gate, norm, epsilon)
    }
}
#[cfg(feature = "cuda")]
fn cuda_post_finalize_strided_fork(
    input: &Tensor,
    gate: &Tensor,
    norm: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    let columns = input.dim(D::Minus1)?;
    if input.dtype() != DType::F32
        || gate.dtype() != DType::BF16
        || norm.dtype() != DType::BF16
        || input.dims() != gate.dims()
        || norm.dims() != [columns]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA post-finalize shape or dtype")
    }
    let (batch, length, heads, width) = input.dims4()?;
    if width == 0 || batch == 0 || length == 0 || heads == 0 || input.stride()[3] != 1 {
        candle_core::bail!("unsupported strided post-finalize layout");
    }
    let columns_arg = i32::try_from(columns)
        .map_err(|_| candle_core::Error::Msg("post-finalize width overflow".into()))?;
    let length_arg =
        u64::try_from(length).map_err(|_| candle_core::Error::Msg("length overflow".into()))?;
    let heads_arg =
        u64::try_from(heads).map_err(|_| candle_core::Error::Msg("heads overflow".into()))?;
    let strides: Vec<u64> = input.stride()[..3]
        .iter()
        .map(|&s| u64::try_from(s).map_err(|_| candle_core::Error::Msg("stride overflow".into())))
        .collect::<Result<_>>()?;
    let gate = gate.contiguous()?;
    let norm = norm.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (norm_storage, norm_layout) = norm.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(gate_storage), Storage::Cuda(norm_storage)) =
        (&*input_storage, &*gate_storage, &*norm_storage)
    else {
        candle_core::bail!("Qwen3.5 post-finalize requires CUDA storage")
    };
    let input_start = input_layout.start_offset();
    let span =
        input
            .dims()
            .iter()
            .zip(input.stride())
            .try_fold(1usize, |span, (&dim, &stride)| {
                dim.checked_sub(1)
                    .and_then(|n| n.checked_mul(stride))
                    .and_then(|n| span.checked_add(n))
                    .ok_or_else(|| candle_core::Error::Msg("post-finalize span overflow".into()))
            })?;
    let input_end = input_start
        .checked_add(span)
        .ok_or_else(|| candle_core::Error::Msg("post-finalize offset overflow".into()))?;
    if input_end > input_storage.as_cuda_slice::<f32>()?.len() {
        candle_core::bail!("post-finalize input span exceeds allocation");
    }
    if !std::sync::Arc::ptr_eq(
        &input_storage.device().cuda_stream(),
        &gate_storage.device().cuda_stream(),
    ) || !std::sync::Arc::ptr_eq(
        &input_storage.device().cuda_stream(),
        &norm_storage.device().cuda_stream(),
    ) {
        candle_core::bail!("post-finalize requires one CUDA stream");
    }
    let (gate_start, gate_end) = gate_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize gate is not contiguous".into())
    })?;
    let (norm_start, norm_end) = norm_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize norm is not contiguous".into())
    })?;
    let source = input_storage
        .as_cuda_slice::<f32>()?
        .slice(input_start..input_end);
    let gate = gate_storage
        .as_cuda_slice::<bf16>()?
        .slice(gate_start..gate_end);
    let norm = norm_storage
        .as_cuda_slice::<bf16>()?
        .slice(norm_start..norm_end);
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let rows = u32::try_from(count / columns)
        .map_err(|_| candle_core::Error::Msg("post-finalize row count overflow".into()))?;
    if count > i32::MAX as usize {
        candle_core::bail!("post-finalize index overflow");
    }
    let mut output = unsafe { device.alloc::<bf16>(count)? };
    let function = device.get_or_load_custom_func(
        "qwen35_post_finalize_strided_fork",
        "sys1_qwen35",
        CUDA_PTX,
    )?;
    let config = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&gate);
    launch.arg(&norm);
    launch.arg(&mut output);
    let columns = columns_arg;
    let epsilon = epsilon as f32;
    launch.arg(&columns);
    launch.arg(&epsilon);
    launch.arg(&length_arg);
    launch.arg(&heads_arg);
    launch.arg(&strides[0]);
    launch.arg(&strides[1]);
    launch.arg(&strides[2]);
    unsafe { launch.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
struct PackedSequences {
    lengths: Vec<usize>,
    cumulative: Tensor,
    starts: Tensor,
    total: usize,
    c64_chunks: Tensor,
    c64_long_ids: Tensor,
    c64_short_ids: Tensor,
    c64_chunk_count: usize,
    c64_long_count: usize,
    c64_short_count: usize,
    c64_geometry_cache: std::sync::Mutex<Option<((usize, usize, usize), bool)>>,
    gate_long_starts: Tensor,
    gate_long_ends: Tensor,
    gate_long_count: usize,
    gate_short_rows: Tensor,
    gate_short_count: usize,
}
#[cfg(feature = "cuda")]
impl PackedSequences {
    fn new(lengths: &[usize], device: &candle_core::Device) -> Result<Self> {
        if lengths.is_empty() || lengths.iter().any(|&n| n == 0) || !device.is_cuda() {
            candle_core::bail!("packed sequences require nonempty positive lengths and CUDA")
        }
        let total = lengths
            .iter()
            .try_fold(0usize, |sum, &n| sum.checked_add(n))
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| candle_core::Error::Msg("packed total overflow".into()))?;
        let mut offsets = vec![0u32];
        let mut starts = Vec::with_capacity(total);
        let mut begin = 0usize;
        for &n in lengths {
            starts.resize(starts.len() + n, begin as u32);
            begin += n;
            offsets.push(begin as u32);
        }
        let mut c64_offsets = vec![0u32];
        let mut long_ids = Vec::new();
        let mut short_ids = Vec::new();
        let mut chunk_count = 0usize;
        for (index, &n) in lengths.iter().enumerate() {
            if n >= (if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90")) {
                128
            } else {
                512
            }) {
                long_ids.push(index as u32);
                chunk_count += n.div_ceil(64);
            } else {
                short_ids.push(index as u32);
            }
            c64_offsets.push(chunk_count as u32);
        }

        long_ids.sort_by_key(|&index| std::cmp::Reverse(lengths[index as usize].div_ceil(64)));
        let long_count = long_ids.len();
        let short_count = short_ids.len();
        if long_ids.is_empty() {
            long_ids.push(0);
        }
        if short_ids.is_empty() {
            short_ids.push(0);
        }
        let mut gate_long_starts = Vec::new();
        let mut gate_long_ends = Vec::new();
        let mut gate_short_rows = Vec::new();
        let mut gate_begin = 0usize;
        for &n in lengths {
            if n >= 1024 {
                for offset in (0..n).step_by(32) {
                    gate_long_starts.push((gate_begin + offset) as u32);
                    gate_long_ends.push((gate_begin + n) as u32);
                }
            } else {
                gate_short_rows.extend((gate_begin..gate_begin + n).map(|i| i as u32));
            }
            gate_begin += n;
        }
        let gate_long_count = gate_long_starts.len();
        if gate_long_starts.is_empty() {
            gate_long_starts.push(0);
            gate_long_ends.push(0);
        }
        let gate_short_count = gate_short_rows.len();
        if gate_short_rows.is_empty() {
            gate_short_rows.push(0);
        }
        Ok(Self {
            lengths: lengths.to_vec(),
            cumulative: Tensor::from_vec(offsets, lengths.len() + 1, device)?,
            starts: Tensor::from_vec(starts, total, device)?,
            total,
            c64_chunks: Tensor::from_vec(c64_offsets, lengths.len() + 1, device)?,
            c64_long_ids: Tensor::from_vec(long_ids, long_count.max(1), device)?,
            c64_short_ids: Tensor::from_vec(short_ids, short_count.max(1), device)?,
            c64_chunk_count: chunk_count,
            c64_long_count: long_count,
            c64_short_count: short_count,
            c64_geometry_cache: std::sync::Mutex::new(None),
            gate_long_starts: Tensor::from_vec(gate_long_starts, gate_long_count.max(1), device)?,
            gate_long_ends: Tensor::from_vec(gate_long_ends, gate_long_count.max(1), device)?,
            gate_long_count,
            gate_short_rows: Tensor::from_vec(gate_short_rows, gate_short_count.max(1), device)?,
            gate_short_count,
        })
    }
}

#[cfg(feature = "cuda")]
fn cuda_conv_silu_packed_sequences(
    input: &Tensor,
    weights: &Tensor,
    sequences: &PackedSequences,
) -> Result<Tensor> {
    let (batch, length, channels) = input.dims3()?;
    if batch != 1
        || length != sequences.total
        || length
            .checked_mul(channels)
            .is_none_or(|n| n > i32::MAX as usize)
    {
        candle_core::bail!("packed convolution shape/index overflow")
    }
    let (weight_channels, groups, kernel) = weights.dims3()?;
    if weight_channels != channels || groups != 1 || kernel == 0 || input.dtype() != weights.dtype()
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA convolution shape or dtype")
    }
    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (weight_storage, weight_layout) = weights.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(weight_storage)) =
        (&*input_storage, &*weight_storage)
    else {
        candle_core::bail!("Qwen3.5 convolution requires CUDA storage")
    };
    let (input_start, input_end) = input_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 convolution input is not contiguous".into())
    })?;
    let (weight_start, weight_end) = weight_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 convolution weights are not contiguous".into())
    })?;
    let device = input_storage.device().clone();
    let (starts_storage, starts_layout) = sequences.starts.storage_and_layout();
    let Storage::Cuda(starts_storage) = &*starts_storage else {
        candle_core::bail!("packed metadata must be CUDA")
    };
    let (start, end) = starts_layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("noncontiguous starts".into()))?;
    let starts = starts_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let count = batch * length * channels;
    let config = LaunchConfig {
        grid_dim: (count.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let count_arg = count as i32;
    let length_arg = length as i32;
    let channels_arg = channels as i32;
    let kernel_arg = kernel as i32;
    let output = match input.dtype() {
        DType::BF16 => {
            let source = input_storage
                .as_cuda_slice::<bf16>()?
                .slice(input_start..input_end);
            let filters = weight_storage
                .as_cuda_slice::<bf16>()?
                .slice(weight_start..weight_end);
            let output = unsafe { device.alloc::<bf16>(count)? };
            let function =
                device.get_or_load_custom_func("sys1_packed_conv_bf16", "sys1_qwen35", CUDA_PTX)?;
            let mut launch = function.builder();
            launch.arg(&source);
            launch.arg(&filters);
            launch.arg(&output);
            launch.arg(&starts);
            launch.arg(&count_arg);
            launch.arg(&length_arg);
            launch.arg(&channels_arg);
            launch.arg(&kernel_arg);
            unsafe { launch.launch(config) }.w()?;
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device))
        }
        DType::F32 => {
            let source = input_storage
                .as_cuda_slice::<f32>()?
                .slice(input_start..input_end);
            let filters = weight_storage
                .as_cuda_slice::<f32>()?
                .slice(weight_start..weight_end);
            let output = unsafe { device.alloc::<f32>(count)? };
            let function =
                device.get_or_load_custom_func("sys1_packed_conv_f32", "sys1_qwen35", CUDA_PTX)?;
            let mut launch = function.builder();
            launch.arg(&source);
            launch.arg(&filters);
            launch.arg(&output);
            launch.arg(&starts);
            launch.arg(&count_arg);
            launch.arg(&length_arg);
            launch.arg(&channels_arg);
            launch.arg(&kernel_arg);
            unsafe { launch.launch(config) }.w()?;
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device))
        }
        dtype => candle_core::bail!("Unsupported Qwen3.5 CUDA convolution dtype: {dtype:?}"),
    };
    Ok(Tensor::from_storage(
        output,
        (batch, length, channels),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn cuda_delta_rule_packed_sequences(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    subwarp16: bool,
) -> Result<Tensor> {
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    if !subwarp16 || batch != 1 || length != sequences.total {
        candle_core::bail!("packed delta unsupported layout")
    }
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let (metadata_storage, metadata_layout) = sequences.cumulative.storage_and_layout();
    let Storage::Cuda(metadata_storage) = &*metadata_storage else {
        candle_core::bail!("packed metadata must be CUDA")
    };
    let (start, end) = offsets(metadata_layout)?;
    let cumulative = metadata_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let device = q_storage.device().clone();
    let ampere_prefetch8 = matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "86"))
        && length >= 512
        && heads == 32
        && key_heads == 16
        && value_dim == 128
        && q_start.is_multiple_of(4)
        && k_start.is_multiple_of(4)
        && v_start.is_multiple_of(4);

    let ampere_warp32 = (matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
        && length >= 512)
        || ampere_prefetch8;
    let hopper_columns2 = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") && length >= 1024;
    let hopper_prefetch8 = hopper_columns2 && value_dim == 128;

    let hopper_prefetch4 =
        hopper_prefetch8 && heads >= 32 && sequences.lengths.len().saturating_mul(heads) >= 64;
    let precomputed_decay = if ampere_warp32 {
        let values = unsafe { device.alloc::<f32>(length * heads)? };
        let function =
            device.get_or_load_custom_func("qwen35_preexp_gate_f32", "sys1_qwen35", CUDA_PTX)?;
        let count = (length * heads) as u32;
        let mut launch = function.builder();
        launch.arg(&gate);
        launch.arg(&values);
        launch.arg(&count);
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }.w()?;
        Some(values)
    } else {
        None
    };
    let output = unsafe { device.alloc::<f32>(batch * heads * length * value_dim)? };
    let name = if ampere_prefetch8 {
        "sys1_packed_delta_prefetch8_preexp_f32"
    } else if hopper_prefetch4 {
        "sys1_packed_delta_prefetch8_columns4_preexp_f32"
    } else if hopper_prefetch8 {
        "sys1_packed_delta_prefetch8_preexp_f32"
    } else if hopper_columns2 {
        "sys1_packed_delta_warp32_cols2_preexp_f32"
    } else if ampere_warp32 {
        "sys1_packed_delta_warp32_preexp_f32"
    } else {
        "sys1_packed_delta_subwarp16_f32"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (
            (sequences.lengths.len() * heads) as u32,
            value_dim.div_ceil(if ampere_prefetch8 {
                8
            } else if hopper_prefetch4 {
                16
            } else if hopper_columns2 {
                8
            } else if ampere_warp32 {
                4
            } else if subwarp16 {
                8
            } else {
                4
            }) as u32,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&query);
    launch.arg(&key);
    launch.arg(&value);
    if let Some(decay) = &precomputed_decay {
        launch.arg(decay);
    } else {
        launch.arg(&gate);
    }
    launch.arg(&beta);
    launch.arg(&output);
    launch.arg(&cumulative);
    let length = length as i32;
    let heads_arg = heads as i32;
    let key_heads_arg = key_heads as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&heads_arg);
    launch.arg(&key_heads_arg);
    launch.arg(&value_dim);
    unsafe { launch.launch(config) }.w()?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length as usize, heads, value_dim as usize),
        BackpropOp::none(),
        false,
    );
    Ok(output)
}

#[cfg(feature = "cuda")]
struct ChunkScratch {
    stream: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    w: Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
    u: Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
    c: Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
    decays: Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
    snapshots: Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
}

#[cfg(feature = "cuda")]
fn grow_chunk_buffer<'a>(
    buffer: &'a mut Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
    device: &candle_core::CudaDevice,
    count: usize,
) -> Result<&'a mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>> {
    if buffer.as_ref().is_none_or(|x| x.len() < count) {
        *buffer = Some(unsafe { device.alloc::<f32>(count)? });
    }
    buffer
        .as_mut()
        .ok_or_else(|| candle_core::Error::Msg("missing chunk buffer".into()))
}

#[cfg(feature = "cuda")]
static CHUNK64_SCRATCH: std::sync::Mutex<Option<ChunkScratch>> = std::sync::Mutex::new(None);
#[cfg(feature = "cuda")]
fn cuda_chunk64_direct_fork(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    tile_columns: usize,
    profiling: bool,
) -> Result<Tensor> {
    if [q, k, v, g, beta].iter().any(|x| x.dtype() != DType::F32) {
        candle_core::bail!("chunk probe F32 inputs required");
    }
    let state_rows = 128usize;
    let carry_shared = 58400u32;
    if tile_columns != 32 {
        candle_core::bail!("C64 state requires N32");
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let device = q_storage.device().clone();

    if tile_columns != 32
        || batch == 0
        || length == 0
        || value_dim > 128
        || [
            q.elem_count(),
            k.elem_count(),
            v.elem_count(),
            g.elem_count(),
        ]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk probe requires nonempty F32 K128,V<=128,small indices");
    }
    let instances = batch
        .checked_mul(heads)
        .and_then(|x| x.checked_mul(length.div_ceil(64)))
        .ok_or_else(|| candle_core::Error::Msg("chunk count overflow".into()))?;
    let w_count = instances
        .checked_mul(64 * 128)
        .ok_or_else(|| candle_core::Error::Msg("chunk W overflow".into()))?;
    let u_count = instances
        .checked_mul(64 * value_dim)
        .ok_or_else(|| candle_core::Error::Msg("chunk U overflow".into()))?;
    let c_count = instances
        .checked_mul(64 * 64)
        .ok_or_else(|| candle_core::Error::Msg("chunk C overflow".into()))?;
    let d_count = instances
        .checked_mul(128)
        .ok_or_else(|| candle_core::Error::Msg("decay count overflow".into()))?;
    if [w_count, u_count, c_count, d_count]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk scratch index overflow");
    }
    let result = (|| -> Result<Tensor> {
        let mut slot = CHUNK64_SCRATCH
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned chunk scratch".into()))?;
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
        {
            *slot = Some(ChunkScratch {
                stream,
                w: None,
                u: None,
                c: None,
                decays: None,
                snapshots: None,
            });
        }
        let workspace = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing chunk scratch".into()))?;
        let ChunkScratch {
            w,
            u,
            c,
            decays,
            snapshots,
            ..
        } = workspace;
        if profiling {
            device.cuda_stream().synchronize().w()?;
        }
        let mut stage_started = Instant::now();
        let w = grow_chunk_buffer(w, &device, w_count)?;
        let u = grow_chunk_buffer(u, &device, u_count)?;
        let c = grow_chunk_buffer(c, &device, c_count)?;
        let decays = grow_chunk_buffer(decays, &device, d_count)?;
        let output_count = batch
            .checked_mul(length)
            .and_then(|n| n.checked_mul(heads))
            .and_then(|n| n.checked_mul(value_dim))
            .ok_or_else(|| candle_core::Error::Msg("register-carry C64 output overflow".into()))?;
        if output_count > i32::MAX as usize {
            candle_core::bail!("padded output index overflow");
        }
        let snapshot_count = 1usize;
        if snapshot_count > i32::MAX as usize {
            candle_core::bail!("snapshot index overflow");
        }
        let snapshots = grow_chunk_buffer(snapshots, &device, snapshot_count)?;
        let mut output = unsafe { device.alloc::<f32>(output_count)? };
        if profiling {
            println!(
                "C64_FLAGS_WORKSPACE retained_bytes={} output_bytes={} single_entry_per_process=true owned_stream_identity=true output_never_reused=true amortized_warm_operator_not_serving=true",
                (w.len() + u.len() + c.len() + decays.len() + snapshots.len()) * 4,
                output_count * 4
            );
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=scratch batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        let length_arg = length as i32;
        let heads_arg = heads as i32;
        let key_heads_arg = key_heads as i32;
        let value_dim_arg = value_dim as i32;
        {
            let function = device.get_or_load_custom_func(
                ("sys1_c64mma_prepare_f32"),
                "sys1_chunk_delta",
                CUDA_PTX,
            )?;
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,86528).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&value);
            launch.arg(&gate);
            launch.arg(&beta);
            launch.arg(&mut *w);
            launch.arg(&mut *u);
            launch.arg(&mut *c);
            launch.arg(&mut *decays);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (instances as u32, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 86528,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=prepare batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        {
            let name = "sys1_c64flag_uniform32_f32";
            let function = device.get_or_load_custom_func(name, "sys1_chunk_delta", CUDA_PTX)?;
            if profiling {
                println!(
                    "C64_FLAGS_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} diagnostic_not_serving=true",
                    function.num_regs().w()?,
                    function.local_size_bytes().w()?,
                    function.shared_size_bytes().w()?
                );
            }
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,carry_shared as i32).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&gate);
            launch.arg(&*w);
            launch.arg(&*u);
            launch.arg(&*c);
            launch.arg(&*decays);
            launch.arg(&mut output);
            launch.arg(&mut *snapshots);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        (batch * heads) as u32,
                        value_dim.div_ceil(tile_columns) as u32,
                        1,
                    ),
                    block_dim: (((state_rows / 16) * (tile_columns / 8) * 32) as u32, 1, 1),
                    shared_mem_bytes: carry_shared,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=state batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
        }

        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (batch, length, heads, value_dim),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    })();
    if result.is_err() {
        if let Ok(mut slot) = CHUNK64_SCRATCH.lock() {
            slot.take();
        }
    }
    result
}

#[cfg(feature = "cuda")]
fn cuda_c64_varlen_fork(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    profiling: bool,
) -> Result<Tensor> {
    if [q, k, v, g, beta].iter().any(|x| x.dtype() != DType::F32) {
        candle_core::bail!("chunk probe F32 inputs required");
    }
    let state_rows = 128usize;
    let tile_columns = 32usize;
    let carry_shared = 58400u32;
    if q.dim(0)? != 1 || q.dim(1)? != sequences.total {
        candle_core::bail!("segmented C64 shape mismatch")
    }
    if sequences.c64_chunk_count == 0 {
        return cuda_delta_rule_packed_sequences(q, k, v, g, beta, sequences, true);
    }
    if [
        k,
        v,
        g,
        beta,
        &sequences.cumulative,
        &sequences.c64_chunks,
        &sequences.c64_long_ids,
        &sequences.c64_short_ids,
    ]
    .iter()
    .any(|t| !q.device().same_device(t.device()))
    {
        candle_core::bail!("segmented C64 device mismatch")
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let (cumulative_storage, cumulative_layout) = sequences.cumulative.storage_and_layout();
    let Storage::Cuda(cumulative_storage) = &*cumulative_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(cumulative_layout)?;
    let cumulative = cumulative_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let (chunk_offsets_storage, chunk_offsets_layout) = sequences.c64_chunks.storage_and_layout();
    let Storage::Cuda(chunk_offsets_storage) = &*chunk_offsets_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(chunk_offsets_layout)?;
    let chunk_offsets = chunk_offsets_storage
        .as_cuda_slice::<u32>()?
        .slice(start..end);
    let (long_ids_storage, long_ids_layout) = sequences.c64_long_ids.storage_and_layout();
    let Storage::Cuda(long_ids_storage) = &*long_ids_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(long_ids_layout)?;
    let long_ids = long_ids_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let (short_ids_storage, short_ids_layout) = sequences.c64_short_ids.storage_and_layout();
    let Storage::Cuda(short_ids_storage) = &*short_ids_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(short_ids_layout)?;
    let short_ids = short_ids_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let device = q_storage.device().clone();

    if tile_columns != 32
        || batch == 0
        || length == 0
        || value_dim > 128
        || [
            q.elem_count(),
            k.elem_count(),
            v.elem_count(),
            g.elem_count(),
        ]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk probe requires nonempty F32 K128,V<=128,small indices");
    }
    let instances = heads
        .checked_mul(sequences.c64_chunk_count)
        .ok_or_else(|| candle_core::Error::Msg("chunk count overflow".into()))?;
    let w_count = instances
        .checked_mul(64 * 128)
        .ok_or_else(|| candle_core::Error::Msg("chunk W overflow".into()))?;
    let u_count = instances
        .checked_mul(64 * value_dim)
        .ok_or_else(|| candle_core::Error::Msg("chunk U overflow".into()))?;
    let c_count = instances
        .checked_mul(64 * 64)
        .ok_or_else(|| candle_core::Error::Msg("chunk C overflow".into()))?;
    let d_count = instances
        .checked_mul(128)
        .ok_or_else(|| candle_core::Error::Msg("decay count overflow".into()))?;
    if [w_count, u_count, c_count, d_count]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk scratch index overflow");
    }
    let result = (|| -> Result<Tensor> {
        let mut slot = CHUNK64_SCRATCH
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned chunk scratch".into()))?;
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
        {
            *slot = Some(ChunkScratch {
                stream,
                w: None,
                u: None,
                c: None,
                decays: None,
                snapshots: None,
            });
        }
        let workspace = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing chunk scratch".into()))?;
        let ChunkScratch {
            w,
            u,
            c,
            decays,
            snapshots,
            ..
        } = workspace;
        if profiling {
            device.cuda_stream().synchronize().w()?;
        }
        let mut stage_started = Instant::now();
        let w = grow_chunk_buffer(w, &device, w_count)?;
        let u = grow_chunk_buffer(u, &device, u_count)?;
        let c = grow_chunk_buffer(c, &device, c_count)?;
        let decays = grow_chunk_buffer(decays, &device, d_count)?;
        let output_count = length
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(value_dim))
            .ok_or_else(|| candle_core::Error::Msg("segmented C64 output overflow".into()))?;
        if output_count > i32::MAX as usize {
            candle_core::bail!("padded output index overflow");
        }
        let snapshot_count = 1usize;
        if snapshot_count > i32::MAX as usize {
            candle_core::bail!("snapshot index overflow");
        }
        let snapshots = grow_chunk_buffer(snapshots, &device, snapshot_count)?;
        let mut output = unsafe { device.alloc::<f32>(output_count)? };
        if profiling {
            println!(
                "C64_CARRY_WORKSPACE retained_bytes={} output_bytes={} single_entry_per_process=true owned_stream_identity=true output_never_reused=true amortized_warm_operator_not_serving=true",
                (w.len() + u.len() + c.len() + decays.len() + snapshots.len()) * 4,
                output_count * 4
            );
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=scratch batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        let sequence_count = sequences.lengths.len() as i32;
        let length_arg = length as i32;
        let heads_arg = heads as i32;
        let key_heads_arg = key_heads as i32;
        let value_dim_arg = value_dim as i32;
        {
            let function = device.get_or_load_custom_func(
                ("sys1_c64v_prepare_f32"),
                "sys1_chunk_delta",
                CUDA_PTX,
            )?;
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,86528).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&value);
            launch.arg(&gate);
            launch.arg(&beta);
            launch.arg(&mut *w);
            launch.arg(&mut *u);
            launch.arg(&mut *c);
            launch.arg(&mut *decays);
            launch.arg(&cumulative);
            launch.arg(&chunk_offsets);
            launch.arg(&sequence_count);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (instances as u32, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 86528,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=prepare batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        {
            if sequences.c64_short_count > 0 {
                let function = device.get_or_load_custom_func(
                    "sys1_c64v_short_f32",
                    "sys1_c64v_short",
                    CUDA_PTX,
                )?;
                let mut launch = function.builder();
                launch
                    .arg(&query)
                    .arg(&key)
                    .arg(&value)
                    .arg(&gate)
                    .arg(&beta)
                    .arg(&mut output)
                    .arg(&cumulative)
                    .arg(&short_ids)
                    .arg(&length_arg)
                    .arg(&heads_arg)
                    .arg(&key_heads_arg)
                    .arg(&value_dim_arg);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (
                            (sequences.c64_short_count * heads) as u32,
                            value_dim.div_ceil(8) as u32,
                            1,
                        ),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .w()?;
            }
            let name = "sys1_c64flag_varlen32_f32";
            let function = device.get_or_load_custom_func(name, "sys1_chunk_delta", CUDA_PTX)?;
            if profiling {
                println!(
                    "C64_CARRY_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} diagnostic_not_serving=true",
                    function.num_regs().w()?,
                    function.local_size_bytes().w()?,
                    function.shared_size_bytes().w()?
                );
            }
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,carry_shared as i32).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&gate);
            launch.arg(&*w);
            launch.arg(&*u);
            launch.arg(&*c);
            launch.arg(&*decays);
            launch.arg(&mut output);
            launch.arg(&mut *snapshots);
            launch.arg(&cumulative);
            launch.arg(&chunk_offsets);
            launch.arg(&long_ids);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        (sequences.c64_long_count * heads) as u32,
                        value_dim.div_ceil(tile_columns) as u32,
                        1,
                    ),
                    block_dim: (((state_rows / 16) * (tile_columns / 8) * 32) as u32, 1, 1),
                    shared_mem_bytes: carry_shared,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=state batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
        }

        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (1, length, heads, value_dim),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    })();
    if result.is_err() {
        if let Ok(mut slot) = CHUNK64_SCRATCH.lock() {
            slot.take();
        }
    }
    result
}

#[cfg(feature = "cuda")]
fn cuda_gate_tensor16_rows32_fork(
    input: &Tensor,
    linear: &Linear,
    tensor_core: bool,
    simt: SimtProjection,
) -> Result<Tensor> {
    let (batch, length, inner) = input.dims3()?;
    let (columns, weight_inner) = linear.weight().dims2()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| candle_core::Error::Msg("Gate projection row count overflow".into()))?;
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| candle_core::Error::Msg("Gate projection output size overflow".into()))?;
    if input.dtype() != DType::BF16
        || linear.weight().dtype() != DType::BF16
        || (linear.bias().is_some()
            && !tensor_core
            && !matches!(
                simt,
                SimtProjection::Compensated | SimtProjection::TiledCompensated
            ))
        || inner != weight_inner
        || rows == 0
        || columns == 0
        || inner == 0
        || !input.device().same_device(linear.weight().device())
        || input.elem_count() > i32::MAX as usize
        || linear.weight().elem_count() > i32::MAX as usize
        || count > i32::MAX as usize
    {
        candle_core::bail!("Unsupported stable BF16 gate projection shape, dtype or device")
    }
    if let Some(bias) = linear.bias() {
        if bias.dtype() != DType::BF16
            || bias.dims() != [columns]
            || !input.device().same_device(bias.device())
        {
            candle_core::bail!(
                "Stable BF16 projection requires matching bias dtype, shape and device"
            )
        }
    }
    let input = input.contiguous()?;
    let weight = linear.weight().contiguous()?;
    let bias = linear.bias().map(Tensor::contiguous).transpose()?;
    let bias_storage = bias.as_ref().map(Tensor::storage_and_layout);
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("Stable gate projection requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("Gate projection storage is not contiguous".into())
        })
    };
    let (xstart, xend) = offsets(xl)?;
    let (wstart, wend) = offsets(wl)?;
    let x = xs.as_cuda_slice::<bf16>()?.slice(xstart..xend);
    let w = ws.as_cuda_slice::<bf16>()?.slice(wstart..wend);
    let b = if let Some((storage, layout)) = &bias_storage {
        let Storage::Cuda(storage) = &**storage else {
            candle_core::bail!("Stable BF16 projection requires CUDA bias storage")
        };
        let (start, end) = offsets(layout)?;
        Some(storage.as_cuda_slice::<bf16>()?.slice(start..end))
    } else {
        None
    };
    let device = xs.device().clone();
    let mut output = unsafe { device.alloc::<bf16>(count)? };

    let tile_rows = 32usize;

    #[cfg(test)]
    let tile_rows = if tensor_core && matches!(simt, SimtProjection::TensorRows32Probe) {
        32
    } else if tensor_core && matches!(simt, SimtProjection::TensorRows128Probe) {
        128
    } else {
        tile_rows
    };
    let kernel = if !tensor_core && matches!(simt, SimtProjection::TiledCompensated) {
        if b.is_some() {
            "qwen35_biased_tiled_compensated_projection_bf16"
        } else {
            "qwen35_tiled_compensated_projection_bf16"
        }
    } else if b.is_some() && !tensor_core {
        "qwen35_biased_compensated_projection_bf16"
    } else if b.is_some() && tile_rows == 128 {
        "qwen35_biased_projection_rows128_bf16"
    } else if b.is_some() && tile_rows == 32 {
        "qwen35_biased_projection_rows32_bf16"
    } else if b.is_some() {
        "qwen35_biased_projection_bf16"
    } else if tensor_core && tile_rows == 128 {
        "sys1_gate_nativevec_rows32_bf16"
    } else if tensor_core && tile_rows == 32 {
        "sys1_gate_nativevec_rows32_bf16"
    } else if tensor_core {
        "sys1_gate_nativevec_rows32_bf16"
    } else {
        match simt {
            #[cfg(test)]
            SimtProjection::TensorRows32Probe | SimtProjection::TensorRows128Probe => {
                "qwen35_gate_projection_bf16"
            }
            SimtProjection::Compensated => "qwen35_gate_projection_bf16",
            SimtProjection::TiledCompensated => "qwen35_tiled_compensated_projection_bf16",
            SimtProjection::GuardedDouble => "qwen35_guarded_double_projection_bf16",
            SimtProjection::GuardedTensorDouble => "qwen35_guarded_tensor_double_projection_bf16",
            SimtProjection::GuardedTensorDoubleRows16 => {
                "qwen35_guarded_tensor_double_rows16_projection_bf16"
            }
        }
    };
    let function = device.get_or_load_custom_func(kernel, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: if tensor_core {
            (
                rows.div_ceil(tile_rows) as u32,
                columns.div_ceil(64) as u32,
                1,
            )
        } else if matches!(simt, SimtProjection::TiledCompensated) {
            (rows.div_ceil(4) as u32, columns.div_ceil(4) as u32, 1)
        } else if simt.is_tensor_double() {
            let row_tile = if matches!(simt, SimtProjection::GuardedTensorDoubleRows16) {
                16
            } else {
                8
            };
            (
                rows.div_ceil(row_tile) as u32,
                columns.div_ceil(32) as u32,
                1,
            )
        } else {
            (rows as u32, columns.div_ceil(8) as u32, 1)
        },
        block_dim: (
            if tensor_core {
                tile_rows.min(32) as u32 * 8
            } else if simt.is_tensor_double() || matches!(simt, SimtProjection::TiledCompensated) {
                128
            } else {
                256
            },
            1,
            1,
        ),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&x);
    launch.arg(&w);
    if let Some(bias) = &b {
        launch.arg(bias);
    }
    launch.arg(&mut output);
    let rows = rows as i32;
    if tensor_core || simt.is_tensor_double() || matches!(simt, SimtProjection::TiledCompensated) {
        launch.arg(&rows);
    }
    let inner = inner as i32;
    let columns_arg = columns as i32;
    let radius_coefficient = if !tensor_core
        && !matches!(
            simt,
            SimtProjection::Compensated | SimtProjection::TiledCompensated
        ) {
        Some(guarded_double_radius_coefficient_impl(
            inner as usize,
            simt.is_tensor_double(),
        ))
    } else {
        None
    };
    launch.arg(&inner);
    launch.arg(&columns_arg);
    if let Some(coefficient) = &radius_coefficient {
        launch.arg(coefficient);
    }
    unsafe { launch.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length, columns),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn cuda_gate_tensor16_original_packed_fork(
    input: &Tensor,
    linear: &Linear,
    tensor_core: bool,
    simt: SimtProjection,
    sequences: &PackedSequences,
) -> Result<Tensor> {
    let (batch, length, inner) = input.dims3()?;
    if batch != 1
        || length != sequences.total
        || !input
            .device()
            .same_device(sequences.gate_long_starts.device())
        || !input
            .device()
            .same_device(sequences.gate_long_ends.device())
        || !input
            .device()
            .same_device(sequences.gate_short_rows.device())
        || linear.bias().is_some()
    {
        candle_core::bail!("segmented narrow-gate shape/metadata mismatch");
    }
    if sequences.gate_short_count == sequences.total {
        return cuda_gate_projection_bf16(input, linear);
    }

    let (columns, weight_inner) = linear.weight().dims2()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| candle_core::Error::Msg("Gate projection row count overflow".into()))?;
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| candle_core::Error::Msg("Gate projection output size overflow".into()))?;
    if input.dtype() != DType::BF16
        || linear.weight().dtype() != DType::BF16
        || (linear.bias().is_some()
            && !tensor_core
            && !matches!(
                simt,
                SimtProjection::Compensated | SimtProjection::TiledCompensated
            ))
        || inner != weight_inner
        || rows == 0
        || columns == 0
        || inner == 0
        || !input.device().same_device(linear.weight().device())
        || input.elem_count() > i32::MAX as usize
        || linear.weight().elem_count() > i32::MAX as usize
        || count > i32::MAX as usize
    {
        candle_core::bail!("Unsupported stable BF16 gate projection shape, dtype or device")
    }
    if let Some(bias) = linear.bias() {
        if bias.dtype() != DType::BF16
            || bias.dims() != [columns]
            || !input.device().same_device(bias.device())
        {
            candle_core::bail!(
                "Stable BF16 projection requires matching bias dtype, shape and device"
            )
        }
    }
    let input = input.contiguous()?;
    let weight = linear.weight().contiguous()?;
    let bias = linear.bias().map(Tensor::contiguous).transpose()?;
    let bias_storage = bias.as_ref().map(Tensor::storage_and_layout);
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("Stable gate projection requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("Gate projection storage is not contiguous".into())
        })
    };
    let (xstart, xend) = offsets(xl)?;
    let (wstart, wend) = offsets(wl)?;
    let x = xs.as_cuda_slice::<bf16>()?.slice(xstart..xend);
    let w = ws.as_cuda_slice::<bf16>()?.slice(wstart..wend);
    let b = if let Some((storage, layout)) = &bias_storage {
        let Storage::Cuda(storage) = &**storage else {
            candle_core::bail!("Stable BF16 projection requires CUDA bias storage")
        };
        let (start, end) = offsets(layout)?;
        Some(storage.as_cuda_slice::<bf16>()?.slice(start..end))
    } else {
        None
    };
    let (starts_storage, starts_layout) = sequences.gate_long_starts.storage_and_layout();
    let (ends_storage, ends_layout) = sequences.gate_long_ends.storage_and_layout();
    let (ids_storage, ids_layout) = sequences.gate_short_rows.storage_and_layout();
    let (Storage::Cuda(starts_storage), Storage::Cuda(ends_storage), Storage::Cuda(ids_storage)) =
        (&*starts_storage, &*ends_storage, &*ids_storage)
    else {
        candle_core::bail!("CUDA gate metadata required")
    };
    let (ss, se) = offsets(starts_layout)?;
    let (es, ee) = offsets(ends_layout)?;
    let (is, ie) = offsets(ids_layout)?;
    let tile_starts = starts_storage.as_cuda_slice::<u32>()?.slice(ss..se);
    let tile_ends = ends_storage.as_cuda_slice::<u32>()?.slice(es..ee);
    let short_ids = ids_storage.as_cuda_slice::<u32>()?.slice(is..ie);
    let device = xs.device().clone();
    let mut output = unsafe { device.alloc::<bf16>(count)? };

    let tile_rows = 32usize;

    #[cfg(test)]
    let tile_rows = if tensor_core && matches!(simt, SimtProjection::TensorRows32Probe) {
        32
    } else if tensor_core && matches!(simt, SimtProjection::TensorRows128Probe) {
        128
    } else {
        tile_rows
    };
    let kernel = if !tensor_core && matches!(simt, SimtProjection::TiledCompensated) {
        if b.is_some() {
            "qwen35_biased_tiled_compensated_projection_bf16"
        } else {
            "qwen35_tiled_compensated_projection_bf16"
        }
    } else if b.is_some() && !tensor_core {
        "qwen35_biased_compensated_projection_bf16"
    } else if b.is_some() && tile_rows == 128 {
        "qwen35_biased_projection_rows128_bf16"
    } else if b.is_some() && tile_rows == 32 {
        "qwen35_biased_projection_rows32_bf16"
    } else if b.is_some() {
        "qwen35_biased_projection_bf16"
    } else if tensor_core && tile_rows == 128 {
        "sys1_gate_nativevec_packed_bf16"
    } else if tensor_core && tile_rows == 32 {
        "sys1_gate_nativevec_packed_bf16"
    } else if tensor_core {
        "sys1_gate_nativevec_packed_bf16"
    } else {
        match simt {
            #[cfg(test)]
            SimtProjection::TensorRows32Probe | SimtProjection::TensorRows128Probe => {
                "qwen35_gate_projection_bf16"
            }
            SimtProjection::Compensated => "qwen35_gate_projection_bf16",
            SimtProjection::TiledCompensated => "qwen35_tiled_compensated_projection_bf16",
            SimtProjection::GuardedDouble => "qwen35_guarded_double_projection_bf16",
            SimtProjection::GuardedTensorDouble => "qwen35_guarded_tensor_double_projection_bf16",
            SimtProjection::GuardedTensorDoubleRows16 => {
                "qwen35_guarded_tensor_double_rows16_projection_bf16"
            }
        }
    };
    let function = device.get_or_load_custom_func(kernel, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: if tensor_core {
            (
                sequences.gate_long_count as u32,
                columns.div_ceil(64) as u32,
                1,
            )
        } else if matches!(simt, SimtProjection::TiledCompensated) {
            (rows.div_ceil(4) as u32, columns.div_ceil(4) as u32, 1)
        } else if simt.is_tensor_double() {
            let row_tile = if matches!(simt, SimtProjection::GuardedTensorDoubleRows16) {
                16
            } else {
                8
            };
            (
                rows.div_ceil(row_tile) as u32,
                columns.div_ceil(32) as u32,
                1,
            )
        } else {
            (rows as u32, columns.div_ceil(8) as u32, 1)
        },
        block_dim: (
            if tensor_core {
                tile_rows.min(32) as u32 * 8
            } else if simt.is_tensor_double() || matches!(simt, SimtProjection::TiledCompensated) {
                128
            } else {
                256
            },
            1,
            1,
        ),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&x);
    launch.arg(&w);
    if let Some(bias) = &b {
        launch.arg(bias);
    }
    launch.arg(&mut output);
    let rows = rows as i32;
    if tensor_core || simt.is_tensor_double() || matches!(simt, SimtProjection::TiledCompensated) {
        launch.arg(&rows);
    }
    let inner = inner as i32;
    let columns_arg = columns as i32;
    let radius_coefficient = if !tensor_core
        && !matches!(
            simt,
            SimtProjection::Compensated | SimtProjection::TiledCompensated
        ) {
        Some(guarded_double_radius_coefficient_impl(
            inner as usize,
            simt.is_tensor_double(),
        ))
    } else {
        None
    };
    launch.arg(&inner);
    launch.arg(&columns_arg);
    launch.arg(&tile_starts);
    launch.arg(&tile_ends);
    if let Some(coefficient) = &radius_coefficient {
        launch.arg(coefficient);
    }
    unsafe { launch.launch(config) }.w()?;
    if sequences.gate_short_count > 0 {
        let function = device.get_or_load_custom_func(
            "sys1_gate_tensor16_short_bf16",
            "sys1_qwen35",
            CUDA_PTX,
        )?;
        let mut launch = function.builder();
        let short_count = sequences.gate_short_count as i32;
        launch
            .arg(&x)
            .arg(&w)
            .arg(&mut output)
            .arg(&short_count)
            .arg(&inner)
            .arg(&columns_arg)
            .arg(&short_ids);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    sequences.gate_short_count.div_ceil(4) as u32,
                    columns.div_ceil(8) as u32,
                    1,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
    }

    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length, columns),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn cuda_gate_tensor16_packed_fork(
    input: &Tensor,
    linear: &Linear,
    tensor_core: bool,
    simt: SimtProjection,
    sequences: &PackedSequences,
) -> Result<Tensor> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
        && tensor_core
        && input.device().is_cuda()
        && input.dtype() == DType::BF16
        && linear.weight().dtype() == DType::BF16
        && linear.bias().is_none()
        && linear.weight().dims() == [64, 4096]
        && sequences.gate_short_count >= 4096
    {
        return cuda_gate_gather_small_fork(input, linear, tensor_core, simt, sequences);
    }
    cuda_gate_tensor16_original_packed_fork(input, linear, tensor_core, simt, sequences)
}

#[cfg(feature = "cuda")]
fn cuda_conv_qkv_fork(
    input: &Tensor,
    weights: &Tensor,
    key_heads: usize,
    value_heads: usize,
    sequences: Option<&PackedSequences>,
) -> Result<(Tensor, Tensor, Tensor)> {
    Ok(cuda_conv_qkv_impl(input, weights, key_heads, value_heads, sequences, false)?.0)
}

#[cfg(feature = "cuda")]
fn cuda_conv_qkv_prepared_fork(
    input: &Tensor,
    weights: &Tensor,
    key_heads: usize,
    value_heads: usize,
    sequences: &PackedSequences,
) -> Result<((Tensor, Tensor, Tensor), Option<native_aot_v91::PreparedQv>)> {
    cuda_conv_qkv_impl(
        input,
        weights,
        key_heads,
        value_heads,
        Some(sequences),
        true,
    )
}

#[cfg(feature = "cuda")]
fn cuda_conv_qkv_impl(
    input: &Tensor,
    weights: &Tensor,
    key_heads: usize,
    value_heads: usize,
    sequences: Option<&PackedSequences>,
    prepare_qv: bool,
) -> Result<((Tensor, Tensor, Tensor), Option<native_aot_v91::PreparedQv>)> {
    let (batch, length, channels) = input.dims3()?;
    let (wc, groups, kernel) = weights.dims3()?;
    if batch == 0
        || length == 0
        || key_heads == 0
        || value_heads == 0
        || value_heads % key_heads != 0
        || input.dtype() != DType::BF16
        || weights.dtype() != DType::BF16
        || !input.device().same_device(weights.device())
        || input.elem_count() > i32::MAX as usize
        || wc != channels
        || groups != 1
        || kernel == 0
        || kernel > i32::MAX as usize
        || key_heads
            .checked_mul(2)
            .and_then(|n| n.checked_add(value_heads))
            .and_then(|n| n.checked_mul(128))
            != Some(channels)
    {
        candle_core::bail!("unsupported fused convolution/QKV shape");
    }
    if let Some(s) = sequences {
        if batch != 1 || length != s.total || !input.device().same_device(s.starts.device()) {
            candle_core::bail!("fused convolution packed boundary mismatch");
        }
    }
    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weights.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("fused convolution requires CUDA");
    };
    let (x0, x1) = xl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("convolution input offsets".into()))?;
    let (w0, w1) = wl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("convolution weight offsets".into()))?;
    let source = xs.as_cuda_slice::<bf16>()?.slice(x0..x1);
    let filters = ws.as_cuda_slice::<bf16>()?.slice(w0..w1);
    let device = xs.device().clone();
    let q_count = batch * length * key_heads * 128;
    let v_count = batch * length * value_heads * 128;
    let mut q = unsafe { device.alloc::<f32>(q_count)? };
    let mut k = unsafe { device.alloc::<f32>(q_count)? };
    let mut v = unsafe { device.alloc::<f32>(v_count)? };
    let rows = (q_count / 128) as u32;
    let kh = key_heads as i32;
    let vh = value_heads as i32;
    let scale = 128f64.powf(-0.5) as f32;
    let length_arg = length as i32;
    let kernel_arg = kernel as i32;

    let lookup157 = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && batch * length >= 513
        && kernel == 4
        && key_heads == 16
        && value_heads == 32
        && x0 % 4 == 0
        && w0 % 8 == 0;
    let mut prepared = if prepare_qv && lookup157 {
        if let Some(sequences) = sequences {
            native_aot_v91::prepare_qv(input.device(), &sequences.lengths)?
        } else {
            None
        }
    } else {
        None
    };
    let name = if prepared.is_some() {
        "sys1_convcast_qkv_packed"
    } else {
        match (lookup157, sequences.is_some()) {
            (true, true) => "sys1_conv_lookup156_packed_fork",
            (true, false) => "sys1_conv_lookup156_uniform_fork",
            (false, true) => "sys1_conv_qkv_packed_fork",
            (false, false) => "sys1_conv_qkv_uniform_fork",
        }
    };
    let function = device.get_or_load_custom_func(name, "sys1_conv_qkv", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(4), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut run =
        |table: Option<&candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>>| -> Result<()> {
            let prepared_limit = prepared
                .as_ref()
                .map_or(0., native_aot_v91::PreparedQv::limit);
            let mut launch = function.builder();
            launch.arg(&source);
            launch.arg(&filters);
            launch.arg(&mut q);
            launch.arg(&mut k);
            launch.arg(&mut v);
            macro_rules! tail {
                () => {{
                    launch.arg(&rows);
                    launch.arg(&kh);
                    launch.arg(&vh);
                    launch.arg(&scale);
                    launch.arg(&length_arg);
                    launch.arg(&kernel_arg);
                    if let Some(table) = table {
                        launch.arg(table);
                    }
                    if let Some(ticket) = prepared.as_mut() {
                        let (qhalf, khalf, vhalf, _, bad) = ticket.outputs();
                        launch
                            .arg(qhalf)
                            .arg(khalf)
                            .arg(vhalf)
                            .arg(&prepared_limit)
                            .arg(bad);
                    }
                }};
            }

            if let Some(s) = sequences {
                let (ss, sl) = s.starts.storage_and_layout();
                let Storage::Cuda(ss) = &*ss else {
                    candle_core::bail!("starts require CUDA");
                };
                let (s0, s1) = sl
                    .contiguous_offsets()
                    .ok_or_else(|| candle_core::Error::Msg("start offsets".into()))?;
                let starts = ss.as_cuda_slice::<u32>()?.slice(s0..s1);
                launch.arg(&starts);
                tail!();
                unsafe { launch.launch(config) }.w()?;
            } else {
                tail!();
                unsafe { launch.launch(config) }.w()?;
            }
            Ok(())
        };
    if lookup157 {
        #[cfg(test)]
        CONV_LOOKUP157_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::kernels::activation::with_sigmoid_lookup_fork(&device, |table| run(Some(table)))?;
    } else {
        run(None)?;
    }
    let wrap = |data, heads| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(data, device.clone())),
            (batch, length, heads, 128),
            BackpropOp::none(),
            false,
        )
    };
    let query = wrap(q, key_heads);
    let key = wrap(k, key_heads);
    let value = wrap(v, value_heads);
    if let Some(ticket) = prepared.as_mut() {
        ticket.bind(&query, &key, &value);
        #[cfg(test)]
        CONV_PREPARED_QV_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(((query, key, value), prepared))
}

#[cfg(feature = "cuda")]
fn cuda_delta_rule_token_major_bfout_fork(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    subwarp16: bool,
) -> Result<Tensor> {
    if !subwarp16 {
        candle_core::bail!("BF16-output scalar requires selected subwarp16");
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let device = q_storage.device().clone();
    let output = unsafe { device.alloc::<bf16>(batch * heads * length * value_dim)? };

    let reuse = cuda_short_value_reuse_selected_fork(
        &device,
        2,
        heads,
        value_dim,
        (batch as u128) * (length as u128),
        length,
    )?;
    let short_columns = if reuse { 32 } else { 8 };
    let name = if reuse {
        "sys1_shortvalues_2_bf16"
    } else {
        "sys1_uniform_delta_bfout_fork"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (
            (batch * heads) as u32,
            value_dim.div_ceil(short_columns) as u32,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&query);
    launch.arg(&key);
    launch.arg(&value);
    launch.arg(&gate);
    launch.arg(&beta);
    launch.arg(&output);
    let length = length as i32;
    let heads_arg = heads as i32;
    let key_heads_arg = key_heads as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&heads_arg);
    launch.arg(&key_heads_arg);
    launch.arg(&value_dim);
    unsafe { launch.launch(config) }.w()?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, heads, length as usize, value_dim as usize),
        BackpropOp::none(),
        false,
    );
    output.transpose(1, 2)
}

#[cfg(feature = "cuda")]
fn cuda_delta_rule_packed_bfout_fork(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    subwarp16: bool,
) -> Result<Tensor> {
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    if !subwarp16 || batch != 1 || length != sequences.total {
        candle_core::bail!("packed delta unsupported layout")
    }
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let (metadata_storage, metadata_layout) = sequences.cumulative.storage_and_layout();
    let Storage::Cuda(metadata_storage) = &*metadata_storage else {
        candle_core::bail!("packed metadata must be CUDA")
    };
    let (start, end) = offsets(metadata_layout)?;
    let cumulative = metadata_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let device = q_storage.device().clone();
    let output = unsafe { device.alloc::<bf16>(batch * heads * length * value_dim)? };

    let longest = *sequences.lengths.iter().max().unwrap_or(&0);
    let staged = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && value_dim == 128
        && heads >= 32
        && (65..512).contains(&longest)
        && q_start % 4 == 0
        && k_start % 4 == 0
        && v_start % 4 == 0;
    let precomputed_decay = if staged {
        let values = unsafe { device.alloc::<f32>(length * heads)? };
        let function =
            device.get_or_load_custom_func("qwen35_preexp_gate_f32", "sys1_qwen35", CUDA_PTX)?;
        let count = (length * heads) as u32;
        let mut launch = function.builder();
        launch.arg(&gate);
        launch.arg(&values);
        launch.arg(&count);
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }.w()?;
        Some(values)
    } else {
        None
    };
    let reuse = !staged
        && cuda_short_value_reuse_selected_fork(
            &device,
            1,
            heads,
            value_dim,
            sequences.total as u128,
            longest,
        )?;
    let short_columns = if staged {
        16
    } else if reuse {
        32
    } else {
        8
    };
    let name = if staged {
        "sys1_packed_delta_prefetch8_columns4_preexp_bf16"
    } else if reuse {
        "sys1_shortvalues_1_bf16"
    } else {
        "sys1_packed_delta_bfout_fork"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (
            (sequences.lengths.len() * heads) as u32,
            value_dim.div_ceil(short_columns) as u32,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&query);
    launch.arg(&key);
    launch.arg(&value);
    if let Some(decay) = &precomputed_decay {
        launch.arg(decay);
    } else {
        launch.arg(&gate);
    }
    launch.arg(&beta);
    launch.arg(&output);
    launch.arg(&cumulative);
    let length = length as i32;
    let heads_arg = heads as i32;
    let key_heads_arg = key_heads as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&heads_arg);
    launch.arg(&key_heads_arg);
    launch.arg(&value_dim);
    unsafe { launch.launch(config) }.w()?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length as usize, heads, value_dim as usize),
        BackpropOp::none(),
        false,
    );
    Ok(output)
}

#[cfg(feature = "cuda")]
fn cuda_c64bfout_uniform_fork(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    tile_columns: usize,
    profiling: bool,
) -> Result<Tensor> {
    if q.dims().len() == 4
        && v.dims().len() == 4
        && q.dim(2)? == 16
        && q.dim(3)? == 128
        && v.dim(2)? == 32
        && v.dim(3)? == 128
        && q.dim(0)? > 0
        && q.dim(1)? > 0
    {
        let (batch, length, _, _) = q.dims4()?;
        let total = batch
            .checked_mul(length)
            .ok_or_else(|| candle_core::Error::Msg("uniform AOT token overflow".into()))?;
        let packed_q = q.reshape((1, total, 16, 128))?;
        let packed_k = k.reshape((1, total, 16, 128))?;
        let packed_v = v.reshape((1, total, 32, 128))?;
        let packed_g = g.reshape((1, total, 32))?;
        let packed_beta = beta.reshape((1, total, 32))?;

        if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80")
            && length >= 128
            && tile_columns == 32
            && [q, k, v, g, beta].iter().all(|x| x.dtype() == DType::F32)
        {
            let sequences = PackedSequences::new(&vec![length; batch], q.device())?;
            return cuda_c64bfout_varlen_fork(
                &packed_q,
                &packed_k,
                &packed_v,
                &packed_g,
                &packed_beta,
                &sequences,
                profiling,
            )?
            .reshape((batch, length, 32, 128));
        }
        if let Some(output) = native_aot_v91::try_forward(
            &packed_q,
            &packed_k,
            &packed_v,
            &packed_g,
            &packed_beta,
            &vec![length; batch],
        )? {
            return output.reshape((batch, length, 32, 128));
        }
    }

    if [q, k, v, g, beta].iter().any(|x| x.dtype() != DType::F32) {
        candle_core::bail!("chunk probe F32 inputs required");
    }
    if tile_columns != 32 {
        candle_core::bail!("C64 state requires N32 caller contract");
    }
    let experimental_bf16 = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90");
    let (tile_columns, state_rows, carry_shared) = if experimental_bf16 {
        (64usize, 64usize, 98336u32)
    } else {
        (32usize, 128usize, 58400u32)
    };
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let device = q_storage.device().clone();

    if ![32usize, 64].contains(&tile_columns)
        || batch == 0
        || length == 0
        || value_dim > 128
        || [
            q.elem_count(),
            k.elem_count(),
            v.elem_count(),
            g.elem_count(),
        ]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk probe requires nonempty F32 K128,V<=128,small indices");
    }
    let instances = batch
        .checked_mul(heads)
        .and_then(|x| x.checked_mul(length.div_ceil(64)))
        .ok_or_else(|| candle_core::Error::Msg("chunk count overflow".into()))?;
    let w_count = instances
        .checked_mul(64 * 128)
        .ok_or_else(|| candle_core::Error::Msg("chunk W overflow".into()))?;
    let u_count = instances
        .checked_mul(64 * value_dim)
        .ok_or_else(|| candle_core::Error::Msg("chunk U overflow".into()))?;
    let c_count = instances
        .checked_mul(64 * 64)
        .ok_or_else(|| candle_core::Error::Msg("chunk C overflow".into()))?;
    let d_count = instances
        .checked_mul(128)
        .ok_or_else(|| candle_core::Error::Msg("decay count overflow".into()))?;
    if [w_count, u_count, c_count, d_count]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk scratch index overflow");
    }
    let result = (|| -> Result<Tensor> {
        let mut slot = CHUNK64_SCRATCH
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned chunk scratch".into()))?;
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
        {
            *slot = Some(ChunkScratch {
                stream,
                w: None,
                u: None,
                c: None,
                decays: None,
                snapshots: None,
            });
        }
        let workspace = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing chunk scratch".into()))?;
        let ChunkScratch {
            w,
            u,
            c,
            decays,
            snapshots,
            ..
        } = workspace;
        if profiling {
            device.cuda_stream().synchronize().w()?;
        }
        let mut stage_started = Instant::now();
        let w = grow_chunk_buffer(w, &device, w_count)?;
        let u = grow_chunk_buffer(u, &device, u_count)?;
        let c = grow_chunk_buffer(c, &device, c_count)?;
        let decays = grow_chunk_buffer(decays, &device, d_count)?;
        let output_count = batch
            .checked_mul(length)
            .and_then(|n| n.checked_mul(heads))
            .and_then(|n| n.checked_mul(value_dim))
            .ok_or_else(|| candle_core::Error::Msg("register-carry C64 output overflow".into()))?;
        if output_count > i32::MAX as usize {
            candle_core::bail!("padded output index overflow");
        }
        let guarded = !experimental_bf16
            && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("90" | "120"));
        let snapshot_count = if guarded { instances } else { 1usize };
        if snapshot_count > i32::MAX as usize {
            candle_core::bail!("snapshot index overflow");
        }
        let snapshots = grow_chunk_buffer(snapshots, &device, snapshot_count)?;
        let mut output = unsafe { device.alloc::<bf16>(output_count)? };
        if profiling {
            println!(
                "C64_FLAGS_WORKSPACE retained_bytes={} output_bytes={} single_entry_per_process=true owned_stream_identity=true output_never_reused=true amortized_warm_operator_not_serving=true",
                (w.len() + u.len() + c.len() + decays.len() + snapshots.len()) * 4,
                output_count * 2
            );
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=scratch batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        let length_arg = length as i32;
        let heads_arg = heads as i32;
        let key_heads_arg = key_heads as i32;
        let value_dim_arg = value_dim as i32;
        {
            let function = device.get_or_load_custom_func(
                if guarded {
                    "sys1_c64zguard_uniform_prepare_f32"
                } else {
                    if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("90" | "120")) {
                        if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                            "sys1_c64prepvload2_uniform_prepare_f32"
                        } else {
                            "sys1_c64sub8vector_uniform_prepare_f32"
                        }
                    } else {
                        "sys1_c64slice_uniform_prepare_f32"
                    }
                },
                "sys1_chunk_delta",
                CUDA_PTX,
            )?;
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,49664).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&value);
            launch.arg(&gate);
            launch.arg(&beta);
            launch.arg(&mut *w);
            launch.arg(&mut *u);
            launch.arg(&mut *c);
            launch.arg(&mut *decays);
            if guarded {
                launch.arg(&mut *snapshots);
            }
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (instances as u32, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 49664,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=prepare batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        {
            let name = if experimental_bf16 {
                "sys1_c64fp16phase3vec2_n64_uniform_f32"
            } else if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("90" | "120")) {
                "sys1_c64zguard_n32_uniform32_f32"
            } else {
                "sys1_c64bfout_uniform32_f32"
            };
            let function = device.get_or_load_custom_func(name, "sys1_chunk_delta", CUDA_PTX)?;
            if profiling {
                println!(
                    "C64_FLAGS_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} diagnostic_not_serving=true",
                    function.num_regs().w()?,
                    function.local_size_bytes().w()?,
                    function.shared_size_bytes().w()?
                );
            }
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,carry_shared as i32).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&gate);
            launch.arg(&*w);
            launch.arg(&*u);
            launch.arg(&*c);
            launch.arg(&*decays);
            launch.arg(&mut output);
            launch.arg(&mut *snapshots);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        (batch * heads) as u32,
                        value_dim.div_ceil(tile_columns) as u32,
                        1,
                    ),
                    block_dim: (((state_rows / 16) * (tile_columns / 8) * 32) as u32, 1, 1),
                    shared_mem_bytes: carry_shared,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=state batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
        }

        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (batch, length, heads, value_dim),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    })();
    if result.is_err() {
        if let Ok(mut slot) = CHUNK64_SCRATCH.lock() {
            slot.take();
        }
    }
    result
}

#[cfg(feature = "cuda")]
fn cuda_c64bfout_varlen_fork(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    profiling: bool,
) -> Result<Tensor> {
    cuda_c64bfout_varlen_prepared_fork(q, k, v, g, beta, sequences, profiling, None)
}

#[cfg(feature = "cuda")]
fn cuda_c64bfout_varlen_prepared_fork(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    profiling: bool,
    prepared: Option<native_aot_v91::PreparedQv>,
) -> Result<Tensor> {
    let output = if let Some(ticket) = prepared {
        native_aot_v91::try_forward_prepared(q, k, v, g, beta, &sequences.lengths, ticket)?
    } else {
        native_aot_v91::try_forward(q, k, v, g, beta, &sequences.lengths)?
    };
    if let Some(output) = output {
        return Ok(output);
    }

    if [q, k, v, g, beta].iter().any(|x| x.dtype() != DType::F32) {
        candle_core::bail!("chunk probe F32 inputs required");
    }

    if q.dim(0)? != 1 || q.dim(1)? != sequences.total {
        candle_core::bail!("segmented C64 shape mismatch")
    }
    if sequences.c64_chunk_count == 0 {
        return cuda_delta_rule_packed_bfout_fork(q, k, v, g, beta, sequences, true);
    }
    if [
        k,
        v,
        g,
        beta,
        &sequences.cumulative,
        &sequences.c64_chunks,
        &sequences.c64_long_ids,
        &sequences.c64_short_ids,
    ]
    .iter()
    .any(|t| !q.device().same_device(t.device()))
    {
        candle_core::bail!("segmented C64 device mismatch")
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let (cumulative_storage, cumulative_layout) = sequences.cumulative.storage_and_layout();
    let Storage::Cuda(cumulative_storage) = &*cumulative_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(cumulative_layout)?;
    let cumulative = cumulative_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let (chunk_offsets_storage, chunk_offsets_layout) = sequences.c64_chunks.storage_and_layout();
    let Storage::Cuda(chunk_offsets_storage) = &*chunk_offsets_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(chunk_offsets_layout)?;
    let chunk_offsets = chunk_offsets_storage
        .as_cuda_slice::<u32>()?
        .slice(start..end);
    let (long_ids_storage, long_ids_layout) = sequences.c64_long_ids.storage_and_layout();
    let Storage::Cuda(long_ids_storage) = &*long_ids_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(long_ids_layout)?;
    let long_ids = long_ids_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let (short_ids_storage, short_ids_layout) = sequences.c64_short_ids.storage_and_layout();
    let Storage::Cuda(short_ids_storage) = &*short_ids_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(short_ids_layout)?;
    let short_ids = short_ids_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let device = q_storage.device().clone();
    let sm_count = cuda_c64_sm_count_fork(&device)?;
    let wide_blocks = sequences
        .c64_long_count
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(value_dim.div_ceil(64)))
        .ok_or_else(|| candle_core::Error::Msg("wide recurrent grid overflow".into()))?;
    let experimental_bf16 = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90");

    let sm80_prepack =
        option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80") && sequences.c64_long_count > 0;

    let use_prepack = sm80_prepack
        || (experimental_bf16
            && sequences
                .lengths
                .iter()
                .filter(|&&n| n >= 128)
                .copied()
                .sum::<usize>()
                >= 4096);

    let sm80_full01 = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80");
    let use_wide = sm80_full01
        || experimental_bf16
        || cuda_c64_scheduled_geometry_fork(sequences, heads, value_dim, sm_count)?;
    let _ = wide_blocks;
    let (tile_columns, state_rows, carry_shared) = if use_prepack {
        (32usize, 64usize, 49184u32)
    } else if use_wide {
        (
            64usize,
            64usize,
            if matches!(
                option_env!("SYS1_CUDA_COMPUTE_CAP"),
                Some("80" | "90" | "120")
            ) {
                98336u32
            } else {
                82976u32
            },
        )
    } else {
        (32usize, 128usize, 58400u32)
    };
    if use_wide {
        static NOTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        NOTED.get_or_init(|| {
            tracing::info!(
                value_tile = tile_columns,
                threads = (state_rows / 16) * (tile_columns / 8) * 32,
                shared_bytes = carry_shared,
                sm_count,
                "Using automatic recurrent carry geometry"
            )
        });
    }

    if ![32usize, 64].contains(&tile_columns)
        || batch == 0
        || length == 0
        || value_dim > 128
        || [
            q.elem_count(),
            k.elem_count(),
            v.elem_count(),
            g.elem_count(),
        ]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk probe requires nonempty F32 K128,V<=128,small indices");
    }
    let instances = heads
        .checked_mul(sequences.c64_chunk_count)
        .ok_or_else(|| candle_core::Error::Msg("chunk count overflow".into()))?;
    let w_count = instances
        .checked_mul(64 * 128)
        .ok_or_else(|| candle_core::Error::Msg("chunk W overflow".into()))?;
    let u_count = instances
        .checked_mul(64 * value_dim)
        .ok_or_else(|| candle_core::Error::Msg("chunk U overflow".into()))?;
    let c_count = instances
        .checked_mul(64 * 64)
        .ok_or_else(|| candle_core::Error::Msg("chunk C overflow".into()))?;
    let d_count = instances
        .checked_mul(128)
        .ok_or_else(|| candle_core::Error::Msg("decay count overflow".into()))?;
    if [w_count, u_count, c_count, d_count]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk scratch index overflow");
    }
    let result = (|| -> Result<Tensor> {
        let mut slot = CHUNK64_SCRATCH
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned chunk scratch".into()))?;
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
        {
            *slot = Some(ChunkScratch {
                stream,
                w: None,
                u: None,
                c: None,
                decays: None,
                snapshots: None,
            });
        }
        let workspace = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing chunk scratch".into()))?;
        let ChunkScratch {
            w,
            u,
            c,
            decays,
            snapshots,
            ..
        } = workspace;
        if profiling {
            device.cuda_stream().synchronize().w()?;
        }
        let mut stage_started = Instant::now();
        let w = grow_chunk_buffer(w, &device, w_count)?;
        let u = grow_chunk_buffer(u, &device, u_count)?;
        let c = grow_chunk_buffer(c, &device, c_count)?;
        let decays = grow_chunk_buffer(decays, &device, d_count)?;
        let output_count = length
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(value_dim))
            .ok_or_else(|| candle_core::Error::Msg("segmented C64 output overflow".into()))?;
        if output_count > i32::MAX as usize {
            candle_core::bail!("padded output index overflow");
        }
        let guarded =
            matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("90" | "120")) && !use_wide;
        let snapshot_count = if guarded { instances } else { 1usize };
        if snapshot_count > i32::MAX as usize {
            candle_core::bail!("snapshot index overflow");
        }
        let snapshots = grow_chunk_buffer(snapshots, &device, snapshot_count)?;
        let mut output = unsafe { device.alloc::<bf16>(output_count)? };
        let packed_count = instances
            .checked_mul(28672)
            .ok_or_else(|| candle_core::Error::Msg("prepacked C64 allocation overflow".into()))?;
        let mut prepacked = if use_prepack {
            Some(unsafe { device.alloc::<u16>(packed_count)? })
        } else {
            None
        };
        let mut pack_flags = if use_prepack {
            Some(unsafe { device.alloc::<u32>(instances)? })
        } else {
            None
        };

        if profiling {
            println!(
                "C64_CARRY_WORKSPACE retained_bytes={} output_bytes={} single_entry_per_process=true owned_stream_identity=true output_never_reused=true amortized_warm_operator_not_serving=true",
                (w.len() + u.len() + c.len() + decays.len() + snapshots.len()) * 4,
                output_count * 2
            );
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=scratch batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        let sequence_count = sequences.lengths.len() as i32;
        let length_arg = length as i32;
        let heads_arg = heads as i32;
        let key_heads_arg = key_heads as i32;
        let value_dim_arg = value_dim as i32;
        {
            let function = device.get_or_load_custom_func(
                if use_prepack {
                    "sys1_c64storepack15_varlen_prepare_f32"
                } else if guarded {
                    "sys1_c64zguard_varlen_prepare_f32"
                } else {
                    if matches!(
                        option_env!("SYS1_CUDA_COMPUTE_CAP"),
                        Some("80" | "90" | "120")
                    ) {
                        if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                            "sys1_c64prepvload2_varlen_prepare_f32"
                        } else {
                            "sys1_c64sub8vector_varlen_prepare_f32"
                        }
                    } else {
                        "sys1_c64slice_varlen_prepare_f32"
                    }
                },
                "sys1_chunk_delta",
                CUDA_PTX,
            )?;
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,49664).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&value);
            launch.arg(&gate);
            launch.arg(&beta);
            launch.arg(&mut *w);
            launch.arg(&mut *u);
            launch.arg(&mut *c);
            launch.arg(&mut *decays);
            if guarded {
                launch.arg(&mut *snapshots);
            }
            launch.arg(&cumulative);
            launch.arg(&chunk_offsets);
            launch.arg(&sequence_count);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            if use_prepack {
                launch.arg(prepacked.as_mut().ok_or_else(|| {
                    candle_core::Error::Msg("missing fused packed C64 buffer".into())
                })?);
                launch.arg(
                    pack_flags
                        .as_mut()
                        .ok_or_else(|| candle_core::Error::Msg("missing fused C64 flags".into()))?,
                );
            }
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (instances as u32, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 49664,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=prepare batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        {
            if sequences.c64_short_count > 0 {
                let short_tokens = sequences
                    .lengths
                    .iter()
                    .filter(|&&n| {
                        n < (if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                            128
                        } else {
                            512
                        })
                    })
                    .map(|&n| n as u128)
                    .sum();
                let longest_short = sequences
                    .lengths
                    .iter()
                    .filter(|&&n| {
                        n < (if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                            128
                        } else {
                            512
                        })
                    })
                    .copied()
                    .max()
                    .unwrap_or(0);
                let reuse = cuda_short_value_reuse_selected_fork(
                    &device,
                    0,
                    heads,
                    value_dim,
                    short_tokens,
                    longest_short,
                )?;
                let short_columns = if reuse { 32 } else { 8 };
                let name = if reuse {
                    "sys1_shortvalues_0_bf16"
                } else {
                    "sys1_c64bfout_short_bf16"
                };
                let function = device.get_or_load_custom_func(name, "sys1_c64v_short", CUDA_PTX)?;
                let mut launch = function.builder();
                launch
                    .arg(&query)
                    .arg(&key)
                    .arg(&value)
                    .arg(&gate)
                    .arg(&beta)
                    .arg(&mut output)
                    .arg(&cumulative)
                    .arg(&short_ids)
                    .arg(&length_arg)
                    .arg(&heads_arg)
                    .arg(&key_heads_arg)
                    .arg(&value_dim_arg);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (
                            (sequences.c64_short_count * heads) as u32,
                            value_dim.div_ceil(short_columns) as u32,
                            1,
                        ),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .w()?;
            }
            let name = if use_prepack {
                "sys1_c64packed32v40_n32_varlen_f32"
            } else if experimental_bf16 {
                "sys1_c64fp16phase3vec2_n64_varlen_f32"
            } else if use_wide {
                if matches!(
                    option_env!("SYS1_CUDA_COMPUTE_CAP"),
                    Some("80" | "90" | "120")
                ) {
                    "sys1_c64full01_n64_varlen_f32"
                } else {
                    "sys1_c64wide_varlen_f32"
                }
            } else if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("90" | "120")) {
                "sys1_c64zguard_n32_varlen32_f32"
            } else {
                "sys1_c64bfout_varlen32_f32"
            };
            let function = device.get_or_load_custom_func(name, "sys1_chunk_delta", CUDA_PTX)?;
            if profiling {
                println!(
                    "C64_CARRY_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} diagnostic_not_serving=true",
                    function.num_regs().w()?,
                    function.local_size_bytes().w()?,
                    function.shared_size_bytes().w()?
                );
            }
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,carry_shared as i32).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&gate);
            launch.arg(&*w);
            launch.arg(&*u);
            launch.arg(&*c);
            launch.arg(&*decays);
            launch.arg(&mut output);
            launch.arg(&mut *snapshots);
            launch.arg(&cumulative);
            launch.arg(&chunk_offsets);
            launch.arg(&long_ids);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            if use_prepack {
                launch.arg(
                    prepacked.as_ref().ok_or_else(|| {
                        candle_core::Error::Msg("missing packed C64 input".into())
                    })?,
                );
                launch.arg(
                    pack_flags
                        .as_ref()
                        .ok_or_else(|| candle_core::Error::Msg("missing C64 flag input".into()))?,
                );
            }

            let value_tiles = value_dim.div_ceil(tile_columns);
            let sequence_heads = sequences.c64_long_count.checked_mul(heads).ok_or_else(|| {
                candle_core::Error::Msg("carry sequence-head grid overflow".into())
            })?;
            let (carry_x, carry_y) = if use_prepack {
                (
                    sequence_heads.checked_mul(value_tiles).ok_or_else(|| {
                        candle_core::Error::Msg("value-fast carry grid overflow".into())
                    })?,
                    1usize,
                )
            } else {
                (sequence_heads, value_tiles)
            };
            let carry_x = u32::try_from(carry_x)
                .map_err(|_| candle_core::Error::Msg("carry x grid exceeds u32".into()))?;
            let carry_y = u32::try_from(carry_y)
                .map_err(|_| candle_core::Error::Msg("carry y grid exceeds u32".into()))?;
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (carry_x, carry_y, 1),
                    block_dim: (((state_rows / 16) * (tile_columns / 8) * 32) as u32, 1, 1),
                    shared_mem_bytes: carry_shared,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=state batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
        }

        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (1, length, heads, value_dim),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    })();
    if result.is_err() {
        if let Ok(mut slot) = CHUNK64_SCRATCH.lock() {
            slot.take();
        }
    }
    result
}

#[cfg(feature = "cuda")]
fn cuda_post_finalize_bf16_input_fork(
    input: &Tensor,
    gate: &Tensor,
    norm: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    let columns = input.dim(D::Minus1)?;
    if input.dtype() != DType::BF16
        || gate.dtype() != DType::BF16
        || norm.dtype() != DType::BF16
        || input.dims() != gate.dims()
        || norm.dims() != [columns]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA post-finalize shape or dtype")
    }
    let (batch, length, heads, width) = input.dims4()?;
    if width == 0 || batch == 0 || length == 0 || heads == 0 || input.stride()[3] != 1 {
        candle_core::bail!("unsupported strided post-finalize layout");
    }
    let columns_arg = i32::try_from(columns)
        .map_err(|_| candle_core::Error::Msg("post-finalize width overflow".into()))?;
    let length_arg =
        u64::try_from(length).map_err(|_| candle_core::Error::Msg("length overflow".into()))?;
    let heads_arg =
        u64::try_from(heads).map_err(|_| candle_core::Error::Msg("heads overflow".into()))?;
    let strides: Vec<u64> = input.stride()[..3]
        .iter()
        .map(|&s| u64::try_from(s).map_err(|_| candle_core::Error::Msg("stride overflow".into())))
        .collect::<Result<_>>()?;
    let gate = gate.contiguous()?;
    let norm = norm.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (norm_storage, norm_layout) = norm.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(gate_storage), Storage::Cuda(norm_storage)) =
        (&*input_storage, &*gate_storage, &*norm_storage)
    else {
        candle_core::bail!("Qwen3.5 post-finalize requires CUDA storage")
    };
    let input_start = input_layout.start_offset();
    let span =
        input
            .dims()
            .iter()
            .zip(input.stride())
            .try_fold(1usize, |span, (&dim, &stride)| {
                dim.checked_sub(1)
                    .and_then(|n| n.checked_mul(stride))
                    .and_then(|n| span.checked_add(n))
                    .ok_or_else(|| candle_core::Error::Msg("post-finalize span overflow".into()))
            })?;
    let input_end = input_start
        .checked_add(span)
        .ok_or_else(|| candle_core::Error::Msg("post-finalize offset overflow".into()))?;
    if input_end > input_storage.as_cuda_slice::<bf16>()?.len() {
        candle_core::bail!("post-finalize input span exceeds allocation");
    }
    if !std::sync::Arc::ptr_eq(
        &input_storage.device().cuda_stream(),
        &gate_storage.device().cuda_stream(),
    ) || !std::sync::Arc::ptr_eq(
        &input_storage.device().cuda_stream(),
        &norm_storage.device().cuda_stream(),
    ) {
        candle_core::bail!("post-finalize requires one CUDA stream");
    }
    let (gate_start, gate_end) = gate_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize gate is not contiguous".into())
    })?;
    let (norm_start, norm_end) = norm_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize norm is not contiguous".into())
    })?;
    let source = input_storage
        .as_cuda_slice::<bf16>()?
        .slice(input_start..input_end);
    let gate = gate_storage
        .as_cuda_slice::<bf16>()?
        .slice(gate_start..gate_end);
    let norm = norm_storage
        .as_cuda_slice::<bf16>()?
        .slice(norm_start..norm_end);
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let rows = u32::try_from(count / columns)
        .map_err(|_| candle_core::Error::Msg("post-finalize row count overflow".into()))?;
    if count > i32::MAX as usize {
        candle_core::bail!("post-finalize index overflow");
    }
    let mut output = unsafe { device.alloc::<bf16>(count)? };
    let function = device.get_or_load_custom_func(
        "sys1_post_finalize_bf16_input_fork",
        "sys1_qwen35",
        CUDA_PTX,
    )?;
    let config = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&gate);
    launch.arg(&norm);
    launch.arg(&mut output);
    let columns = columns_arg;
    let epsilon = epsilon as f32;
    launch.arg(&columns);
    launch.arg(&epsilon);
    launch.arg(&length_arg);
    launch.arg(&heads_arg);
    launch.arg(&strides[0]);
    launch.arg(&strides[1]);
    launch.arg(&strides[2]);
    unsafe { launch.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
struct Bf16PostF32LookupFork {
    stream: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    table: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
}

#[cfg(feature = "cuda")]
static BF16_POST_F32_LOOKUP_FORK: std::sync::Mutex<Option<Bf16PostF32LookupFork>> =
    std::sync::Mutex::new(None);

#[cfg(all(test, feature = "cuda"))]
static BF16_POST_F32_LOOKUP_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "cuda")]
fn cuda_post_finalize_bf16_warp_fork(
    input: &Tensor,
    gate: &Tensor,
    norm: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    cuda_post_finalize_bf16_warp_impl_fork(
        input,
        gate,
        norm,
        epsilon,
        option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90"),
    )
}

#[cfg(feature = "cuda")]
fn cuda_post_finalize_bf16_warp_impl_fork(
    input: &Tensor,
    gate: &Tensor,
    norm: &Tensor,
    epsilon: f64,
    use_lookup: bool,
) -> Result<Tensor> {
    let columns = input.dim(D::Minus1)?;
    if columns != 128 {
        return cuda_post_finalize_bf16_input_fork(input, gate, norm, epsilon);
    }

    let (batch, length, heads, width) = input.dims4()?;
    let merged_gate = use_lookup
        && heads == 32
        && width == 128
        && gate.dims() == [batch, length, 4096]
        && gate.stride() == [length * 12288, 12288, 1];
    if input.dtype() != DType::BF16
        || gate.dtype() != DType::BF16
        || norm.dtype() != DType::BF16
        || (input.dims() != gate.dims() && !merged_gate)
        || norm.dims() != [columns]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA post-finalize shape or dtype")
    }
    if width == 0 || batch == 0 || length == 0 || heads == 0 || input.stride()[3] != 1 {
        candle_core::bail!("unsupported strided post-finalize layout");
    }
    let columns_arg = i32::try_from(columns)
        .map_err(|_| candle_core::Error::Msg("post-finalize width overflow".into()))?;
    let length_arg =
        u64::try_from(length).map_err(|_| candle_core::Error::Msg("length overflow".into()))?;
    let heads_arg =
        u64::try_from(heads).map_err(|_| candle_core::Error::Msg("heads overflow".into()))?;
    let strides: Vec<u64> = input.stride()[..3]
        .iter()
        .map(|&s| u64::try_from(s).map_err(|_| candle_core::Error::Msg("stride overflow".into())))
        .collect::<Result<_>>()?;
    let gate = if merged_gate {
        gate.clone()
    } else {
        gate.contiguous()?
    };
    let norm = norm.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (norm_storage, norm_layout) = norm.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(gate_storage), Storage::Cuda(norm_storage)) =
        (&*input_storage, &*gate_storage, &*norm_storage)
    else {
        candle_core::bail!("Qwen3.5 post-finalize requires CUDA storage")
    };
    let input_start = input_layout.start_offset();
    let span =
        input
            .dims()
            .iter()
            .zip(input.stride())
            .try_fold(1usize, |span, (&dim, &stride)| {
                dim.checked_sub(1)
                    .and_then(|n| n.checked_mul(stride))
                    .and_then(|n| span.checked_add(n))
                    .ok_or_else(|| candle_core::Error::Msg("post-finalize span overflow".into()))
            })?;
    let input_end = input_start
        .checked_add(span)
        .ok_or_else(|| candle_core::Error::Msg("post-finalize offset overflow".into()))?;
    if input_end > input_storage.as_cuda_slice::<bf16>()?.len() {
        candle_core::bail!("post-finalize input span exceeds allocation");
    }
    if !std::sync::Arc::ptr_eq(
        &input_storage.device().cuda_stream(),
        &gate_storage.device().cuda_stream(),
    ) || !std::sync::Arc::ptr_eq(
        &input_storage.device().cuda_stream(),
        &norm_storage.device().cuda_stream(),
    ) {
        candle_core::bail!("post-finalize requires one CUDA stream");
    }
    let (gate_start, gate_end) = if merged_gate {
        let start = gate_layout.start_offset();
        let end = (batch - 1)
            .checked_mul(length * 12288)
            .and_then(|n| n.checked_add((length - 1) * 12288))
            .and_then(|n| n.checked_add(4096))
            .and_then(|n| start.checked_add(n))
            .ok_or_else(|| candle_core::Error::Msg("merged gate span overflow".into()))?;
        if end > gate_storage.as_cuda_slice::<bf16>()?.len() {
            candle_core::bail!("merged gate span exceeds allocation")
        }
        (start, end)
    } else {
        gate_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("Qwen3.5 post-finalize gate is not contiguous".into())
        })?
    };
    let (norm_start, norm_end) = norm_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize norm is not contiguous".into())
    })?;
    let source = input_storage
        .as_cuda_slice::<bf16>()?
        .slice(input_start..input_end);
    let gate = gate_storage
        .as_cuda_slice::<bf16>()?
        .slice(gate_start..gate_end);
    let norm = norm_storage
        .as_cuda_slice::<bf16>()?
        .slice(norm_start..norm_end);
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let rows = u32::try_from(count / columns)
        .map_err(|_| candle_core::Error::Msg("post-finalize row count overflow".into()))?;
    if count > i32::MAX as usize {
        candle_core::bail!("post-finalize index overflow");
    }
    let mut output = unsafe { device.alloc::<bf16>(count)? };
    let mut lookup_slot = if use_lookup {
        Some(
            BF16_POST_F32_LOOKUP_FORK
                .lock()
                .map_err(|_| candle_core::Error::Msg("poisoned post SiLU lookup".into()))?,
        )
    } else {
        None
    };
    if let Some(slot) = lookup_slot.as_mut() {
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|entry| !std::sync::Arc::ptr_eq(&entry.stream, &stream))
        {
            let mut table = unsafe { device.alloc::<f32>(65536)? };
            let init = device.get_or_load_custom_func(
                "sys1_build_bfpost_silu_f32_lookup_fork",
                "sys1_qwen35",
                CUDA_PTX,
            )?;
            let mut builder = init.builder();
            builder.arg(&mut table);
            unsafe { builder.launch(LaunchConfig::for_num_elems(65536)) }.w()?;
            **slot = Some(Bf16PostF32LookupFork { stream, table });
        }
    }
    let function_name = if merged_gate {
        "sys1_hopper_qkvz_post_lookup_strided_gate"
    } else if use_lookup {
        "sys1_bfpostwarp128_lookup_fork"
    } else {
        "sys1_bfpostwarp128_fork"
    };
    let function = device.get_or_load_custom_func(function_name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&gate);
    if let Some(slot) = lookup_slot.as_ref() {
        let entry = slot
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("missing post SiLU lookup".into()))?;
        launch.arg(&entry.table);
    }
    launch.arg(&norm);
    launch.arg(&mut output);
    let columns = columns_arg;
    let epsilon = epsilon as f32;
    launch.arg(&columns);
    launch.arg(&epsilon);
    launch.arg(&length_arg);
    launch.arg(&heads_arg);
    launch.arg(&strides[0]);
    launch.arg(&strides[1]);
    launch.arg(&strides[2]);
    let rows_arg = u64::from(rows);
    launch.arg(&rows_arg);
    let gate_strides = [(length as u64) * 12288, 12288u64, 128u64];
    if merged_gate {
        launch
            .arg(&gate_strides[0])
            .arg(&gate_strides[1])
            .arg(&gate_strides[2]);
    }
    unsafe { launch.launch(config) }.w()?;
    #[cfg(test)]
    if use_lookup {
        BF16_POST_F32_LOOKUP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(test)]
    if merged_gate {
        hopper_qkvz::STRIDED_POST.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn cuda_c64_sm_count_fork(device: &candle_core::CudaDevice) -> Result<usize> {
    type Context = std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>;
    static CACHE: std::sync::Mutex<Option<(Context, usize)>> = std::sync::Mutex::new(None);
    let context = device.cuda_stream().context().clone();
    let mut slot = CACHE
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned CUDA SM count cache".into()))?;
    if let Some((old, count)) = slot.as_ref() {
        if std::sync::Arc::ptr_eq(old, &context) {
            return Ok(*count);
        }
    }
    let count=context.attribute(candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).w()?;
    let count = usize::try_from(count)
        .map_err(|_| candle_core::Error::Msg("invalid CUDA SM count".into()))?;
    if count == 0 {
        candle_core::bail!("zero CUDA SM count");
    }
    *slot = Some((context, count));
    Ok(count)
}

#[cfg(feature = "cuda")]
fn cuda_c64_schedule_makespan_fork(
    lengths: &[usize],
    heads: usize,
    tiles: usize,
    sms: usize,
) -> Result<u64> {
    if sms == 0 {
        candle_core::bail!("zero SM count");
    }
    let mut chunks = lengths
        .iter()
        .filter(|&&n| n >= 512)
        .map(|&n| n.div_ceil(64) as u64)
        .collect::<Vec<_>>();
    chunks.sort_unstable_by(|a, b| b.cmp(a));
    let mut slots = std::collections::BinaryHeap::from(vec![std::cmp::Reverse(0u64); sms]);

    for _ in 0..tiles {
        for &chunks in &chunks {
            for _ in 0..heads {
                let mut earliest = slots
                    .peek_mut()
                    .ok_or_else(|| candle_core::Error::Msg("empty CTA schedule".into()))?;
                earliest.0 = earliest
                    .0
                    .checked_add(chunks)
                    .ok_or_else(|| candle_core::Error::Msg("CTA schedule overflow".into()))?;
            }
        }
    }
    Ok(slots.into_iter().map(|x| x.0).max().unwrap_or(0))
}

#[cfg(feature = "cuda")]
fn cuda_c64_scheduled_geometry_fork(
    sequences: &PackedSequences,
    heads: usize,
    value_dim: usize,
    sms: usize,
) -> Result<bool> {
    if value_dim != 128 || sequences.c64_long_count == 0 {
        return Ok(false);
    }
    let key = (heads, value_dim, sms);
    let mut cache = sequences
        .c64_geometry_cache
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned recurrent geometry cache".into()))?;
    if let Some((old, wide)) = *cache {
        if old == key {
            return Ok(wide);
        }
    }
    let narrow = cuda_c64_schedule_makespan_fork(&sequences.lengths, heads, 4, sms)?;
    let wide = cuda_c64_schedule_makespan_fork(&sequences.lengths, heads, 2, sms)?;

    let use_wide = u128::from(narrow) * 3 >= u128::from(wide) * 4;
    *cache = Some((key, use_wide));
    Ok(use_wide)
}

#[cfg(feature = "cuda")]
fn cuda_full_norm_rope_fork(
    input: &Tensor,
    weight: &Tensor,
    rotary: &RotaryTensors,
    epsilon: f64,
) -> Result<Tensor> {
    let (batch, length, heads, width) = input.dims4()?;
    let strides = input.stride();
    let row_stride = strides[2];
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || weight.dims() != [width]
        || rotary.cos.dtype() != DType::BF16
        || rotary.sin.dtype() != DType::BF16
        || !(32..=1024).contains(&width)
        || rotary.dimension > width
        || !rotary.dimension.is_multiple_of(2)
        || rotary.cos.dims() != [1, length, 1, rotary.dimension]
        || rotary.sin.dims() != rotary.cos.dims()
        || strides[3] != 1
        || row_stride < width
        || strides[1] != heads * row_stride
        || strides[0] != length * heads * row_stride
        || batch == 0
        || length == 0
        || heads == 0
        || !input.device().is_cuda()
        || !input.device().same_device(weight.device())
        || !input.device().same_device(rotary.cos.device())
        || !input.device().same_device(rotary.sin.device())
    {
        candle_core::bail!("unsupported full attention fused norm/rotary layout");
    }
    let weight = weight.contiguous()?;
    let cosine = rotary.cos.contiguous()?;
    let sine = rotary.sin.contiguous()?;
    let (x_storage, x_layout) = input.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();
    let (c_storage, c_layout) = cosine.storage_and_layout();
    let (s_storage, s_layout) = sine.storage_and_layout();
    let (Storage::Cuda(x), Storage::Cuda(w), Storage::Cuda(c), Storage::Cuda(s)) =
        (&*x_storage, &*w_storage, &*c_storage, &*s_storage)
    else {
        candle_core::bail!("full norm/rotary CUDA storage required")
    };
    let rows = batch
        .checked_mul(length)
        .and_then(|n| n.checked_mul(heads))
        .ok_or_else(|| candle_core::Error::Msg("full norm/rotary rows overflow".into()))?;
    let span = (rows - 1)
        .checked_mul(row_stride)
        .and_then(|n| n.checked_add(width))
        .ok_or_else(|| candle_core::Error::Msg("full norm/rotary source span overflow".into()))?;
    let offset = x_layout.start_offset();
    let end = offset
        .checked_add(span)
        .ok_or_else(|| candle_core::Error::Msg("full norm/rotary offset overflow".into()))?;
    let input = x.as_cuda_slice::<bf16>()?.slice(offset..end);
    let contiguous = |layout: &candle_core::Layout| {
        layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("noncontiguous full norm/rotary parameter".into())
        })
    };
    let (a, b) = contiguous(w_layout)?;
    let weight = w.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = contiguous(c_layout)?;
    let cosine = c.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = contiguous(s_layout)?;
    let sine = s.as_cuda_slice::<bf16>()?.slice(a..b);
    let device = x.device().clone();
    let count = rows
        .checked_mul(width)
        .ok_or_else(|| candle_core::Error::Msg("full norm/rotary output overflow".into()))?;
    let output = unsafe { device.alloc::<bf16>(count)? };
    let function =
        device.get_or_load_custom_func("sys1_full_norm_rope_bf16", "sys1_qwen35", CUDA_PTX)?;
    let integer = |n| {
        i32::try_from(n)
            .map_err(|_| candle_core::Error::Msg("full norm/rotary launch index overflow".into()))
    };
    let width_arg = integer(width)?;
    let rotary_arg = integer(rotary.dimension)?;
    let heads_arg = integer(heads)?;
    let length_arg = integer(length)?;
    let stride_arg = integer(row_stride)?;
    let epsilon = epsilon as f32;
    let mut launch = function.builder();
    launch.arg(&input);
    launch.arg(&weight);
    launch.arg(&cosine);
    launch.arg(&sine);
    launch.arg(&output);
    launch.arg(&width_arg);
    launch.arg(&rotary_arg);
    launch.arg(&heads_arg);
    launch.arg(&length_arg);
    launch.arg(&stride_arg);
    launch.arg(&epsilon);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (
                u32::try_from(rows).map_err(|_| {
                    candle_core::Error::Msg("full norm/rotary grid overflow".into())
                })?,
                1,
                1,
            ),
            block_dim: (width.next_power_of_two().clamp(32, 1024) as u32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length, heads, width),
        BackpropOp::none(),
        false,
    ))
}

fn full_attention_norm_rotary_fork(
    input: &Tensor,
    norm: &RmsNorm,
    rotary: &RotaryTensors,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if input.device().is_cuda()
        && input.dtype() == DType::BF16
        && norm.weight.dtype() == DType::BF16
    {
        let (batch, length, heads, width) = input.dims4()?;
        let strides = input.stride();
        let stride = strides[2];
        if batch > 0
            && length > 0
            && heads > 0
            && (32..=1024).contains(&width)
            && strides[3] == 1
            && stride >= width
            && strides[1] == heads * stride
            && strides[0] == length * heads * stride
            && rotary.dimension <= width
            && rotary.dimension.is_multiple_of(2)
            && rotary.cos.dtype() == DType::BF16
            && rotary.sin.dtype() == DType::BF16
        {
            return cuda_full_norm_rope_fork(input, &norm.weight, rotary, norm.eps);
        }
    }
    rotary.apply(&norm.forward(input)?)
}

#[cfg(feature = "cuda")]
impl Layer {
    fn forward_deferred_fork(
        &self,
        residual: Tensor,
        update: Option<Tensor>,
        rotary: Option<&RotaryTensors>,
        sequences: Option<&PackedSequences>,
    ) -> Result<(Tensor, Tensor)> {
        let x = &residual;
        let profile = layer_profiling_enabled();
        if profile {
            x.device().synchronize()?;
        }
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "start")?;
        let started = Instant::now();
        let had_update = update.is_some();
        let (x, normalized) = match update {
            Some(mlp) => self.input_norm.forward_residual(x, &mlp)?,
            None => (x.clone(), self.input_norm.forward(x)?),
        };

        drop(residual);
        if had_update && self.index > 0 {
            crate::numerics::capture(&format!("layer-{:02}-output", self.index - 1), &x)?;
        }
        crate::numerics::capture(&format!("layer-{:02}-normalized", self.index), &normalized)?;
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "norm")?;
        if profile {
            x.device().synchronize()?;
        }
        let norm_ms = started.elapsed().as_secs_f64() * 1_000.;
        let started = Instant::now();
        let mixed = match (&self.mixer, sequences) {
            (Mixer::Full(layer), Some(sequences)) => layer.forward_packed(
                &normalized,
                rotary.ok_or_else(|| {
                    candle_core::Error::Msg("Missing Qwen3.5 rotary tensors".into())
                })?,
                self.index,
                sequences,
            )?,
            (Mixer::Full(layer), None) => layer.forward(
                &normalized,
                rotary.ok_or_else(|| {
                    candle_core::Error::Msg("Missing Qwen3.5 rotary tensors".into())
                })?,
                self.index,
            )?,
            (Mixer::Linear(layer), Some(sequences)) => {
                layer.forward_packed(&normalized, self.index, sequences)?
            }
            (Mixer::Linear(layer), None) => layer.forward(&normalized, self.index)?,
        };
        crate::numerics::capture(&format!("layer-{:02}-mixed", self.index), &mixed)?;
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "mixer_tail")?;
        if profile {
            x.device().synchronize()?;
        }
        let mixer_ms = started.elapsed().as_secs_f64() * 1_000.;
        let started = Instant::now();
        let (x, post_normalized) = self.post_norm.forward_residual(&x, &mixed)?;
        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "residual_and_post_norm")?;
        crate::numerics::capture(
            &format!("layer-{:02}-post-normalized", self.index),
            &post_normalized,
        )?;
        let mlp_output = self.mlp.forward(&post_normalized, self.index)?;
        crate::numerics::capture(&format!("layer-{:02}-mlp-output", self.index), &mlp_output)?;

        #[cfg(feature = "cuda")]
        event_profile::mark(x.device(), self.index, "mlp_residual_deferred")?;
        if profile {
            x.device().synchronize()?;
            tracing::info!(
                layer = self.index,
                mixer = match &self.mixer {
                    Mixer::Full(_) => "full",
                    Mixer::Linear(_) => "linear",
                },
                norm_ms,
                mixer_ms,
                mlp_ms = started.elapsed().as_secs_f64() * 1_000.,
                "Qwen3.5 layer profile"
            );
        }
        Ok((x, mlp_output))
    }
}

#[cfg(feature = "cuda")]
impl Decoder {
    fn forward_deferred_fork(
        &self,
        embeds: &Tensor,
        rotary: Option<&RotaryTensors>,
        sequences: Option<&PackedSequences>,
    ) -> Result<Tensor> {
        let mut residual = embeds.clone();
        let mut update = None;
        for layer in &self.layers {
            let (next, mlp) = layer.forward_deferred_fork(residual, update, rotary, sequences)?;
            residual = next;
            update = Some(mlp);
        }
        match update {
            Some(mlp) => {
                let (hidden, output) = self.norm.forward_residual(&residual, &mlp)?;
                if let Some(last) = self.layers.last() {
                    crate::numerics::capture(&format!("layer-{:02}-output", last.index), &hidden)?;
                }
                Ok(output)
            }
            None => self.norm.forward(&residual),
        }
    }
}

impl Decoder {
    fn forward_layers_fork(
        &self,
        embeds: &Tensor,
        rotary: Option<&RotaryTensors>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if embeds.device().is_cuda() && embeds.dtype() == DType::BF16 {
            return self.forward_deferred_fork(embeds, rotary, None);
        }

        let mut x = embeds.clone();
        for layer in &self.layers {
            x = layer.forward(&x, rotary)?;
        }
        self.norm.forward(&x)
    }
}

#[cfg(feature = "cuda")]
fn cuda_full_qkv_finish_fork(
    projected: &Tensor,
    separate: Option<(&Tensor, &Tensor)>,
    qw: &Tensor,
    kw: &Tensor,
    rotary: &RotaryTensors,
    heads: usize,
    kv_heads: usize,
    epsilon: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let (batch, length, columns) = projected.dims3()?;
    let width = qw.dim(0)?;
    let merged = separate.is_none();
    let expected = (2 * heads + if merged { 2 * kv_heads } else { 0 }) * width;
    if batch == 0
        || length == 0
        || heads == 0
        || kv_heads == 0
        || columns != expected
        || kw.dims() != [width]
        || !(32..=1024).contains(&width)
        || rotary.dimension > width
        || !rotary.dimension.is_multiple_of(2)
        || rotary.cos.dims() != [1, length, 1, rotary.dimension]
        || rotary.sin.dims() != rotary.cos.dims()
    {
        candle_core::bail!("unsupported fused QKV geometry");
    }
    let k = separate.map(|(k, _)| k).unwrap_or(projected);
    let v = separate.map(|(_, v)| v).unwrap_or(projected);
    if !merged && (k.dims() != [batch, length, kv_heads * width] || v.dims() != k.dims()) {
        candle_core::bail!("separate fused QKV input geometry");
    }
    for value in [projected, k, v, qw, kw, &rotary.cos, &rotary.sin] {
        if value.dtype() != DType::BF16
            || !value.device().same_device(projected.device())
            || !value.is_contiguous()
        {
            candle_core::bail!("fused QKV requires colocated contiguous BF16");
        }
    }
    let (x_store, x_layout) = projected.storage_and_layout();
    let (k_store, k_layout) = k.storage_and_layout();
    let (v_store, v_layout) = v.storage_and_layout();
    let (qw_store, qw_layout) = qw.storage_and_layout();
    let (kw_store, kw_layout) = kw.storage_and_layout();
    let (c_store, c_layout) = rotary.cos.storage_and_layout();
    let (s_store, s_layout) = rotary.sin.storage_and_layout();
    let (
        Storage::Cuda(x),
        Storage::Cuda(k),
        Storage::Cuda(vs),
        Storage::Cuda(qw),
        Storage::Cuda(kw),
        Storage::Cuda(c),
        Storage::Cuda(s),
    ) = (
        &*x_store, &*k_store, &*v_store, &*qw_store, &*kw_store, &*c_store, &*s_store,
    )
    else {
        candle_core::bail!("fused QKV requires CUDA storage")
    };
    let offsets = |l: &candle_core::Layout| {
        l.contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("fused QKV layout".into()))
    };
    let device = x.device().clone();
    let (a, b) = offsets(x_layout)?;
    let x = x.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = offsets(k_layout)?;
    let keys = k.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = offsets(v_layout)?;
    let values = vs.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = offsets(qw_layout)?;
    let qw = qw.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = offsets(kw_layout)?;
    let kw = kw.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = offsets(c_layout)?;
    let cosine = c.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = offsets(s_layout)?;
    let sine = s.as_cuda_slice::<bf16>()?.slice(a..b);
    let tokens = batch
        .checked_mul(length)
        .ok_or_else(|| candle_core::Error::Msg("fused QKV token overflow".into()))?;
    let qcount = tokens
        .checked_mul(heads)
        .and_then(|x| x.checked_mul(width))
        .ok_or_else(|| candle_core::Error::Msg("fused QKV Q overflow".into()))?;
    let kcount = tokens
        .checked_mul(kv_heads)
        .and_then(|x| x.checked_mul(width))
        .ok_or_else(|| candle_core::Error::Msg("fused QKV K overflow".into()))?;
    let count = qcount
        .checked_mul(2)
        .and_then(|x| x.checked_add(kcount.checked_mul(if merged { 2 } else { 1 })?))
        .ok_or_else(|| candle_core::Error::Msg("fused QKV allocation overflow".into()))?;
    let output = unsafe { device.alloc::<bf16>(count)? };
    let integer = |x| {
        i32::try_from(x).map_err(|_| candle_core::Error::Msg("fused QKV index overflow".into()))
    };
    let d = integer(width)?;
    let r = integer(rotary.dimension)?;
    let h = integer(heads)?;
    let kh = integer(kv_heads)?;
    let l = integer(length)?;
    let eps = epsilon as f32;
    let grid = u32::try_from(
        tokens
            .checked_mul(heads + kv_heads)
            .ok_or_else(|| candle_core::Error::Msg("fused QKV grid overflow".into()))?,
    )
    .map_err(|_| candle_core::Error::Msg("fused QKV grid u32 overflow".into()))?;
    let warp_finish = width == 256
        && rotary.dimension == 64
        && heads == 16
        && kv_heads == 4
        && matches!(
            device.cuda_stream().context().compute_capability().w()?,
            (8, 0) | (9, 0)
        );
    let name = if warp_finish {
        if merged {
            "sys1_full_finish_merged_warp128"
        } else {
            "sys1_full_finish_warp127"
        }
    } else if merged {
        "sys1_full_qkv_finish_merged_bf16"
    } else {
        "sys1_full_qkv_finish_separate_bf16"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    crate::kernels::activation::with_sigmoid_lookup_fork(&device, |table| {
        let mut launch = function.builder();
        launch.arg(&x);
        launch.arg(&keys);
        launch.arg(&values);
        launch.arg(&qw);
        launch.arg(&kw);
        launch.arg(&cosine);
        launch.arg(&sine);
        launch.arg(table);
        launch.arg(&output);
        launch.arg(&d);
        launch.arg(&r);
        launch.arg(&h);
        launch.arg(&kh);
        launch.arg(&l);
        launch.arg(&eps);
        let config = if warp_finish {
            LaunchConfig {
                grid_dim: (tokens as u32, 5, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (width.next_power_of_two().clamp(32, 1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            }
        };
        unsafe { launch.launch(config) }.w()?;
        Ok(())
    })?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        count,
        BackpropOp::none(),
        false,
    );
    let q = output
        .narrow(0, 0, qcount)?
        .reshape((batch, length, heads, width))?;
    let k = output
        .narrow(0, qcount, kcount)?
        .reshape((batch, length, kv_heads, width))?;
    let gate =
        output
            .narrow(0, qcount + kcount, qcount)?
            .reshape((batch, length, heads * width))?;
    let v = if merged {
        output
            .narrow(0, 2 * qcount + kcount, kcount)?
            .reshape((batch, length, kv_heads, width))?
    } else {
        v.reshape((batch, length, kv_heads, width))?
    };
    Ok((q, k, v, gate))
}

#[cfg(feature = "cuda")]
fn cuda_gate_gather_small_fork(
    input: &Tensor,
    linear: &Linear,
    tensor_core: bool,
    simt: SimtProjection,
    sequences: &PackedSequences,
) -> Result<Tensor> {
    let (batch, length, inner) = input.dims3()?;
    if batch != 1
        || length != sequences.total
        || !input
            .device()
            .same_device(sequences.gate_long_starts.device())
        || !input
            .device()
            .same_device(sequences.gate_long_ends.device())
        || !input
            .device()
            .same_device(sequences.gate_short_rows.device())
        || linear.bias().is_some()
    {
        candle_core::bail!("segmented narrow-gate shape/metadata mismatch");
    }
    let small_tile = 32usize;

    let (columns, weight_inner) = linear.weight().dims2()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| candle_core::Error::Msg("Gate projection row count overflow".into()))?;
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| candle_core::Error::Msg("Gate projection output size overflow".into()))?;
    if input.dtype() != DType::BF16
        || linear.weight().dtype() != DType::BF16
        || (linear.bias().is_some()
            && !tensor_core
            && !matches!(
                simt,
                SimtProjection::Compensated | SimtProjection::TiledCompensated
            ))
        || inner != weight_inner
        || rows == 0
        || columns == 0
        || inner == 0
        || !input.device().same_device(linear.weight().device())
        || input.elem_count() > i32::MAX as usize
        || linear.weight().elem_count() > i32::MAX as usize
        || count > i32::MAX as usize
    {
        candle_core::bail!("Unsupported stable BF16 gate projection shape, dtype or device")
    }
    if let Some(bias) = linear.bias() {
        if bias.dtype() != DType::BF16
            || bias.dims() != [columns]
            || !input.device().same_device(bias.device())
        {
            candle_core::bail!(
                "Stable BF16 projection requires matching bias dtype, shape and device"
            )
        }
    }
    let input = input.contiguous()?;
    let weight = linear.weight().contiguous()?;
    let bias = linear.bias().map(Tensor::contiguous).transpose()?;
    let bias_storage = bias.as_ref().map(Tensor::storage_and_layout);
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("Stable gate projection requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("Gate projection storage is not contiguous".into())
        })
    };
    let (xstart, xend) = offsets(xl)?;
    let (wstart, wend) = offsets(wl)?;
    let x = xs.as_cuda_slice::<bf16>()?.slice(xstart..xend);
    let w = ws.as_cuda_slice::<bf16>()?.slice(wstart..wend);
    let b = if let Some((storage, layout)) = &bias_storage {
        let Storage::Cuda(storage) = &**storage else {
            candle_core::bail!("Stable BF16 projection requires CUDA bias storage")
        };
        let (start, end) = offsets(layout)?;
        Some(storage.as_cuda_slice::<bf16>()?.slice(start..end))
    } else {
        None
    };
    let (starts_storage, starts_layout) = sequences.gate_long_starts.storage_and_layout();
    let (ends_storage, ends_layout) = sequences.gate_long_ends.storage_and_layout();
    let (ids_storage, ids_layout) = sequences.gate_short_rows.storage_and_layout();
    let (Storage::Cuda(starts_storage), Storage::Cuda(ends_storage), Storage::Cuda(ids_storage)) =
        (&*starts_storage, &*ends_storage, &*ids_storage)
    else {
        candle_core::bail!("CUDA gate metadata required")
    };
    let (ss, se) = offsets(starts_layout)?;
    let (es, ee) = offsets(ends_layout)?;
    let (is, ie) = offsets(ids_layout)?;
    let tile_starts = starts_storage.as_cuda_slice::<u32>()?.slice(ss..se);
    let tile_ends = ends_storage.as_cuda_slice::<u32>()?.slice(es..ee);
    let short_ids = ids_storage.as_cuda_slice::<u32>()?.slice(is..ie);
    let device = xs.device().clone();
    let mut output = unsafe { device.alloc::<bf16>(count)? };

    let tile_rows = 32usize;

    #[cfg(test)]
    let tile_rows = if tensor_core && matches!(simt, SimtProjection::TensorRows32Probe) {
        32
    } else if tensor_core && matches!(simt, SimtProjection::TensorRows128Probe) {
        128
    } else {
        tile_rows
    };
    let kernel = if !tensor_core && matches!(simt, SimtProjection::TiledCompensated) {
        if b.is_some() {
            "qwen35_biased_tiled_compensated_projection_bf16"
        } else {
            "qwen35_tiled_compensated_projection_bf16"
        }
    } else if b.is_some() && !tensor_core {
        "qwen35_biased_compensated_projection_bf16"
    } else if b.is_some() && tile_rows == 128 {
        "qwen35_biased_projection_rows128_bf16"
    } else if b.is_some() && tile_rows == 32 {
        "qwen35_biased_projection_rows32_bf16"
    } else if b.is_some() {
        "qwen35_biased_projection_bf16"
    } else if tensor_core && tile_rows == 128 {
        "sys1_gate_nativevec_packed_bf16"
    } else if tensor_core && tile_rows == 32 {
        "sys1_gate_nativevec_packed_bf16"
    } else if tensor_core {
        "sys1_gate_nativevec_packed_bf16"
    } else {
        match simt {
            #[cfg(test)]
            SimtProjection::TensorRows32Probe | SimtProjection::TensorRows128Probe => {
                "qwen35_gate_projection_bf16"
            }
            SimtProjection::Compensated => "qwen35_gate_projection_bf16",
            SimtProjection::TiledCompensated => "qwen35_tiled_compensated_projection_bf16",
            SimtProjection::GuardedDouble => "qwen35_guarded_double_projection_bf16",
            SimtProjection::GuardedTensorDouble => "qwen35_guarded_tensor_double_projection_bf16",
            SimtProjection::GuardedTensorDoubleRows16 => {
                "qwen35_guarded_tensor_double_rows16_projection_bf16"
            }
        }
    };
    let function = device.get_or_load_custom_func(kernel, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: if tensor_core {
            (
                sequences.gate_long_count as u32,
                columns.div_ceil(64) as u32,
                1,
            )
        } else if matches!(simt, SimtProjection::TiledCompensated) {
            (rows.div_ceil(4) as u32, columns.div_ceil(4) as u32, 1)
        } else if simt.is_tensor_double() {
            let row_tile = if matches!(simt, SimtProjection::GuardedTensorDoubleRows16) {
                16
            } else {
                8
            };
            (
                rows.div_ceil(row_tile) as u32,
                columns.div_ceil(32) as u32,
                1,
            )
        } else {
            (rows as u32, columns.div_ceil(8) as u32, 1)
        },
        block_dim: (
            if tensor_core {
                tile_rows.min(32) as u32 * 8
            } else if simt.is_tensor_double() || matches!(simt, SimtProjection::TiledCompensated) {
                128
            } else {
                256
            },
            1,
            1,
        ),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&x);
    launch.arg(&w);
    if let Some(bias) = &b {
        launch.arg(bias);
    }
    launch.arg(&mut output);
    let rows = rows as i32;
    if tensor_core || simt.is_tensor_double() || matches!(simt, SimtProjection::TiledCompensated) {
        launch.arg(&rows);
    }
    let inner = inner as i32;
    let columns_arg = columns as i32;
    let radius_coefficient = if !tensor_core
        && !matches!(
            simt,
            SimtProjection::Compensated | SimtProjection::TiledCompensated
        ) {
        Some(guarded_double_radius_coefficient_impl(
            inner as usize,
            simt.is_tensor_double(),
        ))
    } else {
        None
    };
    launch.arg(&inner);
    launch.arg(&columns_arg);
    launch.arg(&tile_starts);
    launch.arg(&tile_ends);
    if let Some(coefficient) = &radius_coefficient {
        launch.arg(coefficient);
    }
    if sequences.gate_long_count > 0 {
        unsafe { launch.launch(config) }.w()?;
    }
    if sequences.gate_short_count > 0 {
        let function = device.get_or_load_custom_func(
            "sys1_gate_gathervector_small_32_fork",
            "sys1_qwen35",
            CUDA_PTX,
        )?;
        let mut launch = function.builder();
        let short_count = sequences.gate_short_count as i32;
        launch
            .arg(&x)
            .arg(&w)
            .arg(&mut output)
            .arg(&short_count)
            .arg(&inner)
            .arg(&columns_arg)
            .arg(&short_ids);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    sequences.gate_short_count.div_ceil(small_tile) as u32,
                    columns.div_ceil(64) as u32,
                    1,
                ),
                block_dim: ((small_tile * 8) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
    }

    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length, columns),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn cuda_short_value_reuse_selected_fork(
    device: &candle_core::CudaDevice,
    kind: usize,
    heads: usize,
    value_dim: usize,
    tokens: u128,
    longest: usize,
) -> Result<bool> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120") || longest == 0 {
        return Ok(false);
    }
    let sm_count = cuda_c64_sm_count_fork(device)?;
    let blocks = (heads as u128) * (value_dim.div_ceil(32) as u128);
    if blocks * tokens < (sm_count as u128) * 2 * (longest as u128) {
        return Ok(false);
    }
    type Context = std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>;
    static CACHE: std::sync::Mutex<Option<(Context, [usize; 3])>> = std::sync::Mutex::new(None);
    let context = device.cuda_stream().context().clone();
    let mut cache = CACHE
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned short value residency".into()))?;
    if !cache
        .as_ref()
        .is_some_and(|(old, _)| std::sync::Arc::ptr_eq(old, &context))
    {
        *cache = Some((context, [0; 3]));
    }
    let (_, resident) = cache.as_mut().unwrap();
    if resident[kind] == 0 {
        let name = [
            "sys1_shortvalues_0_bf16",
            "sys1_shortvalues_1_bf16",
            "sys1_shortvalues_2_bf16",
        ][kind];
        let f = device.get_or_load_custom_func(name, "sys1_short_value_reuse", CUDA_PTX)?;
        resident[kind] = f
            .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
            .w()? as usize;
        if resident[kind] == 0 {
            candle_core::bail!("zero short value residency");
        }
        tracing::info!(
            sm_count,
            kind,
            resident_blocks_per_sm = resident[kind],
            value_columns_per_lane_group = 4,
            "Queried exact short recurrence operand-reuse residency"
        );
    }

    Ok(blocks * tokens
        >= (sm_count as u128) * (resident[kind].div_ceil(2) as u128) * (longest as u128))
}

#[cfg(feature = "cuda")]
fn cuda_conv_vstore_fork(
    input: &Tensor,
    weights: &Tensor,
    key_heads: usize,
    value_heads: usize,
    sequences: Option<&PackedSequences>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (batch, length, channels) = input.dims3()?;
    let (wc, groups, kernel) = weights.dims3()?;
    if batch == 0
        || length == 0
        || key_heads == 0
        || value_heads == 0
        || value_heads % key_heads != 0
        || input.dtype() != DType::BF16
        || weights.dtype() != DType::BF16
        || !input.device().same_device(weights.device())
        || input.elem_count() > i32::MAX as usize
        || wc != channels
        || groups != 1
        || kernel == 0
        || kernel > i32::MAX as usize
        || key_heads
            .checked_mul(2)
            .and_then(|n| n.checked_add(value_heads))
            .and_then(|n| n.checked_mul(128))
            != Some(channels)
    {
        candle_core::bail!("unsupported fused convolution/QKV shape");
    }
    if let Some(s) = sequences {
        if batch != 1 || length != s.total || !input.device().same_device(s.starts.device()) {
            candle_core::bail!("fused convolution packed boundary mismatch");
        }
    }
    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weights.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("fused convolution requires CUDA");
    };
    let (x0, x1) = xl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("convolution input offsets".into()))?;
    let (w0, w1) = wl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("convolution weight offsets".into()))?;
    let source = xs.as_cuda_slice::<bf16>()?.slice(x0..x1);
    let filters = ws.as_cuda_slice::<bf16>()?.slice(w0..w1);
    let device = xs.device().clone();
    let q_count = batch * length * key_heads * 128;
    let v_count = batch * length * value_heads * 128;
    let mut q = unsafe { device.alloc::<f32>(q_count)? };
    let mut k = unsafe { device.alloc::<f32>(q_count)? };
    let mut v = unsafe { device.alloc::<bf16>(v_count)? };
    let rows = (q_count / 128) as u32;
    let kh = key_heads as i32;
    let vh = value_heads as i32;
    let scale = 128f64.powf(-0.5) as f32;
    let length_arg = length as i32;
    let kernel_arg = kernel as i32;
    let name = if sequences.is_some() {
        "sys1_conv_vstore_packed_fork"
    } else {
        "sys1_conv_vstore_uniform_fork"
    };
    let function = device.get_or_load_custom_func(name, "sys1_conv_qkv", CUDA_PTX)?;
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&filters);
    launch.arg(&mut q);
    launch.arg(&mut k);
    launch.arg(&mut v);
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(4), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    macro_rules! tail {
        () => {{
            launch.arg(&rows);
            launch.arg(&kh);
            launch.arg(&vh);
            launch.arg(&scale);
            launch.arg(&length_arg);
            launch.arg(&kernel_arg);
        }};
    }
    if let Some(s) = sequences {
        let (ss, sl) = s.starts.storage_and_layout();
        let Storage::Cuda(ss) = &*ss else {
            candle_core::bail!("starts require CUDA");
        };
        let (s0, s1) = sl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("start offsets".into()))?;
        let starts = ss.as_cuda_slice::<u32>()?.slice(s0..s1);
        launch.arg(&starts);
        tail!();
        unsafe { launch.launch(config) }.w()?;
    } else {
        tail!();
        unsafe { launch.launch(config) }.w()?;
    }
    let wrap = |data, heads| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(data, device.clone())),
            (batch, length, heads, 128),
            BackpropOp::none(),
            false,
        )
    };
    let v = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(v, device.clone())),
        (batch, length, value_heads, 128),
        BackpropOp::none(),
        false,
    );
    Ok((wrap(q, key_heads), wrap(k, key_heads), v))
}

#[cfg(feature = "cuda")]
fn cuda_vstore_short_selected_fork(
    device: &candle_core::CudaDevice,
    kind: usize,
    heads: usize,
    value_dim: usize,
    tokens: u128,
    longest: usize,
) -> Result<bool> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120") || longest == 0 {
        return Ok(false);
    }
    let sm_count = cuda_c64_sm_count_fork(device)?;
    let blocks = (heads as u128) * (value_dim.div_ceil(32) as u128);
    if blocks * tokens < (sm_count as u128) * 2 * (longest as u128) {
        return Ok(false);
    }
    type Context = std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>;
    static CACHE: std::sync::Mutex<Option<(Context, [usize; 3])>> = std::sync::Mutex::new(None);
    let context = device.cuda_stream().context().clone();
    let mut cache = CACHE
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned short value residency".into()))?;
    if !cache
        .as_ref()
        .is_some_and(|(old, _)| std::sync::Arc::ptr_eq(old, &context))
    {
        *cache = Some((context, [0; 3]));
    }
    let (_, resident) = cache.as_mut().unwrap();
    if resident[kind] == 0 {
        let name = [
            "sys1_vstoreshort_0_bf16",
            "sys1_vstoreshort_1_bf16",
            "sys1_vstoreshort_2_bf16",
        ][kind];
        let f = device.get_or_load_custom_func(name, "sys1_short_value_reuse", CUDA_PTX)?;
        resident[kind] = f
            .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
            .w()? as usize;
        if resident[kind] == 0 {
            candle_core::bail!("zero short value residency");
        }
        tracing::info!(
            sm_count,
            kind,
            resident_blocks_per_sm = resident[kind],
            value_columns_per_lane_group = 4,
            "Queried exact short recurrence operand-reuse residency"
        );
    }

    Ok(blocks * tokens
        >= (sm_count as u128) * (resident[kind].div_ceil(2) as u128) * (longest as u128))
}

#[cfg(feature = "cuda")]
fn cuda_vstore_delta_packed_fork(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    subwarp16: bool,
) -> Result<Tensor> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120") {
        candle_core::bail!("vstore full GDN probe requires SM120");
    }

    if [q, k, g, beta].iter().any(|x| x.dtype() != DType::F32) || v.dtype() != DType::BF16 {
        candle_core::bail!("vstore packed input types");
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    if !subwarp16 || batch != 1 || length != sequences.total {
        candle_core::bail!("packed delta unsupported layout")
    }
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<bf16>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let (metadata_storage, metadata_layout) = sequences.cumulative.storage_and_layout();
    let Storage::Cuda(metadata_storage) = &*metadata_storage else {
        candle_core::bail!("packed metadata must be CUDA")
    };
    let (start, end) = offsets(metadata_layout)?;
    let cumulative = metadata_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let device = q_storage.device().clone();
    let output = unsafe { device.alloc::<bf16>(batch * heads * length * value_dim)? };

    let reuse = cuda_vstore_short_selected_fork(
        &device,
        1,
        heads,
        value_dim,
        sequences.total as u128,
        *sequences.lengths.iter().max().unwrap_or(&0),
    )?;
    let short_columns = if reuse { 32 } else { 8 };
    let name = if reuse {
        "sys1_vstoreshort_1_bf16"
    } else {
        "sys1_vstore_packed_delta_bfout_fork"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (
            (sequences.lengths.len() * heads) as u32,
            value_dim.div_ceil(short_columns) as u32,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&query);
    launch.arg(&key);
    launch.arg(&value);
    launch.arg(&gate);
    launch.arg(&beta);
    launch.arg(&output);
    launch.arg(&cumulative);
    let length = length as i32;
    let heads_arg = heads as i32;
    let key_heads_arg = key_heads as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&heads_arg);
    launch.arg(&key_heads_arg);
    launch.arg(&value_dim);
    unsafe { launch.launch(config) }.w()?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length as usize, heads, value_dim as usize),
        BackpropOp::none(),
        false,
    );
    Ok(output)
}

#[cfg(feature = "cuda")]
fn cuda_vstore_gdn_varlen_fork(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    profiling: bool,
) -> Result<Tensor> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120") {
        candle_core::bail!("vstore full GDN probe requires SM120");
    }

    if [q, k, g, beta].iter().any(|x| x.dtype() != DType::F32) || v.dtype() != DType::BF16 {
        candle_core::bail!("chunk probe F32 inputs required");
    }

    if q.dim(0)? != 1 || q.dim(1)? != sequences.total {
        candle_core::bail!("segmented C64 shape mismatch")
    }
    if sequences.c64_chunk_count == 0 {
        return cuda_vstore_delta_packed_fork(q, k, v, g, beta, sequences, true);
    }
    if [
        k,
        v,
        g,
        beta,
        &sequences.cumulative,
        &sequences.c64_chunks,
        &sequences.c64_long_ids,
        &sequences.c64_short_ids,
    ]
    .iter()
    .any(|t| !q.device().same_device(t.device()))
    {
        candle_core::bail!("segmented C64 device mismatch")
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<bf16>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let (cumulative_storage, cumulative_layout) = sequences.cumulative.storage_and_layout();
    let Storage::Cuda(cumulative_storage) = &*cumulative_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(cumulative_layout)?;
    let cumulative = cumulative_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let (chunk_offsets_storage, chunk_offsets_layout) = sequences.c64_chunks.storage_and_layout();
    let Storage::Cuda(chunk_offsets_storage) = &*chunk_offsets_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(chunk_offsets_layout)?;
    let chunk_offsets = chunk_offsets_storage
        .as_cuda_slice::<u32>()?
        .slice(start..end);
    let (long_ids_storage, long_ids_layout) = sequences.c64_long_ids.storage_and_layout();
    let Storage::Cuda(long_ids_storage) = &*long_ids_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(long_ids_layout)?;
    let long_ids = long_ids_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let (short_ids_storage, short_ids_layout) = sequences.c64_short_ids.storage_and_layout();
    let Storage::Cuda(short_ids_storage) = &*short_ids_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(short_ids_layout)?;
    let short_ids = short_ids_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let device = q_storage.device().clone();
    let sm_count = cuda_c64_sm_count_fork(&device)?;
    let wide_blocks = sequences
        .c64_long_count
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(value_dim.div_ceil(64)))
        .ok_or_else(|| candle_core::Error::Msg("wide recurrent grid overflow".into()))?;
    let use_wide = cuda_c64_scheduled_geometry_fork(sequences, heads, value_dim, sm_count)?;
    let _ = wide_blocks;
    let (tile_columns, state_rows, carry_shared) = if use_wide {
        (
            64usize,
            64usize,
            if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") {
                98336u32
            } else {
                82976u32
            },
        )
    } else {
        (32usize, 128usize, 58400u32)
    };
    if use_wide {
        static NOTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        NOTED.get_or_init(|| {
            tracing::info!(
                value_tile = 64,
                threads = 1024,
                shared_bytes = carry_shared,
                sm_count,
                "Using wider recurrent value tiles selected by cached CTA schedule cost"
            )
        });
    }

    if ![32usize, 64].contains(&tile_columns)
        || batch == 0
        || length == 0
        || value_dim > 128
        || [
            q.elem_count(),
            k.elem_count(),
            v.elem_count(),
            g.elem_count(),
        ]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk probe requires nonempty F32 K128,V<=128,small indices");
    }
    let instances = heads
        .checked_mul(sequences.c64_chunk_count)
        .ok_or_else(|| candle_core::Error::Msg("chunk count overflow".into()))?;
    let w_count = instances
        .checked_mul(64 * 128)
        .ok_or_else(|| candle_core::Error::Msg("chunk W overflow".into()))?;
    let u_count = instances
        .checked_mul(64 * value_dim)
        .ok_or_else(|| candle_core::Error::Msg("chunk U overflow".into()))?;
    let c_count = instances
        .checked_mul(64 * 64)
        .ok_or_else(|| candle_core::Error::Msg("chunk C overflow".into()))?;
    let d_count = instances
        .checked_mul(128)
        .ok_or_else(|| candle_core::Error::Msg("decay count overflow".into()))?;
    if [w_count, u_count, c_count, d_count]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk scratch index overflow");
    }
    let result = (|| -> Result<Tensor> {
        let mut slot = CHUNK64_SCRATCH
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned chunk scratch".into()))?;
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
        {
            *slot = Some(ChunkScratch {
                stream,
                w: None,
                u: None,
                c: None,
                decays: None,
                snapshots: None,
            });
        }
        let workspace = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing chunk scratch".into()))?;
        let ChunkScratch {
            w,
            u,
            c,
            decays,
            snapshots,
            ..
        } = workspace;
        if profiling {
            device.cuda_stream().synchronize().w()?;
        }
        let mut stage_started = Instant::now();
        let w = grow_chunk_buffer(w, &device, w_count)?;
        let u = grow_chunk_buffer(u, &device, u_count)?;
        let c = grow_chunk_buffer(c, &device, c_count)?;
        let decays = grow_chunk_buffer(decays, &device, d_count)?;
        let output_count = length
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(value_dim))
            .ok_or_else(|| candle_core::Error::Msg("segmented C64 output overflow".into()))?;
        if output_count > i32::MAX as usize {
            candle_core::bail!("padded output index overflow");
        }
        let guarded = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") && !use_wide;
        let snapshot_count = if guarded { instances } else { 1usize };
        if snapshot_count > i32::MAX as usize {
            candle_core::bail!("snapshot index overflow");
        }
        let snapshots = grow_chunk_buffer(snapshots, &device, snapshot_count)?;
        let mut output = unsafe { device.alloc::<bf16>(output_count)? };
        if profiling {
            println!(
                "C64_CARRY_WORKSPACE retained_bytes={} output_bytes={} single_entry_per_process=true owned_stream_identity=true output_never_reused=true amortized_warm_operator_not_serving=true",
                (w.len() + u.len() + c.len() + decays.len() + snapshots.len()) * 4,
                output_count * 2
            );
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=scratch batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        let sequence_count = sequences.lengths.len() as i32;
        let length_arg = length as i32;
        let heads_arg = heads as i32;
        let key_heads_arg = key_heads as i32;
        let value_dim_arg = value_dim as i32;
        {
            let function = device.get_or_load_custom_func(
                if guarded {
                    "sys1_c64vstoreguard_varlen_prepare_f32"
                } else {
                    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") {
                        "sys1_c64vstore_varlen_prepare_f32"
                    } else {
                        "sys1_c64slice_varlen_prepare_f32"
                    }
                },
                "sys1_chunk_delta",
                CUDA_PTX,
            )?;
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,49664).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&value);
            launch.arg(&gate);
            launch.arg(&beta);
            launch.arg(&mut *w);
            launch.arg(&mut *u);
            launch.arg(&mut *c);
            launch.arg(&mut *decays);
            if guarded {
                launch.arg(&mut *snapshots);
            }
            launch.arg(&cumulative);
            launch.arg(&chunk_offsets);
            launch.arg(&sequence_count);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (instances as u32, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 49664,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=prepare batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        {
            if sequences.c64_short_count > 0 {
                let short_tokens = sequences
                    .lengths
                    .iter()
                    .filter(|&&n| n < 512)
                    .map(|&n| n as u128)
                    .sum();
                let longest_short = sequences
                    .lengths
                    .iter()
                    .filter(|&&n| n < 512)
                    .copied()
                    .max()
                    .unwrap_or(0);
                let reuse = cuda_vstore_short_selected_fork(
                    &device,
                    0,
                    heads,
                    value_dim,
                    short_tokens,
                    longest_short,
                )?;
                let short_columns = if reuse { 32 } else { 8 };
                let name = if reuse {
                    "sys1_vstoreshort_0_bf16"
                } else {
                    "sys1_vstore_short_bf16"
                };
                let function = device.get_or_load_custom_func(name, "sys1_c64v_short", CUDA_PTX)?;
                let mut launch = function.builder();
                launch
                    .arg(&query)
                    .arg(&key)
                    .arg(&value)
                    .arg(&gate)
                    .arg(&beta)
                    .arg(&mut output)
                    .arg(&cumulative)
                    .arg(&short_ids)
                    .arg(&length_arg)
                    .arg(&heads_arg)
                    .arg(&key_heads_arg)
                    .arg(&value_dim_arg);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (
                            (sequences.c64_short_count * heads) as u32,
                            value_dim.div_ceil(short_columns) as u32,
                            1,
                        ),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .w()?;
            }
            let name = if use_wide {
                if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") {
                    "sys1_c64full01_n64_varlen_f32"
                } else {
                    "sys1_c64wide_varlen_f32"
                }
            } else if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") {
                "sys1_c64zguard_n32_varlen32_f32"
            } else {
                "sys1_c64bfout_varlen32_f32"
            };
            let function = device.get_or_load_custom_func(name, "sys1_chunk_delta", CUDA_PTX)?;
            if profiling {
                println!(
                    "C64_CARRY_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} diagnostic_not_serving=true",
                    function.num_regs().w()?,
                    function.local_size_bytes().w()?,
                    function.shared_size_bytes().w()?
                );
            }
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,carry_shared as i32).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&gate);
            launch.arg(&*w);
            launch.arg(&*u);
            launch.arg(&*c);
            launch.arg(&*decays);
            launch.arg(&mut output);
            launch.arg(&mut *snapshots);
            launch.arg(&cumulative);
            launch.arg(&chunk_offsets);
            launch.arg(&long_ids);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        (sequences.c64_long_count * heads) as u32,
                        value_dim.div_ceil(tile_columns) as u32,
                        1,
                    ),
                    block_dim: (((state_rows / 16) * (tile_columns / 8) * 32) as u32, 1, 1),
                    shared_mem_bytes: carry_shared,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=state batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
        }

        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (1, length, heads, value_dim),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    })();
    if result.is_err() {
        if let Ok(mut slot) = CHUNK64_SCRATCH.lock() {
            slot.take();
        }
    }
    result
}

#[cfg(all(test, feature = "cuda"))]
thread_local! {
    static VSTORAGE_MODEL_REPLAY:std::cell::Cell<bool>=const{std::cell::Cell::new(true)};
    static VSTORAGE_MODEL_CALLS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};
}

#[cfg(feature = "cuda")]
fn cuda_vstore_gdn_uniform_fork(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    tile_columns: usize,
    profiling: bool,
) -> Result<Tensor> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120") {
        candle_core::bail!("vstore full GDN probe requires SM120");
    }

    if [q, k, g, beta].iter().any(|x| x.dtype() != DType::F32) || v.dtype() != DType::BF16 {
        candle_core::bail!("chunk probe F32 inputs required");
    }
    let state_rows = 128usize;
    let carry_shared = 58400u32;
    if tile_columns != 32 {
        candle_core::bail!("C64 state requires N32");
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<bf16>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let device = q_storage.device().clone();

    if tile_columns != 32
        || batch == 0
        || length == 0
        || value_dim > 128
        || [
            q.elem_count(),
            k.elem_count(),
            v.elem_count(),
            g.elem_count(),
        ]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk probe requires nonempty F32 K128,V<=128,small indices");
    }
    let instances = batch
        .checked_mul(heads)
        .and_then(|x| x.checked_mul(length.div_ceil(64)))
        .ok_or_else(|| candle_core::Error::Msg("chunk count overflow".into()))?;
    let w_count = instances
        .checked_mul(64 * 128)
        .ok_or_else(|| candle_core::Error::Msg("chunk W overflow".into()))?;
    let u_count = instances
        .checked_mul(64 * value_dim)
        .ok_or_else(|| candle_core::Error::Msg("chunk U overflow".into()))?;
    let c_count = instances
        .checked_mul(64 * 64)
        .ok_or_else(|| candle_core::Error::Msg("chunk C overflow".into()))?;
    let d_count = instances
        .checked_mul(128)
        .ok_or_else(|| candle_core::Error::Msg("decay count overflow".into()))?;
    if [w_count, u_count, c_count, d_count]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk scratch index overflow");
    }
    let result = (|| -> Result<Tensor> {
        let mut slot = CHUNK64_SCRATCH
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned chunk scratch".into()))?;
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
        {
            *slot = Some(ChunkScratch {
                stream,
                w: None,
                u: None,
                c: None,
                decays: None,
                snapshots: None,
            });
        }
        let workspace = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing chunk scratch".into()))?;
        let ChunkScratch {
            w,
            u,
            c,
            decays,
            snapshots,
            ..
        } = workspace;
        if profiling {
            device.cuda_stream().synchronize().w()?;
        }
        let mut stage_started = Instant::now();
        let w = grow_chunk_buffer(w, &device, w_count)?;
        let u = grow_chunk_buffer(u, &device, u_count)?;
        let c = grow_chunk_buffer(c, &device, c_count)?;
        let decays = grow_chunk_buffer(decays, &device, d_count)?;
        let output_count = batch
            .checked_mul(length)
            .and_then(|n| n.checked_mul(heads))
            .and_then(|n| n.checked_mul(value_dim))
            .ok_or_else(|| candle_core::Error::Msg("register-carry C64 output overflow".into()))?;
        if output_count > i32::MAX as usize {
            candle_core::bail!("padded output index overflow");
        }
        let guarded = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120");
        let snapshot_count = if guarded { instances } else { 1usize };
        if snapshot_count > i32::MAX as usize {
            candle_core::bail!("snapshot index overflow");
        }
        let snapshots = grow_chunk_buffer(snapshots, &device, snapshot_count)?;
        let mut output = unsafe { device.alloc::<bf16>(output_count)? };
        if profiling {
            println!(
                "C64_FLAGS_WORKSPACE retained_bytes={} output_bytes={} single_entry_per_process=true owned_stream_identity=true output_never_reused=true amortized_warm_operator_not_serving=true",
                (w.len() + u.len() + c.len() + decays.len() + snapshots.len()) * 4,
                output_count * 2
            );
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=scratch batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        let length_arg = length as i32;
        let heads_arg = heads as i32;
        let key_heads_arg = key_heads as i32;
        let value_dim_arg = value_dim as i32;
        {
            let function = device.get_or_load_custom_func(
                if guarded {
                    "sys1_c64vstoreguard_uniform_prepare_f32"
                } else {
                    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") {
                        "sys1_c64vstore_uniform_prepare_f32"
                    } else {
                        "sys1_c64slice_uniform_prepare_f32"
                    }
                },
                "sys1_chunk_delta",
                CUDA_PTX,
            )?;
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,49664).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&value);
            launch.arg(&gate);
            launch.arg(&beta);
            launch.arg(&mut *w);
            launch.arg(&mut *u);
            launch.arg(&mut *c);
            launch.arg(&mut *decays);
            if guarded {
                launch.arg(&mut *snapshots);
            }
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (instances as u32, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 49664,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=prepare batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        {
            let name = if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") {
                "sys1_c64zguard_n32_uniform32_f32"
            } else {
                "sys1_c64bfout_uniform32_f32"
            };
            let function = device.get_or_load_custom_func(name, "sys1_chunk_delta", CUDA_PTX)?;
            if profiling {
                println!(
                    "C64_FLAGS_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} diagnostic_not_serving=true",
                    function.num_regs().w()?,
                    function.local_size_bytes().w()?,
                    function.shared_size_bytes().w()?
                );
            }
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,carry_shared as i32).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&gate);
            launch.arg(&*w);
            launch.arg(&*u);
            launch.arg(&*c);
            launch.arg(&*decays);
            launch.arg(&mut output);
            launch.arg(&mut *snapshots);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        (batch * heads) as u32,
                        value_dim.div_ceil(tile_columns) as u32,
                        1,
                    ),
                    block_dim: (((state_rows / 16) * (tile_columns / 8) * 32) as u32, 1, 1),
                    shared_mem_bytes: carry_shared,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=state batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
        }

        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (batch, length, heads, value_dim),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    })();
    if result.is_err() {
        if let Ok(mut slot) = CHUNK64_SCRATCH.lock() {
            slot.take();
        }
    }
    result
}

#[derive(Clone)]
struct Linear {
    inner: CandleLinear,
    #[cfg(feature = "cuda")]
    nn_weight: std::sync::Arc<std::sync::Mutex<Option<Tensor>>>,
}
impl Linear {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            inner: CandleLinear::new(weight, bias),
            #[cfg(feature = "cuda")]
            nn_weight: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }
    #[cfg(feature = "cuda")]
    fn cached_nn(&self) -> Result<Option<Tensor>> {
        Ok(self
            .nn_weight
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned model-owned NN weight".into()))?
            .clone())
    }
    #[cfg(feature = "cuda")]
    fn prepared_nn(&self) -> Result<Tensor> {
        let mut slot = self
            .nn_weight
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned model-owned NN weight".into()))?;
        if slot.is_none() {
            let prepared = self.weight().t()?.contiguous()?;
            #[cfg(test)]
            QWEN_NN_WEIGHT_BYTES.with(|slot| slot.set(slot.get() + self.weight().elem_count() * 2));
            *slot = Some(prepared);
        }
        Ok(slot.as_ref().unwrap().clone())
    }
}
impl std::ops::Deref for Linear {
    type Target = CandleLinear;
    fn deref(&self) -> &CandleLinear {
        &self.inner
    }
}
impl candle_core::Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_core::Module::forward(&self.inner, x)
    }
}

#[cfg(all(test, feature = "cuda"))]
thread_local! {
    static QWEN_NN_REPLAY:std::cell::Cell<bool>=const{std::cell::Cell::new(true)};
    static QWEN_NN_CALLS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};
    static QWEN_NN_WEIGHT_BYTES:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};
}

#[cfg(feature = "cuda")]
fn nn_geometry_available(inner: usize, columns: usize) -> bool {
    matches!(
        (inner, columns),
        (4096, 12288) | (12288, 4096) | (4096, 8192) | (4096, 4096)
    )
}

#[cfg(feature = "cuda")]
fn nn_layout_selected(rows: usize, inner: usize, columns: usize) -> bool {
    match (inner, columns) {
        (4096, 12288) | (12288, 4096) => rows >= 4096,
        (4096, 8192) => rows >= 1536,
        (4096, 4096) => rows >= 8193,
        _ => false,
    }
}

#[cfg(feature = "cuda")]
fn try_nn_projection(input: &Tensor, linear: &Linear) -> Result<Option<Tensor>> {
    #[cfg(test)]
    let enabled = QWEN_NN_REPLAY.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let enabled = true;
    if !enabled
        || linear.bias().is_some()
        || !input.device().is_cuda()
        || input.dtype() != DType::BF16
        || option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120")
    {
        return Ok(None);
    }
    let inner = input.dim(D::Minus1)?;
    let columns = linear.weight().dim(0)?;
    let rows = input.elem_count() / inner.max(1);
    if !nn_layout_selected(rows, inner, columns) {
        return Ok(None);
    }
    let Some(weight) = linear.cached_nn()? else {
        return Ok(None);
    };
    let actual_m = long_down_actual_m(rows, inner, columns, true)
        || (columns >= 2048
            && columns >= inner.saturating_mul(2)
            && (rows <= 512 || columns == inner.saturating_mul(2)))
        || (rows >= 8193 && columns >= 256);
    #[cfg(test)]
    QWEN_NN_CALLS.with(|slot| slot.set(slot.get() + 1));
    Ok(Some(crate::kernels::blaslt::project_transposed_weight(
        input, &weight, !actual_m,
    )?))
}

#[cfg(feature = "cuda")]
fn nn_capacity_preserved(
    free: usize,
    extra: usize,
    bytes_per_token: usize,
    target_tokens: usize,
) -> bool {
    match free.checked_sub(extra) {
        Some(after) => after.saturating_sub(after / 4) / bytes_per_token.max(1) >= target_tokens,
        None => false,
    }
}

#[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
impl Mlp {
    fn fused_geometry_available(&self) -> bool {
        self.gate.bias().is_none()
            && self.up.bias().is_none()
            && crate::kernels::fused_gate_up::available(self.gate.weight(), self.up.weight())
    }
    fn cached_fused(&self) -> Result<Option<Tensor>> {
        Ok(self
            .fused_weight
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned model-owned fused MLP weight".into()))?
            .clone())
    }
    fn prepared_fused(&self) -> Result<Tensor> {
        let mut slot = self
            .fused_weight
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned model-owned fused MLP weight".into()))?;
        if slot.is_none() {
            *slot = Some(crate::kernels::fused_gate_up::prepare(
                self.gate.weight(),
                self.up.weight(),
            )?);
        }
        Ok(slot.as_ref().unwrap().clone())
    }
}

#[cfg(feature = "cuda")]
impl Decoder {
    fn prepare_nn_weights_base(
        &self,
        target_tokens: usize,
        bytes_per_token: usize,
    ) -> Result<usize> {
        if !matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "120")) {
            return Ok(0);
        }
        let mut candidates: Vec<&Linear> = Vec::new();
        let mut fused_mlps: Vec<&Mlp> = Vec::new();
        for layer in &self.layers {
            if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80") {
                #[cfg(sys1_cuda_fused_ffn)]
                if layer.mlp.fused_geometry_available() && layer.mlp.cached_fused()?.is_none() {
                    fused_mlps.push(&layer.mlp);
                }
                continue;
            }
            #[cfg(sys1_cuda_fused_ffn)]
            let fused = layer.mlp.fused_geometry_available();
            #[cfg(not(sys1_cuda_fused_ffn))]
            let fused = false;
            if fused {
                #[cfg(sys1_cuda_fused_ffn)]
                if layer.mlp.cached_fused()?.is_none() {
                    fused_mlps.push(&layer.mlp);
                }
                candidates.push(&layer.mlp.down);
            } else {
                candidates.extend([&layer.mlp.gate, &layer.mlp.up, &layer.mlp.down]);
            }
            match &layer.mixer {
                Mixer::Linear(mixer) => candidates.extend([&mixer.qkv, &mixer.z, &mixer.output]),
                Mixer::Full(mixer) => candidates.extend([&mixer.q, &mixer.output]),
            }
        }
        let mut weights = Vec::new();
        let mut extra = 0usize;
        let mut fused_bytes = 0usize;
        for linear in candidates {
            if linear.bias().is_some() || linear.weight().dtype() != DType::BF16 {
                continue;
            }
            let (columns, inner) = linear.weight().dims2()?;
            if !nn_geometry_available(inner, columns) || linear.cached_nn()?.is_some() {
                continue;
            }
            extra = extra
                .checked_add(linear.weight().elem_count().checked_mul(2).ok_or_else(|| {
                    candle_core::Error::Msg("NN weight byte count overflow".into())
                })?)
                .ok_or_else(|| {
                    candle_core::Error::Msg("NN total weight byte count overflow".into())
                })?;
            weights.push(linear);
        }
        for mlp in &fused_mlps {
            fused_bytes =
                fused_bytes
                    .checked_add(mlp.gate.weight().elem_count().checked_mul(4).ok_or_else(
                        || candle_core::Error::Msg("paired MLP weight bytes overflow".into()),
                    )?)
                    .ok_or_else(|| {
                        candle_core::Error::Msg("paired total weight bytes overflow".into())
                    })?;
        }
        extra = extra
            .checked_add(fused_bytes)
            .ok_or_else(|| candle_core::Error::Msg("optional layout bytes overflow".into()))?;
        if weights.is_empty() && fused_mlps.is_empty() {
            return Ok(0);
        }
        let reference = weights
            .first()
            .map(|linear| linear.weight())
            .or_else(|| fused_mlps.first().map(|mlp| mlp.gate.weight()))
            .ok_or_else(|| candle_core::Error::Msg("missing layout reference".into()))?;
        let candle_core::Device::Cuda(cuda) = reference.device() else {
            return Ok(0);
        };
        let (free, _) = cuda
            .cuda_stream()
            .context()
            .mem_get_info()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        if !nn_capacity_preserved(free, extra, bytes_per_token, target_tokens) {
            tracing::info!(
                free_bytes = free,
                extra_weight_bytes = extra,
                fused_mlp_weight_bytes = fused_bytes,
                target_tokens,
                "Qwen optional physical layouts skipped to preserve activation capacity"
            );
            return Ok(0);
        }
        #[cfg(sys1_cuda_fused_ffn)]
        for mlp in fused_mlps {
            let _ = mlp.prepared_fused()?;
        }
        for linear in weights {
            let _ = linear.prepared_nn()?;
        }
        tracing::info!(
            extra_weight_bytes = extra,
            fused_mlp_weight_bytes = fused_bytes,
            target_tokens,
            "Qwen optional physical layouts prepared without reducing token capacity"
        );
        Ok(extra)
    }
}

#[cfg(all(test, feature = "cuda", sys1_cuda_fused_ffn))]
thread_local! {static FUSED_MLP_CALLS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};}

impl LinearAttention {
    fn project_qkvz(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        #[cfg(feature = "cuda")]
        if let Some(output) = self.project_merged_hopper_qkvz(x)? {
            return Ok(output);
        }
        #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
        if x.device().is_cuda()
            && x.dtype() == DType::BF16
            && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
        {
            let (batch, length, _) = x.dims3()?;
            let rows = batch
                .checked_mul(length)
                .ok_or_else(|| candle_core::Error::Msg("QKV/Z rows overflow".into()))?;
            #[cfg(test)]
            let enabled = QKVZ_DUAL_ENABLED.with(std::cell::Cell::get);
            #[cfg(not(test))]
            let enabled = true;
            if let Some(tile) = crate::kernels::fused_qkvz::select_tile(rows).filter(|_| enabled) {
                let packed = self
                    .qkvz_weight
                    .lock()
                    .map_err(|_| {
                        candle_core::Error::Msg("poisoned model-owned QKV/Z weight".into())
                    })?
                    .clone();
                if let Some(packed) = packed {
                    let (qkv, z) = crate::kernels::fused_qkvz::forward(x, &packed, tile)?;
                    #[cfg(test)]
                    QKVZ_DUAL_CALLS.with(|c| c.set(c.get() + 1));
                    return Ok((qkv, Some(z)));
                }
            }
        }
        Ok((linear_attention_projection_forward(x, &self.qkv)?, None))
    }

    #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
    fn qkvz_available(&self) -> bool {
        self.qkv.bias().is_none()
            && self.z.bias().is_none()
            && crate::kernels::fused_qkvz::available(self.qkv.weight(), self.z.weight())
    }
}

#[cfg(all(test, feature = "cuda", sys1_cuda_fused_ffn))]
thread_local! {
    static QKVZ_DUAL_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    static QKVZ_DUAL_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(feature = "cuda")]
impl Decoder {
    pub(crate) fn prepare_nn_weights(
        &self,
        target_tokens: usize,
        bytes_per_token: usize,
    ) -> Result<usize> {
        let base = self.prepare_nn_weights_base(target_tokens, bytes_per_token)?;
        #[cfg(sys1_cuda_fused_ffn)]
        {
            if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120") {
                return Ok(base);
            }
            let mut candidates = Vec::new();
            let mut extra = 0usize;
            for layer in &self.layers {
                if let Mixer::Linear(mixer) = &layer.mixer {
                    if !mixer.qkvz_available() {
                        continue;
                    }
                    if mixer.qkv.cached_nn()?.is_none() || mixer.z.cached_nn()?.is_none() {
                        return Ok(base);
                    }
                    if mixer
                        .qkvz_weight
                        .lock()
                        .map_err(|_| {
                            candle_core::Error::Msg("poisoned model-owned QKV/Z weight".into())
                        })?
                        .is_some()
                    {
                        continue;
                    }
                    let bytes = mixer
                        .qkv
                        .weight()
                        .elem_count()
                        .checked_add(mixer.z.weight().elem_count())
                        .and_then(|n| n.checked_mul(2))
                        .ok_or_else(|| {
                            candle_core::Error::Msg("QKV/Z packed bytes overflow".into())
                        })?;
                    extra = extra.checked_add(bytes).ok_or_else(|| {
                        candle_core::Error::Msg("QKV/Z total bytes overflow".into())
                    })?;
                    candidates.push(mixer);
                }
            }
            let Some(first) = candidates.first() else {
                return Ok(base);
            };
            let candle_core::Device::Cuda(cuda) = first.qkv.weight().device() else {
                return Ok(base);
            };
            let (free, _) = cuda
                .cuda_stream()
                .context()
                .mem_get_info()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            if !nn_capacity_preserved(free, extra, bytes_per_token, target_tokens) {
                tracing::info!(
                    free_bytes = free,
                    extra_qkvz_weight_bytes = extra,
                    target_tokens,
                    "Optional direct QKV/Z layout skipped; parent layouts and token capacity retained"
                );
                return Ok(base);
            }
            for mixer in candidates {
                let packed =
                    crate::kernels::fused_qkvz::prepare(mixer.qkv.weight(), mixer.z.weight())?;
                *mixer.qkvz_weight.lock().map_err(|_| {
                    candle_core::Error::Msg("poisoned model-owned QKV/Z weight".into())
                })? = Some(packed);
            }
            let total = base.checked_add(extra).ok_or_else(|| {
                candle_core::Error::Msg("total optional weight bytes overflow".into())
            })?;
            tracing::info!(
                parent_new_weight_bytes = base,
                extra_qkvz_weight_bytes = extra,
                total_new_weight_bytes = total,
                target_tokens,
                "Direct QKV/Z layouts prepared in addition to all original layouts"
            );
            return Ok(total);
        }
        #[cfg(not(sys1_cuda_fused_ffn))]
        {
            Ok(base)
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
static CONV_LOOKUP157_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(test, feature = "cuda"))]
static CONV_PREPARED_QV_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(test, feature = "cuda", sys1_cuda_direct_ffn))]
thread_local! {static DIRECT_MLP_CALLS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};}

fn full_attention_gate_mul(
    attention: &Tensor,
    gate: &Tensor,
    batch: usize,
    length: usize,
    width: usize,
) -> Result<Tensor> {
    let flat = attention.reshape((batch, length, width))?;
    #[cfg(feature = "cuda")]
    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
        if let Some(output) = crate::kernels::pointwise_mul::try_mul(&flat, gate)? {
            return Ok(output);
        }
    }
    &flat * gate
}
