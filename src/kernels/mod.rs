pub(crate) mod activation;
#[cfg(feature = "cuda")]
pub(crate) mod biased_projection;
#[cfg(feature = "cuda")]
pub(crate) mod blaslt;
pub(crate) mod linear;
#[cfg(feature = "metal")]
pub(crate) mod metal;
#[cfg(feature = "cuda")]
pub(crate) mod stable_attention;

#[cfg(feature = "cuda")]
pub(crate) mod reduction;

#[cfg(feature = "cuda")]
pub(crate) mod span_mean;

#[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
pub(crate) mod fused_gate_up;

#[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
pub(crate) mod fused_qkvz;

#[cfg(feature = "cuda")]
pub(crate) mod guarded_dgemm;

#[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
pub(crate) mod fused_gate_up_direct;

#[cfg(all(feature = "cuda", sys1_cuda_fp64_emulation))]
pub(crate) mod emulated_dgemm;

#[cfg(feature = "cuda")]
pub(crate) mod head_blaslt;

#[cfg(feature = "cuda")]
pub(crate) mod pointwise_mul;
