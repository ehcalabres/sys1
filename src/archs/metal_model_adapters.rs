use super::{modernbert, qwen35_text};
use crate::models::AttentionImplementation;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub(crate) enum QwenDecoder {
    V3(qwen35_text::Decoder),
    MainMetal(qwen35_text::main_metal::Decoder),
}

impl QwenDecoder {
    pub(crate) fn load(
        config: &qwen35_text::TextConfig,
        vb: VarBuilder,
        attention: AttentionImplementation,
    ) -> Result<Self> {
        if vb.device().is_metal() {
            return qwen35_text::main_metal::Decoder::load(config, vb, attention)
                .map(Self::MainMetal);
        }
        qwen35_text::Decoder::load(config, vb, attention).map(Self::V3)
    }
    pub(crate) fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::V3(model) => model.forward(ids),
            Self::MainMetal(model) => model.forward(ids),
        }
    }
    pub(crate) fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::V3(model) => model.embed(ids),
            Self::MainMetal(model) => model.embed(ids),
        }
    }
    pub(crate) fn forward_embeds(&self, embeds: &Tensor, positions: &[[u32; 3]]) -> Result<Tensor> {
        match self {
            Self::V3(model) => model.forward_embeds(embeds, positions),
            Self::MainMetal(model) => model.forward_embeds(embeds, positions),
        }
    }
}

pub(crate) enum ModernBertEncoder {
    V3(modernbert::Encoder),
    MainMetal(modernbert::main_metal::Encoder),
}

impl ModernBertEncoder {
    pub(crate) fn load(
        vb: VarBuilder,
        config: &modernbert::Config,
        compute_dtype: candle_core::DType,
        attention: AttentionImplementation,
    ) -> Result<Self> {
        if vb.device().is_metal() {
            return modernbert::main_metal::Encoder::load(vb, config, compute_dtype, attention)
                .map(Self::MainMetal);
        }
        modernbert::Encoder::load(vb, config, compute_dtype, attention).map(Self::V3)
    }
    pub(crate) fn forward(
        &self,
        ids: &Tensor,
        mask: &Tensor,
        lengths: &[usize],
        has_padding: bool,
    ) -> Result<Tensor> {
        match self {
            Self::V3(model) => model.forward(ids, mask, lengths, has_padding),
            Self::MainMetal(model) => model.forward(ids, mask, lengths, has_padding),
        }
    }
}
