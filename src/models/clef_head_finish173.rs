use candle_core::{DType, Result, Tensor};

pub fn grouped_final_scores(
    raw_priors: &[Tensor],
    field_cos: &Tensor,
    residual_output: &Tensor,
    prior_scale: &Tensor,
    joint_scale: &Tensor,
    gate: &Tensor,
) -> Result<(Tensor, Tensor)> {
    if raw_priors.is_empty()
        || raw_priors
            .iter()
            .any(|p| p.rank() != 1 || p.elem_count() == 0 || p.dtype() != DType::F32)
        || field_cos.dtype() != DType::F32
        || [prior_scale, joint_scale, gate]
            .iter()
            .any(|s| s.dtype() != DType::F32 || s.elem_count() != 1)
    {
        candle_core::bail!("nonempty FP32 prior vectors, field cosines and scalar scales required")
    }
    let rows = raw_priors.iter().try_fold(0usize, |sum, p| {
        sum.checked_add(p.elem_count())
            .ok_or_else(|| candle_core::Error::Msg("prior length overflow".into()))
    })?;
    if field_cos.dims() != [rows] || residual_output.dims() != [rows, 1] {
        candle_core::bail!("final-head field/residual population mismatch")
    }
    #[cfg(test)]
    if field_cos.device().is_cuda() {
        eprintln!(
            "HOPPER_FINISH173 options={} questions={}",
            rows,
            raw_priors.len()
        );
    }
    let raw_prior = if raw_priors.len() == 1 {
        raw_priors[0].clone()
    } else {
        Tensor::cat(&raw_priors.iter().collect::<Vec<_>>(), 0)?
    };
    let prior = raw_prior.broadcast_mul(prior_scale)?;
    let residual = residual_output.squeeze(1)?.to_dtype(DType::F32)?;
    let joint = (field_cos.broadcast_mul(joint_scale)? + residual)?;
    let scores = (&prior + joint.broadcast_mul(gate)?)?;

    Ok((scores, prior))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn bits(t: &Tensor) -> Result<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    fn original(
        priors: &[Tensor],
        field_cos: &Tensor,
        residual: &Tensor,
        prior_scale: &Tensor,
        joint_scale: &Tensor,
        gate: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut scores = Vec::new();
        let mut scaled = Vec::new();
        let mut start = 0;
        for raw in priors {
            let count = raw.elem_count();
            let prior = raw.broadcast_mul(prior_scale)?;
            let field = field_cos.narrow(0, start, count)?;
            let residual = residual
                .narrow(0, start, count)?
                .squeeze(1)?
                .to_dtype(DType::F32)?;
            let joint = (field.broadcast_mul(joint_scale)? + residual)?;
            scores.push((&prior + joint.broadcast_mul(gate)?)?);
            scaled.push(prior);
            start += count;
        }
        Ok((
            Tensor::cat(&scores.iter().collect::<Vec<_>>(), 0)?,
            Tensor::cat(&scaled.iter().collect::<Vec<_>>(), 0)?,
        ))
    }
}
