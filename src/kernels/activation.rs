use candle_core::{DType, Result, Tensor};

pub(crate) fn gelu_erf(input: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if input.device().is_metal() {
        return input.gelu_erf();
    }
    match input.dtype() {
        DType::F16 | DType::BF16 => input
            .to_dtype(DType::F32)?
            .gelu_erf()?
            .to_dtype(input.dtype()),
        _ => input.gelu_erf(),
    }
}

pub(crate) fn silu(input: &Tensor) -> Result<Tensor> {
    if input.device().is_cuda() && input.dtype() == DType::BF16 {
        #[cfg(feature = "cuda")]
        {
            return cuda_bf16_activation(input, None, 0);
        }
        #[cfg(not(feature = "cuda"))]
        {
            input.to_dtype(DType::F32)?.silu()?.to_dtype(input.dtype())
        }
    } else {
        input.silu()
    }
}

pub(crate) fn sigmoid(input: &Tensor) -> Result<Tensor> {
    if input.device().is_cuda() && input.dtype() == DType::BF16 {
        #[cfg(feature = "cuda")]
        {
            return cuda_bf16_activation(input, None, 1);
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle_nn::ops::sigmoid(&input.to_dtype(DType::F32)?)?.to_dtype(input.dtype())
        }
    } else {
        candle_nn::ops::sigmoid(input)
    }
}

#[cfg(feature = "cuda")]
fn cuda_bf16_activation_original_fork(
    input: &Tensor,
    up: Option<&Tensor>,
    mode: i32,
) -> Result<Tensor> {
    use candle_core::{
        CudaStorage, Storage,
        backend::BackendStorage,
        cuda_backend::{
            WrapErr,
            cudarc::driver::{LaunchConfig, PushKernelArg},
        },
        op::BackpropOp,
    };
    use half::bf16;
    if input.dtype() != DType::BF16 || !(0..=2).contains(&mode) || (mode == 2) != up.is_some() {
        candle_core::bail!("unsupported BF16 activation mode/input");
    }
    if let Some(up) = up {
        if up.dims() != input.dims()
            || up.dtype() != input.dtype()
            || !up.device().same_device(input.device())
        {
            candle_core::bail!("SiLU product requires identical colocated BF16 shapes");
        }
    }
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    let count = u32::try_from(input.elem_count())
        .map_err(|_| candle_core::Error::Msg("activation count overflow".into()))?;
    let input = input.contiguous()?;
    let up = up.unwrap_or(&input).contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (us, ul) = up.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(us)) = (&*xs, &*us) else {
        candle_core::bail!("BF16 activation requires CUDA storage");
    };
    let (xstart, xend) = xl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("activation input noncontiguous".into()))?;
    let (ustart, uend) = ul
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("activation up noncontiguous".into()))?;
    let x = xs.as_cuda_slice::<bf16>()?.slice(xstart..xend);
    let u = us.as_cuda_slice::<bf16>()?.slice(ustart..uend);
    let device = xs.device().clone();
    let output = unsafe { device.alloc::<bf16>(count as usize)? };
    let function = device.get_or_load_custom_func(
        "sys1_activation_bf16",
        "sys1_activation",
        include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx")),
    )?;
    let mut launch = function.builder();
    launch.arg(&x);
    launch.arg(&u);
    launch.arg(&output);
    launch.arg(&count);
    launch.arg(&mode);
    unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn mul_silu(input: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if input.device().is_cuda() && input.dtype() == DType::BF16 {
        return cuda_bf16_activation(input, Some(up), 2);
    }
    silu(input)? * up
}

#[cfg(feature = "cuda")]
fn cuda_bf16_activation_lookup_fork(
    input: &Tensor,
    up: Option<&Tensor>,
    mode: i32,
) -> Result<Tensor> {
    use candle_core::{
        CudaStorage, Storage,
        backend::BackendStorage,
        cuda_backend::{
            WrapErr,
            cudarc::driver::{LaunchConfig, PushKernelArg},
        },
        op::BackpropOp,
    };
    use half::bf16;
    if input.dtype() != DType::BF16 || !(0..=2).contains(&mode) || (mode == 2) != up.is_some() {
        candle_core::bail!("unsupported BF16 activation mode/input");
    }
    if let Some(up) = up {
        if up.dims() != input.dims()
            || up.dtype() != input.dtype()
            || !up.device().same_device(input.device())
        {
            candle_core::bail!("SiLU product requires identical colocated BF16 shapes");
        }
    }
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    let count = u32::try_from(input.elem_count())
        .map_err(|_| candle_core::Error::Msg("activation count overflow".into()))?;
    let input = input.contiguous()?;
    let up = up.unwrap_or(&input).contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (us, ul) = up.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(us)) = (&*xs, &*us) else {
        candle_core::bail!("BF16 activation requires CUDA storage");
    };
    let (xstart, xend) = xl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("activation input noncontiguous".into()))?;
    let (ustart, uend) = ul
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("activation up noncontiguous".into()))?;
    let x = xs.as_cuda_slice::<bf16>()?.slice(xstart..xend);
    let u = us.as_cuda_slice::<bf16>()?.slice(ustart..uend);
    let device = xs.device().clone();
    let mut output = unsafe { device.alloc::<bf16>(count as usize)? };
    let stream = device.cuda_stream();
    let mut slot = ACTIVATION_LOOKUP_FORK
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned activation lookup".into()))?;
    if slot
        .as_ref()
        .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
    {
        let mut table = unsafe { device.alloc::<bf16>(131072)? };
        let build = device.get_or_load_custom_func(
            "sys1_build_activation_lookup_bf16",
            "sys1_activation_lookup",
            include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx")),
        )?;
        let mut init = build.builder();
        init.arg(&mut table);
        unsafe { init.launch(LaunchConfig::for_num_elems(131072)) }.w()?;
        *slot = Some(ActivationLookupFork { stream, table });
    }
    let table = &slot
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing lookup table".into()))?
        .table;
    let function = device.get_or_load_custom_func(
        "sys1_activation_lookup_bf16",
        "sys1_activation_lookup",
        include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx")),
    )?;
    let mut launch = function.builder();
    launch.arg(&x);
    launch.arg(&u);
    launch.arg(table);
    launch.arg(&mut output);
    launch.arg(&count);
    launch.arg(&mode);
    unsafe { launch.launch(LaunchConfig::for_num_elems(count.div_ceil(8))) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}
#[cfg(feature = "cuda")]
struct ActivationLookupFork {
    stream: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    table: candle_core::cuda_backend::cudarc::driver::CudaSlice<half::bf16>,
}

#[cfg(feature = "cuda")]
static ACTIVATION_LOOKUP_FORK: std::sync::Mutex<Option<ActivationLookupFork>> =
    std::sync::Mutex::new(None);

#[cfg(feature = "cuda")]
fn cuda_bf16_activation(input: &Tensor, up: Option<&Tensor>, mode: i32) -> Result<Tensor> {
    if matches!(
        option_env!("SYS1_CUDA_COMPUTE_CAP"),
        Some("80" | "90" | "120")
    ) {
        cuda_bf16_activation_lookup_fork(input, up, mode)
    } else {
        cuda_bf16_activation_original_fork(input, up, mode)
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn with_sigmoid_lookup_fork<R>(
    device: &candle_core::CudaDevice,
    run: impl FnOnce(&candle_core::cuda_backend::cudarc::driver::CudaSlice<half::bf16>) -> Result<R>,
) -> Result<R> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    let stream = device.cuda_stream();
    let mut slot = ACTIVATION_LOOKUP_FORK
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned activation lookup".into()))?;
    if slot
        .as_ref()
        .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
    {
        let mut table = unsafe { device.alloc::<half::bf16>(131072)? };
        let build = device.get_or_load_custom_func(
            "sys1_build_activation_lookup_bf16",
            "sys1_activation_lookup",
            include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx")),
        )?;
        let mut init = build.builder();
        init.arg(&mut table);
        unsafe { init.launch(LaunchConfig::for_num_elems(131072)) }.w()?;
        *slot = Some(ActivationLookupFork { stream, table });
    }
    run(&slot
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing lookup table".into()))?
        .table)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuda_bf16_silu_sigmoid_use_single_output_rounding() -> Result<()> {
        let device = crate::device::load().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        if !device.is_cuda() {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if let candle_core::Device::Cuda(cuda) = &device {
            if cuda
                .cuda_stream()
                .context()
                .compute_capability()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                .0
                < 8
            {
                println!("ACTIVATION_OPMATH_SKIPPED BF16 requires SM80");
                return Ok(());
            }
        }
        let input = Tensor::from_vec(
            (0..2049).map(|i| i as f32 / 128. - 8.).collect::<Vec<_>>(),
            2049,
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let f32_input = input.to_dtype(DType::F32)?;
        let observed = [silu(&input)?, sigmoid(&input)?];
        let expected = [
            f32_input.silu()?.to_dtype(DType::BF16)?,
            candle_nn::ops::sigmoid(&f32_input)?.to_dtype(DType::BF16)?,
        ];
        let xs = f32_input.to_vec1::<f32>()?;
        for (kind, (a, b)) in observed.iter().zip(&expected).enumerate() {
            let av = a.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let bv = b.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            assert_eq!(
                av.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                bv.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            for (x, y) in xs.iter().zip(av) {
                let sig = 1. / (1. + (-(*x as f64)).exp());
                let reference = if kind == 0 { *x as f64 * sig } else { sig };
                assert!(
                    y.is_finite() && (y as f64 - reference).abs() <= 1e-5 + 0.016 * reference.abs()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn promoted_gelu_matches_native_torch_fixtures() -> Result<()> {
        let root = crate::assets::fixtures("activation")?;
        let device =
            crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        for (name, rtol) in [("bf16", 0.016f32), ("f16", 0.001), ("f32", 1.3e-6)] {
            #[cfg(feature = "cuda")]
            if name == "bf16" {
                if let candle_core::Device::Cuda(cuda) = &device {
                    let (major, _) = cuda
                        .cuda_stream()
                        .context()
                        .compute_capability()
                        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                    if major < 8 {
                        continue;
                    }
                }
            }
            let data = candle_core::safetensors::load(
                root.join(format!("gelu-erf-{name}.safetensors")),
                &device,
            )?;
            let actual = gelu_erf(&data["input"])?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let expected = data["expected"].to_dtype(DType::F32)?.to_vec1::<f32>()?;
            assert_eq!(actual.len(), 2049);
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    actual.is_finite()
                        && (*actual - expected).abs() <= 1e-5 + rtol * expected.abs(),
                    "{name} GELU at{index}: {actual} vs {expected}"
                );
            }
        }
        Ok(())
    }
}
