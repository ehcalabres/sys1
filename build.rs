#[cfg(feature = "cuda")]
#[path = "src/assets.rs"]
mod assets;

use std::{env, fs, path::PathBuf, process::Command};

fn kernel_sources(extension: &str) -> Vec<PathBuf> {
    let directory = PathBuf::from("src/kernels");
    println!("cargo:rerun-if-changed={}", directory.display());
    let mut sources: Vec<_> = fs::read_dir(&directory)
        .expect("failed to read kernel directory")
        .map(|entry| entry.expect("failed to read kernel entry").path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == extension))
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "no .{extension} kernels found");
    for source in &sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }
    sources
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(sys1_cuda_fused_ffn)");
    println!("cargo:rustc-check-cfg=cfg(sys1_cuda_fp64_emulation)");
    println!("cargo:rustc-check-cfg=cfg(sys1_cuda_direct_ffn)");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    if env::var_os("CARGO_FEATURE_METAL").is_some()
        && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
    {
        let sources = kernel_sources("metal");
        let output_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let library = output_dir.join("kernels.metallib");
        let mut air_files = Vec::with_capacity(sources.len());
        for source in sources {
            let air = output_dir
                .join(source.file_stem().unwrap())
                .with_extension("air");
            let metal = Command::new("xcrun")
                .args(["--toolchain", "Metal", "metal", "-std=metal3.0", "-c"])
                .arg(&source)
                .arg("-o")
                .arg(&air)
                .status()
                .expect("failed to run the Metal compiler");
            assert!(metal.success(), "failed to compile {}", source.display());
            air_files.push(air);
        }
        let link = Command::new("xcrun")
            .args(["--toolchain", "Metal", "metallib"])
            .args(&air_files)
            .arg("-o")
            .arg(&library)
            .status()
            .expect("failed to run metallib");
        assert!(link.success(), "failed to link Metal kernels");
        println!(
            "cargo:rustc-env=SYS1_KERNEL_METALLIB_PATH={}",
            library.display()
        );
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        let capability = compute_capability();
        if capability == 86 {
            let version = Command::new("nvcc")
                .arg("--version")
                .output()
                .expect("failed to inspect CUDA toolkit for FP64 emulation");
            assert!(version.status.success(), "nvcc --version failed");
            let version = String::from_utf8_lossy(&version.stdout);
            let major = version
                .split("release ")
                .nth(1)
                .and_then(|v| v.split('.').next())
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(0);
            let cuda_root = [
                "CUDA_HOME",
                "CUDA_PATH",
                "CUDA_ROOT",
                "CUDA_TOOLKIT_ROOT_DIR",
            ]
            .iter()
            .find_map(env::var_os)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
            let header = cuda_root.join("include/cublas_api.h");
            println!("cargo:rerun-if-changed={}", header.display());
            let header = fs::read_to_string(header).unwrap_or_default();
            if major >= 13
                && [
                    "cublasSetFixedPointEmulationMantissaControl",
                    "cublasSetFixedPointEmulationMantissaBitOffset",
                    "cublasSetFixedPointEmulationMaxMantissaBitCount",
                ]
                .iter()
                .all(|name| header.contains(name))
            {
                println!("cargo:rustc-cfg=sys1_cuda_fp64_emulation");
            }
        }

        compile_assets(capability);
        println!("cargo:rustc-env=SYS1_CUDA_COMPUTE_CAP={capability}");
        #[cfg(feature = "cuda")]
        compile_fused_gate_up(capability);
        #[cfg(feature = "cuda")]
        compile_direct_gate_up(capability);
        let sources: Vec<_> = kernel_sources("cu")
            .into_iter()
            .filter(|source| {
                if capability >= 80 {
                    return true;
                }
                let name = source.file_name().unwrap().to_str().unwrap();
                !(name.starts_with("qwen35_c64_")
                    || name.starts_with("qwen35_z_")
                    || matches!(
                        name,
                        "qwen35_early_bf16_output_fork.cu" | "qwen35_gate_native_vector_fork.cu"
                    ))
            })
            .collect();
        let output_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let unit = output_dir.join("kernels.cu");
        let includes = sources
            .iter()
            .map(|path| {
                format!(
                    "#include \"{}\"\n",
                    path.file_name().unwrap().to_str().unwrap()
                )
            })
            .collect::<String>();
        fs::write(&unit, includes).expect("failed to generate CUDA translation unit");
        let output = output_dir.join("cuda.ptx");
        let status = Command::new("nvcc")
            .args(["-ptx", &format!("-arch=compute_{capability}"), "-I"])
            .arg("src/kernels")
            .arg(&unit)
            .arg("-o")
            .arg(&output)
            .status()
            .expect("failed to run nvcc for CUDA kernels");
        assert!(
            status.success(),
            "failed to compile CUDA kernels for compute capability {capability}"
        );
    }
}

fn compute_capability() -> u32 {
    match env::var("CUDA_COMPUTE_CAP") {
        Ok(value) => parse_compute_capability(&value),
        Err(env::VarError::NotPresent) => {
            let output = Command::new("nvidia-smi")
                .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
                .output()
                .expect("failed to detect CUDA compute capability with nvidia-smi; set CUDA_COMPUTE_CAP when building without a GPU");
            assert!(
                output.status.success(),
                "nvidia-smi failed to detect CUDA compute capability; set CUDA_COMPUTE_CAP when building without a GPU: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let value = String::from_utf8(output.stdout)
                .expect("nvidia-smi returned invalid UTF-8 for CUDA compute capability");
            parse_compute_capability(value.lines().next().unwrap_or_default())
        }
        Err(error) => panic!("invalid CUDA_COMPUTE_CAP: {error}"),
    }
}

fn parse_compute_capability(value: &str) -> u32 {
    let normalized = value.trim().to_ascii_lowercase();
    let normalized = normalized
        .strip_prefix("sm_")
        .unwrap_or(&normalized)
        .trim_end_matches(['a', 'f'])
        .replace('.', "");
    normalized.parse::<u32>().unwrap_or_else(|_| {
        panic!(
            "invalid CUDA_COMPUTE_CAP {:?}; expected values such as 80, 89, or 90",
            value
        )
    })
}

#[cfg(feature = "cuda")]
fn compile_fused_gate_up(capability: u32) {
    if !matches!(capability, 80 | 120) {
        return;
    }
    println!("cargo:rerun-if-changed=src/kernels/fused_qkvz.cuh");
    let source = "src/kernels/fused_gate_up.cuh";
    println!("cargo:rerun-if-changed={source}");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let includes = cudaforge::DependencyManager::new()
        .with_cutlass(Some("7d49e6c7e2f8896c47f586706e67e1fb215529dc"))
        .fetch_all(&out)
        .expect("failed to fetch pinned CUTLASS headers");
    let status = Command::new("nvcc")
        .args(["-x", "cu", "-ptx", "-std=c++17", "-O3", "-lineinfo"])
        .arg(format!("-arch=compute_{capability}"))
        .args(includes)
        .arg(source)
        .arg("-o")
        .arg(out.join("fused_gate_up.ptx"))
        .status()
        .expect("failed to compile fused MLP PTX");
    assert!(status.success(), "fused MLP PTX compile failed");
    println!("cargo:rustc-cfg=sys1_cuda_fused_ffn");
}

fn compile_assets(capability: u32) {
    println!("cargo:rustc-check-cfg=cfg(sys1_cuda_fa4_sm90)");
    #[cfg(feature = "cuda")]
    {
        let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        for architecture in ["qwen35_aot_assets_sm80", "qwen35_aot_assets_v91"] {
            fs::create_dir_all(out.join(architecture)).expect("create CUDA asset directory");
            for index in 0..5 {
                let name = format!("kernels/{architecture}/{index}.cubin");
                let source = assets::get(&name).expect("download pinned CUDA asset");
                fs::copy(
                    source,
                    out.join(architecture).join(format!("{index}.cubin")),
                )
                .expect("stage CUDA asset");
            }
        }
        if capability == 90 {
            let source = assets::get("attention_fa4_assets_v112/fa4_sm90_bf16_d256.fatbin")
                .expect("download pinned FA4 image");
            fs::copy(source, out.join("fa4_sm90_bf16_d256.fatbin")).expect("stage FA4 image");
            println!("cargo:rustc-cfg=sys1_cuda_fa4_sm90");
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = capability;
}

#[cfg(feature = "cuda")]
fn compile_direct_gate_up(capability: u32) {
    if capability != 86 {
        return;
    }
    let source = "src/kernels/fused_gate_up_sm86_direct.cuh";
    println!("cargo:rerun-if-changed={source}");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let includes = cudaforge::DependencyManager::new()
        .with_cutlass(Some("7d49e6c7e2f8896c47f586706e67e1fb215529dc"))
        .fetch_all(&out)
        .expect("pinned CUTLASS headers");
    let status = Command::new("nvcc")
        .args([
            "-x",
            "cu",
            "-ptx",
            "-std=c++17",
            "-O3",
            "-lineinfo",
            "-arch=compute_86",
        ])
        .args(includes)
        .arg(source)
        .arg("-o")
        .arg(out.join("fused_gate_up_direct.ptx"))
        .status()
        .expect("compile original-weight MLP");
    assert!(status.success(), "direct MLP compilation failed");
    println!("cargo:rustc-cfg=sys1_cuda_direct_ffn");
}
