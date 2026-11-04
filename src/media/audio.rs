//! LFM3-Omni audio frontend: bounded decoding and the checkpoint's log-mel transform.
use crate::schema::MediaInput;
use anyhow::{Context, ensure};
use candle_core::{Device, Tensor};
use rustfft::{FftPlanner, num_complex::Complex32};
use std::{f64::consts::PI, io::Write};

const RATE: usize = 16000;
const FFT: usize = 400;
const HOP: usize = 160;
const MELS: usize = 128;

pub(crate) fn decode(input: &MediaInput, max_tokens: usize) -> anyhow::Result<Vec<f32>> {
    let bytes = super::input_bytes(input, "audio")?;
    let mut file = tempfile::NamedTempFile::new()?;
    file.write_all(&bytes)?;
    file.flush()?;
    let path = file.path().to_string_lossy().into_owned();
    let probe_args = [
        "-v".into(),
        "error".into(),
        "-select_streams".into(),
        "a:0".into(),
        "-show_entries".into(),
        "stream=sample_rate,channels".into(),
        "-of".into(),
        "json".into(),
        path.clone(),
    ];
    let probe: serde_json::Value =
        serde_json::from_slice(&super::run_audio_tool("ffprobe", &probe_args, 1 << 20)?)?;
    let stream = &probe["streams"][0];
    let rate: usize = stream["sample_rate"]
        .as_str()
        .context("audio sample rate is unavailable")?
        .parse()?;
    let channels = stream["channels"]
        .as_u64()
        .context("audio channel count is unavailable")? as usize;
    ensure!(
        (1..=384000).contains(&rate) && (1..=64).contains(&channels),
        "unsupported audio sample rate or channel count"
    );
    let samples = max_tokens
        .checked_mul(8)
        .and_then(|n| n.checked_mul(HOP))
        .context("audio budget overflow")?;
    let source_samples = samples
        .checked_add(FFT)
        .and_then(|n| n.checked_mul(rate))
        .context("audio budget overflow")?
        .div_ceil(RATE);
    let max_bytes = source_samples
        .checked_mul(channels)
        .and_then(|n| n.checked_mul(4))
        .context("audio budget overflow")?
        .min(128 << 20);
    let args = [
        "-v".into(),
        "error".into(),
        "-i".into(),
        path,
        "-vn".into(),
        "-f".into(),
        "f32le".into(),
        "pipe:1".into(),
    ];
    let pcm = super::run_audio_tool("ffmpeg", &args, max_bytes)?;
    ensure!(pcm.len().is_multiple_of(4), "invalid decoded audio size");
    let interleaved: Vec<f32> = pcm
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    ensure!(
        interleaved.len().is_multiple_of(channels),
        "decoded audio has incomplete channel frames"
    );
    let mono: Vec<f32> = interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    let waveform = resample(&mono, rate, RATE);
    ensure!(!waveform.is_empty(), "audio contains no samples");
    ensure!(
        waveform.iter().all(|x| x.is_finite()),
        "audio contains non-finite samples"
    );
    ensure!(
        frame_count(waveform.len()).div_ceil(8) <= max_tokens,
        "audio exceeds --max-model-len token budget"
    );
    Ok(waveform)
}
/// The reference runtime's Hann-windowed sinc (torchaudio defaults). Unlike a
/// decoder's default resampler, this keeps the model frontend consistent across
/// WAV, compressed files, channel counts and original sample rates.
fn resample(wave: &[f32], rate: usize, target: usize) -> Vec<f32> {
    if rate == target {
        return wave.to_vec();
    }
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    };
    let common = gcd(rate, target);
    let old = rate / common;
    let new = target / common;
    let cut = old.min(new) as f32 * 0.99;
    let width = (6. * old as f32 / cut).ceil() as isize;
    let kernels: Vec<Vec<(isize, f32)>> = (0..new)
        .map(|phase| {
            let center = phase as f32 / new as f32 * old as f32;
            let lo = (center - width as f32).floor() as isize;
            let hi = (center + width as f32).ceil() as isize;
            (lo.max(-width)..=hi.min(width + old as isize - 1))
                .filter_map(|k| {
                    let t = (-(phase as f32) / new as f32 + k as f32 / old as f32) * cut;
                    if t.abs() >= 6. {
                        return None;
                    }
                    let window = (t * std::f32::consts::PI / 6. / 2.).cos().powi(2);
                    let angle = t * std::f32::consts::PI;
                    let sinc = if angle == 0. { 1. } else { angle.sin() / angle };
                    Some((k, sinc * window * (cut / old as f32)))
                })
                .collect()
        })
        .collect();
    let count = (wave.len() * new).div_ceil(old);
    (0..count)
        .map(|i| {
            let base = (i / new * old) as isize;
            kernels[i % new]
                .iter()
                .map(|&(k, w)| {
                    let index = base + k;
                    if index < 0 {
                        0.
                    } else {
                        wave.get(index as usize).copied().unwrap_or(0.) * w
                    }
                })
                .sum()
        })
        .collect()
}
pub(crate) fn frame_count(samples: usize) -> usize {
    if samples <= FFT {
        1
    } else {
        1 + (samples - FFT).div_ceil(HOP)
    }
}

/// Slaney's piecewise-linear/log frequency scale and area-normalized triangles.
fn filters() -> Vec<Vec<f32>> {
    let hz_to_mel = |hz: f64| {
        if hz < 1000. {
            hz / (200. / 3.)
        } else {
            15. + (hz / 1000.).ln() / (6.4f64.ln() / 27.)
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel < 15. {
            mel * (200. / 3.)
        } else {
            1000. * ((mel - 15.) * (6.4f64.ln() / 27.)).exp()
        }
    };
    let high = hz_to_mel(8000.);
    let freqs: Vec<_> = (0..MELS + 2)
        .map(|i| mel_to_hz(high * i as f64 / (MELS + 1) as f64))
        .collect();
    (0..MELS)
        .map(|m| {
            (0..=FFT / 2)
                .map(|k| {
                    let hz = k as f64 * RATE as f64 / FFT as f64;
                    let up = (hz - freqs[m]) / (freqs[m + 1] - freqs[m]);
                    let down = (freqs[m + 2] - hz) / (freqs[m + 2] - freqs[m + 1]);
                    (up.min(down).max(0.) * 2. / (freqs[m + 2] - freqs[m])) as f32
                })
                .collect()
        })
        .collect()
}
pub(crate) fn log_mel(waveform: &[f32], device: &Device) -> anyhow::Result<Tensor> {
    ensure!(
        waveform.iter().all(|x| x.is_finite()),
        "audio contains non-finite samples"
    );
    let frames = frame_count(waveform.len());
    let window: Vec<f32> = (0..FFT)
        .map(|i| (0.5 - 0.5 * (2. * PI * i as f64 / FFT as f64).cos()) as f32)
        .collect();
    let bank = filters();
    let fft = FftPlanner::new().plan_fft_forward(FFT);
    let mut buffer = vec![Complex32::new(0., 0.); FFT];
    let mut result = vec![0f32; MELS * frames];
    for frame in 0..frames {
        for (i, value) in buffer.iter_mut().enumerate() {
            *value = Complex32::new(
                waveform.get(frame * HOP + i).copied().unwrap_or(0.) * window[i],
                0.,
            );
        }
        fft.process(&mut buffer);
        for (mel, filter) in bank.iter().enumerate() {
            let power: f32 = filter
                .iter()
                .zip(&buffer)
                .map(|(w, x)| w * x.norm_sqr())
                .sum();
            result[mel * frames + frame] = (power.max(1e-12).log10() + 4.) / 4.;
        }
    }
    Ok(Tensor::from_vec(result, (MELS, frames), device)?)
}

#[cfg(test)]
pub(crate) fn fixture_wav(rate: usize, samples: usize) -> Vec<u8> {
    let data_len = (samples * 2) as u32;
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&(rate as u32).to_le_bytes());
    wav.extend_from_slice(&(rate as u32 * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..samples {
        let noise = (((i * 37 + 13) % 997) as i32 - 498) * 2;
        let beep = (rate / 5 <= i && i < 2 * rate / 5) || (3 * rate / 5 <= i && i < 4 * rate / 5);
        let tone = if beep {
            if (i / (rate / 1000).max(1)).is_multiple_of(2) {
                9000
            } else {
                -9000
            }
        } else {
            0
        };
        wav.extend_from_slice(&((noise + tone) as i16).to_le_bytes());
    }
    wav
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn silence_padding_and_token_lengths() -> anyhow::Result<()> {
        for (n, frames) in [(1, 1), (400, 1), (401, 2), (16000, 99)] {
            assert_eq!(frame_count(n), frames);
            let mel = log_mel(&vec![0.; n], &Device::Cpu)?;
            assert_eq!(mel.dims(), &[128, frames]);
            assert!(
                mel.flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|v| (*v + 2.).abs() < 1e-6)
            );
        }
        Ok(())
    }
    #[test]
    fn decoder_preserves_pcm_and_rejects_over_budget_clips() -> anyhow::Result<()> {
        let bytes = fixture_wav(16000, 16000);
        let decoded = decode(&MediaInput::Bytes(bytes.clone()), 13)?;
        assert_eq!(decoded.len(), 16000);
        assert_eq!(decoded[0], -970. / 32768.);
        assert!(decode(&MediaInput::Bytes(bytes), 1).is_err());
        Ok(())
    }
    #[test]
    fn resampling_keeps_dc_and_exact_duration() {
        let data = resample(&vec![0.5; 8000], 8000, 16000);
        assert_eq!(data.len(), 16000);
        assert!(data[100..15900].iter().all(|v| (*v - 0.5).abs() < 0.001));
    }
}
