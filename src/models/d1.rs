mod parsing;
mod pointer;
mod typed;
use super::{AttentionImplementation, DecisionModel};
use crate::{
    archs::{
        lfm3::{self, Config},
        qwen35_text::VisionConfig,
        qwen35_vision::{images, video},
    },
    schema::{ApiError, DecisionRequest, DecisionResponse, Usage},
    tokenizer::{self, Tokenizer},
};
use anyhow::{Context, ensure};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

#[derive(Deserialize)]
struct Calibration {
    #[serde(default = "one")]
    default: f64,
    #[serde(default)]
    temperatures: HashMap<String, f64>,
    #[serde(default)]
    tables: HashMap<String, Vec<[f64; 2]>>,
}
fn one() -> f64 {
    1.
}
impl Calibration {
    fn temperature(&self, kind: &str, n: usize) -> f64 {
        let bucket = match kind {
            "noul" => "noul".into(),
            "score" => format!("score:{n}"),
            _ => format!(
                "choice:{}",
                if n <= 3 {
                    "small"
                } else if n <= 8 {
                    "mid"
                } else if n <= 26 {
                    "large"
                } else {
                    "inventory"
                }
            ),
        };
        self.temperatures
            .get(&bucket)
            .copied()
            .unwrap_or(self.default)
    }
    fn through(&self, name: &str, p: f64) -> f64 {
        self.map(name, p, false)
    }
    fn level(&self, name: &str, p: f64) -> f64 {
        self.map(name, p, true)
    }
    fn multiple(&self, p: f64) -> f64 {
        self.through("multiple", p)
    }
    fn map(&self, name: &str, p: f64, inverse: bool) -> f64 {
        let Some(knots) = self.tables.get(name).filter(|v| !v.is_empty()) else {
            return p;
        };
        let x = usize::from(inverse);
        let y = 1 - x;
        let i = knots.partition_point(|v| v[x] < p);
        if i == 0 {
            return knots[0][y];
        }
        if i == knots.len() {
            return knots[i - 1][y];
        }
        let a = knots[i - 1];
        let b = knots[i];
        a[y] + (p - a[x]) * (b[y] - a[y]) / (b[x] - a[x])
    }
}
struct Question {
    name: String,
    kind: String,
    labels: Vec<String>,
    criteria: Value,
    suffix: String,
    slots: Vec<usize>,
    forms: Vec<Vec<u32>>,
    multiple: bool,
    readout: typed::Readout,
    slot_forms: Vec<typed::Forms>,
    audio: Option<Tensor>,
}

pub struct D1 {
    tokenizer: Tokenizer,
    offset_tokenizer: tokenizers_offsets::Tokenizer,
    specials: Vec<String>,
    config: Config,
    text: lfm3::TextModel,
    vision: lfm3::VisionModel,
    audio: Option<lfm3::AudioModel>,
    calibration: Calibration,
    device: Device,
    dtype: DType,
    max_length: usize,
}
impl D1 {
    pub fn load(
        path: &Path,
        dtype: Option<DType>,
        max_length: Option<usize>,
        attention: AttentionImplementation,
    ) -> anyhow::Result<Self> {
        ensure!(
            matches!(
                attention,
                AttentionImplementation::Auto | AttentionImplementation::Eager
            ),
            "d1 currently supports auto or eager attention"
        );
        let config: Config = serde_json::from_slice(&fs::read(path.join("config.json"))?)?;
        validate_config(&config)?;
        let max_length =
            max_length.unwrap_or(16384.min(config.text_config.max_position_embeddings));
        ensure!(
            max_length > 0 && max_length <= config.text_config.max_position_embeddings,
            "invalid d1 --max-model-len"
        );
        let architecture = if config.audio_encoder_config.is_some() {
            super::Architecture::Lfm3Omni
        } else {
            super::Architecture::Lfm3Vl
        };
        let dtype = super::dtype::resolve(path, architecture, dtype)?;
        let device = crate::device::load()?;
        ensure!(
            !device.is_cpu() || dtype == DType::F32,
            "d1 on CPU requires --dtype f32"
        );
        let raw = fs::read(path.join("tokenizer.json"))?;
        let tokenizer = tokenizer::from_qwen_json(&raw)?;
        for (name, id) in [
            ("<|startoftext|>", config.bos_token_id),
            ("<|vision_start|>", config.vision_start_token_id),
            ("<|vision_end|>", config.vision_end_token_id),
            ("<|image|>", config.image_token_id),
            ("<|video|>", config.video_token_id),
        ] {
            ensure!(
                tokenizer.token_to_id(name) == Some(id),
                "config token {name} differs from tokenizer"
            );
        }
        if let Some(id) = config.audio_token_id {
            ensure!(
                tokenizer.token_to_id("<|audio|>") == Some(id),
                "audio token differs from tokenizer"
            );
        }

        let tok: Value = serde_json::from_slice(&raw)?;
        let mut specials: Vec<String> = tok["added_tokens"]
            .as_array()
            .context("missing added tokens")?
            .iter()
            .filter(|v| v["special"] == true)
            .filter_map(|v| v["content"].as_str().map(str::to_owned))
            .collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.len()));
        let shards: Vec<_> = fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name().and_then(|s| s.to_str()).is_some_and(|s| {
                    s == "model.safetensors"
                        || s.starts_with("model-") && s.ends_with(".safetensors")
                })
            })
            .collect();
        ensure!(!shards.is_empty(), "no d1 Safetensors weights found");
        // SAFETY: checkpoint files must remain immutable while mapped.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, dtype, &device)? };
        let text = lfm3::TextModel::load(&config.text_config, vb.pp("model.language_model"))?;
        let vision = lfm3::VisionModel::load(&config, vb.pp("model"))?;
        let audio = config
            .audio_encoder_config
            .as_ref()
            .map(|c| lfm3::AudioModel::load(c, config.text_config.hidden_size, vb.pp("model")))
            .transpose()?;
        let calibration: Calibration = if path.join("calibration.json").exists() {
            serde_json::from_slice(&fs::read(path.join("calibration.json"))?)?
        } else {
            Calibration {
                default: 1.,
                temperatures: HashMap::new(),
                tables: HashMap::new(),
            }
        };
        ensure!(
            calibration.default.is_finite()
                && calibration.default > 0.
                && calibration
                    .temperatures
                    .values()
                    .all(|t| t.is_finite() && *t > 0.),
            "invalid d1 calibration temperature"
        );
        validate_processor(path, &config)?;
        Ok(Self {
            tokenizer,
            offset_tokenizer: tokenizers_offsets::Tokenizer::from_bytes(&raw)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?,
            specials,
            config,
            text,
            vision,
            audio,
            calibration,
            device,
            dtype,
            max_length,
        })
    }
    fn plain(&self, text: &str) -> String {
        self.plain_offsets(text).0
    }
    fn plain_offsets(&self, text: &str) -> (String, Vec<usize>) {
        let mut out = text.to_owned();
        let mut boundaries: Vec<_> = (0..=text.chars().count()).collect();
        loop {
            let next = self
                .specials
                .iter()
                .filter_map(|special| out.find(special).map(|at| (at, special)))
                .min_by(|(a, x), (b, y)| a.cmp(b).then_with(|| y.len().cmp(&x.len())));
            let Some((at, special)) = next else { break };
            let pos = at + special.chars().next().unwrap().len_utf8();
            let ch = out[..pos].chars().count();
            boundaries.insert(ch, boundaries[ch]);
            out.insert(pos, '\u{200b}');
        }
        (out, boundaries)
    }
    fn ids(&self, forms: &[String]) -> anyhow::Result<Vec<u32>> {
        let mut result = Vec::new();
        for text in forms {
            let ids = self.tokenizer.encode(text)?;
            if ids.len() == 1 && !result.contains(&ids[0]) {
                result.push(ids[0]);
            }
        }
        Ok(result)
    }
    fn aliases(&self, labels: &[String]) -> anyhow::Result<Vec<(String, u32)>> {
        let native = labels
            .iter()
            .all(|s| s.trim().chars().count() == 1 && s.trim().chars().all(char::is_alphabetic));
        let codes: Vec<String> = if native {
            labels.iter().map(|s| s.trim().into()).collect()
        } else {
            (0..labels.len())
                .map(|i| {
                    if i < 26 {
                        ((b'A' + i as u8) as char).to_string()
                    } else {
                        format!(
                            "{}{}",
                            (b'A' + ((i - 26) / 26) as u8) as char,
                            (b'A' + ((i - 26) % 26) as u8) as char
                        )
                    }
                })
                .collect()
        };
        let pool: Vec<String> =
            (b'A'..=b'Z')
                .map(|c| (c as char).to_string())
                .chain((0..100).map(|i| format!("{i:02}")))
                .chain((b'a'..=b'z').map(|c| (c as char).to_string()))
                .chain((0..200).map(|i| format!("#{i}")))
                .chain((b'A'..=b'Z').flat_map(|a| {
                    (b'A'..=b'Z').map(move |b| format!("{}{}", a as char, b as char))
                }))
                .collect();
        let mut used = HashSet::new();
        let mut result = Vec::new();
        for code in codes {
            let mut found = None;
            for raw in std::iter::once(&code).chain(pool.iter()) {
                let ids = self.tokenizer.encode(raw)?;
                if ids.len() == 1 && used.insert(ids[0]) {
                    found = Some((raw.clone(), ids[0]));
                    break;
                }
            }
            result.push(found.context("no single-token option alias")?);
        }
        Ok(result)
    }
    fn question(&self, name: &str, q: &Value) -> anyhow::Result<Question> {
        let kind = q["type"]
            .as_str()
            .context("missing question type")?
            .to_owned();
        let spoken = crate::schema::MediaInput::spoken(&q["instructions"])?;
        let audio = spoken
            .as_ref()
            .map(|input| self.encode_audio(input))
            .transpose()?;
        let said = if audio.is_some() {
            "\u{e000}".into()
        } else {
            match &q["instructions"] {
                Value::Null => name.to_owned(),
                v if crate::schema::is_d1_typed(&kind) && !v.is_string() => parsing::python_repr(v),
                v => as_text(v),
            }
        };
        crate::schema::validate_d1_question(q).map_err(anyhow::Error::msg)?;
        let construction = self.typed_question(q, &said)?;
        let mut readout = typed::Readout::Legacy;
        let mut slot_forms = Vec::new();
        let mut markers = Vec::new();
        let mut after = false;
        let multiple = q["multiple"].as_bool().unwrap_or(false);
        ensure!(
            !multiple || kind == "choice",
            "multiple is supported only for choice questions"
        );
        let mut forms = Vec::new();
        let mut labels = Vec::new();
        let yesno = || -> anyhow::Result<Vec<Vec<u32>>> {
            Ok(vec![
                self.ids(&["yes".into(), "Yes".into(), "YES".into()])?,
                self.ids(&["no".into(), "No".into(), "NO".into()])?,
            ])
        };
        let body = match kind.as_str() {
            "noul" => {
                forms = yesno()?;
                let extra = if q["criteria"].as_object().is_some_and(|o| !o.is_empty()) {
                    format!(
                        "\nYes: {}\nNo: {}",
                        python_text(&q["criteria"]["true"]),
                        python_text(&q["criteria"]["false"])
                    )
                } else {
                    String::new()
                };
                format!("{said}{extra}\n\nReply with yes or no only.")
            }
            "choice" => {
                let options = q["criteria"]
                    .as_object()
                    .context("choice criteria must be an object")?;
                ensure!(
                    !options.is_empty() && options.len() <= 255,
                    "choice requires 1 to 255 options"
                );
                labels = options.keys().cloned().collect();
                let mut lines = Vec::new();
                if multiple {
                    forms = yesno()?;
                    for label in &labels {
                        let desc = options[label]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .map(str::to_owned)
                            .unwrap_or_else(|| label.clone());
                        lines.push(format!("{}: {desc}", serde_json::to_string(label)?));
                    }
                    format!(
                        "{said}\n\nOptions:\n{}\n\nReply with yes or no for each option.",
                        lines.join("\n")
                    )
                } else {
                    for ((code, tid), label) in self.aliases(&labels)?.into_iter().zip(&labels) {
                        let mut group = vec![tid];
                        for id in self.ids(&[format!(" {code}")])? {
                            if !group.contains(&id) {
                                group.push(id);
                            }
                        }
                        forms.push(group);
                        let desc = options[label]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .map(str::to_owned)
                            .unwrap_or_else(|| {
                                if options[label].is_null() {
                                    label.replace('_', " ")
                                } else {
                                    as_text(&options[label])
                                }
                            });
                        lines.push(format!("{code} {desc}"));
                    }
                    format!(
                        "{said}\n\nOptions:\n{}\n\nReply with the option code only.",
                        lines.join("\n")
                    )
                }
            }
            "score" => {
                let levels = q["criteria"]
                    .as_array()
                    .context("score criteria must be an array")?;
                ensure!(
                    (2..=10).contains(&levels.len()),
                    "d1 score requires 2 to 10 levels"
                );
                let mut lines = Vec::new();
                for (i, value) in levels.iter().enumerate() {
                    labels.push(i.to_string());
                    forms.push(self.ids(&[i.to_string()])?);
                    lines.push(format!("{i} {}", as_text(value)));
                }
                format!(
                    "{said}\n\n{}\n\nReply with a single digit 0-{} only.",
                    lines.join("\n"),
                    levels.len() - 1
                )
            }
            _ => {
                let c = construction.context(format!("unsupported d1 question type {kind}"))?;
                readout = c.readout;
                forms = c.forms;
                slot_forms = c.slot_forms;
                markers = c.markers;
                after = c.after;
                c.body
            }
        };
        ensure!(
            forms.iter().all(|g| !g.is_empty()),
            "tokenizer lacks single-token answer forms"
        );
        let mut suffix = format!(
            "{}<|im_end|>\n<|im_start|>assistant\n<think></think>",
            self.plain(&body)
        );
        if let Some(audio) = &audio {
            suffix = suffix.replace('\u{e000}', &self.audio_markup(audio.dim(0)?));
        }
        if multiple {
            markers = labels
                .iter()
                .map(|s| serde_json::to_string(s).unwrap())
                .collect();
        }
        let mut ends = Vec::new();
        for (i, name) in markers.iter().enumerate() {
            suffix.push_str(&format!(
                "{}{}:",
                if i > 0 { "\n" } else { "" },
                self.plain(name)
            ));
            ends.push(suffix.len());
        }
        if after {
            let candidate = format!("{suffix}\n");
            let enc = self
                .offset_tokenizer
                .encode(candidate.as_str(), false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let last = enc.get_ids().last().context("empty marker encoding")?;
            let decoded = self
                .offset_tokenizer
                .decode(&[*last], false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            if decoded.trim_end().ends_with(':') {
                suffix = candidate;
            }
        }
        let encoded = self
            .offset_tokenizer
            .encode(suffix.as_str(), false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let slots = if markers.is_empty() {
            vec![encoded.len() - 1]
        } else {
            ends.iter()
                .map(|end| {
                    let matching: Vec<_> = encoded
                        .get_offsets()
                        .iter()
                        .enumerate()
                        .filter(|(_, (a, b))| *a < *end && *end <= *b)
                        .collect();
                    ensure!(
                        matching.len() == 1,
                        "answer marker must have an exact colon position"
                    );
                    let (i, &(_, stop)) = matching[0];
                    ensure!(
                        suffix[*end..stop].trim().is_empty(),
                        "marker overlaps next text"
                    );
                    Ok(i)
                })
                .collect::<anyhow::Result<Vec<_>>>()?
        };
        Ok(Question {
            name: name.into(),
            kind,
            labels,
            criteria: q["criteria"].clone(),
            suffix,
            slots,
            forms,
            multiple,
            readout,
            slot_forms,
            audio,
        })
    }
    fn encode_audio(&self, input: &crate::schema::MediaInput) -> anyhow::Result<Tensor> {
        let model = self
            .audio
            .as_ref()
            .context("audio is not supported by this model")?;
        let waveform = crate::media::audio::decode(input, self.max_length)?;
        let mel = crate::media::audio::log_mel(&waveform, &self.device)?.to_dtype(self.dtype)?;
        Ok(model.forward(&mel)?)
    }
    fn audio_markup(&self, count: usize) -> String {
        format!(
            "<|vision_start|>{}<|vision_end|>",
            "<|audio|>".repeat(count)
        )
    }
    // Qwen and LFM3 share the channel/temporal/merge-block patch layout.
    // Reuse its bounded FFmpeg decoding with the LFM3 sizing policy.
    fn media_config(&self) -> VisionConfig {
        let c = &self.config.vision_config;
        VisionConfig {
            dtype: None,
            depth: c.num_hidden_layers,
            hidden_size: c.hidden_size,
            intermediate_size: c.intermediate_size,
            num_heads: c.num_attention_heads,
            in_channels: c.in_channels,
            patch_size: c.patch_size,
            temporal_patch_size: c.temporal_patch_size,
            spatial_merge_size: c.spatial_merge_size,
            out_hidden_size: self.config.text_config.hidden_size,
            num_position_embeddings: 0,
        }
    }
    fn predict(&self, r: DecisionRequest) -> anyhow::Result<DecisionResponse> {
        ensure!(!r.questions.is_empty(), "questions must not be empty");
        ensure!(
            !r.questions
                .values()
                .any(|q| matches!(q["type"].as_str(), Some("box" | "boxes")))
                || r.images.len() == 1,
            "box and boxes questions require exactly one image"
        );
        let qs = r
            .questions
            .iter()
            .map(|(n, q)| self.question(n, q))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut prefix = vec![self.config.bos_token_id];
        prefix.extend(self.tokenizer.encode("<|im_start|>user\n")?);
        let media_config = self.media_config();
        let mut features = Vec::new();
        if !r.images.is_empty() {
            let data = images::decode_images_with_cap(
                &r.images,
                &media_config,
                self.max_length,
                self.config.d1_vision_max_pixels,
            )?;
            let n = data.patches.len() / (3 * 2 * self.config.vision_config.patch_size.pow(2));
            let patches = Tensor::from_vec(
                data.patches,
                (n, 3 * 2 * self.config.vision_config.patch_size.pow(2)),
                &self.device,
            )?
            .to_dtype(self.dtype)?;
            let encoded = self.vision.forward(&patches, &data.grids)?;
            let mut offset = 0;
            for &[t, h, w] in &data.grids {
                let count = t * h * w / self.config.vision_config.spatial_merge_size.pow(2);
                prefix.push(self.config.vision_start_token_id);
                prefix.extend(std::iter::repeat_n(self.config.image_token_id, count));
                prefix.push(self.config.vision_end_token_id);
                features.push(encoded.narrow(0, offset, count)?);
                offset += count;
            }
        }
        if !r.videos.is_empty() {
            let data = video::decode_videos_with_policy(
                &r.videos,
                &media_config,
                self.max_length,
                true,
                if self.audio.is_some() { 384 } else { 768 },
            )?;
            let n = data.patches.len() / (3 * 2 * self.config.vision_config.patch_size.pow(2));
            let patches = Tensor::from_vec(
                data.patches,
                (n, 3 * 2 * self.config.vision_config.patch_size.pow(2)),
                &self.device,
            )?
            .to_dtype(self.dtype)?;
            let encoded = self.vision.forward(&patches, &data.grids)?;
            let mut offset = 0;
            for (&[t, h, w], timestamps) in data.grids.iter().zip(&data.timestamps) {
                // The chat template wraps the entire timestamped clip as well
                // as each temporal patch group emitted by the processor.
                prefix.push(self.config.vision_start_token_id);
                let count = h * w / self.config.vision_config.spatial_merge_size.pow(2);
                for &timestamp in timestamps.iter().take(t) {
                    prefix.extend(
                        self.tokenizer
                            .encode(&format!("<{timestamp:.1} seconds>"))?,
                    );
                    prefix.push(self.config.vision_start_token_id);
                    prefix.extend(std::iter::repeat_n(self.config.video_token_id, count));
                    prefix.push(self.config.vision_end_token_id);
                    features.push(encoded.narrow(0, offset, count)?);
                    offset += count;
                }
                prefix.push(self.config.vision_end_token_id);
            }
        }
        for input in &r.audio {
            let encoded = self.encode_audio(input)?;
            let count = encoded.dim(0)?;
            prefix.push(self.config.vision_start_token_id);
            prefix.extend(std::iter::repeat_n(
                self.config.audio_token_id.context("missing audio token")?,
                count,
            ));
            prefix.push(self.config.vision_end_token_id);
            features.push(encoded);
        }
        let mut prefix_text = self
            .offset_tokenizer
            .decode(&prefix, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        if !r.state.is_null() {
            prefix_text.push_str(&format!(
                "{}\nQUESTION:\n",
                self.plain(&format!("{}\n\n", as_text(&r.state)))
            ));
        }
        let mut answers = Map::new();
        let mut tokens = 0;
        for chunk in qs.chunks(self.config.d1_set_size) {
            let mut ids = Vec::new();
            let mut groups = Vec::new();
            let mut slots = Vec::new();
            let mut chunk_features = features.clone();
            let first_text = format!("{prefix_text}{}", chunk[0].suffix);
            let state_positions = if chunk
                .iter()
                .any(|q| matches!(q.readout, typed::Readout::Read { .. }))
            {
                Some(pointer::StatePositions::new(
                    self,
                    &r.state,
                    &prefix_text,
                    &first_text,
                )?)
            } else {
                None
            };
            for (i, q) in chunk.iter().enumerate() {
                let own = if i == 0 {
                    self.tokenizer.encode(&first_text)?
                } else {
                    self.tokenizer.encode(&format!(
                        "{}{}",
                        if r.state.is_null() { "" } else { "QUESTION:\n" },
                        q.suffix
                    ))?
                };
                let suffix_len = self.tokenizer.encode(&q.suffix)?.len();
                ensure!(
                    own.len() >= suffix_len,
                    "question token boundaries disagree"
                );
                let start = ids.len() + own.len() - suffix_len;
                slots.push(q.slots.iter().map(|j| start + j).collect::<Vec<_>>());
                if i == 0 {
                    let state = own.len() - suffix_len;
                    groups.extend(std::iter::repeat_n(0, state));
                    groups.extend(std::iter::repeat_n(1, suffix_len));
                } else {
                    groups.extend(std::iter::repeat_n(i + 1, own.len()));
                }
                ids.extend(own);
                if let Some(audio) = &q.audio {
                    chunk_features.push(audio.clone());
                }
            }
            ensure!(
                ids.len() <= self.max_length,
                "input exceeds --max-model-len {}",
                self.max_length
            );
            tokens += ids.len();
            let embeddings = self
                .text
                .embedding
                .forward(&Tensor::new(ids.as_slice(), &self.device)?)?;
            let mut pieces = Vec::new();
            let mut at = 0;
            let mut read = 0;
            let features = if chunk_features.is_empty() {
                None
            } else {
                Some(Tensor::cat(&chunk_features, 0)?)
            };
            while at < ids.len() {
                let is_media = |id: u32| {
                    id == self.config.image_token_id
                        || id == self.config.video_token_id
                        || self.config.audio_token_id == Some(id)
                };
                let start = at;
                let media = is_media(ids[at]);
                while at < ids.len() && is_media(ids[at]) == media {
                    at += 1;
                }
                if media {
                    let features = features
                        .as_ref()
                        .context("media tokens have no encoded features")?;
                    pieces.push(features.narrow(0, read, at - start)?);
                    read += at - start;
                } else {
                    pieces.push(embeddings.narrow(0, start, at - start)?);
                }
            }
            ensure!(
                features.as_ref().map_or(0, |f| f.dims()[0]) == read,
                "media feature and token counts differ"
            );
            let hidden = self.text.forward(Tensor::cat(&pieces, 0)?, &groups)?;
            for (q, positions) in chunk.iter().zip(slots) {
                if let typed::Readout::Read { unit } = &q.readout {
                    let answer = state_positions
                        .as_ref()
                        .context("missing state-position map")?
                        .decode(
                            &hidden,
                            &positions,
                            &q.kind,
                            unit.as_deref(),
                            &self.calibration,
                        )?;
                    answers.insert(q.name.clone(), answer);
                    continue;
                }
                let mut distributions = Vec::new();
                for (slot, position) in positions.into_iter().enumerate() {
                    let forms = q.slot_forms.get(slot).unwrap_or(&q.forms);
                    let all: Vec<u32> = forms.iter().flatten().copied().collect();
                    let weights = self
                        .text
                        .embedding
                        .embeddings()
                        .index_select(&Tensor::new(all.as_slice(), &self.device)?, 0)?;
                    let scores = weights
                        .matmul(&hidden.get(position)?.unsqueeze(1)?)?
                        .flatten_all()?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?;
                    let mut best = Vec::new();
                    let mut offset = 0;
                    for group in forms {
                        best.push(
                            scores[offset..offset + group.len()]
                                .iter()
                                .copied()
                                .fold(f32::NEG_INFINITY, f32::max)
                                as f64,
                        );
                        offset += group.len();
                    }
                    let temperature = if q.multiple || !matches!(q.readout, typed::Readout::Legacy)
                    {
                        1.
                    } else {
                        self.calibration.temperature(&q.kind, q.forms.len())
                    };
                    ensure!(
                        best.iter().all(|v| v.is_finite()),
                        "d1 produced non-finite answer scores"
                    );
                    distributions.push(probabilities(&best, temperature));
                }
                let answer = typed::answer(
                    &q.readout,
                    &distributions,
                    &self.calibration,
                    self.config.d1_box_edges == "interp",
                )?
                .unwrap_or_else(|| self.answer(q, &distributions));
                answers.insert(q.name.clone(), answer);
            }
        }
        Ok(DecisionResponse {
            model: String::new(),
            answers,
            usage: Usage {
                input_tokens: tokens,
                output_tokens: 0,
            },
        })
    }
    fn answer(&self, q: &Question, p: &[Vec<f64>]) -> Value {
        if q.kind == "noul" {
            return json!({"type":"noul","noul":p[0][0]});
        }
        if q.multiple {
            let ps: Vec<_> = p.iter().map(|p| self.calibration.multiple(p[0])).collect();
            let chosen: Vec<_> = q
                .labels
                .iter()
                .zip(&ps)
                .filter(|(_, p)| **p >= 0.5)
                .map(|(n, _)| n.clone())
                .collect();
            let probabilities: Map<_, _> = q
                .labels
                .iter()
                .cloned()
                .zip(ps.into_iter().map(|p| json!(p)))
                .collect();
            return json!({"type":"choice","choices":chosen,"probabilities":probabilities});
        }
        let ps = &p[0];
        let best = ps
            .iter()
            .enumerate()
            .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
            .unwrap()
            .0;
        let probs: Map<_, _> = q
            .labels
            .iter()
            .cloned()
            .zip(ps.iter().map(|p| json!(p)))
            .collect();
        if q.kind == "choice" {
            json!({"type":"choice","choice":q.labels[best],"confidence":ps[best],"probabilities":probs})
        } else {
            let mut total = 0.;
            let median = ps
                .iter()
                .position(|p| {
                    total += p;
                    total >= 0.5
                })
                .unwrap_or(ps.len() - 1);
            let legend: Map<_, _> = q
                .labels
                .iter()
                .cloned()
                .zip(q.criteria.as_array().unwrap().iter().cloned())
                .collect();
            json!({"type":"score","score":median as f64,"confidence":ps[best],"probabilities":probs,"legend":legend})
        }
    }
}
fn as_text(v: &Value) -> String {
    v.as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| serde_json::to_string_pretty(v).unwrap())
}
fn python_text(v: &Value) -> String {
    if v.is_null() {
        "None".into()
    } else {
        as_text(v)
    }
}
fn probabilities(scores: &[f64], temperature: f64) -> Vec<f64> {
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<_> = scores
        .iter()
        .map(|x| ((x - max) / temperature).exp())
        .collect();
    let total: f64 = e.iter().sum();
    e.into_iter().map(|x| x / total).collect()
}
fn validate_config(c: &Config) -> anyhow::Result<()> {
    ensure!(
        matches!(c.model_type.as_str(), "lfm3_vl" | "lfm3_omni"),
        "unsupported d1 model type"
    );
    ensure!(
        (c.model_type == "lfm3_omni") == c.audio_encoder_config.is_some(),
        "audio encoder configuration does not match model type"
    );
    if let Some(a) = &c.audio_encoder_config {
        validate_audio_config(a)?;
        ensure!(c.audio_token_id.is_some(), "missing audio token");
    }
    ensure!(
        c.d1_prompt == 1 && matches!(c.d1_box_edges.as_str(), "median" | "interp"),
        "unsupported d1 typed prompt revision"
    );
    ensure!(c.tie_word_embeddings, "d1 requires tied output embeddings");
    ensure!(c.d1_vision_max_pixels > 0, "invalid d1 image pixel cap");
    let t = &c.text_config;
    let v = &c.vision_config;
    ensure!(
        c.d1_attention == "set" && c.d1_codes == 2 && (2..=128).contains(&c.d1_set_size),
        "unsupported d1 prompt/attention configuration"
    );
    ensure!(
        c.projector_hidden_act == "gelu",
        "unsupported projector activation"
    );
    ensure!(
        t.num_hidden_layers == t.layer_types.len() && v.num_hidden_layers == v.layer_types.len(),
        "layer count mismatch"
    );
    ensure!(
        t.hidden_size > 0
            && t.num_attention_heads > 0
            && t.num_key_value_heads > 0
            && t.head_dim > 0
            && t.head_dim.is_multiple_of(2)
            && t.num_attention_heads.is_multiple_of(t.num_key_value_heads)
            && t.conv_l_cache > 0
            && t.sliding_window > 0,
        "invalid text dimensions"
    );
    ensure!(
        v.num_attention_heads > 0
            && v.num_key_value_heads > 0
            && v.hidden_size.is_multiple_of(v.num_attention_heads * 4)
            && v.num_attention_heads.is_multiple_of(v.num_key_value_heads)
            && v.conv_kernel_size % 2 == 1
            && v.patch_size > 0
            && v.spatial_merge_size > 0
            && v.in_channels == 3
            && v.temporal_patch_size == 2,
        "unsupported vision dimensions"
    );
    ensure!(
        t.layer_types
            .iter()
            .all(|s| matches!(s.as_str(), "conv" | "full_attention" | "sliding_attention"))
            && v.layer_types
                .iter()
                .all(|s| matches!(s.as_str(), "conv" | "full_attention")),
        "unsupported layer type"
    );
    for name in ["full_attention", "sliding_attention"] {
        let rope = t
            .rope_parameters
            .get(name)
            .context("missing text RoPE configuration")?;
        ensure!(
            rope.rope_type == "default"
                && rope.rope_theta > 0.
                && (0.0..=1.0).contains(&rope.partial_rotary_factor),
            "unsupported text RoPE"
        );
    }
    ensure!(
        v.rope_parameters.rope_type == "axial" && v.rope_parameters.rope_theta > 0.,
        "unsupported vision RoPE"
    );
    Ok(())
}
fn validate_audio_config(c: &lfm3::AudioConfig) -> anyhow::Result<()> {
    let t = &c.text_config;
    ensure!(
        c.num_mel_bins == 128
            && c.conv_channels > 0
            && c.hidden_size == t.hidden_size
            && c.sliding_window > 0,
        "unsupported audio encoder dimensions"
    );
    ensure!(
        t.hidden_size > 0
            && t.num_attention_heads > 0
            && t.num_key_value_heads > 0
            && t.hidden_size.is_multiple_of(t.num_attention_heads * 2)
            && t.num_attention_heads.is_multiple_of(t.num_key_value_heads)
            && t.conv_l_cache > 0
            && t.block_multiple_of > 0
            && t.layer_types.len() == t.num_hidden_layers
            && t.layer_types
                .iter()
                .all(|s| matches!(s.as_str(), "conv" | "full_attention")),
        "unsupported audio backbone"
    );
    ensure!(
        t.rope_parameters.rope_type == "default" && t.rope_parameters.rope_theta > 0.,
        "unsupported audio RoPE"
    );
    Ok(())
}
fn validate_processor(path: &Path, c: &Config) -> anyhow::Result<()> {
    let p: Value = serde_json::from_slice(&fs::read(path.join("processor_config.json"))?)?;
    for (section, name) in [
        ("image_processor", "Lfm3VlImageProcessor"),
        ("video_processor", "Lfm3VlVideoProcessor"),
    ] {
        let v = &p[section];
        let key = if section == "image_processor" {
            "image_processor_type"
        } else {
            "video_processor_type"
        };
        ensure!(
            v[key] == name
                && v["patch_size"] == c.vision_config.patch_size
                && v["merge_size"] == c.vision_config.spatial_merge_size
                && v["temporal_patch_size"] == 2
                && v["image_mean"] == json!([0.5, 0.5, 0.5])
                && v["image_std"] == json!([0.5, 0.5, 0.5]),
            "unsupported d1 {section}"
        );
    }
    for section in ["image_processor", "video_processor"] {
        let v = &p[section];
        for flag in ["do_convert_rgb", "do_normalize", "do_rescale", "do_resize"] {
            ensure!(v[flag] == true, "unsupported d1 {section}.{flag}");
        }
        ensure!(
            v["resample"] == 3 && v["rescale_factor"] == json!(1.0 / 255.0),
            "unsupported d1 resampling"
        );
    }
    ensure!(
        p["image_processor"]["size"]
            == json!({"shortest_edge":65536,"longest_edge":if c.audio_encoder_config.is_some(){2097152}else{16777216}}),
        "unsupported d1 image size policy"
    );
    let v = &p["video_processor"];
    ensure!(
        v["size"] == json!({"shortest_edge":4096,"longest_edge":25165824})
            && v["fps"] == 2.0
            && v["min_frames"] == 4
            && v["max_frames"]
                == if c.audio_encoder_config.is_some() {
                    384
                } else {
                    768
                }
            && v["do_sample_frames"] == true
            && v["cap_pixels_per_frame"] == false,
        "unsupported d1 video sampling policy"
    );
    if c.audio_encoder_config.is_some() {
        ensure!(
            p["audio_markers"] == true && p["vision_markers"] == true,
            "unsupported Omni media markers"
        );
        let f = &p["feature_extractor"];
        let expected = json!({"feature_extractor_type":"Lfm3OmniFeatureExtractor","sampling_rate":16000,"feature_size":128,"n_fft":400,"hop_length":160,"win_length":400,"log_floor":-12.0,"normalization_offset":4.0,"normalization_divisor":4.0,"encoder_stride":8,"padding_value":0.0,"padding_side":"right"});
        for (key, value) in expected.as_object().unwrap() {
            ensure!(&f[key] == value, "unsupported audio feature setting {key}");
        }
    }
    Ok(())
}
impl DecisionModel for D1 {
    fn supports_d1_typed(&self) -> bool {
        true
    }

    fn supports_audio(&self) -> bool {
        self.audio.is_some()
    }

    fn supports_images(&self) -> bool {
        true
    }
    fn supports_videos(&self) -> bool {
        true
    }
    fn predict_batch(
        &self,
        requests: Vec<DecisionRequest>,
    ) -> Vec<Result<DecisionResponse, ApiError>> {
        requests
            .into_iter()
            .map(|r| self.predict(r).map_err(|e| ApiError::new(e.to_string())))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(actual: &Value, expected: &Value, path: &str) {
        match (actual, expected) {
            (Value::Number(a), Value::Number(b)) => assert!(
                (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 0.0002,
                "{path}: {a} != {b}"
            ),
            (Value::Array(a), Value::Array(b)) => {
                assert_eq!(a.len(), b.len(), "{path}");
                for (i, (x, y)) in a.iter().zip(b).enumerate() {
                    close(x, y, &format!("{path}[{i}]"));
                }
            }
            (Value::Object(a), Value::Object(b)) => {
                assert_eq!(a.len(), b.len(), "{path}");
                for (k, v) in b {
                    close(&a[k], v, &format!("{path}.{k}"));
                }
            }
            _ => assert_eq!(actual, expected, "{path}"),
        }
    }
    #[test]
    #[ignore = "requires SYS1_D1_PATH checkpoint (~4GB), FFmpeg and FFprobe"]
    fn d1_reference_parity() -> anyhow::Result<()> {
        let path = std::env::var("SYS1_D1_PATH").context("set SYS1_D1_PATH")?;
        reference_parity(&path, include_str!("fixtures/d1_reference.json"))
    }
    #[test]
    #[ignore = "requires SYS1_D1_OMNI_PATH checkpoint (~4GB), FFmpeg and FFprobe"]
    fn d1_omni_reference_parity() -> anyhow::Result<()> {
        let path = std::env::var("SYS1_D1_OMNI_PATH").context("set SYS1_D1_OMNI_PATH")?;
        reference_parity(&path, include_str!("fixtures/d1_omni_reference.json"))
    }
    fn reference_parity(path: &str, fixtures: &str) -> anyhow::Result<()> {
        let model = D1::load(
            Path::new(&path),
            Some(DType::F32),
            Some(4096),
            AttentionImplementation::Eager,
        )?;
        let fixtures: Vec<Value> = serde_json::from_str(fixtures)?;
        for fixture in fixtures {
            let name = fixture["name"].as_str().unwrap();
            if std::env::var("SYS1_REFERENCE_CASE").is_ok_and(|filter| !name.contains(&filter)) {
                continue;
            }
            let mut request: DecisionRequest = serde_json::from_value(fixture["request"].clone())?;
            if name == "typed_boxes" {
                request.images.push(crate::schema::MediaInput::Bytes(
                    include_bytes!("fixtures/d1_cats.png").to_vec(),
                ));
            }
            if name == "image" || name == "mixed" {
                let mut ppm = b"P6\n256 256\n255\n".to_vec();
                for _ in 0..256 * 256 {
                    ppm.extend_from_slice(&[255, 0, 0]);
                }
                request.images.push(crate::schema::MediaInput::Bytes(ppm));
            }
            if name == "video" {
                let mut frames = Vec::new();
                for rgb in [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]] {
                    for _ in 0..64 * 64 {
                        frames.extend_from_slice(&rgb);
                    }
                }
                let args = [
                    "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x64", "-r", "2",
                    "-i", "pipe:0", "-c:v", "ffv1", "-f", "matroska", "pipe:1",
                ]
                .map(str::to_owned);
                let video = crate::media::run_media_tool("ffmpeg", &args, &frames)?;
                request.videos.push(crate::schema::MediaInput::Bytes(video));
            }
            if let Some(clips) = fixture["audio_fixtures"].as_array() {
                for descriptor in clips {
                    let rate = descriptor[0].as_u64().unwrap() as usize;
                    let samples = descriptor[1].as_u64().unwrap() as usize;
                    let bytes = crate::media::audio::fixture_wav(rate, samples);
                    if name == "spoken" {
                        request.questions["answer"]["instructions"] =
                            json!({"type":"audio","bytes":bytes});
                    } else {
                        request.audio.push(crate::schema::MediaInput::Bytes(bytes));
                    }
                }
            }
            let response = model.predict(request)?;
            println!("{name}: {}", serde_json::to_string(&response)?);
            assert_eq!(
                response.usage.input_tokens,
                fixture["expected"]["usage"]["input_tokens"]
                    .as_u64()
                    .unwrap() as usize,
                "{name}"
            );
            close(
                &json!(response.answers),
                &fixture["expected"]["answers"],
                name,
            );
        }
        for images in [
            vec![],
            vec![
                crate::schema::MediaInput::Bytes(vec![]),
                crate::schema::MediaInput::Bytes(vec![]),
            ],
        ] {
            let mut request: DecisionRequest = serde_json::from_value(
                json!({"state":null,"questions":{"q":{"type":"box","instructions":"Where?"}}}),
            )?;
            request.images = images;
            assert!(
                model
                    .predict(request)
                    .unwrap_err()
                    .to_string()
                    .contains("exactly one image")
            );
        }
        Ok(())
    }
    #[test]
    fn calibrated_softmax_is_stable() {
        let p = probabilities(&[10000., 10002.], 2.);
        assert!((p[1] - 0.73105857863).abs() < 1e-9);
    }
}
