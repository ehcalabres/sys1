//! Parameter-free state-position readout, matching upstream pointer.py/json_spans.py.
use super::D1;
use anyhow::{Context, ensure};
use candle_core::{DType, Tensor};
use serde_json::{Value, json};

pub(super) struct StatePositions {
    text: String,
    offsets: Vec<Option<(usize, usize)>>,
    paths: Vec<(usize, usize, String)>,
}
impl StatePositions {
    pub fn new(model: &D1, state: &Value, prefix: &str, full: &str) -> anyhow::Result<Self> {
        ensure!(
            full.starts_with(prefix),
            "pointer prompt must start with state prefix"
        );
        let text = if state.is_null() {
            String::new()
        } else {
            super::as_text(state)
        };
        let (escaped, boundaries) = model.plain_offsets(&format!("{text}\n\n"));
        let tail = format!("{escaped}\nQUESTION:\n");
        let start = if state.is_null() {
            prefix.chars().count()
        } else {
            ensure!(prefix.ends_with(&tail), "pointer state differs from prefix");
            prefix.chars().count() - tail.chars().count()
        };
        let stop = start
            + boundaries
                .partition_point(|b| *b <= text.chars().count())
                .saturating_sub(1);
        let encoded = model
            .offset_tokenizer
            .encode(full, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        ensure!(
            encoded.get_ids() == model.tokenizer.encode(full)?.as_slice(),
            "pointer token offsets differ from native tokenizer IDs"
        );
        let byte_to_char = |at: usize| full[..at].chars().count();
        let chars: Vec<_> = full.chars().collect();
        let mut offsets = Vec::new();
        for &(a, b) in encoded.get_offsets() {
            let a = byte_to_char(a);
            let b = byte_to_char(b);
            let lo = a.max(start);
            let hi = b.min(stop);
            let span = if lo < hi
                && chars[a..lo].iter().all(|c| c.is_whitespace())
                && chars[hi..b].iter().all(|c| c.is_whitespace())
            {
                let left = boundaries[lo - start];
                let right = boundaries[hi - start];
                if left < right
                    && right <= text.chars().count()
                    && text
                        .chars()
                        .skip(left)
                        .take(right - left)
                        .any(|c| !c.is_whitespace())
                {
                    Some((left, right))
                } else {
                    None
                }
            } else {
                None
            };
            offsets.push(span);
        }
        ensure!(
            !offsets.is_empty() && offsets[0].is_none(),
            "pointer BOS must be a template token"
        );
        let paths = if state.is_string() || state.is_null() {
            vec![]
        } else {
            string_paths(&text)?
        };
        Ok(Self {
            text,
            offsets,
            paths,
        })
    }
    pub fn decode(
        &self,
        hidden: &Tensor,
        slots: &[usize],
        kind: &str,
        unit: Option<&str>,
        cal: &super::Calibration,
    ) -> anyhow::Result<Value> {
        ensure!(slots.len() == 2, "extraction requires start/end slots");
        let width = self.offsets.iter().rposition(Option::is_some).unwrap_or(0) + 1;
        let query = Tensor::stack(&[hidden.get(slots[0])?, hidden.get(slots[1])?], 0)?
            .to_dtype(DType::F32)?;
        let keys = hidden
            .narrow(0, 0, width)?
            .to_dtype(DType::F32)?
            .t()?
            .contiguous()?;
        let scores = (query.matmul(&keys)? / (hidden.dim(1)? as f64).sqrt())?.to_vec2::<f32>()?;
        let mut logp = Vec::new();
        for row in scores {
            let row: Vec<f64> = row
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    if i == 0 || self.offsets[i].is_some() {
                        *x as f64
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect();
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z = row.iter().map(|x| (x - max).exp()).sum::<f64>().ln() + max;
            logp.push(row.iter().map(|x| x - z).collect::<Vec<_>>());
        }
        let raw_found = -logp[0][0].exp_m1();
        let mut picked = None;
        let cap = if kind == "text" { 512 } else { 32 };
        if raw_found >= 0.5 && width > 1 {
            let mut best = f64::NEG_INFINITY;
            for first in 1..width {
                for last in first..(first + cap).min(width) {
                    let score = logp[0][first] + logp[1][last];
                    if score > best {
                        best = score;
                        picked = Some((first, last));
                    }
                }
            }
        }
        let found = cal.through(&format!("found:{kind}"), raw_found);
        let mut text = None;
        let mut span = None;
        let mut path = None;
        if found >= 0.5
            && let Some((first, last)) = picked
        {
            let (mut a, _) = self.offsets[first].context("pointer start outside state")?;
            let (_, mut b) = self.offsets[last].context("pointer end outside state")?;
            let chars: Vec<_> = self.text.chars().collect();
            while a < b && chars[a].is_whitespace() {
                a += 1;
            }
            while b > a && chars[b - 1].is_whitespace() {
                b -= 1;
            }
            let held: Vec<_> = self
                .paths
                .iter()
                .filter(|(start, end, _)| *start < b && a < *end)
                .collect();
            if held.len() == 1 {
                let (start, end, p) = held[0];
                a = a.max(*start);
                b = b.min(*end);
                path = Some(p.clone());
                let slice: String = chars[a..b].iter().collect();
                text =
                    Some(serde_json::from_str::<String>(&format!("\"{slice}\"")).unwrap_or(slice));
            } else {
                text = Some(chars[a..b].iter().collect::<String>());
            }
            span = Some([a, b]);
        }
        let value = match text.as_deref() {
            None => Value::Null,
            Some(t) => match kind {
                "number" => super::parsing::parse_number(t, &self.text, unit),
                "date" => {
                    super::parsing::parse_date(t, &self.text).map_or(Value::Null, Value::String)
                }
                _ => json!(t),
            },
        };
        let mut out = json!({"type":kind,kind:value,"span":span,"found":found});
        if kind != "text" {
            out["text"] = json!(text);
        }
        if found >= 0.5
            && let Some(path) = path
        {
            out["path"] = json!(path);
        }
        if kind == "number" {
            out["unit"] = json!(
                unit.filter(|unit| !unit.is_empty())
                    .map(str::to_owned)
                    .or_else(|| text.as_deref().and_then(super::parsing::written_unit))
            );
        }
        Ok(out)
    }
}
fn string_paths(text: &str) -> anyhow::Result<Vec<(usize, usize, String)>> {
    fn skip(bytes: &[u8], at: &mut usize) {
        while bytes.get(*at).is_some_and(u8::is_ascii_whitespace) {
            *at += 1;
        }
    }
    fn string_end(bytes: &[u8], at: usize) -> usize {
        let mut i = at + 1;
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => i += 2,
                b'"' => return i + 1,
                _ => i += 1,
            }
        }
        i
    }
    fn walk(
        text: &str,
        at: &mut usize,
        path: String,
        out: &mut Vec<(usize, usize, String)>,
    ) -> anyhow::Result<()> {
        let bytes = text.as_bytes();
        skip(bytes, at);
        match bytes[*at] {
            b'{' => {
                *at += 1;
                skip(bytes, at);
                while bytes[*at] != b'}' {
                    let end = string_end(bytes, *at);
                    let key: String = serde_json::from_str(&text[*at..end])?;
                    *at = end;
                    skip(bytes, at);
                    *at += 1;
                    walk(
                        text,
                        at,
                        format!("{path}/{}", key.replace('~', "~0").replace('/', "~1")),
                        out,
                    )?;
                    skip(bytes, at);
                    if bytes[*at] == b',' {
                        *at += 1;
                        skip(bytes, at);
                    }
                }
                *at += 1;
            }
            b'[' => {
                *at += 1;
                skip(bytes, at);
                let mut i = 0;
                while bytes[*at] != b']' {
                    walk(text, at, format!("{path}/{i}"), out)?;
                    i += 1;
                    skip(bytes, at);
                    if bytes[*at] == b',' {
                        *at += 1;
                        skip(bytes, at);
                    }
                }
                *at += 1;
            }
            b'"' => {
                let end = string_end(bytes, *at);
                out.push((
                    text[..*at + 1].chars().count(),
                    text[..end - 1].chars().count(),
                    path,
                ));
                *at = end;
            }
            _ => {
                while bytes.get(*at).is_some_and(|b| {
                    !b.is_ascii_whitespace() && *b != b',' && *b != b']' && *b != b'}'
                }) {
                    *at += 1;
                }
            }
        }
        Ok(())
    }
    let mut out = vec![];
    walk(text, &mut 0, String::new(), &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pointer_reads_trimmed_unicode_spans_and_preserves_missing() -> anyhow::Result<()> {
        let state = StatePositions {
            text: " Zoë ".into(),
            offsets: vec![None, Some((0, 3)), Some((3, 5)), None, None],
            paths: vec![],
        };
        let hidden = Tensor::from_slice(
            &[-1f32, -1., 1., 0., 0., 1., 10., 0., 0., 10.],
            (5, 2),
            &candle_core::Device::Cpu,
        )?;
        let cal = super::super::Calibration {
            default: 1.,
            temperatures: Default::default(),
            tables: Default::default(),
        };
        let answer = state.decode(&hidden, &[3, 4], "text", None, &cal)?;
        assert_eq!(answer["text"], "Zoë");
        assert_eq!(answer["span"], json!([1, 4]));
        let empty = StatePositions {
            text: String::new(),
            offsets: vec![None; 5],
            paths: vec![],
        };
        let answer = empty.decode(&hidden, &[3, 4], "number", None, &cal)?;
        assert!(answer["number"].is_null());
        assert!(answer["span"].is_null());
        assert_eq!(answer["found"].as_f64(), Some(0.));
        Ok(())
    }
    #[test]
    fn empty_unit_uses_the_written_currency() -> anyhow::Result<()> {
        let state = StatePositions {
            text: "£12".into(),
            offsets: vec![None, Some((0, 3)), None],
            paths: vec![],
        };
        let hidden = Tensor::from_slice(
            &[-1f32, -1., 1., 1., 10., 10.],
            (3, 2),
            &candle_core::Device::Cpu,
        )?;
        let cal = super::super::Calibration {
            default: 1.,
            temperatures: Default::default(),
            tables: Default::default(),
        };
        let answer = state.decode(&hidden, &[2, 2], "number", Some(""), &cal)?;
        assert_eq!(answer["unit"], "£");
        assert_eq!(answer["number"], 12);
        Ok(())
    }
    #[test]
    fn json_paths_escape_keys_and_count_characters() -> anyhow::Result<()> {
        let text = "{\"a/b~c\":[\"Zoë\\nÅ\"]}";
        let paths = string_paths(text)?;
        assert_eq!(paths, vec![(11, 17, "/a~1b~0c/0".into())]);
        Ok(())
    }
}
