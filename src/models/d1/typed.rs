//! Prompt, grid and calibrated box readouts from the pinned d1 runtime.
use super::{Calibration, D1};
use anyhow::ensure;
use serde_json::{Value, json};
pub(super) type Forms = Vec<Vec<u32>>;
#[derive(Default)]
pub(super) enum Readout {
    #[default]
    Legacy,
    Read {
        unit: Option<String>,
    },
    Estimate {
        grid: Grid,
        unit: Option<String>,
    },
    Box {
        grid: Grid,
    },
    Boxes {
        grid: Grid,
    },
}
pub(super) struct Construction {
    pub body: String,
    pub markers: Vec<String>,
    pub slot_forms: Vec<Forms>,
    pub forms: Forms,
    pub readout: Readout,
    pub after: bool,
}
#[derive(Clone)]
pub(super) struct Grid {
    pub low: f64,
    pub high: f64,
    pub size: usize,
}
impl Grid {
    pub fn step(&self) -> f64 {
        (self.high - self.low) / (self.size - 1) as f64
    }
    pub fn values(&self) -> Vec<f64> {
        (0..self.size)
            .map(|i| {
                if i + 1 == self.size {
                    self.high
                } else {
                    self.low + i as f64 * self.step()
                }
            })
            .collect()
    }
    pub fn quantile(&self, p: &[f64], level: f64) -> f64 {
        let total: f64 = p.iter().sum();
        let mut mass = 0.;
        for (v, p) in self.values().into_iter().zip(p) {
            mass += p / total;
            if mass >= level {
                return v;
            }
        }
        self.high
    }
}
pub(super) fn g(v: f64) -> String {
    if v == 0. {
        return if v.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    let scientific = format!("{v:.5e}");
    let (mantissa, exp) = scientific.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let trim = |s: String| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            s
        }
    };
    if (-4..6).contains(&exp) {
        trim(format!("{:.*}", (5 - exp).max(0) as usize, v))
    } else {
        format!("{}e{:+03}", trim(mantissa.into()), exp)
    }
}
impl D1 {
    fn number_forms(&self, grid: &Grid, scale: f64) -> anyhow::Result<Option<Forms>> {
        if self.config.d1_numbers {
            let mut groups = Vec::new();
            let mut valid = true;
            for value in grid.values() {
                let value = value * scale;
                let n = value.round_ties_even();
                if (value - n).abs() > 1e-6_f64.max(1e-9 * value.abs()) {
                    valid = false;
                    break;
                }
                let raw = format!("{n:.0}");
                if self.tokenizer.encode(&raw)?.len() != 1 {
                    valid = false;
                    break;
                }
                let ids = self.ids(&[raw.clone(), format!(" {raw}")])?;
                if ids.is_empty() {
                    valid = false;
                    break;
                }
                groups.push(ids);
            }
            if valid {
                return Ok(Some(groups));
            }
        }
        Ok(None)
    }
    fn grid_forms(&self, grid: &Grid, scale: f64) -> anyhow::Result<(Forms, bool, Vec<String>)> {
        if let Some(groups) = self.number_forms(grid, scale)? {
            return Ok((groups, true, vec![]));
        }
        if grid.size <= 10 {
            return Ok((
                (0..grid.size)
                    .map(|i| self.ids(&[i.to_string()]))
                    .collect::<anyhow::Result<_>>()?,
                false,
                (0..grid.size).map(|i| i.to_string()).collect(),
            ));
        }
        let labels = (0..grid.size).map(|i| i.to_string()).collect::<Vec<_>>();
        let mut forms = vec![];
        let mut codes = vec![];
        for (code, id) in self.aliases(&labels)? {
            let mut group = vec![id];
            for id in self.ids(&[format!(" {code}")])? {
                if !group.contains(&id) {
                    group.push(id);
                }
            }
            forms.push(group);
            codes.push(code);
        }
        Ok((forms, false, codes))
    }
    pub(super) fn typed_question(
        &self,
        q: &Value,
        said: &str,
    ) -> anyhow::Result<Option<Construction>> {
        let kind = q["type"].as_str().unwrap_or("");
        if !matches!(kind, "text" | "number" | "date" | "box" | "boxes") {
            return Ok(None);
        }
        crate::schema::validate_d1_question(q).map_err(anyhow::Error::msg)?;
        let unit = q["unit"].as_str().map(str::to_owned);
        if matches!(kind, "text" | "date") || kind == "number" && q.get("range").is_none() {
            let unit_text = if kind == "number" {
                unit.as_ref()
                    .filter(|s| !s.is_empty())
                    .map(|s| format!(" (unit: {s})"))
                    .unwrap_or_default()
            } else {
                String::new()
            };
            return Ok(Some(Construction {
                body: format!(
                    "{said}\n\nRead the {kind}{unit_text} from the state, or nothing if it is absent."
                ),
                markers: vec!["start".into(), "end".into()],
                slot_forms: vec![],
                forms: vec![],
                readout: Readout::Read { unit },
                after: false,
            }));
        }
        if kind == "number" {
            let low = q["range"][0].as_f64().unwrap();
            let high = q["range"][1].as_f64().unwrap();
            let size = q["step"].as_f64().map_or(128, |step| {
                (((high - low) / step).round_ties_even() as usize)
                    .saturating_add(1)
                    .min(128)
            });
            let grid = Grid { low, high, size };
            let (forms, numbers, codes) = self.grid_forms(&grid, 1.)?;
            let unit_text = unit.as_ref().map(|u| format!(" {u}")).unwrap_or_default();
            let body = if numbers {
                format!(
                    "{said}\n\nEstimate from {} to {}{unit_text}, in steps of {}. Reply with the number only.",
                    g(low),
                    g(high),
                    g(grid.step())
                )
            } else {
                format!(
                    "{said}\n\nEstimate from {} to {}{unit_text} on {} equally spaced values, step {}: {} is {}, {} is {}. Reply with the code only.",
                    g(low),
                    g(high),
                    size,
                    g(grid.step()),
                    codes[0],
                    g(low),
                    codes.last().unwrap(),
                    g(high)
                )
            };
            return Ok(Some(Construction {
                body,
                markers: vec![],
                slot_forms: vec![],
                forms,
                readout: Readout::Estimate { grid, unit },
                after: false,
            }));
        }
        let numeric = Grid {
            low: 0.,
            high: 0.999,
            size: 1000,
        };
        let numeric_forms = self.number_forms(&numeric, 1000.)?;
        let numbers = numeric_forms.is_some();
        let (grid, forms, legend) = if let Some(forms) = numeric_forms {
            (numeric,forms,"Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom from its top edge, left and right from its left edge.".into())
        } else {
            let grid = Grid {
                low: 0.,
                high: 1.,
                size: 128,
            };
            let (forms, _, codes) = self.grid_forms(&grid, 1.)?;
            let legend = format!(
                "Each edge is a code on a 128-value scale: {} is 0 (top or left), {} is 1 (bottom or right). Values are equally spaced.",
                codes[0],
                codes.last().unwrap()
            );
            (grid, forms, legend)
        };
        let yesno = vec![
            self.ids(&["yes".into(), "Yes".into(), "YES".into()])?,
            self.ids(&["no".into(), "No".into(), "NO".into()])?,
        ];
        let blocks = if kind == "box" {
            1
        } else {
            q["max"].as_u64().unwrap_or(16) as usize
        };
        let mut slot_forms = vec![];
        let mut markers = vec![];
        for k in 1..=blocks {
            for name in ["present", "top", "bottom", "left", "right"] {
                markers.push(if kind == "box" {
                    name.into()
                } else {
                    format!("{name} {k}")
                });
                slot_forms.push(if name == "present" {
                    yesno.clone()
                } else {
                    forms.clone()
                });
            }
        }
        let body = if kind == "box" {
            format!(
                "{said}\n\nPresence is yes or no. {legend} Reply with yes or no at present, then one {} at top, bottom, left and right.",
                if numbers { "number" } else { "code" }
            )
        } else {
            format!(
                "{said}\n\nEvery one of them, {}in reading order (left edge, then top). Present k is yes when there are at least k. {legend} Reply with yes or no at each present, then its top, bottom, left and right.",
                if self.config.d1_prompt == 1 {
                    "16 at most, "
                } else {
                    ""
                }
            )
        };
        Ok(Some(Construction {
            body,
            markers,
            slot_forms,
            forms: vec![],
            readout: if kind == "box" {
                Readout::Box { grid }
            } else {
                Readout::Boxes { grid }
            },
            after: kind == "boxes" && (blocks < 16 || self.config.d1_prompt > 1),
        }))
    }
}
fn ordered(a: f64, b: f64, grid: &Grid) -> (f64, f64) {
    if a < b {
        return (a, b);
    }
    let half = grid.step() / 2.;
    let mid = ((a + b) / 2.).max(half).min(1. - half);
    (mid - half, mid + half)
}
fn rectangle(edges: [f64; 4], grid: &Grid) -> [f64; 4] {
    let (t, b) = ordered(edges[0], edges[1], grid);
    let (l, r) = ordered(edges[2], edges[3], grid);
    [t, b, l, r]
}
fn box_json(b: [f64; 4]) -> Value {
    json!({"top":b[0],"bottom":b[1],"left":b[2],"right":b[3]})
}
fn area(b: [f64; 4]) -> f64 {
    (b[1] - b[0]) * (b[3] - b[2])
}
fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let overlap =
        (a[1].min(b[1]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[2].max(b[2])).max(0.);
    overlap / (area(a) + area(b) - overlap)
}
fn radical(mut i: usize, base: usize) -> f64 {
    let mut v = 0.;
    let mut scale = 1.;
    while i > 0 {
        scale /= base as f64;
        v += (i % base) as f64 * scale;
        i /= base;
    }
    v
}
fn confidence(edges: &[Vec<f64>], b: [f64; 4], grid: &Grid) -> f64 {
    let cdfs: Vec<Vec<f64>> = edges
        .iter()
        .map(|p| {
            let total: f64 = p.iter().sum();
            let mut sum = 0.;
            p.iter()
                .map(|p| {
                    sum += p / total;
                    sum
                })
                .collect()
        })
        .collect();
    let values = grid.values();
    let mut hits = 0;
    for i in 1..=512 {
        let mut draw = [0.; 4];
        for (axis, base) in [2, 3, 5, 7].into_iter().enumerate() {
            let at = cdfs[axis]
                .partition_point(|p| *p < radical(i, base))
                .min(grid.size - 1);
            draw[axis] = values[at];
        }
        if iou(rectangle(draw, grid), b) >= 0.5 {
            hits += 1;
        }
    }
    hits as f64 / 512.
}
fn medians(edges: &[Vec<f64>], grid: &Grid, interp: bool) -> [f64; 4] {
    let mut b = [0.; 4];
    for (axis, p) in edges.iter().enumerate() {
        b[axis] = if !interp {
            grid.quantile(p, 0.5)
        } else {
            let total: f64 = p.iter().sum();
            let mut mass = 0.;
            let mut result = grid.high;
            for (v, p) in grid.values().into_iter().zip(p) {
                let share = p / total;
                if share > 0. && mass + share >= 0.5 {
                    result = (v - grid.step() / 2. + (0.5 - mass) / share * grid.step())
                        .clamp(grid.low, grid.high);
                    break;
                }
                mass += share;
            }
            result
        };
    }
    rectangle(b, grid)
}
pub(super) fn answer(
    mode: &Readout,
    p: &[Vec<f64>],
    cal: &Calibration,
    interp: bool,
) -> anyhow::Result<Option<Value>> {
    Ok(match mode {
        Readout::Estimate { grid, unit } => Some(
            json!({"type":"number","number":grid.quantile(&p[0],cal.level("estimate",0.5)),"interval":[grid.quantile(&p[0],cal.level("estimate",0.1)),grid.quantile(&p[0],cal.level("estimate",0.9))],"unit":unit}),
        ),
        Readout::Box { grid } => {
            ensure!(p.len() == 5, "box needs five slots");
            let found = cal.through("found:box", p[0][0]);
            Some(if found < 0.5 {
                json!({"type":"box","box":null,"found":found,"confidence":0.0})
            } else {
                let b = medians(&p[1..], grid, interp);
                json!({"type":"box","box":box_json(b),"found":found,"confidence":cal.through("box",confidence(&p[1..],b,grid))})
            })
        }
        Readout::Boxes { grid } => {
            let mut presence = vec![];
            let mut below = 1f64;
            for chunk in p.as_chunks::<5>().0 {
                below = below.min(cal.through("found:boxes", chunk[0][0]));
                presence.push(below);
            }
            let mut boxes = vec![];
            for (k, &found) in presence.iter().enumerate() {
                if found < 0.5 {
                    break;
                }
                let edges = &p[5 * k + 1..5 * k + 5];
                let b = medians(edges, grid, interp);
                boxes.push((b,json!({"box":box_json(b),"found":found,"confidence":cal.through("boxes",confidence(edges,b,grid))})));
            }
            boxes.sort_by(|a, b| a.0[2].total_cmp(&b.0[2]).then(a.0[0].total_cmp(&b.0[0])));
            let count = boxes.len();
            Some(
                json!({"type":"boxes","boxes":boxes.into_iter().map(|(_,v)|v).collect::<Vec<_>>(),"count":count,"at_least":count>0&&count==presence.len(),"presence":presence}),
            )
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn calibration() -> Calibration {
        Calibration {
            default: 1.,
            temperatures: Default::default(),
            tables: Default::default(),
        }
    }
    #[test]
    fn boxes_are_ordered_truncated_and_calibrated() -> anyhow::Result<()> {
        let grid = Grid {
            low: 0.,
            high: 1.,
            size: 3,
        };
        let mut p = Vec::new();
        // Two certain full-height boxes, supplied rightmost first.
        for edges in [[0, 2, 1, 2], [0, 2, 0, 1]] {
            p.push(vec![0.9, 0.1]);
            for at in edges {
                let mut row = vec![0.; 3];
                row[at] = 1.;
                p.push(row);
            }
        }
        let out = answer(
            &Readout::Boxes { grid: grid.clone() },
            &p,
            &calibration(),
            false,
        )?
        .unwrap();
        assert_eq!(out["count"], 2);
        assert_eq!(out["at_least"], true);
        assert_eq!(out["presence"], json!([0.9, 0.9]));
        assert_eq!(out["boxes"][0]["box"]["left"], 0.0);
        assert_eq!(out["boxes"][1]["box"]["left"], 0.5);
        let mut cal = calibration();
        cal.tables
            .insert("found:boxes".into(), vec![[0., 0.], [1., 0.4]]);
        let out = answer(&Readout::Boxes { grid }, &p, &cal, false)?.unwrap();
        assert_eq!(out["count"], 0);
        assert_eq!(out["at_least"], false);
        Ok(())
    }
    #[test]
    fn crossed_edges_and_interpolated_medians_follow_upstream() {
        let grid = Grid {
            low: 0.,
            high: 1.,
            size: 3,
        };
        let b = rectangle([0.9, 0.1, 0.5, 0.5], &grid);
        assert_eq!(b, [0.25, 0.75, 0.25, 0.75]);
        assert_eq!(grid.quantile(&[0.1, 0.8, 0.1], 0.5), 0.5);
        let b = medians(&vec![vec![0.8, 0.1, 0.1]; 4], &grid, true);
        assert!(b[0] >= 0. && b[1] > b[0]);
    }
}
