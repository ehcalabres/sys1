//! Strict written-number/date parsing ported from the pinned d1 parsing.py.
use fancy_regex::{Captures, Regex};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, OnceLock},
};
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};
fn regex(pattern: &str) -> Arc<Regex> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<Regex>>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    cache
        .entry(pattern.into())
        .or_insert_with(|| Arc::new(Regex::new(pattern).expect("valid upstream expression")))
        .clone()
}
fn cap<'a>(p: &str, s: &'a str) -> Option<Captures<'a, str>> {
    regex(p).captures(s).ok().flatten()
}
fn full<'a>(p: &str, s: &'a str) -> Option<Captures<'a, str>> {
    cap(&format!("\\A(?:{p})\\z"), s)
}
fn has(p: &str, s: &str) -> bool {
    regex(p).is_match(s).unwrap_or(false)
}
fn all(p: &str, s: &str) -> Vec<Vec<String>> {
    regex(p)
        .captures_iter(s)
        .filter_map(Result::ok)
        .map(|c| {
            (0..c.len())
                .map(|i| c.get(i).map_or("", |m| m.as_str()).into())
                .collect()
        })
        .collect()
}
fn sub(p: &str, s: &str, to: &str) -> String {
    regex(p).replace_all(s, to).into_owned()
}
fn at<'a>(c: &Captures<'a, str>, i: usize) -> &'a str {
    c.get(i).map_or("", |m| m.as_str())
}
fn one<T: Eq + std::hash::Hash>(mut set: HashSet<T>) -> Option<T> {
    if set.len() == 1 {
        set.drain().next()
    } else {
        None
    }
}
fn fold(s: &str) -> String {
    s.to_lowercase()
        .replace('ß', "ss")
        .nfkd()
        .filter(|c| !is_combining_mark(*c))
        .collect()
}
fn currency_pattern() -> String {
    let mut names: Vec<_> = CURRENCIES
        .iter()
        .flat_map(|(_, names)| names.iter().copied())
        .collect();
    names.sort_by_key(|n| std::cmp::Reverse(n.chars().count()));
    format!(
        "(?i:{})",
        names
            .into_iter()
            .map(|c| {
                let escaped = fancy_regex::escape(c);
                if c.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
                    format!(r"\b{escaped}(?![a-z])")
                } else {
                    escaped.into_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("|")
    )
}
pub(super) fn written_unit(text: &str) -> Option<String> {
    if text.trim_end().ends_with('%') {
        return Some("%".into());
    }
    let markers = all(&currency_pattern(), text);
    if markers.len() != 1 {
        return None;
    }
    CURRENCIES
        .iter()
        .flat_map(|(_, n)| n.iter())
        .find(|n| n.to_lowercase() == markers[0][0].to_lowercase())
        .map(|s| s.to_string())
}
fn currency_marks(text: &str) -> HashSet<String> {
    all(&currency_pattern(), text)
        .iter()
        .filter_map(|m| {
            CURRENCIES
                .iter()
                .find(|(_, names)| {
                    names
                        .iter()
                        .any(|n| n.to_lowercase() == m[0].to_lowercase())
                })
                .and_then(|(mark, _)| mark.map(str::to_owned))
        })
        .collect()
}
const GROUPED: &str =
    r"(?<![\d.,])\d{1,3}(?:([,. \u{00a0}\u{202f}'’])\d{3})+(?!\1)([.,])\d+(?![\d.,])";
const SHORT: &str = r"(?<![\d.,])\d+([.,])\d{1,2}(?![\d.,])";
const REPEATED: &str = r"(?<![\d.,])\d{1,3}(?:([.,])\d{3}){2,}(?![\d.,])";
fn decimal_mark(written: &str, context: &str) -> Option<String> {
    let context = sub(r"<\?[^>]*\?>", context, "");
    let mut numbers: HashSet<_> = all(GROUPED, &context)
        .iter()
        .map(|m| m[2].clone())
        .chain(all(SHORT, &context).iter().map(|m| m[1].clone()))
        .collect();
    numbers.extend(
        all(REPEATED, &context)
            .iter()
            .map(|m| if m[1] == "." { ",".into() } else { ".".into() }),
    );
    let locales=all(r#"\b(?:locale|language|lang)["']?\s*[=:]\s*["']?(en|fr|de|es|it|pt|nl|pl|ru|sv|ja|zh|ko)(?:[-_][a-z]{2})?\b"#,&context.to_lowercase()).iter().map(|m|if matches!(m[1].as_str(),"en"|"ja"|"zh"|"ko"){".".into()}else{",".into()}).collect();
    for evidence in [
        currency_marks(written),
        numbers,
        currency_marks(&context),
        locales,
    ] {
        if !evidence.is_empty() {
            return one(evidence);
        }
    }
    None
}
fn digits(text: &str, written: &str, context: &str) -> Option<String> {
    full(r"\d[\d., '’]*", text)?;
    if has(r"[ '’]", text) {
        let c = full(r"(\d{1,3}(?:[ '’]\d{3})+)([.,]\d+)?", text)?;
        return digits(
            &(sub(r"[ '’]", at(&c, 1), "") + at(&c, 2)),
            written,
            context,
        );
    }
    if full(r"\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?", text).is_some()
        && has(
            r"(?i)₹|\bINR\b|\bRs\b|lakh|crore",
            &format!("{written}\n{context}"),
        )
    {
        return Some(text.replace(',', ""));
    }
    if text.contains('.') && text.contains(',') {
        let (point, group) = if text.rfind('.') > text.rfind(',') {
            ('.', ',')
        } else {
            (',', '.')
        };
        full(
            &format!(
                r"\d{{1,3}}(?:{}\d{{3}})+{}\d+",
                fancy_regex::escape(&group.to_string()),
                fancy_regex::escape(&point.to_string())
            ),
            text,
        )?;
        return Some(text.replace(group, "").replace(point, "."));
    }
    if !text.contains('.') && !text.contains(',') {
        return Some(text.into());
    }
    let mark = if text.contains('.') { "." } else { "," };
    let parts: Vec<_> = text.split(mark).collect();
    if parts.iter().any(|p| p.is_empty()) {
        return None;
    }
    let point = decimal_mark(written, context);
    if parts.len() > 2 || point.as_deref().is_some_and(|p| p != mark) {
        return if (1..=3).contains(&parts[0].chars().count())
            && parts[1..].iter().all(|p| p.chars().count() == 3)
        {
            Some(parts.join(""))
        } else {
            None
        };
    }
    if point.is_none() && parts[1].chars().count() == 3 && parts[0].chars().count() <= 3 {
        return None;
    }
    Some(parts.join("."))
}
pub(super) fn parse_number(text: &str, context: &str, unit: Option<&str>) -> Value {
    parse_number_inner(text, context, unit).unwrap_or(Value::Null)
}
fn parse_number_inner(text: &str, context: &str, unit: Option<&str>) -> Option<Value> {
    let currency = currency_pattern();
    let context = format!(
        "{context}{}",
        unit.filter(|u| full(&currency, u.trim()).is_some())
            .map(|u| format!("\n{u}"))
            .unwrap_or_default()
    );
    let written = text;
    let mut text = text
        .trim()
        .replace('−', "-")
        .replace(['\u{a0}', '\u{202f}'], " ");
    if text.ends_with('%') {
        text.pop();
        text = text.trim_end().into();
    }
    let negative = text.starts_with('(') && text.ends_with(')');
    if negative {
        text = text[1..text.len() - 1].trim().into();
    }
    let mut sign = None;
    let mut had_currency = false;
    for _ in 0..2 {
        if !had_currency
            && let Some(c) = full(&format!(r"({currency})\s*(.*)|(.*?)\s*({currency})"), &text)
        {
            let next = if !at(&c, 1).is_empty() {
                at(&c, 2)
            } else {
                at(&c, 3)
            }
            .trim()
            .to_string();
            text = next;
            had_currency = true;
        }
        if sign.is_none() && (text.starts_with('-') || text.starts_with('+')) {
            sign = text.chars().next();
            text = text[1..].trim().into();
        }
    }
    if negative && sign.is_some() {
        return None;
    }
    let scale_pattern = format!(
        r"(?i)\s*({})\.?$",
        SCALES
            .iter()
            .map(|(s, _)| fancy_regex::escape(s))
            .collect::<Vec<_>>()
            .join("|")
    );
    let mut power = 0;
    if let Some(c) = cap(&scale_pattern, &text) {
        let word = at(&c, 1);
        if word == "Billion" {
            return None;
        }
        power = SCALES
            .iter()
            .find(|(s, _)| *s == word.to_lowercase())
            .map(|(_, p)| *p)?;
        text = text[..c.get(0)?.start()].into();
    } else if let Some(c) = cap(r"(?i)(?<=\d)(k|m|bn|b)$", &text) {
        if !has(&currency, &format!("{written}\n{context}"))
            && !has(
                r"(?i)\b(?:money|price|invoice|revenue|cost|count|quantity|population|people|items|units|users|downloads)\b",
                &context,
            )
        {
            return None;
        }
        power = match at(&c, 1).to_lowercase().as_str() {
            "k" => 3,
            "m" => 6,
            _ => 9,
        };
        text = text[..c.get(0)?.start()].into();
    }
    let normalized = normalize_digits(&digits(&text, written, &context)?);
    let (integer, fraction) = normalized.split_once('.').unwrap_or((&normalized, ""));
    let mut value = format!("{integer}{fraction}");
    // Decimal multiplication in upstream uses Python's default 28 significant
    // digits and ROUND_HALF_EVEN, even when the scale is one.
    value = value.trim_start_matches('0').to_owned();
    if value.is_empty() {
        value = "0".into();
    }
    if value.len() > 28 {
        let discarded = value[28..].to_owned();
        let n = discarded.len();
        let mut kept = value.as_bytes()[..28].to_vec();
        let up = discarded.as_bytes()[0] > b'5'
            || discarded.as_bytes()[0] == b'5'
                && (discarded.as_bytes()[1..].iter().any(|b| *b != b'0')
                    || (kept[27] - b'0') % 2 == 1);
        if up {
            let mut i = kept.len();
            while i > 0 && kept[i - 1] == b'9' {
                kept[i - 1] = b'0';
                i -= 1;
            }
            if i == 0 {
                kept.insert(0, b'1');
            } else {
                kept[i - 1] += 1;
            }
        }
        value = String::from_utf8(kept).unwrap() + &"0".repeat(n);
    }

    let shift = power as isize - fraction.len() as isize;
    if shift >= 0 {
        value.push_str(&"0".repeat(shift as usize));
    } else {
        let places = (-shift) as usize;
        if value.len() <= places {
            value = format!("{}{}", "0".repeat(places + 1 - value.len()), value);
        }
        value.insert(value.len() - places, '.');
        value = value.trim_end_matches('0').trim_end_matches('.').into();
    }
    value = value.trim_start_matches('0').into();
    if value.is_empty() {
        value = "0".into();
    } else if value.starts_with('.') {
        value.insert(0, '0');
    }
    if (negative || sign == Some('-')) && value != "0" {
        value.insert(0, '-');
    }
    let result: Value = serde_json::from_str(&value).ok()?;
    if value.contains('.') {
        let number = result.as_f64().filter(|v| v.is_finite())?;
        return Some(serde_json::json!(number));
    }
    Some(result)
}
const NUMERIC_DATE: &str = r"(?<!\d)(\d{1,4})([/.-])(\d{1,2})\2(\d{2,4})(?!\d)";
fn months() -> HashMap<String, u32> {
    let mut out = HashMap::new();
    let mut abbreviations: HashMap<String, HashSet<u32>> = HashMap::new();
    for (month, names) in MONTH_DATA.iter().enumerate() {
        for name in names.split_whitespace() {
            out.insert(name.into(), month as u32 + 1);
            abbreviations
                .entry(name.chars().take(3).collect())
                .or_default()
                .insert(month as u32 + 1);
        }
    }
    for (name, set) in abbreviations {
        if let Some(month) = one(set) {
            out.insert(name, month);
        }
    }
    for (name, month) in [("janv", 1), ("fevr", 2), ("juil", 7), ("sept", 9)] {
        out.insert(name.into(), month);
    }
    out
}
fn iso(year: u32, month: u32, day: u32) -> Option<String> {
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        _ => return None,
    };
    if !(1..=9999).contains(&year) || day == 0 || day > days {
        return None;
    }
    Some(format!("{year:04}-{month:02}-{day:02}"))
}
fn date_order(context: &str) -> Option<bool> {
    // true: day/month/year
    let folded = fold(context);
    let day = r"(?:dd?|jj?|tt?|gg?|day|jour|tag|dia|giorno)";
    let month = r"(?:mm?|month|mois|monat|mes|mese)";
    let year = r"(?:yy(?:yy)?|aa(?:aa)?|jj(?:jj)?|year|annee|jahr|ano|anno)";
    let mut explicit = HashSet::new();
    for (a, b, dmy) in [(day, month, true), (month, day, false)] {
        if has(&format!(r"\b{a}\s*[/.-]\s*{b}\s*[/.-]\s*{year}\b"), &folded) {
            explicit.insert(dmy);
        }
    }
    if !explicit.is_empty() {
        return one(explicit);
    }
    let locales = all(
        r#"\b(?:locale|language|lang|langue|sprache|idioma|lingua)["']?\s*[=:]\s*["']?(en(?:[-_](?:us|gb|au|ca))?|fr(?:[-_]fr)?|de(?:[-_]de)?|es(?:[-_]es)?|it(?:[-_]it)?)\b"#,
        &folded,
    );
    if !locales.is_empty() {
        return one(locales
            .iter()
            .map(|m| !matches!(m[1].replace('_', "-").as_str(), "en" | "en-us"))
            .collect());
    }
    let mut orders = HashSet::new();
    for c in all(NUMERIC_DATE, context) {
        let a: u32 = c[1].parse().unwrap_or(0);
        let b: u32 = c[3].parse().unwrap_or(0);
        let y: u32 = c[4].parse().unwrap_or(0);
        if c[4].len() != 4 || a == 0 || b == 0 || a > 31 || b > 31 {
            continue;
        }
        if (a > 12) != (b > 12) {
            if iso(y, b, a).is_some() {
                orders.insert(true);
            }
            if iso(y, a, b).is_some() {
                orders.insert(false);
            }
        }
    }
    one(orders)
}
pub(super) fn parse_date(text: &str, context: &str) -> Option<String> {
    parse_date_depth(&normalize_digits(text), &normalize_digits(context), 0)
}
fn parse_date_depth(text: &str, context: &str, depth: usize) -> Option<String> {
    if depth > 3 {
        return None;
    }
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    if let Some(c) = full(
        r"(.+?)[T\s]+(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)",
        text,
    ) {
        let mut date = parse_date_depth(at(&c, 1), context, depth + 1)?;
        let time = at(&c, 2);
        let t = full(
            r"(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?(Z|[+-]\d{2}:\d{2})?",
            time,
        )?;
        let mut h: u32 = at(&t, 1).parse().ok()?;
        let m: u32 = at(&t, 2).parse().ok()?;
        let sec: u32 = if at(&t, 3).is_empty() {
            0
        } else {
            at(&t, 3).parse().ok()?
        };
        if h > 24 || m > 59 || sec > 59 {
            return None;
        }
        let fraction = at(&t, 4);
        let micro = format!("{fraction:0<6}")
            .chars()
            .take(6)
            .collect::<String>();
        let fraction = if micro == "000000" {
            String::new()
        } else {
            format!(".{micro}")
        };
        if h == 24 {
            if m != 0 || sec != 0 || micro != "000000" {
                return None;
            }
            let y: u32 = date[..4].parse().ok()?;
            let month: u32 = date[5..7].parse().ok()?;
            let day: u32 = date[8..10].parse().ok()?;
            date = iso(y, month, day + 1).or_else(|| {
                if month == 12 {
                    iso(y + 1, 1, 1)
                } else {
                    iso(y, month + 1, 1)
                }
            })?;
            h = 0;
        }
        let zone = at(&t, 5);
        let zone = if zone == "Z" { "+00:00" } else { zone };
        let zone = if zone.is_empty() {
            String::new()
        } else {
            let hour: u32 = zone[1..3].parse().ok()?;
            let minute: u32 = zone[4..].parse().ok()?;
            let total = hour * 60 + minute;
            if total >= 24 * 60 {
                return None;
            }
            format!(
                "{}{:02}:{:02}",
                if total == 0 || zone.starts_with('+') {
                    "+"
                } else {
                    "-"
                },
                total / 60,
                total % 60
            )
        };
        return Some(format!("{date}T{h:02}:{m:02}:{sec:02}{fraction}{zone}"));
    }
    if let Some(c) = full(NUMERIC_DATE, text) {
        let a: u32 = at(&c, 1).parse().ok()?;
        let b: u32 = at(&c, 3).parse().ok()?;
        let mut year: u32 = at(&c, 4).parse().ok()?;
        if at(&c, 1).len() == 4 {
            return iso(a, b, year);
        }
        if at(&c, 4).len() != 4 {
            if at(&c, 4).len() != 2 {
                return None;
            }
            let mut centuries = HashSet::new();
            let mut complete: Vec<_> = all(NUMERIC_DATE, context)
                .into_iter()
                .filter(|m| m[1].len() == 4 || m[4].len() == 4)
                .map(|m| m[0].clone())
                .collect();
            let month_names = months();
            let mut names: Vec<_> = month_names.keys().cloned().collect();
            names.sort_by_key(|n| std::cmp::Reverse(n.len()));
            let names = names.join("|");
            let pattern = format!(
                r"\b(?:\d{{1,2}}\w*[ .]+(?:de\s+)?(?:{names})[ ,.]+(?:de\s+)?\d{{4}}|(?:{names})[ .]+\d{{1,2}}\w*[ ,.]+\d{{4}})\b"
            );
            complete.extend(
                all(&pattern, &fold(context))
                    .into_iter()
                    .map(|m| m[0].clone()),
            );
            for date in complete {
                if let Some(date) = parse_date_depth(&date, context, depth + 1) {
                    centuries.insert(date[..4].parse::<u32>().ok()? / 100 * 100);
                }
            }
            year += one(centuries)?;
        }
        let order = date_order(context);
        let mut values = HashSet::new();
        if order != Some(false)
            && let Some(v) = iso(year, b, a)
        {
            values.insert(v);
        }
        if order != Some(true)
            && let Some(v) = iso(year, a, b)
        {
            values.insert(v);
        }
        return one(values);
    }
    if let Some(c) = full(
        r"(\d{4})\s*[年년]\s*(\d{1,2})\s*[月월]\s*(\d{1,2})\s*[日일]",
        text,
    ) {
        return iso(
            at(&c, 1).parse().ok()?,
            at(&c, 2).parse().ok()?,
            at(&c, 3).parse().ok()?,
        );
    }
    let words = fold(text);
    let words = sub(r"(?<=\d)(?:st|nd|rd|th|er)\b", &words, "");
    let words = sub(r"(?<=\d)\.?(?:o|a|°)\b", &words, "");
    let words = sub(r"\b(?:de|of|the|le|el|il|den)\b", &words, " ");
    let words = sub(r"[,.]", &words, " ");
    let mut words: Vec<_> = words.split_whitespace().collect();
    if words.first().is_some_and(|w| {
        WEEKDAYS
            .split_whitespace()
            .any(|d| d == w.strip_suffix("-feira").unwrap_or(w))
    }) {
        words.remove(0);
    }
    if words.len() != 3 {
        return None;
    }
    let names = months();
    let (day, month, year) = if let Some(&month) = names.get(words[0]) {
        (words[1], month, words[2])
    } else {
        (words[0], *names.get(words[1])?, words[2])
    };
    if year.len() != 4 {
        return None;
    }
    iso(year.parse().ok()?, month, day.parse().ok()?)
}

const MONTH_DATA: &[&str] = &[
    "january janvier januar enero gennaio janeiro januari stycznia styczen января",
    "february fevrier februar febrero febbraio fevereiro februari lutego luty февраля",
    "march mars marz marzo marco maart marca марта",
    "april avril abril aprile kwietnia kwiecien апреля",
    "may mai mayo maggio maio mei maj maja мая",
    "june juin juni junio giugno junho czerwca czerwiec июня",
    "july juillet juli julio luglio julho lipca lipiec июля",
    "august aout agosto augustus augusti sierpnia sierpien августа",
    "september septembre septiembre settembre setembro wrzesnia wrzesien сентября",
    "october octobre oktober octubre ottobre outubro pazdziernika pazdziernik октября",
    "november novembre noviembre novembro listopada listopad ноября",
    "december decembre dezember diciembre dicembre dezembro grudnia grudzien декабря",
];
const WEEKDAYS: &str = "dienstag dimanche domenica domingo donnerstag freitag fri friday giovedi jeudi jueves lundi lunedi lunes mardi martedi martes mercoledi mercredi miercoles mittwoch mon monday montag quarta quinta sabado sabato samedi samstag sat saturday segunda sexta sonntag sun sunday terca thu thur thurs thursday tue tues tuesday vendredi venerdi viernes wed wednesday";
const CURRENCIES: &[(Option<&str>, &[&str])] = &[
    (
        Some("."),
        &[
            "$", "£", "₹", "USD", "GBP", "CAD", "AUD", "NZD", "INR", "Rs.", "Rs", "MXN", "CNY",
            "RMB", "元", "HKD", "SGD",
        ],
    ),
    (
        Some(","),
        &[
            "R$", "BRL", "₽", "RUB", "zł", "PLN", "kr", "SEK", "NOK", "DKK", "₺", "TRY",
        ],
    ),
    (
        Some(""),
        &[
            "¥", "￥", "円", "₩", "₫", "Rp.", "Rp", "IDR", "JPY", "KRW", "VND", "CLP",
        ],
    ),
    (None, &["€", "EUR", "CHF"]),
];
const SCALES: &[(&str, u32)] = &[
    ("milliarden", 9),
    ("milliards", 9),
    ("milliarde", 9),
    ("thousand", 3),
    ("millions", 6),
    ("millones", 6),
    ("billions", 9),
    ("milliard", 9),
    ("miliardo", 9),
    ("miliardi", 9),
    ("trillion", 12),
    ("tausend", 3),
    ("million", 6),
    ("milhões", 6),
    ("milione", 6),
    ("milioni", 6),
    ("billion", 9),
    ("crores", 7),
    ("millón", 6),
    ("milhão", 6),
    ("lakhs", 5),
    ("crore", 7),
    ("lakh", 5),
    ("млрд", 9),
    ("mil", 3),
    ("тыс", 3),
    ("mio", 6),
    ("mln", 6),
    ("млн", 6),
    ("mrd", 9),
    ("千", 3),
    ("万", 4),
    ("萬", 4),
    ("亿", 8),
    ("億", 8),
];

fn normalize_digits(text: &str) -> String {
    text.chars()
        .map(|c| {
            let n = c as u32;
            DIGIT_ZEROS
                .iter()
                .find(|zero| n >= **zero && n < **zero + 10)
                .and_then(|z| char::from_u32(48 + n - *z))
                .unwrap_or(c)
        })
        .collect()
}
const DIGIT_ZEROS: &[u32] = &[
    48, 1632, 1776, 1984, 2406, 2534, 2662, 2790, 2918, 3046, 3174, 3302, 3430, 3558, 3664, 3792,
    3872, 4160, 4240, 6112, 6160, 6470, 6608, 6784, 6800, 6992, 7088, 7232, 7248, 42528, 43216,
    43264, 43472, 43504, 43600, 44016, 65296, 66720, 68912, 69734, 69872, 69942, 70096, 70384,
    70736, 70864, 71248, 71360, 71472, 71904, 72016, 72784, 73040, 73120, 73552, 92768, 92864,
    93008, 120782, 120792, 120802, 120812, 120822, 123200, 123632, 124144, 125264, 130032,
];

/// Python's f-string rendering of JSON-valued instructions. Legacy sys1 question
/// rendering is intentionally unchanged; upstream typed prompts use this form.
pub(super) fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(v) => {
            if *v {
                "True".into()
            } else {
                "False".into()
            }
        }
        Value::String(text) => {
            let quote = if text.contains('\'') && !text.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut out = String::new();
            out.push(quote);
            for c in text.chars() {
                match c {
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    '\\' => out.push_str("\\\\"),
                    c if c == quote => {
                        out.push('\\');
                        out.push(c);
                    }
                    c if c.is_control()
                        || c.is_whitespace() && c != ' '
                        || matches!(c as u32,0x200b..=0x200f|0x202a..=0x202e|0x2060..=0x206f|0xe000..=0xf8ff|0xfeff) =>
                    {
                        let n = c as u32;
                        out.push_str(&if n <= 255 {
                            format!("\\x{n:02x}")
                        } else if n <= 65535 {
                            format!("\\u{n:04x}")
                        } else {
                            format!("\\U{n:08x}")
                        });
                    }
                    c => out.push(c),
                }
            }
            out.push(quote);
            out
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(k, v)| format!(
                    "{}: {}",
                    python_repr(&Value::String(k.clone())),
                    python_repr(v)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Number(number) => {
            let raw = number.to_string();
            if !raw.contains(['.', 'e', 'E']) {
                return raw;
            }
            let value = number.as_f64().unwrap();
            let scientific = format!("{value:e}");
            let (mantissa, exp) = scientific.split_once('e').unwrap();
            let exp: i32 = exp.parse().unwrap();
            if !(-4..16).contains(&exp) {
                format!("{mantissa}e{exp:+03}")
            } else {
                let mut s = value.to_string();
                if !s.contains('.') {
                    s.push_str(".0");
                }
                s
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_upstream_written_values() {
        let fixtures: Value =
            serde_json::from_str(include_str!("../fixtures/d1_parsing_reference.json")).unwrap();
        for case in fixtures["numbers"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            let context = case["context"].as_str().unwrap();
            let actual = parse_number(text, context, case["unit"].as_str());
            if case["expected"].is_number()
                && case["expected"].to_string().contains(['.', 'e', 'E'])
            {
                assert_eq!(actual.as_f64(), case["expected"].as_f64(), "{case}");
            } else {
                assert_eq!(actual, case["expected"], "{case}");
            }
            assert_eq!(
                serde_json::json!(written_unit(text)),
                case["written_unit"],
                "{case}"
            );
        }
        for case in fixtures["dates"].as_array().unwrap() {
            assert_eq!(
                serde_json::json!(parse_date(
                    case["text"].as_str().unwrap(),
                    case["context"].as_str().unwrap()
                )),
                case["expected"],
                "{case}"
            );
        }
    }
}

#[cfg(test)]
mod instruction_tests {
    use super::*;
    #[test]
    fn structured_instructions_keep_python_prompt_rendering() {
        let value = serde_json::json!({"ask":["Alice's refund",true,null,"\n",1.0,1e-7]});
        assert_eq!(
            python_repr(&value),
            r#"{'ask': ["Alice's refund", True, None, '\n', 1.0, 1e-07]}"#
        );
    }
}
