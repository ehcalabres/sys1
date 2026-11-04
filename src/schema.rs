use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use utoipa::ToSchema;

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct DecisionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub state: Value,
    /// Experimental image inputs, accepted only by models with image support. Each item may be a public HTTP(S) URL, base64 string or data URL, JSON byte array, or object containing url, base64, or bytes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<MediaInput>,
    /// Experimental video inputs, accepted only by models with video support. Each item may be a public HTTP(S) URL, base64 string or data URL, JSON byte array, or object containing url, base64, or bytes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub videos: Vec<MediaInput>,
    /// Experimental audio clips, accepted only by models with audio support; same sources as images/videos.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub audio: Vec<MediaInput>,
    #[schema(value_type = Object)]
    pub questions: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum MediaInput {
    Text(String),
    Bytes(Vec<u8>),
    Url(MediaUrl),
    Base64(MediaBase64),
    ByteObject(MediaBytes),
}

impl MediaInput {
    /// Spoken instructions accept only the same bounded sources as top-level media.
    pub(crate) fn spoken(value: &Value) -> Result<Option<Self>, serde_json::Error> {
        if value.get("type").and_then(Value::as_str) != Some("audio") {
            return Ok(None);
        }
        let mut source = value.as_object().unwrap().clone();
        source.remove("type");
        serde_json::from_value(Value::Object(source)).map(Some)
    }
}

impl From<&str> for MediaInput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for MediaInput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct MediaUrl {
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct MediaBase64 {
    pub base64: String,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct MediaBytes {
    pub bytes: Vec<u8>,
}

#[derive(ToSchema)]
pub struct SystemOneRequest {
    pub model: Option<String>,
    pub state: Value,
    /// Experimental image inputs, accepted only by models with image support.
    pub images: Vec<MediaInput>,
    /// Experimental video inputs, accepted only by models with video support.
    pub videos: Vec<MediaInput>,
    /// Experimental audio clips, accepted only by models with audio support.
    pub audio: Vec<MediaInput>,
    #[schema(value_type = Object)]
    pub questions: Map<String, Value>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ValidationError {
    pub loc: Vec<String>,
    pub msg: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HTTPValidationError {
    pub detail: Vec<ValidationError>,
}

impl HTTPValidationError {
    pub fn single(loc: Vec<String>, msg: impl Into<String>, kind: &'static str) -> Self {
        Self {
            detail: vec![ValidationError {
                loc,
                msg: msg.into(),
                kind,
            }],
        }
    }
}

fn invalid(path: Vec<String>, message: impl Into<String>) -> HTTPValidationError {
    HTTPValidationError::single(path, message, "value_error")
}

fn content(value: &Value, nullable: bool) -> bool {
    matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_))
        || (nullable && value.is_null())
}

impl TryFrom<Value> for SystemOneRequest {
    type Error = HTTPValidationError;

    fn try_from(input: Value) -> Result<Self, Self::Error> {
        let body = input
            .as_object()
            .ok_or_else(|| invalid(vec!["body".into()], "Input should be an object"))?;
        let required = |name: &str| {
            body.get(name).ok_or_else(|| {
                HTTPValidationError::single(
                    vec!["body".into(), name.into()],
                    "Field required",
                    "missing",
                )
            })
        };
        let model = match body.get("model") {
            None | Some(Value::Null) => None,
            Some(Value::String(model)) => Some(model.clone()),
            Some(_) => {
                return Err(invalid(
                    vec!["body".into(), "model".into()],
                    "Input should be a string or null",
                ));
            }
        };
        let state = required("state")?;
        let has_media = ["images", "videos", "audio"].iter().any(|name| {
            body.get(*name)
                .and_then(Value::as_array)
                .is_some_and(|v| !v.is_empty())
        });
        if !content(state, false) && !(state.is_null() && has_media) {
            return Err(invalid(
                vec!["body".into(), "state".into()],
                "Input should be a string, object, or array",
            ));
        }
        let media = |name: &str| -> Result<Vec<MediaInput>, HTTPValidationError> {
            match body.get(name) {
                None => Ok(Vec::new()),
                Some(value) => serde_json::from_value(value.clone())
                    .map_err(|error| invalid(vec!["body".into(), name.into()], error.to_string())),
            }
        };
        let images = media("images")?;
        let videos = media("videos")?;
        let audio = media("audio")?;
        let questions = required("questions")?.as_object().ok_or_else(|| {
            invalid(
                vec!["body".into(), "questions".into()],
                "Input should be an object",
            )
        })?;
        if questions.is_empty() {
            return Err(invalid(
                vec!["body".into(), "questions".into()],
                "Object should have at least one property",
            ));
        }
        for (name, question) in questions {
            let path = vec!["body".into(), "questions".into(), name.clone()];
            let object = question
                .as_object()
                .ok_or_else(|| invalid(path.clone(), "Input should be an object"))?;
            let kind = object.get("type").ok_or_else(|| {
                HTTPValidationError::single(
                    [path.clone(), vec!["type".into()]].concat(),
                    "Field required",
                    "missing",
                )
            })?;
            let kind = kind.as_str().ok_or_else(|| {
                invalid(
                    [path.clone(), vec!["type".into()]].concat(),
                    "Input should be a string",
                )
            })?;
            if !matches!(
                kind,
                "noul" | "choice" | "score" | "text" | "number" | "date" | "box" | "boxes"
            ) {
                return Err(invalid(
                    [path.clone(), vec!["type".into()]].concat(),
                    "Unsupported question type",
                ));
            }
            if let Some(instructions) = object.get("instructions")
                && !content(instructions, true)
            {
                return Err(invalid(
                    [path.clone(), vec!["instructions".into()]].concat(),
                    "Input should be a string, object, array, or null",
                ));
            }
            if let Some(instructions) = object.get("instructions") {
                MediaInput::spoken(instructions).map_err(|error| {
                    invalid(
                        [path.clone(), vec!["instructions".into()]].concat(),
                        error.to_string(),
                    )
                })?;
            }
            validate_d1_question(question).map_err(|error| invalid(path.clone(), error))?;
            let criteria = object.get("criteria");
            let criteria_path = [path, vec!["criteria".into()]].concat();
            match kind {
                "noul" => {
                    if let Some(criteria) = criteria
                        && !criteria.is_null()
                    {
                        let object = criteria.as_object().ok_or_else(|| {
                            invalid(criteria_path.clone(), "Input should be an object or null")
                        })?;
                        for (name, value) in object {
                            if matches!(name.as_str(), "true" | "false") && !content(value, true) {
                                return Err(invalid(
                                    [criteria_path.clone(), vec![name.clone()]].concat(),
                                    "Input should be a string, object, array, or null",
                                ));
                            }
                        }
                    }
                }
                "choice" => {
                    let criteria = criteria.ok_or_else(|| {
                        HTTPValidationError::single(
                            criteria_path.clone(),
                            "Field required",
                            "missing",
                        )
                    })?;
                    let object = criteria.as_object().ok_or_else(|| {
                        invalid(criteria_path.clone(), "Input should be an object")
                    })?;
                    if object.is_empty() || object.len() > 255 {
                        return Err(invalid(criteria_path, "Choice requires 1 to 255 options"));
                    }
                    for (name, value) in object {
                        if !content(value, true) {
                            return Err(invalid(
                                [criteria_path.clone(), vec![name.clone()]].concat(),
                                "Input should be a string, object, array, or null",
                            ));
                        }
                    }
                }
                "score" => {
                    let criteria = criteria.ok_or_else(|| {
                        HTTPValidationError::single(
                            criteria_path.clone(),
                            "Field required",
                            "missing",
                        )
                    })?;
                    let values = criteria.as_array().ok_or_else(|| {
                        invalid(criteria_path.clone(), "Input should be an array")
                    })?;
                    if values.is_empty() || values.len() > 10 {
                        return Err(invalid(criteria_path, "Score requires 1 to 10 levels"));
                    }
                    for (index, value) in values.iter().enumerate() {
                        if !content(value, false) {
                            return Err(invalid(
                                [criteria_path.clone(), vec![index.to_string()]].concat(),
                                "Input should be a string, object, or array",
                            ));
                        }
                    }
                }
                "text" | "number" | "date" | "box" | "boxes" => {}
                _ => unreachable!(),
            }
        }
        Ok(Self {
            model,
            state: state.clone(),
            images,
            videos,
            audio,
            questions: questions.clone(),
        })
    }
}

pub(crate) fn is_d1_typed(kind: &str) -> bool {
    matches!(kind, "text" | "number" | "date" | "box" | "boxes")
}
/// Common validation for both API routes and direct native d1 calls.
pub(crate) fn validate_d1_question(q: &Value) -> Result<(), String> {
    let kind = q["type"].as_str().unwrap_or("");
    if is_d1_typed(kind)
        && (q["instructions"].is_null()
            || q["instructions"]
                .as_str()
                .is_some_and(|s| s.trim().is_empty()))
    {
        return Err("Typed d1 questions require nonempty instructions".into());
    }
    if let Some(unit) = q.get("unit")
        && !unit.is_null()
        && !unit.is_string()
    {
        return Err("unit must be a string or null".into());
    }
    if let Some(max) = q.get("max")
        && (kind != "boxes" || !max.as_u64().is_some_and(|n| (1..=16).contains(&n)))
    {
        return Err(
            "max belongs to boxes and must be an integer from 1 to 16 for these checkpoints".into(),
        );
    }
    if let Some(range) = q.get("range") {
        if kind != "number" || range.as_array().is_none_or(|v| v.len() != 2) {
            return Err("range belongs to number and must be [low, high]".into());
        }
        let low = range[0]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("range values must be finite numbers")?;
        let high = range[1]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("range values must be finite numbers")?;
        if low >= high || !(high - low).is_finite() {
            return Err("range must run from low to high".into());
        }
        if let Some(step) = q.get("step") {
            let step = step
                .as_f64()
                .filter(|v| v.is_finite() && *v > 0.)
                .ok_or("step must be a positive finite number")?;
            let n = (high - low) / step;
            if !n.is_finite()
                || n.round_ties_even() < 1.
                || (n - n.round_ties_even()).abs() > 1e-9 * n.abs().max(n.round_ties_even().abs())
            {
                return Err("step must divide the declared range".into());
            }
        }
    } else if q.get("step").is_some() {
        return Err("step requires a number range".into());
    }
    if let Some(multiple) = q.get("multiple")
        && (!multiple.is_boolean() || multiple == true && kind != "choice")
    {
        return Err("multiple is a boolean belonging to choice".into());
    }
    Ok(())
}

impl From<SystemOneRequest> for DecisionRequest {
    fn from(request: SystemOneRequest) -> Self {
        Self {
            model: request.model,
            state: request.state,
            images: request.images,
            videos: request.videos,
            audio: request.audio,
            questions: request.questions,
        }
    }
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct DecisionResponse {
    pub model: String,
    #[schema(value_type = Object)]
    pub answers: Map<String, Value>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct ApiError {
    pub error: String,
    #[serde(skip)]
    #[schema(ignore)]
    status: u16,
    #[serde(skip)]
    #[schema(ignore)]
    unknown_model: bool,
}

impl ApiError {
    pub fn new(message: impl Into<String>) -> Self {
        Self::with_status(message, 400)
    }

    pub fn unknown_model(model: &str) -> Self {
        Self {
            error: format!("unknown model: {model}"),
            status: 400,
            unknown_model: true,
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::with_status(message, 503)
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self::with_status(message, 504)
    }

    pub fn payload_too_large(message: impl Into<String>) -> Self {
        Self::with_status(message, 413)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::with_status(message, 500)
    }

    pub fn status(&self) -> u16 {
        self.status
    }

    pub fn is_unknown_model(&self) -> bool {
        self.unknown_model
    }

    fn with_status(message: impl Into<String>, status: u16) -> Self {
        Self {
            error: message.into(),
            status,
            unknown_model: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_typed_d1_questions_without_changing_legacy_criteria() {
        for question in [
            json!({"type":"text","instructions":"Who?"}),
            json!({"type":"number","instructions":"How much?","unit":"EUR"}),
            json!({"type":"number","instructions":"Count?","range":[0,10],"step":1}),
            json!({"type":"number","instructions":"Estimate?","range":[-1,1],"step":0.25}),
            json!({"type":"date","instructions":"When?"}),
            json!({"type":"box","instructions":"Where?"}),
            json!({"type":"boxes","instructions":"Every cat.","max":16}),
            json!({"type":"noul"}),
            json!({"type":"choice","criteria":{"x":null}}),
            json!({"type":"score","criteria":["one"]}),
        ] {
            assert!(
                SystemOneRequest::try_from(json!({"state":"document","questions":{"q":question}}))
                    .is_ok(),
                "{question}"
            );
        }
        for question in [
            json!({"type":"text"}),
            json!({"type":"date","instructions":" "}),
            json!({"type":"number","instructions":"Count?","range":[1,1]}),
            json!({"type":"number","instructions":"Count?","range":[0,1],"step":0.3}),
            json!({"type":"number","instructions":"Count?","range":[0,1],"step":0}),
            json!({"type":"number","instructions":"Count?","range":[false,1]}),
            json!({"type":"number","instructions":"Count?","step":1}),
            json!({"type":"number","instructions":"Count?","unit":12}),
            json!({"type":"box","instructions":"Where?","range":[0,1]}),
            json!({"type":"boxes","instructions":"Every cat.","max":0}),
            json!({"type":"boxes","instructions":"Every cat.","max":17}),
            json!({"type":"boxes","instructions":"Every cat.","max":true}),
            json!({"type":"box","instructions":"Where?","max":1}),
        ] {
            assert!(
                SystemOneRequest::try_from(json!({"state":"document","questions":{"q":question}}))
                    .is_err(),
                "{question}"
            );
        }
    }

    #[test]
    fn validates_audio_only_states_and_spoken_instruction_sources() {
        let request=SystemOneRequest::try_from(json!({"state":null,"audio":[{"bytes":[1,2]}],"questions":{"q":{"type":"noul","instructions":{"type":"audio","base64":"AA=="}}}})).unwrap();
        assert_eq!(request.audio.len(), 1);
        assert!(
            SystemOneRequest::try_from(json!({"state":null,"questions":{"q":{"type":"noul"}}}))
                .is_err()
        );
        assert!(SystemOneRequest::try_from(json!({"state":"x","questions":{"q":{"type":"noul","instructions":{"type":"audio","path":"/etc/passwd"}}}})).is_err());
    }

    #[test]
    fn accepts_each_json_media_representation() {
        assert!(matches!(
            serde_json::from_value::<MediaInput>(json!("data:image/png;base64,AA==")).unwrap(),
            MediaInput::Text(_)
        ));
        assert!(matches!(
            serde_json::from_value::<MediaInput>(json!([0, 1, 2])).unwrap(),
            MediaInput::Bytes(_)
        ));
        assert!(matches!(
            serde_json::from_value::<MediaInput>(json!({"bytes": [0, 1, 2]})).unwrap(),
            MediaInput::ByteObject(_)
        ));
        assert!(matches!(
            serde_json::from_value::<MediaInput>(json!({"base64": "AA=="})).unwrap(),
            MediaInput::Base64(_)
        ));
        assert!(matches!(
            serde_json::from_value::<MediaInput>(json!({"url": "https://example.com/a.mp4"}))
                .unwrap(),
            MediaInput::Url(_)
        ));
    }
}
