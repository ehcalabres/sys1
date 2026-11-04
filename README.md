<div align="center">
  <img
    src="https://github.com/user-attachments/assets/8b2774a7-6439-422e-bbd3-3fdd85117c5d"
    alt="sys1"
    width="1200"
  />
  <br/>
  <em>
    Blazing fast structured decisions for System One open-models self-hosted with a TypeSafe AI compatible API, written in Rust.
  </em>
</div>

## Features

- `tokio`, `axum` and `serde`, the usual suspects
- System One compatible API Spec
- `candle` with [`tokenizers` release candidate](https://huggingface.co/blog/tokenizers-v1)!
- Dynamic, token-based batching
- SDPA on CPU, Metal, and CUDA
- Flash Attention on Ampere, Ada Lovelace, and Hopper
- Blazing fast inference for ModernBERT and Qwen3.5
- Multimodal decisions from text, images, and video when supported by the model

## Get started

Rust and Cargo 1.98.1, mandatory. `ffmpeg` and `ffprobe` 6.1.1 or higher for video inputs.

If you run on CUDA you also need `nvcc`. And if you run on Metal, you also need `xcode` and `xcodebuild -downloadComponent MetalToolchain`.

```bash
cargo install sys1 --features cpu
# cargo install sys1 --no-default-features --features metal
# cargo install sys1 --no-default-features --features cuda
# cargo install sys1 --no-default-features --features cuda,flash-attn-2 # Ampere, Ada Lovelace, or Hopper
```

Then run it with any of the supported models (more coming soon!).

- [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) for English text, guardrails, email triage
- [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) for 100+ languages, ~2.2x faster
- [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) for typed-decisions workflows
- [`LiquidAI/d1-2B`](https://huggingface.co/LiquidAI/d1-2B) for text, image, and video decisions (native Candle inference)
- [`Cloudflare/clef`](https://huggingface.co/Cloudflare/clef) for text, image, and video decisions
- [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) for text, image, and video decisions (smaller and faster than the 27B variant)

```bash
sys1 --model-id convaiinnovations/laya --dtype auto
```

And, just query your Jev-compatible API at `/v1/systemone` (or `/v1/decide`).

```bash
curl http://localhost:3000/v1/systemone \
    -H "Content-Type: application/json" \
    -d '{
      "model": "convaiinnovations/laya",
      "state": {
        "message": "I was charged twice for invoice 4411. Please refund me today."
      },
      "questions": {
        "route": {
          "type": "choice",
          "instructions": "Where should this ticket go?",
          "criteria": {
            "billing": "payments, refunds, invoices",
            "bug": "the product is broken",
            "account": "login or access"
          }
        },
        "urgency": {
          "type": "score",
          "instructions": "How urgent is this message?",
          "criteria": [
            "routine, no rush",
            "today",
            "urgent",
            "critical, about to churn"
          ]
        },
        "escalate": {
          "type": "noul",
          "instructions": "Escalate to a human immediately?"
        }
      }
    }'
```

Or, install the [TypeSafe AI SDK for Python](https://github.com/typesafe-ai/typesafe-sdk-python), or the [TypeSafe AI SDK for Javascript / Typescript](https://github.com/typesafe-ai/typesafe-sdk-js).

```python
from typesafe_sdk import Choice, TypeSafeClient

client = TypeSafeClient(base_url="http://localhost:3000", api_key="-")

response = client.system_one(
    model="convaiinnovations/laya",
    state={"message": "I was charged twice for invoice 4411."},
    questions={
        "route": Choice(
            instructions="Where should this ticket go?",
            criteria={"billing": None, "bug": None, "account": None},
        ),
    },
)
print(response.choices["route"].choice)
```

## References

- [TypeSafe AI API](https://api.typesafe.ai)
- [TypeSafe AI Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [Laya: Multilingual, non-autoregressive System 1 decision engine](https://github.com/NandhaKishorM/laya)
- [Introducing Clef: our open-source decision models, and new RL fine-tuning platform](https://blog.cloudflare.com/clef-decision-models/)

## LiquidAI d1-2B

Run `sys1 --model-id LiquidAI/d1-2B --dtype f32` on CPU, or use a CUDA/Metal
build with `--dtype auto` for the checkpoint's BF16 precision. Inference is native
Rust/Candle and requires no Python runtime or remote model code.

Text, images, and video use the existing API with `noul`, `choice` (including
`multiple: true`), `score`, `text`, `number`, `date`, `box`, and `boxes` questions.
d1 scores require 2 to 10 levels.
The native backend preserves set attention, answer-token pooling, temperature
calibration, and multi-choice calibration. Requests are processed independently;
questions within each request are packed in the checkpoint's groups of 16.

Use `--attention auto` or `--attention eager`. Attention uses bounded query chunks;
Flash Attention is not yet supported for d1's set masks. `--max-model-len` defaults
to 16384. Media decoding uses FFmpeg/FFprobe, as for Clef.

The optional CPU parity test compares text, structured JSON, multi-choice,
question grouping, images, video, extraction, numeric grids, and boxes against the
pinned reference checkpoint:

```bash
SYS1_D1_PATH=/path/to/d1-2B cargo test d1_reference_parity -- --ignored
```

Expected outputs are stored in `src/models/fixtures/d1_reference.json`; the
optional development script `scripts/d1_reference.py` regenerates them using
LiquidAI's pinned Python implementation. Media fixtures include synthetic frames
and the COCO cats-and-remotes example;
FFmpeg resizing can differ slightly from the reference's Pillow/Torchvision
interpolation on other images.

## LiquidAI d1-Omni-2B

Omni reuses the native d1 text, vision, and decision implementation and adds a
native audio encoder. It retains image and video support. Run on CPU with:

```bash
sys1 --model-id LiquidAI/d1-Omni-2B --dtype f32
```

The `audio` array accepts the same URL, base64, data URL, byte-array, or source
object forms as `images` and `videos`. `state` may be `null` when media supply the
state. Audio is decoded with FFmpeg/FFprobe, averaged to mono, resampled to 16 kHz,
and converted to the checkpoint's log-mel features in Rust. Other models reject
audio inputs. Example request body:

```json
{
  "state": null,
  "audio": [{"url": "https://example.com/call.wav"}],
  "questions": {
    "speech": {"type": "noul", "instructions": "Does someone speak in the recording?"}
  }
}
```

A spoken question uses an encoded audio source in `instructions`, for example
`{"type":"audio","base64":"..."}` or `{"type":"audio","url":"https://example.com/question.wav"}`.
Local filesystem paths are not accepted through the API. Audio files are inputs
for decisions; sys1 does not generate speech or transcripts. Video soundtracks
are not automatically extracted; supply audio separately when needed.

Omni uses its own weights and calibration. Its processor caps videos at 384
sampled frames, compared with 768 for d1-2B. Both models use the same context-length
and attention options documented above. Every complete request must fit the
configured token budget, including audio and spoken questions.

```bash
SYS1_D1_OMNI_PATH=/path/to/d1-Omni-2B cargo test d1_omni_reference_parity -- --ignored
```

The optional `scripts/d1_omni_reference.py` regenerates reference outputs for text,
images, video, audio, resampling, multiple clips, mixed image/audio input, and spoken
questions. Python is used only to generate validation fixtures.

## Typed d1 answers

The following types are available on both d1-2B and d1-Omni-2B. Other backends
reject them explicitly. Existing `noul`, `choice`, and `score` readouts remain
unchanged.

| Question | Required configuration | Answer fields |
| --- | --- | --- |
| `text` | `instructions` | `text`, `span`, `found`, optional `path` |
| `number` extraction | `instructions`, optional `unit` | `number`, `text`, `span`, `found`, `unit`, optional `path` |
| `date` | `instructions` | ISO `date`, `text`, `span`, `found`, optional `path` |
| `number` estimation | `instructions`, `range: [low, high]`, optional `step` and `unit` | median `number`, central 80% `interval`, `unit` |
| `box` | `instructions`, exactly one image | normalized `box`, `found`, `confidence` |
| `boxes` | `instructions`, exactly one image, optional integer `max` from 1 to 16 | `boxes`, `count`, `at_least`, `presence` |

All typed questions require nonempty instructions. `step` must be positive and
divide the range; estimates use at most 128 equally spaced values, even when a
finer step is requested. Without `step`, the grid has 128 values. Use `[0, 10]`
with `step: 1` for integer counts. Estimation reads the model's numeric/code-token
distribution; it does not extract a written number from the state.

Extraction uses start/end pointers over the state and the checkpoint's presence
calibration. Missing values return null values and spans. `found` describes span
presence: an ambiguous written number/date can return a null parsed value while
retaining its original `text`, `span`, and `found`. Parsing applies explicit scale
words, currency/locale evidence and date formats, without unit conversion or
inventing missing date components.

Spans are half-open Unicode character offsets in the rendered state: the original
string, or JSON rendered with two-space indentation. Values inside one JSON string
are unescaped and may include an RFC6901 `path`; offsets still index the rendered
JSON. Text reads span at most 512 tokens, number/date reads at most 32.

Box edges are `top`, `bottom`, `left`, and `right`, as fractions of image height
and width. Missing single boxes return `box: null` and zero confidence. Multiple
boxes are sorted by left edge, then top edge; `presence[k]` is the calibrated
probability of at least k+1 instances, kept non-increasing. `at_least: true` means
all requested slots were present, so the count can be higher. These pinned
checkpoints default to 16 slots and use prompt version 1. Video-only or audio-only
box requests are rejected.

For example, the cats-and-remotes request is now accepted:

```bash
curl http://localhost:3000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": null,
    "images": ["http://images.cocodataset.org/val2017/000000039769.jpg"],
    "questions": {
      "cats": {"type": "number", "instructions": "How many cats are there?", "range": [0, 10], "step": 1},
      "remote": {"type": "box", "instructions": "Where is the remote control on the left?"},
      "every_cat": {"type": "boxes", "instructions": "Every cat.", "max": 4}
    }
  }'
```

Reference scripts enforce the checkpoint revisions used by the fixtures. Prepare
complete local reference snapshots (including their Python code) and regenerate:

```bash
hf download LiquidAI/d1-2B --revision 652f920e544ea37f81e4b0ce89b81d4ed0da90b1 --local-dir /tmp/d1-reference
hf download LiquidAI/d1-Omni-2B --revision 81222dae8bf8bdd7f903643b5b2a8803bf2e3176 --local-dir /tmp/d1-omni-reference
python scripts/d1_reference.py /tmp/d1-reference src/models/fixtures/d1_reference.json
python scripts/d1_omni_reference.py /tmp/d1-omni-reference src/models/fixtures/d1_omni_reference.json
python scripts/d1_parsing_reference.py /tmp/d1-reference src/models/fixtures/d1_parsing_reference.json
```

Then run the optional parity tests above with `SYS1_D1_PATH` and
`SYS1_D1_OMNI_PATH`. They check all answer fields, coordinates, spans, missing
values, calibration, and exact input-token counts. Probabilities use a 0.0002
absolute tolerance against CPU float32 upstream outputs. `SYS1_REFERENCE_CASE`
can select a case by name (for example `typed_json` or `typed_boxes`). Standard
`cargo test --locked` also runs parser oracles, validation, pointer and box
readout tests without downloading model weights. GPU parity remains unverified.

To request counts and draw the predicted boxes over an image, run the standalone
client script (its Pillow dependency is installed automatically by uv):

```bash
uv run scripts/d1_image_boxes.py
uv run scripts/d1_image_boxes.py photo.jpg --output annotated.png --show
uv run scripts/d1_image_boxes.py 'https://example.com/parking.jpg' --objects cars --locate 'the red car' --max-count 30
```

It defaults to the COCO cats image and `http://localhost:3001/v1/systemone`.
Use `--endpoint` for another server, `--model` for a served model name, and
`--response-json result.json` to save the complete response. Cyan boxes mark
all requested objects; the orange box marks the additional `--locate` target.
