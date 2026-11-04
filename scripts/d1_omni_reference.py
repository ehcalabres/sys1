"""Generate native d1-Omni parity fixtures; optional development tool only.

Usage: python scripts/d1_omni_reference.py CHECKPOINT OUTPUT_JSON
Reference: LiquidAI/d1-Omni-2B@81222dae8bf8bdd7f903643b5b2a8803bf2e3176.
Requires the same packages as d1_reference.py. WAV fixtures are constructed from
integer samples so Rust and Python read identical waveforms.
"""
import json
import struct
import sys
import tempfile
import wave
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel
from d1_reference import cases, typed_cases, load_reference


def pcm(rate, samples):
    result = []
    for i in range(samples):
        noise = ((i * 37 + 13) % 997 - 498) * 2
        beep = (rate // 5 <= i < 2 * rate // 5 or 3 * rate // 5 <= i < 4 * rate // 5)
        tone = (9000 if (i // max(1, rate // 1000)) % 2 == 0 else -9000) if beep else 0
        result.append(noise + tone)
    return result


def wav(path, rate, samples):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(struct.pack(f"<{samples}h", *pcm(rate, samples)))


def main():
    model = load_reference(sys.argv[1], "81222dae8bf8bdd7f903643b5b2a8803bf2e3176")
    result = []
    for name, state, questions, media in list(cases()) + list(typed_cases()):
        out = model.system_one(state, questions, media=media)
        result.append({"name": name, "request": {"state": state, "questions": questions}, "expected": out})
        print(name, out["usage"], flush=True)
    qs = {
        "speech": {"type": "noul", "instructions": "Does someone speak in the recording?"},
        "sound": {"type": "choice", "instructions": "Which sound is heard?", "criteria": {
            "beeps": "Electronic beeps", "speech": "A person speaking", "music": "Music"}},
        "volume": {"type": "score", "instructions": "How loud is the audio?", "criteria": ["quiet", "medium", "loud"]},
    }
    with tempfile.TemporaryDirectory() as folder:
        for name, descriptors, mixed, spoken in [
            ("audio", [[16000, 16000]], False, False),
            ("audio_resampled", [[8000, 12000]], False, False),
            ("audio_multiple", [[16000, 16000], [16000, 8000]], False, False),
            ("mixed", [[16000, 16000]], True, False),
            ("spoken", [[16000, 16000]], False, True),
        ]:
            media = []
            for i, (rate, samples) in enumerate(descriptors):
                path = Path(folder) / f"{i}.wav"
                wav(path, rate, samples)
                media.append({"type": "audio", "path": str(path)})
            state = None
            questions = qs
            if mixed:
                media.insert(0, Image.new("RGB", (256, 256), (255, 0, 0)))
                state = "Inspect the picture and listen to the audio."
            if spoken:
                questions = {"answer": {"type": "noul", "instructions": media[0]}}
                media = []
                state = "The sky is blue."
            out = model.system_one(state, questions, media=media)
            # Persist descriptors rather than paths/large base64 media payloads.
            request = {"state": state, "questions": qs if not spoken else {
                "answer": {"type": "noul", "instructions": "SPOKEN_FIXTURE"}}}
            result.append({"name": name, "request": request, "audio_fixtures": descriptors, "expected": out})
            print(name, out["usage"], flush=True)
    with open(sys.argv[2], "w") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
