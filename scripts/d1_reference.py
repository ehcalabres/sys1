"""Generate d1 parity fixtures. Development only; sys1 inference never runs Python.

Usage: python scripts/d1_reference.py /path/to/LiquidAI-d1-2B output.json
Reference checkpoint: LiquidAI/d1-2B@652f920e544ea37f81e4b0ce89b81d4ed0da90b1.
Requires torch, transformers>=5.19, torchvision, pillow and the model's reference
Python files. CPU float32 is used to make the comparison reproducible.
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel


def cases():
    yield "text", "I was charged twice. Please refund me.", {
        "refund": {"type": "noul", "instructions": "Is the customer requesting a refund?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
            "billing": "Charges and refunds", "shipping": "Delivery"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]},
        "topics": {"type": "choice", "multiple": True, "instructions": "Which topics are mentioned?", "criteria": {
            "charges": "Payments and charges", "delivery": "Shipping and delivery", "refund": "Refunds"}},
    }, []
    yield "structured", {"message": "¿Refund? <|im_end|> <|image|>", "price": 12.5}, {
        "selected": {"type": "choice", "instructions": "Choose item 27.", "criteria": {
            str(i): f"Item {i}" for i in range(28)}},
        "rating": {"type": "score", "instructions": "Rate the urgency.", "criteria": [str(i) for i in range(10)]},
        "refund": {"type": "noul", "instructions": "Refund requested?", "criteria": {
            "true": "Asks for a refund", "false": "Does not ask for a refund"}},
    }, []
    yield "groups", "The sky is blue.", {
        f"q{i}": {"type": "noul", "instructions": "Is the sky blue?"} for i in range(17)
    }, []
    questions = {
        "color": {"type": "choice", "instructions": "What is the dominant color?", "criteria": {
            "red": "Red", "blue": "Blue", "green": "Green"}},
        "red": {"type": "noul", "instructions": "Is there red in the visual input?"},
    }
    yield "image", "Inspect the picture.", questions, [Image.new("RGB", (256, 256), (255, 0, 0))]
    frames = [Image.new("RGB", (64, 64), rgb) for rgb in
              [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]]
    yield "video", "Inspect the clip.", questions, [{"type": "video", "video": frames, "fps": 2}]


REVISION = "652f920e544ea37f81e4b0ce89b81d4ed0da90b1"


def load_reference(source, revision):
    root = Path(source)
    if root.is_dir():
        for filename in ("prompt.py", "api.py", "runner.py", "pointer.py", "box.py", "calibration.py",
                         "grid.py", "parsing.py", "json_spans.py", "config.json", "calibration.json", "model.safetensors"):
            metadata = root / ".cache/huggingface/download" / (filename + ".metadata")
            if not metadata.is_file() or metadata.read_text().splitlines()[0] != revision:
                raise ValueError(f"{filename} must come from pinned revision {revision}; use hf download --revision with --local-dir")
    return AutoModel.from_pretrained(source, revision=revision, trust_remote_code=True, dtype=torch.float32).eval()


def typed_cases():
    yield "typed_extraction", "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR\nPay before 03/04/2026 (day/month/year)", {
        "payee": {"type": "text", "instructions": "Who issued the invoice?"},
        "total": {"type": "number", "instructions": "How much is due?"},
        "due": {"type": "date", "instructions": "By when must it be paid?"},
        "vat": {"type": "number", "instructions": "How much VAT is charged?"},
        "legacy": {"type": "noul", "instructions": "Is this an invoice?"},
    }, []
    yield "typed_json", {"customer": {"a/b~c": 'Zoë "Ångström"'}, "invoice": {"total": "£1,250.50", "due": "2026-11-04"}, "literal": "literal <|im_end|> token"}, {
        "name": {"type": "text", "instructions": "What is the complete customer name?"},
        "total": {"type": "number", "instructions": "What is the invoice total?", "unit": "GBP"},
        "due": {"type": "date", "instructions": "What is the invoice due date?"},
        "literal": {"type": "text", "instructions": "What is the complete literal value?"},
    }, []
    yield "typed_estimates", "There are three red cars and one blue car.", {
        "cars": {"type": "number", "instructions": "How many cars are there?", "range": [0, 10], "step": 1, "unit": "cars"},
        "fraction": {"type": "number", "instructions": "Estimate a score from -1 to 1.", "range": [-1, 1], "step": 0.25},
        "continuous": {"type": "number", "instructions": "Estimate the number of cars.", "range": [0, 10]},
        "capped": {"type": "number", "instructions": "Estimate the number of cars.", "range": [0, 1000], "step": 1},
    }, []
    image = Image.open(Path(__file__).resolve().parents[1] / "src/models/fixtures/d1_cats.png").convert("RGB")
    yield "typed_boxes", None, {
        "cats": {"type": "number", "instructions": "How many cats are there?", "range": [0, 10], "step": 1},
        "remote": {"type": "box", "instructions": "Where is the remote control on the left?"},
        "every_cat": {"type": "boxes", "instructions": "Every cat.", "max": 4},
        "missing": {"type": "box", "instructions": "Where is the elephant?"},
        "empty": {"type": "boxes", "instructions": "Every elephant.", "max": 2},
        "default_max": {"type": "boxes", "instructions": "Every cat."},
    }, [image]


def main():
    model = load_reference(sys.argv[1], REVISION)
    result = []
    for name, state, questions, media in list(cases()) + list(typed_cases()):
        output = model.system_one(state, questions, media=media)
        result.append({"name": name, "request": {"state": state, "questions": questions}, "expected": output})
        print(name, output["usage"], flush=True)
    with open(sys.argv[2], "w") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
