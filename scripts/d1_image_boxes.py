#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=11,<14"]
# ///
"""Ask sys1 about an image and save its predicted bounding boxes.

    uv run scripts/d1_image_boxes.py
    uv run scripts/d1_image_boxes.py photo.jpg --output boxes.png
    uv run scripts/d1_image_boxes.py https://example.com/parking.jpg \
        --objects cars --locate 'the red car' --max-count 30

Requires a running d1-2B or d1-Omni-2B sys1 server. No model weights are loaded
by this script. The decoded image is uploaded as PNG, so the server predicts on
exactly the pixels used for drawing (including EXIF orientation correction).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont, ImageOps

DEFAULT_IMAGE = "http://images.cocodataset.org/val2017/000000039769.jpg"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE, help="Image URL or local path (default: COCO cats)")
    parser.add_argument("--endpoint", default="http://localhost:3001/v1/systemone", help="sys1 endpoint URL")
    parser.add_argument("--model", help="Optional served model name; omitted uses the server default")
    parser.add_argument("--objects", default="cats", help="Objects to count and localize (default: cats)")
    parser.add_argument("--locate", default="the remote control on the left", help="Description of one additional object")
    parser.add_argument("--max-boxes", type=int, choices=range(1, 17), default=4, metavar="1..16")
    parser.add_argument("--max-count", type=int, default=10, help="Upper bound of the count estimate")
    parser.add_argument("--output", type=Path, default=Path("annotated.png"), help="Annotated PNG path")
    parser.add_argument("--response-json", type=Path, help="Also save the complete API response")
    parser.add_argument("--timeout", type=float, default=180, help="HTTP timeout in seconds")
    parser.add_argument("--show", action="store_true", help="Open the saved image in the default viewer")
    args = parser.parse_args()
    if args.max_count < 1:
        parser.error("--max-count must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")
    return args


def load_image(source: str, timeout: float) -> Image.Image:
    if urlsplit(source).scheme.lower() in {"http", "https"}:
        request = Request(source, headers={"User-Agent": "sys1-image-boxes/1.0"})
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
    else:
        data = Path(source).read_bytes()
    with Image.open(io.BytesIO(data)) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def predict(image: Image.Image, args: argparse.Namespace) -> dict:
    png = io.BytesIO()
    image.save(png, format="PNG")
    payload = {
        "state": None,
        "images": [{"base64": base64.b64encode(png.getvalue()).decode("ascii")}],
        "questions": {
            "count": {"type": "number", "instructions": f"How many {args.objects} are there?",
                      "range": [0, args.max_count], "step": 1},
            "target": {"type": "box", "instructions": f"Where is {args.locate}?"},
            "instances": {"type": "boxes", "instructions": f"Every instance of {args.objects}.", "max": args.max_boxes},
        },
    }
    if args.model:
        payload["model"] = args.model
    request = Request(args.endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=args.timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise ValueError(f"Unexpected sys1 response: {result!r}")
    return result


def draw_box(image: Image.Image, answer: dict, label: str, color: str, font: ImageFont.ImageFont) -> bool:
    box = answer.get("box")
    if box is None:
        return False
    if not isinstance(box, dict):
        raise ValueError(f"Invalid box for {label}: {box!r}")
    edges = [float(box[key]) for key in ("left", "top", "right", "bottom")]
    if not all(math.isfinite(edge) for edge in edges):
        raise ValueError(f"Non-finite box for {label}")
    left, top, right, bottom = [min(1.0, max(0.0, edge)) for edge in edges]
    if left >= right or top >= bottom:
        raise ValueError(f"Inverted or empty box for {label}: {box!r}")
    width, height = image.size
    # Upstream coordinates are fractions of width/height, not pixel coordinates.
    xy = (min(width - 1, round(left * width)), min(height - 1, round(top * height)),
          min(width - 1, round(right * width)), min(height - 1, round(bottom * height)))
    draw = ImageDraw.Draw(image)
    draw.rectangle(xy, outline=color, width=max(2, width // 200))
    if answer.get("confidence") is not None:
        label += f" | conf {answer['confidence']:.0%}"
    bounds = draw.textbbox((0, 0), label, font=font)
    text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x = max(0, min(xy[0], width - text_width - 8))
    y = max(0, xy[1] - text_height - 8)
    draw.rectangle((x, y, x + text_width + 8, y + text_height + 8), fill=color)
    draw.text((x + 4 - bounds[0], y + 4 - bounds[1]), label, font=font, fill="#111827")
    return True


def annotate(image: Image.Image, result: dict, args: argparse.Namespace) -> Image.Image:
    image = image.copy()
    answers = result["answers"]
    font = ImageFont.load_default(size=max(12, min(24, image.width // 40)))
    instances = answers["instances"]
    for index, box in enumerate(instances.get("boxes", []), 1):
        draw_box(image, box, f"{args.objects} #{index}", "#22d3ee", font)
    draw_box(image, answers["target"], args.locate, "#fb923c", font)
    count = answers["count"].get("number")
    count_label = "unknown" if count is None else f"{count:g}"
    count_lower_bound = "+" if instances.get("at_least") else ""
    lines = [f"{args.objects}: {count_label} | localized: {instances.get('count', 0)}{count_lower_bound}"]
    if answers["target"].get("box") is None:
        lines.append(f"Not found: {args.locate}")
    # Add the caption below the original image so normalized box positions stay aligned.
    line_height = max(20, font.getbbox("Ag")[3] + 8)
    canvas = Image.new("RGB", (image.width, image.height + line_height * len(lines) + 12), "#111827")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, text in enumerate(lines):
        draw.text((8, image.height + 6 + index * line_height), text, font=font, fill="white")
    return canvas


def main() -> int:
    args = arguments()
    try:
        image = load_image(args.image, args.timeout)
        result = predict(image, args)
        annotated = annotate(image, result, args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(args.output, format="PNG")
        if args.response_json:
            args.response_json.parent.mkdir(parents=True, exist_ok=True)
            args.response_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{args.objects}: {result['answers']['count'].get('number')}")
        print(f"Saved {args.output.resolve()}")
        if args.show:
            annotated.show()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        print(f"HTTP {error.code} from {error.url}: {detail}", file=sys.stderr)
        return 1
    except (OSError, URLError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
