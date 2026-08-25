"""Regenerate the Sprint 2 demo images and print their SHA-256 digests.

The mock classifier is keyed by the exact SHA-256 of the decoded image bytes, so
a demo needs images whose digests are in `mock_bioclip_predictions.json`. No
image is committed to the repository - this script generates them, deterministically,
from a few lines of Pillow.

Run it from the repository root with this agent's virtual environment:

    python backend\\agents\\multimodal_recognition_agent\\fixtures\\make_demo_images.py

It writes the images to `fixtures/demo_images/` (git-ignored) and prints each
digest. If a digest ever differs from what the fixture holds - a different
Pillow build encodes PNG slightly differently - paste the printed value into
`mock_bioclip_predictions.json` under the same species, or point
`RECOGNITION_CLASSIFICATION_FIXTURE_PATH` at your own copy.

To register a photograph of your own instead, hash it and add the digest:

    python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" my_photo.jpg
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image

OUTPUT_DIR = Path(__file__).resolve().parent / "demo_images"

# (filename, size, fill colour, format). Fixed values, so the bytes - and
# therefore the digests - are reproducible.
DEMO_IMAGES: tuple[tuple[str, tuple[int, int], tuple[int, int, int], str], ...] = (
    ("demo_identified.png", (256, 256), (196, 148, 62), "PNG"),
    ("demo_uncertain.png", (256, 256), (120, 120, 124), "PNG"),
    ("demo_partial_taxonomy.png", (256, 256), (232, 236, 240), "PNG"),
    ("demo_unknown.png", (256, 256), (12, 34, 56), "PNG"),
)


def render(size: tuple[int, int], colour: tuple[int, int, int], image_format: str) -> bytes:
    image = Image.new("RGB", size, colour)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing demo images to {OUTPUT_DIR}\n")
    for name, size, colour, image_format in DEMO_IMAGES:
        raw = render(size, colour, image_format)
        (OUTPUT_DIR / name).write_bytes(raw)
        print(f"{name:<32} sha256 = {hashlib.sha256(raw).hexdigest()}")
    print(
        "\nThese digests are the keys in fixtures/mock_bioclip_predictions.json. "
        "demo_unknown.png is deliberately absent from it, so it returns not_identified."
    )


if __name__ == "__main__":
    main()
