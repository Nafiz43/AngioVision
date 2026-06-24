"""Image encoding utilities."""

import base64
from pathlib import Path


def encode_image_base64(image_path) -> str:
    """Return the base64-encoded contents of *image_path*.

    Accepts both ``str`` and ``pathlib.Path`` arguments.
    """
    if isinstance(image_path, Path):
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
