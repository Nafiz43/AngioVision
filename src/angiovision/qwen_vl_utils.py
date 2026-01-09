"""Utility helpers that mirror the reference implementation from Qwen."""
from __future__ import annotations

from typing import Dict, List

from PIL import Image


def process_vision_info(messages: List[Dict]) -> Dict[str, List[Image.Image]]:
    """Extract image and video payloads from chat messages.

    The Hugging Face Qwen processor expects images and videos to be passed as
    separate lists. Each message's ``content`` is iterated and grouped by type.
    This mirrors the helper shipped with the official Qwen demos so we avoid a
    runtime dependency on the original repository.
    """

    images: List[Image.Image] = []
    videos: List = []

    for message in messages:
        for item in message.get("content", []):
            if item.get("type") == "image" and item.get("image") is not None:
                images.append(item["image"])
            elif item.get("type") == "video" and item.get("video") is not None:
                videos.append(item["video"])

    return {"images": images, "videos": videos}
