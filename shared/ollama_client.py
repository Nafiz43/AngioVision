"""Thin Ollama API wrappers (chat and generate endpoints)."""

from typing import Any, Dict, List, Optional

import requests


def ollama_chat_with_images(
    prompt: str,
    images_b64: List[str],
    model: str,
    url: str = "http://localhost:11434/api/chat",
    timeout: int = 180,
) -> str:
    """Send a chat request to Ollama with one or more base64-encoded images."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def ollama_chat(
    prompt: str,
    model: str,
    url: str = "http://localhost:11434/api/chat",
    timeout: int = 180,
    images_b64: Optional[List[str]] = None,
) -> str:
    """Send a chat request to Ollama, optionally with images."""
    msg: Dict[str, Any] = {"role": "user", "content": prompt}
    if images_b64:
        msg["images"] = images_b64

    payload = {
        "model": model,
        "messages": [msg],
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def ollama_generate(
    model: str,
    prompt: str,
    image_b64: str,
    url: str = "http://localhost:11434/api/generate",
    timeout: int = 180,
) -> str:
    """Send a generate request to Ollama with a single base64-encoded image."""
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()
    except Exception:
        return "NO"
