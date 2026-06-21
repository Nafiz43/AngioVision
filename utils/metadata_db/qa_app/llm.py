"""Ollama chat model management and synthesis helper."""

import re
import sys
import logging
from typing import Any, Dict, List

try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    print("ERROR: pip install langchain-ollama")
    sys.exit(1)

from . import config
from .state import state

log = logging.getLogger(__name__)


def get_ollama() -> ChatOllama:
    """Get or initialize the Ollama ChatOllama instance (singleton pattern)."""
    if state.ollama is None:
        state.ollama = ChatOllama(model=config.DEFAULT_MODEL)
    return state.ollama


def set_model(model: str) -> None:
    """Update the global Ollama model instance to use a different model."""
    state.ollama = ChatOllama(model=model)
    log.info(f"Model set to: {model}")


def llm_call(messages: List[Dict[str, str]], think: bool = True) -> str:
    """
    Call the Ollama model with the given messages.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        think: Whether to prepend '/no_think' directive to user messages (for qwen3)

    Returns:
        The model's text response with any <think> tags stripped
    """
    ollama = get_ollama()
    lc_messages: List[Any] = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            if not think:
                content = "/no_think\n" + content
            lc_messages.append(HumanMessage(content=content))
    response = ollama.invoke(lc_messages)
    text = response.content
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()
