"""Terminal UI helpers: colors, banners, prompts, report boxes."""

from __future__ import annotations

import os
import textwrap

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
DIM = "\033[2m"


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


def banner(msg: str) -> None:
    try:
        width = min(80, os.get_terminal_size().columns)
    except OSError:
        width = 80
    print(c("─" * width, CYAN))
    print(c(f"  {msg}", BOLD + CYAN))
    print(c("─" * width, CYAN))


def section(msg: str) -> None:
    print(f"\n{c('▶', GREEN)} {c(msg, BOLD)}\n")


def info(msg: str) -> None:
    print(c(f"  ℹ  {msg}", DIM))


def warn(msg: str) -> None:
    print(c(f"  ⚠  {msg}", YELLOW))


def err(msg: str) -> None:
    print(c(f"  ✗  {msg}", RED))


def success(msg: str) -> None:
    print(c(f"  ✔  {msg}", GREEN))


def prompt(msg: str) -> str:
    return input(c(f"\n  ❯  {msg}: ", BOLD + MAGENTA))


def print_report_box(title: str, text: str) -> None:
    bar = "═" * (74 - len(title) - 2)
    print(c(f"  ╔═ {title} {bar}", CYAN))
    for line in textwrap.wrap(text.strip() or "(empty)", width=72):
        print(c("  ║  ", CYAN) + line)
    print(c("  ╚" + "═" * 74, CYAN))
    print()
