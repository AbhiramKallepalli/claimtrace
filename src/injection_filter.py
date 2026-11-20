"""A small filter that flags text that looks like a hidden instruction.

It only FLAGS text (for the log and for review). It never blocks anything,
and the final decision never depends on it. The decision rule in harness.py
is the real defense. This is an extra layer.
"""
import re

PATTERNS = {
    "ignore_instructions": r"\b(ignore|disregard|forget|override|skip)\b.{0,40}\b(rules?|instructions?|results?|estimator|cutoff|everything|above|previous|prior)\b",
    "role_change": r"\b(you are now|act as|from now on|new instructions?)\b",
    "fake_system": r"(<\s*/?\s*system\s*>|^\s*system\s*:|developer message|###\s*instruction)",
    "force_answer": r"\b(answer|say|output|respond|return)\b.{0,40}\b(overturned|upheld|100%)",
    "reveal": r"\b(reveal|print|show)\b.{0,40}\b(system prompt|instructions|decision log|prompt)\b",
    "tool_abuse": r"\b(do not|don't|never)\b.{0,20}search|\bnever say unsure|\bcall the tool\b",
}

_COMPILED = {name: re.compile(p, re.I | re.M | re.S) for name, p in PATTERNS.items()}


def looks_like_injection(text):
    """Return the list of pattern names that match. Empty list means nothing suspicious."""
    return [name for name, rx in _COMPILED.items() if rx.search(text or "")]