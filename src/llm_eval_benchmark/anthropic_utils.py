"""Small shared helpers for talking to the Anthropic Messages API shape.

Both ``AnthropicCandidate`` (candidates.py) and ``LLMJudge``
(judge_anthropic.py) send a ``messages.create(...)`` call and need to read
the resulting text back out. Nothing here imports the real ``anthropic``
package: any object exposing the same minimal shape (``.content`` being a
list of objects with a ``.text`` attribute) works, which is what lets tests
inject lightweight fakes instead of making real network calls.
"""

from __future__ import annotations

from typing import Any


def extract_text(message: Any) -> str:
    """Extract and concatenate the text blocks of an Anthropic Messages API response."""
    blocks = getattr(message, "content", None) or []
    parts = []
    for block in blocks:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)
