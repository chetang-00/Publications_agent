"""Split page text into overlapping chunks that never cross a page boundary.

Sizes are in characters (≈4 characters per token): 3,200 characters ≈ 800 tokens. Splits prefer
paragraph breaks, then sentence ends, and only cut inside a sentence when one is longer than a chunk.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.parsing import ParsedPage

DEFAULT_MAX_CHARS = 3200
DEFAULT_OVERLAP_CHARS = 400
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    index: int
    page: int | None
    text: str
    token_count: int


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _units(text: str, max_chars: int) -> list[str]:
    """Paragraphs, broken into sentences or hard slices only where they exceed max_chars."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            while len(sentence) > max_chars:
                units.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if sentence:
                units.append(sentence)
    return units


def _overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    tail = text[-overlap_chars:]
    if len(text) > overlap_chars and " " in tail:
        tail = tail[tail.index(" ") + 1 :]  # start on a word boundary
    return tail


def _split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for unit in _units(text, max_chars):
        if not current:
            current = unit
        elif len(current) + 1 + len(unit) <= max_chars:
            current = f"{current} {unit}"
        else:
            pieces.append(current)
            tail = _overlap_tail(current, overlap_chars)
            current = f"{tail} {unit}" if tail and len(tail) + 1 + len(unit) <= max_chars else unit
    if current:
        pieces.append(current)
    return pieces


def chunk_pages(
    pages: Sequence[ParsedPage],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for piece in _split(page.text, max_chars, overlap_chars):
            chunks.append(
                Chunk(index=len(chunks), page=page.page, text=piece, token_count=estimate_tokens(piece))
            )
    return chunks
