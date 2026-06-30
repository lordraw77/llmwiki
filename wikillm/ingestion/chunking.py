"""Text chunking utilities.

Long documents are split into overlapping windows before embedding. Overlap
preserves context across boundaries so that a passage straddling two chunks is
still retrievable. The splitter is paragraph-aware: it prefers to break on
blank lines, then on single newlines, then on whitespace, and only as a last
resort mid-token.
"""

from __future__ import annotations

import re
from typing import List

# Split on blank lines first (paragraph boundaries), then single newlines.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_WHITESPACE_RE = re.compile(r"\s+")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split ``text`` into overlapping chunks of roughly ``chunk_size`` chars.

    The algorithm packs whole paragraphs into a chunk until adding the next
    paragraph would exceed ``chunk_size``. Paragraphs longer than ``chunk_size``
    are themselves hard-split with a sliding window. Consecutive chunks share
    ``overlap`` characters of tail/head context.

    Args:
        text: The full text to split. Leading/trailing whitespace is trimmed.
        chunk_size: Target maximum chunk length in characters (must be > 0).
        overlap: Number of trailing characters from one chunk to prepend to the
            next (must satisfy ``0 <= overlap < chunk_size``).

    Returns:
        A list of non-empty chunk strings. An empty/whitespace-only input
        yields an empty list.

    Raises:
        ValueError: If ``chunk_size`` / ``overlap`` are inconsistent.

    Example:
        >>> chunks = chunk_text("a" * 2500, chunk_size=1000, overlap=100)
        >>> [len(c) for c in chunks]
        [1000, 1000, 700]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]

    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            # Flush whatever is buffered, then hard-split the long paragraph.
            if current:
                chunks.append(current)
                current = _carry_overlap(current, overlap)
            for piece in _sliding_window(paragraph, chunk_size, overlap):
                chunks.append(piece)
            current = _carry_overlap(chunks[-1], overlap) if chunks else ""
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            tail = _carry_overlap(current, overlap)
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

    if current:
        chunks.append(current)

    # Deduplicate accidental empty results and return.
    return [chunk for chunk in chunks if chunk.strip()]


def _carry_overlap(chunk: str, overlap: int) -> str:
    """Return the trailing ``overlap`` characters of ``chunk`` (word-aligned).

    Args:
        chunk: The chunk whose tail provides the overlap.
        overlap: Desired overlap length in characters.

    Returns:
        The overlap text, trimmed to start at a word boundary when possible.
    """
    if overlap <= 0 or not chunk:
        return ""
    tail = chunk[-overlap:]
    # Avoid starting the overlap mid-word: drop up to the first whitespace.
    match = _WHITESPACE_RE.search(tail)
    if match and match.end() < len(tail):
        tail = tail[match.end():]
    return tail.strip()


def _sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Hard-split a long string into windows of ``chunk_size`` with overlap.

    Args:
        text: The (long) text to split.
        chunk_size: Window size in characters.
        overlap: Overlap between consecutive windows.

    Returns:
        A list of window strings covering the whole input.
    """
    step = max(1, chunk_size - overlap)
    windows: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        windows.append(text[start:start + chunk_size])
        start += step
    return windows
