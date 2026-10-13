"""Text chunking: a recursive character splitter (LangChain-style).

Strategy: split on the coarsest separator first (paragraphs), then recurse
into finer separators (lines, sentences, words, hard slices) until every
piece fits within `chunk_size`. Pieces are then merged greedily, carrying
a `chunk_overlap` tail forward so adjacent chunks share boundary context.
"""

from __future__ import annotations

from collections.abc import Sequence

# Ordered coarse -> fine. "" means "no separator left; slice by characters".
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


def recursive_character_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str] = _SEPARATORS,
) -> list[str]:
    """Split `text` into overlapping chunks of at most `chunk_size` chars."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    text = text.strip()
    if not text:
        return []

    # Phase 1: recursively split into pieces that each fit within chunk_size.
    pieces = _split_pieces(text, chunk_size, tuple(separators))

    # Phase 2: merge pieces into chunks, preserving overlap between neighbours.
    return _merge_pieces(pieces, chunk_size, chunk_overlap)


def _split_pieces(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    """Recursively break text along separators until pieces fit chunk_size."""
    if len(text) <= chunk_size:
        return [text]

    # Find the first (coarsest) separator that actually occurs in this text.
    for index, separator in enumerate(separators):
        if separator == "":
            # No separator applies (or none found); hard-slice by characters.
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        if separator in text:
            # Re-attach the separator so pieces remain faithful to the source.
            parts = text.split(separator)
            pieces: list[str] = []
            for position, part in enumerate(parts):
                if position < len(parts) - 1:
                    part = part + separator
                if not part:
                    continue
                if len(part) <= chunk_size:
                    pieces.append(part)
                else:
                    # Still oversized: recurse with strictly *finer* separators.
                    # Using index+1 (not index) guarantees progress and prevents
                    # a part re-appended with its own separator from looping.
                    pieces.extend(_split_pieces(part, chunk_size, separators[index + 1 :]))
            return pieces

    return [text]  # unreachable with the default separator list; safety net


def _merge_pieces(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack pieces into chunks <= chunk_size, keeping an overlap tail."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        # If adding this piece would overflow, flush the current chunk first.
        if current and current_len + len(piece) > chunk_size:
            chunks.append("".join(current).strip())
            # Seed the next chunk with a tail of the flushed one (the overlap),
            # so context flows across chunk boundaries.
            tail: list[str] = []
            tail_len = 0
            for previous in reversed(current):
                if tail_len + len(previous) > chunk_overlap:
                    break
                tail.insert(0, previous)
                tail_len += len(previous)
            current, current_len = tail, tail_len
        current.append(piece)
        current_len += len(piece)

    if current:
        chunks.append("".join(current).strip())

    # Drop empty artifacts produced by whitespace-only splits.
    return [chunk for chunk in chunks if chunk]
