"""Aligning model-supplied quotes back onto character offsets in the unit text.

Models paraphrase or re-punctuate quotes even when told not to, so an exact
`str.find` recovers only part of them. Falling back through case-insensitive
and whitespace-normalised matching, then approximate matching, recovers most of
the rest; anything still unmatched is recorded as a unit-level code rather than
a span, which is exactly how the human tool represents the same situation.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

WS = re.compile(r"\s+")


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace, keeping a map from normalised index to original."""
    out: list[str] = []
    index_map: list[int] = []
    previous_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if previous_space:
                continue
            out.append(" ")
            index_map.append(i)
            previous_space = True
        else:
            out.append(ch.casefold())
            index_map.append(i)
            previous_space = False
    while out and out[-1] == " ":
        out.pop()
        index_map.pop()
    return "".join(out), index_map


def locate_quote(
    text: str, quote: str, min_ratio: float = 0.82
) -> tuple[int, int, str] | None:
    """Return (start, end, match_kind) of `quote` inside `text`, or None."""
    if not text or not quote:
        return None
    quote = quote.strip()
    if not quote:
        return None

    start = text.find(quote)
    if start != -1:
        return start, start + len(quote), "exact"

    lowered = text.casefold()
    start = lowered.find(quote.casefold())
    if start != -1:
        return start, start + len(quote), "caseless"

    norm_text, index_map = _normalise_with_map(text)
    norm_quote = WS.sub(" ", quote).strip().casefold()
    if not norm_quote or not norm_text:
        return None

    start = norm_text.find(norm_quote)
    if start != -1:
        end = start + len(norm_quote) - 1
        return index_map[start], index_map[end] + 1, "normalised"

    # Approximate: slide a window the length of the quote and keep the best.
    window = len(norm_quote)
    if window > len(norm_text):
        ratio = SequenceMatcher(None, norm_text, norm_quote).ratio()
        return (0, len(text), "fuzzy") if ratio >= min_ratio else None

    best_ratio, best_start = 0.0, -1
    step = max(1, window // 8)
    for i in range(0, len(norm_text) - window + 1, step):
        ratio = SequenceMatcher(None, norm_text[i : i + window], norm_quote).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i
    if best_start < 0 or best_ratio < min_ratio:
        return None

    # Refine around the best coarse position.
    low = max(0, best_start - step)
    high = min(len(norm_text) - window, best_start + step)
    for i in range(low, high + 1):
        ratio = SequenceMatcher(None, norm_text[i : i + window], norm_quote).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i

    end = min(best_start + window - 1, len(index_map) - 1)
    return index_map[best_start], index_map[end] + 1, "fuzzy"


def span_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Character-level intersection over union of two spans."""
    a_start, a_end = a
    b_start, b_end = b
    if a_end <= a_start or b_end <= b_start:
        return 0.0
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union else 0.0
