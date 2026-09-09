"""Codebook loading, rendering and code-name normalisation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

UNTHEMED = "Uncategorised"


def _clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _normalise(name: str) -> str:
    """Loose key used to match a model's code string to a real code."""
    return re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()


@dataclass(frozen=True)
class Code:
    code: str
    theme: str
    definition: str
    inclusion: str
    exclusion: str
    example: str

    @property
    def display_theme(self) -> str:
        return self.theme or UNTHEMED


class Codebook:
    """The 38 codes the annotator is allowed to use, plus prompt rendering."""

    def __init__(self, codes: list[Code]):
        if not codes:
            raise ValueError("codebook is empty")
        self.codes = codes
        self._by_name = {c.code: c for c in codes}
        self._by_norm: dict[str, str] = {}
        for c in codes:
            # Code names are unique but two differ only in case ("positive
            # content" vs the "Positive content" theme), so keep first-wins.
            self._by_norm.setdefault(_normalise(c.code), c.code)

    def __len__(self) -> int:
        return len(self.codes)

    def __iter__(self):
        return iter(self.codes)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    @property
    def names(self) -> list[str]:
        return [c.code for c in self.codes]

    @property
    def themes(self) -> list[str]:
        seen: list[str] = []
        for c in self.codes:
            if c.display_theme not in seen:
                seen.append(c.display_theme)
        return seen

    def theme_of(self, code: str) -> str:
        entry = self._by_name.get(code)
        return entry.theme if entry else ""

    def resolve(self, raw: str, fuzzy: bool = True) -> str | None:
        """Map a model-produced string onto a canonical code name.

        Returns None when the string cannot be tied to a real code, which is
        how hallucinated labels get dropped instead of silently polluting the
        output.
        """
        raw = _clean(raw)
        if not raw:
            return None
        if raw in self._by_name:
            return raw
        norm = _normalise(raw)
        if norm in self._by_norm:
            return self._by_norm[norm]
        if not fuzzy:
            return None
        match = get_close_matches(norm, list(self._by_norm), n=1, cutoff=0.9)
        return self._by_norm[match[0]] if match else None

    def render(self, include_examples: bool = True) -> str:
        """The codebook block that goes into the system prompt."""
        lines: list[str] = []
        for theme in self.themes:
            lines.append(f"### THEME: {theme}")
            for code in self.codes:
                if code.display_theme != theme:
                    continue
                lines.append(f"- CODE: {code.code}")
                if code.definition:
                    lines.append(f"  definition: {code.definition}")
                if code.inclusion:
                    lines.append(f"  apply when: {code.inclusion}")
                if code.exclusion:
                    lines.append(f"  do not apply when: {code.exclusion}")
                if include_examples and code.example:
                    lines.append(f"  example: {code.example}")
            lines.append("")
        return "\n".join(lines).strip()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.codes])


def load_codebook(path: str | Path, drop_archived: bool = True) -> Codebook:
    frame = pd.read_csv(path)
    required = {"theme", "code", "definition"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"codebook is missing columns: {sorted(missing)}")
    if drop_archived and "archived" in frame.columns:
        archived = frame["archived"].astype(str).str.lower().isin({"true", "1", "yes"})
        frame = frame[~archived]

    codes: list[Code] = []
    for _, row in frame.iterrows():
        name = _clean(row.get("code"))
        if not name:
            continue
        codes.append(
            Code(
                code=name,
                theme=_clean(row.get("theme")),
                definition=_clean(row.get("definition")),
                inclusion=_clean(row.get("inclusion")),
                exclusion=_clean(row.get("exclusion")),
                example=_clean(row.get("example")),
            )
        )
    return Codebook(codes)
