"""The annotation loop: prompt -> model -> validated, span-aligned rows.

The output schema is fixed and backend-neutral. A GPT-5.1 run and a Qwen run
produce byte-compatible CSVs that differ only in the `model` / `backend`
columns and the filename, so they can be concatenated or compared without any
renaming.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from .backends import BackendError, LLMBackend, get_backend
from .codebook import Codebook
from .config import RunConfig
from .prompts import build_messages, build_schema
from .spans import locate_quote

LONG_COLUMNS = [
    "unit_id",
    "user_id",
    "account",
    "unit_text",
    "theme",
    "code",
    "scope",
    "start_offset",
    "end_offset",
    "quote",
    "quote_match",
    "confidence",
    "reason",
    "backend",
    "model",
    "prompt_variant",
    "run_key",
    "annotated_at",
]

UNIT_COLUMNS = [
    "unit_id",
    "user_id",
    "account",
    "unit_text",
    "n_codes",
    "codes",
    "themes",
    "unit_note",
    "status",
    "error",
    "latency_s",
    "input_tokens",
    "output_tokens",
    "from_cache",
    "backend",
    "model",
    "prompt_variant",
    "run_key",
    "annotated_at",
]


@dataclass
class AnnotationRun:
    """Everything one run produced, plus the prompt that produced it."""

    config: RunConfig
    system_prompt: str
    annotations: pd.DataFrame
    units: pd.DataFrame
    dropped_codes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def label_sets(self) -> dict[str, set[str]]:
        """unit_id -> predicted code set, including units that got none."""
        out = {unit_id: set() for unit_id in self.units["unit_id"]}
        for row in self.annotations.itertuples():
            out.setdefault(row.unit_id, set()).add(row.code)
        return out

    @property
    def n_failed(self) -> int:
        return int((self.units["status"] != "ok").sum())

    def save(self, output_dir: Path | str | None = None, prefix: str = "annotations") -> dict[str, Path]:
        """Write the long, per-unit and wide CSVs plus the run manifest."""
        directory = Path(output_dir or self.config.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        key = self.config.run_key

        paths = {
            "annotations": directory / f"{prefix}__{key}.csv",
            "units": directory / f"{prefix}__{key}__units.csv",
            "wide": directory / f"{prefix}__{key}__wide.csv",
            "manifest": directory / f"{prefix}__{key}__manifest.json",
        }
        self.annotations.to_csv(paths["annotations"], index=False)
        self.units.to_csv(paths["units"], index=False)
        self.to_wide().to_csv(paths["wide"])
        paths["manifest"].write_text(
            json.dumps(
                {
                    "config": self.config.to_dict(),
                    "n_units": int(len(self.units)),
                    "n_annotations": int(len(self.annotations)),
                    "n_failed": self.n_failed,
                    "dropped_codes": self.dropped_codes[:50],
                    "system_prompt": self.system_prompt,
                    "written_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )
        return paths

    def to_wide(self, codes: list[str] | None = None) -> pd.DataFrame:
        """Binary unit x code matrix."""
        labels = self.label_sets
        unit_ids = list(self.units["unit_id"])
        columns = codes or sorted({c for s in labels.values() for c in s})
        data = {
            code: [int(code in labels.get(u, ())) for u in unit_ids] for code in columns
        }
        return pd.DataFrame(data, index=pd.Index(unit_ids, name="unit_id"))


class _Cache:
    """Append-only JSONL cache so an interrupted run resumes for free.

    Entries are keyed by unit *and* by a hash of the per-unit context, so a
    retrieval-augmented run does not serve answers produced under a different
    set of retrieved examples.
    """

    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()
        self.entries: dict[str, dict[str, Any]] = {}
        if enabled and path.exists():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.entries[self._key(record["unit_id"], record.get("context_hash", ""))] = record

    @staticmethod
    def _key(unit_id: str, context_hash: str) -> str:
        return f"{unit_id}|{context_hash}" if context_hash else unit_id

    def get(self, unit_id: str, context_hash: str = "") -> dict[str, Any] | None:
        return self.entries.get(self._key(unit_id, context_hash)) if self.enabled else None

    def put(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.entries[self._key(record["unit_id"], record.get("context_hash", ""))] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 1.0
    if confidence > 1.0:  # some models answer on a 0-100 scale
        confidence = confidence / 100.0
    return min(max(confidence, 0.0), 1.0)


def parse_model_answer(
    answer: dict[str, Any],
    unit_text: str,
    codebook: Codebook,
    min_confidence: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Validate one model answer into annotation rows.

    Anything that is not a real code, or that duplicates a code already
    assigned to the unit, is dropped and reported rather than written out.
    """
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen: set[str] = set()

    raw_codes = answer.get("codes")
    if isinstance(raw_codes, dict):
        raw_codes = [raw_codes]
    if not isinstance(raw_codes, list):
        raw_codes = []

    for item in raw_codes:
        if isinstance(item, str):
            item = {"code": item}
        if not isinstance(item, dict):
            dropped.append({"reason": "not_an_object", "value": str(item)[:120]})
            continue

        canonical = codebook.resolve(str(item.get("code", "")))
        if canonical is None:
            dropped.append({"reason": "unknown_code", "value": str(item.get("code"))[:120]})
            continue
        if canonical in seen:
            dropped.append({"reason": "duplicate_code", "value": canonical})
            continue

        confidence = _coerce_confidence(item.get("confidence", 1.0))
        if confidence < min_confidence:
            dropped.append({"reason": "below_min_confidence", "value": canonical})
            continue

        quote = str(item.get("quote") or "").strip()
        located = locate_quote(unit_text, quote) if quote else None
        if located:
            start, end, match_kind = located
            rows.append(
                {
                    "theme": codebook.theme_of(canonical),
                    "code": canonical,
                    "scope": "span",
                    "start_offset": start,
                    "end_offset": end,
                    "quote": unit_text[start:end],
                    "quote_match": match_kind,
                    "confidence": confidence,
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
        else:
            rows.append(
                {
                    "theme": codebook.theme_of(canonical),
                    "code": canonical,
                    "scope": "unit",
                    "start_offset": pd.NA,
                    "end_offset": pd.NA,
                    "quote": "",
                    "quote_match": "unmatched" if quote else "none",
                    "confidence": confidence,
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
        seen.add(canonical)

    note = str(answer.get("unit_note") or "").strip()
    return rows, dropped, note


def annotate_units(
    units: pd.DataFrame,
    codebook: Codebook,
    config: RunConfig,
    system_prompt: str,
    backend: LLMBackend | None = None,
    progress: Callable[[int, int], None] | None = None,
    verify_backend: bool = True,
    context_builder: Callable[[dict[str, Any]], str] | None = None,
) -> AnnotationRun:
    """Annotate every row of `units` (needs `unit_id` and `unit_text`).

    `context_builder` optionally returns per-unit text to prepend to the user
    message, which is how retrieval-augmented prompting is supported without
    rebuilding the (cached, expensive) system prompt for every unit.
    """
    for column in ("unit_id", "unit_text"):
        if column not in units.columns:
            raise ValueError(f"units frame needs a {column!r} column")

    backend = backend or get_backend(config)
    if verify_backend:
        backend.health_check()

    schema = build_schema(codebook)
    fingerprint = config.prompt_fingerprint(system_prompt)
    cache = _Cache(
        config.output_dir / "cache" / f"{config.run_key}__{fingerprint}.jsonl",
        enabled=config.use_cache,
    )

    records = units.to_dict("records")
    total = len(records)
    results: dict[str, dict[str, Any]] = {}
    completed = 0
    counter_lock = threading.Lock()

    def _tick() -> None:
        nonlocal completed
        with counter_lock:
            completed += 1
            if progress:
                progress(completed, total)

    def _work(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        unit_id = record["unit_id"]
        text = str(record.get("unit_text") or "").strip()
        context = context_builder(record) if (context_builder and text) else ""
        context_hash = (
            hashlib.sha256(context.encode("utf-8")).hexdigest()[:12] if context else ""
        )

        cached = cache.get(unit_id, context_hash)
        if cached is not None:
            _tick()
            return unit_id, {**cached, "from_cache": True}

        if not text:
            payload = {
                "unit_id": unit_id,
                "context_hash": context_hash,
                "answer": {"codes": [], "unit_note": ""},
                "status": "empty_text",
                "error": "",
                "latency_s": 0.0,
                "usage": {},
            }
            cache.put(payload)
            _tick()
            return unit_id, {**payload, "from_cache": False}

        try:
            answer, response = backend.complete_json(
                build_messages(system_prompt, text, context), schema
            )
            payload = {
                "unit_id": unit_id,
                "context_hash": context_hash,
                "answer": answer,
                "status": "ok",
                "error": "",
                "latency_s": round(response.latency_s, 3),
                "usage": response.usage,
            }
            cache.put(payload)
        except BackendError as exc:
            payload = {
                "unit_id": unit_id,
                "context_hash": context_hash,
                "answer": {"codes": [], "unit_note": ""},
                "status": "failed",
                "error": str(exc)[:500],
                "latency_s": 0.0,
                "usage": {},
            }
            # Failures are not cached: a re-run should retry them.
        _tick()
        return unit_id, {**payload, "from_cache": False}

    workers = max(1, int(config.max_workers))
    if workers == 1:
        for record in records:
            unit_id, payload = _work(record)
            results[unit_id] = payload
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_work, record) for record in records]
            for future in as_completed(futures):
                unit_id, payload = future.result()
                results[unit_id] = payload

    return _assemble(records, results, codebook, config, system_prompt)


def _assemble(
    records: Iterable[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    codebook: Codebook,
    config: RunConfig,
    system_prompt: str,
) -> AnnotationRun:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {
        "backend": config.backend,
        "model": config.model,
        "prompt_variant": config.prompt_variant,
        "run_key": config.run_key,
        "annotated_at": stamp,
    }

    long_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    all_dropped: list[dict[str, Any]] = []

    for record in records:
        unit_id = record["unit_id"]
        text = str(record.get("unit_text") or "")
        payload = results.get(unit_id, {"answer": {"codes": []}, "status": "missing", "error": "no result"})

        parsed, dropped, note = parse_model_answer(
            payload.get("answer") or {}, text, codebook, config.min_confidence
        )
        for entry in dropped:
            all_dropped.append({"unit_id": unit_id, **entry})

        for row in parsed:
            long_rows.append(
                {
                    "unit_id": unit_id,
                    "user_id": record.get("user_id", ""),
                    "account": record.get("account", ""),
                    "unit_text": text,
                    **row,
                    **common,
                }
            )

        usage = payload.get("usage") or {}
        codes = [row["code"] for row in parsed]
        unit_rows.append(
            {
                "unit_id": unit_id,
                "user_id": record.get("user_id", ""),
                "account": record.get("account", ""),
                "unit_text": text,
                "n_codes": len(codes),
                "codes": "; ".join(codes),
                "themes": "; ".join(dict.fromkeys(row["theme"] for row in parsed if row["theme"])),
                "unit_note": note,
                "status": payload.get("status", "ok"),
                "error": payload.get("error", ""),
                "latency_s": payload.get("latency_s", 0.0),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "from_cache": bool(payload.get("from_cache", False)),
                **common,
            }
        )

    annotations = pd.DataFrame(long_rows, columns=LONG_COLUMNS)
    units_out = pd.DataFrame(unit_rows, columns=UNIT_COLUMNS)
    return AnnotationRun(
        config=config,
        system_prompt=system_prompt,
        annotations=annotations,
        units=units_out,
        dropped_codes=all_dropped,
    )
