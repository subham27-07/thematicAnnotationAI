"""Second-pass adjudication: turn a first coding into a gold-style code set.

The first pass understands the justification but hedges — it applies two related
codes where the humans kept one, and misses the pattern code they did apply.
This pass sees the first-pass codes plus the nearest already-adjudicated
examples and returns the set a human adjudicator would keep.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from .annotate import AnnotationRun, _Cache, _assemble
from .backends import BackendError, get_backend
from .calibration import NeighbourRetriever
from .pipeline import Workspace
from .prompts import build_schema, refine_messages


REFINE_RULES = """\
You are the adjudicator for a thematic coding study. A first coder has already \
proposed codes for one justification. Produce the FINAL code set, matching how \
human adjudicators actually coded this project.

Rules:
1. Usually 1 or 2 codes. Use 3 or more only when the text makes that many \
separate points. Do not stack near-synonyms.
2. Within Type of offensiveness, keep the single most specific code:
   - Threats of violence, not also Harm other person, when they mention a threat.
   - Death wish when they mention wishing someone dead; that is not a generic threat.
   - Hate speech and offensive language absorbs insult, bullying, derogatory \
remarks and Extreme offensive unless the participant used those exact ideas.
   - No offensive content when they say nothing was offensive. Do not also add \
Normal behavior, positive content, or Not offensive at all.
3. Consistent offensive behavior only if they describe a repeated pattern \
(every, multiple, history, numerous, all comments). Majority offesnive only if \
they say most/majority. Mix of offensive and non-offensive only if they \
explicitly note both kinds.
4. Calm and thoughtful only if they call the messages calm, thoughtful or \
eloquent — not merely inoffensive.
5. Do not code a suspension/ban outcome unless they name a specific one \
(temporary, permanent, police). "Should be suspended" is not Permanent ban.
6. Obvious an clear suspension only if they say the decision is obvious, clear \
or certain.
7. You may drop a first-pass code, keep it, or replace it with a better one \
from the codebook. Add a code the first pass missed only when the text and the \
similar human examples clearly support it.
8. Code what the participant wrote, not what you infer about the account.

Return JSON only."""


def refine_run(
    run: AnnotationRun,
    workspace: Workspace,
    progress: Callable[[int, int], None] | None = None,
) -> AnnotationRun:
    """Re-code every unit in `run` with the adjudication pass."""
    config = run.config
    codebook = workspace.codebook
    backend = get_backend(config)
    retriever = NeighbourRetriever(workspace.train)
    first_pass = run.label_sets
    reasons: dict[str, dict[str, str]] = {}
    for row in run.annotations.itertuples():
        reasons.setdefault(row.unit_id, {})[row.code] = str(getattr(row, "reason", "") or "")

    system_prompt = REFINE_RULES + "\n\n## CODEBOOK\n\n" + codebook.render(include_examples=False)
    schema = build_schema(codebook)
    fingerprint = config.prompt_fingerprint(system_prompt) + "-refine-v1"
    cache = _Cache(
        config.output_dir / "cache" / f"refine__{config.run_key}__{fingerprint}.jsonl",
        enabled=config.use_cache,
    )

    records = run.units.to_dict("records")

    def _work(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        unit_id = record["unit_id"]
        text = str(record.get("unit_text") or "").strip()
        proposed = sorted(first_pass.get(unit_id, set()))
        why = reasons.get(unit_id) or {}
        neighbours = retriever.render_context(text, k=8) if text else ""
        context = _refine_context(proposed, why, neighbours)
        context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()[:12]
        cached = cache.get(unit_id, context_hash)
        if cached is not None:
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
            return unit_id, {**payload, "from_cache": False}

        try:
            answer, response = backend.complete_json(
                refine_messages(system_prompt, text, context), schema
            )
            payload = {
                "unit_id": unit_id,
                "context_hash": context_hash,
                "answer": answer,
                "status": "ok",
                "error": "",
                "latency_s": response.latency_s,
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
        return unit_id, {**payload, "from_cache": False}

    results: dict[str, dict[str, Any]] = {}
    workers = max(1, int(config.max_workers))
    if workers == 1:
        for i, record in enumerate(records, start=1):
            unit_id, payload = _work(record)
            results[unit_id] = payload
            if progress:
                progress(i, len(records))
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_work, record) for record in records]
            for future in as_completed(futures):
                unit_id, payload = future.result()
                results[unit_id] = payload
                done += 1
                if progress:
                    progress(done, len(records))

    refined = _assemble(records, results, codebook, config, system_prompt)
    refined.config = config
    return refined


def _refine_context(
    proposed: list[str],
    reasons: dict[str, str],
    neighbours: str,
) -> str:
    if proposed:
        lines = ["FIRST-PASS CODES (review and correct):"]
        for code in proposed:
            reason = str(reasons.get(code) or "").strip()
            lines.append(f"  - {code}" + (f" — {reason}" if reason else ""))
    else:
        lines = ["FIRST-PASS CODES: (none)"]
    if neighbours:
        lines += ["", neighbours]
    return "\n".join(lines)
