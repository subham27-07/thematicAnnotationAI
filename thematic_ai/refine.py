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
from .calibration import CueExampleRetriever, NeighbourRetriever
from .coding_rules import CODING_GUIDANCE, CONTRASTIVE_EXAMPLES, render_cue_hints
from .pipeline import Workspace
from .prompts import build_schema, refine_messages


REFINE_RULES = """\
You are the adjudicator for a thematic coding study. A first coder has already \
proposed codes for one justification. Produce the FINAL code set, matching how \
human adjudicators actually coded this project.

Rules:
1. Usually 1 or 2 codes. Use 3 or more when the text makes that many separate \
points. Do not stack near-synonyms, but DO keep a specific code next to a \
broader one when both ideas are in the text.
2. Type of offensiveness — keep every specific idea that is named:
   - Death wish when they mention wishing someone dead, a death threat, "hope \
you die", or telling someone to die. That is not automatically Threats of \
violence. Add Threats of violence as well only when they also describe a \
threat to commit violence / kill.
   - Threats of violence for threatening to kill/harm, violence, murder.
   - racial remark when they mention racism/racist. Keep Hate speech too if \
slurs or hate speech are named.
   - Extreme offensive when they call the content extreme/severe/brutal.
   - Harm other person when they mention harm, harmful, hurtful, or intent to \
harm. Do not drop it because Threats of violence or Hate speech is present.
   - insult when they say insult/insulting/rude/name-calling and do not mention \
slurs. Do not replace insult with Hate speech.
   - Hate speech and offensive language for slurs, hate speech, homophobia, or \
generic "offensive"/"hateful" language. It does NOT absorb the specific codes \
above.
   - No offensive content when they say nothing was offensive. Do not add it \
when the characterisation is Calm and thoughtful, Normal behavior, or Personal \
opinions and thoughts. Never with an offensiveness code.
3. Patterns of behavior:
   - Consistent offensive behavior when they describe a repeated pattern \
(every, all, both, multiple, history, numerous, frequent, standard behavior, \
counts). Apply it even if Majority offesnive is also applied.
   - Majority offesnive only if they say most/majority.
   - Mix of offensive and non-offensive only if they explicitly note both kinds \
as a mix — not when a few calm posts sit next to offensive ones.
   - one message is enough (offensive) only if ONE message was itself enough. \
Not when several messages were all bad (that is Consistent).
   - Normal behavior when they call the account ordinary/normal/typical.
4. Calm and thoughtful when they call the messages calm, thoughtful, eloquent, \
genuine, supportive, considerate, uplifting, or "nice and cool". Humans never \
combined this with No offensive content — prefer Calm and thoughtful.
5. Decision outcomes: Permanent ban only if they name removal/termination/ban \
from the platform. Consequences beyond suspencion / contact police when they \
name police, arrest, therapy, investigation, or "more than suspension". \
"Should be suspended" is not Permanent ban and not Obvious an clear suspension.
6. Obvious an clear suspension only if they call the suspension decision \
itself obvious, clear or certain. Do not suspend is extremely rare — "no \
reason to suspend" is No offensive content.
7. You may drop a first-pass code, keep it, or replace it. ADD a code the \
first pass missed whenever the wording cues or the similar human examples \
support it. Recovering a missed specific code matters more than leaving the \
first pass unchanged.
8. Code what the participant wrote, not what you infer about the account.

""" + CODING_GUIDANCE + "\n\n" + CONTRASTIVE_EXAMPLES + "\n\nReturn JSON only."


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
    cue_examples = CueExampleRetriever(workspace.train)
    first_pass = run.label_sets
    reasons: dict[str, dict[str, str]] = {}
    for row in run.annotations.itertuples():
        reasons.setdefault(row.unit_id, {})[row.code] = str(getattr(row, "reason", "") or "")

    system_prompt = REFINE_RULES + "\n\n## CODEBOOK\n\n" + codebook.render(include_examples=True)
    schema = build_schema(codebook)
    fingerprint = config.prompt_fingerprint(system_prompt) + "-refine-v2"
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
        neighbours = (
            retriever.render_context(text, k=8, exclude={unit_id}) if text else ""
        )
        cues = cue_examples.render_context(text, exclude={unit_id}) if text else ""
        hints = render_cue_hints(text) if text else ""
        context = _refine_context(proposed, why, neighbours, cues, hints)
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
    cue_examples: str = "",
    cue_hints: str = "",
) -> str:
    if proposed:
        lines = ["FIRST-PASS CODES (review and correct):"]
        for code in proposed:
            reason = str(reasons.get(code) or "").strip()
            lines.append(f"  - {code}" + (f" — {reason}" if reason else ""))
    else:
        lines = ["FIRST-PASS CODES: (none)"]
    extra = [block for block in (cue_hints, cue_examples, neighbours) if block]
    return "\n\n".join(["\n".join(lines), *extra])
