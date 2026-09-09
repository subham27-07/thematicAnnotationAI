"""Prompt construction and the JSON schema every backend is constrained to.

Two variants are supported:

``codebook_only``
    The codebook plus the illustrative examples that ship inside it. No human
    annotations are shown, so this measures how far the written codebook alone
    carries a model.

``few_shot``
    The same codebook plus k worked examples taken from adjudicated human
    annotations. Examples must come from the training split only.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Literal

import pandas as pd

from .codebook import Codebook

PromptVariant = Literal["codebook_only", "few_shot"]

TASK_BRIEF = """\
You are an experienced qualitative researcher performing deductive thematic \
coding for a study on social media content moderation.

Participants reviewed a suspicious account, decided whether to suspend it, and \
wrote a short free-text justification. Your job is to assign codes from a fixed \
codebook to one such justification.

Rules:
1. Use ONLY the codes listed in the codebook below, spelled exactly as given. \
Never invent, merge or rename a code.
2. Assign every code the text supports. Most justifications get one or two \
codes; some get none, and a few get four or more.
3. Code what the participant actually wrote, not what you infer about the \
account. If the participant does not say it, do not code it.
4. A code is justified by evidence in the text. For each code, quote the \
shortest span of the justification that supports it, copied verbatim, \
character for character. If the code is supported by the justification as a \
whole rather than one span, return an empty quote.
5. Respect each code's "do not apply when" guidance, and prefer the most \
specific code available over a general one.
6. If the justification is too short or vague to support any code, return an \
empty list of codes.
7. Give each code a confidence between 0 and 1 reflecting how clearly the text \
supports it.

Return JSON only, matching the required schema."""

OUTPUT_CONTRACT = """\
Respond with a JSON object shaped like:
{"codes": [{"code": "<exact code name>", "quote": "<verbatim span or empty string>",
            "confidence": 0.0, "reason": "<one short clause>"}],
 "unit_note": "<one sentence on anything ambiguous, or empty>"}"""


def build_schema(codebook: Codebook) -> dict[str, Any]:
    """Strict JSON schema; the code enum makes hallucinated labels impossible."""
    return {
        "type": "object",
        "properties": {
            "codes": {
                "type": "array",
                "description": "Codes supported by the justification; may be empty.",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "enum": codebook.names},
                        "quote": {
                            "type": "string",
                            "description": "Verbatim span from the justification, or empty string.",
                        },
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["code", "quote", "confidence", "reason"],
                    "additionalProperties": False,
                },
            },
            "unit_note": {"type": "string"},
        },
        "required": ["codes", "unit_note"],
        "additionalProperties": False,
    }


def select_few_shot_units(
    train_gold: pd.DataFrame,
    adjudicated: pd.DataFrame,
    k: int,
    seed: int = 20260909,
) -> list[str]:
    """Greedily pick k training units that between them cover the most codes.

    Sampling at random would show the model half a dozen variations of "hate
    speech" and never a single rare code, so coverage is maximised explicitly.
    """
    if k <= 0 or train_gold.empty:
        return []

    pool = train_gold.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    labelled = pool[pool["n_gold_codes"] > 0]
    # A short, vague justification that humans left uncoded teaches restraint;
    # a long uncoded one just looks like a labelling mistake to the model.
    empty = pool[(pool["n_gold_codes"] == 0) & (pool["unit_text"].str.len() < 60)]

    chosen: list[str] = []
    covered: set[str] = set()
    remaining = {row.unit_id: set(row.gold_codes) for row in labelled.itertuples()}

    # Reserve one slot for a legitimately uncoded justification when one exists.
    budget = k - 1 if (len(empty) and k > 2) else k

    while remaining and len(chosen) < budget:
        unit_id, gain = max(
            remaining.items(),
            key=lambda kv: (len(kv[1] - covered), -len(kv[1])),
        )
        if not (gain - covered) and len(covered) >= 1:
            break  # nothing new to teach; stop early rather than pad
        chosen.append(unit_id)
        covered |= gain
        remaining.pop(unit_id)

    # Top up with the shortest remaining units so the prompt stays compact.
    if len(chosen) < budget and remaining:
        leftover = sorted(remaining, key=lambda u: len(remaining[u]))
        chosen.extend(leftover[: budget - len(chosen)])

    if budget < k and len(empty):
        chosen.append(empty.iloc[0]["unit_id"])

    return chosen


def render_few_shot(
    unit_ids: Iterable[str],
    gold: pd.DataFrame,
    adjudicated: pd.DataFrame,
) -> str:
    """Format gold units as input/output demonstrations."""
    texts = gold.set_index("unit_id")["unit_text"].to_dict()
    quotes: dict[tuple[str, str], str] = {}
    for row in adjudicated.itertuples():
        quote = "" if pd.isna(getattr(row, "quote", None)) else str(row.quote)
        quotes.setdefault((row.unit_id, row.code), quote)

    label_sets = gold.set_index("unit_id")["gold_codes"].to_dict()

    blocks: list[str] = []
    for i, unit_id in enumerate(unit_ids, start=1):
        if unit_id not in texts:
            continue
        answer = {
            "codes": [
                {
                    "code": code,
                    "quote": quotes.get((unit_id, code), ""),
                    "confidence": 0.9,
                    "reason": "",
                }
                for code in label_sets.get(unit_id, [])
            ],
            "unit_note": "",
        }
        blocks.append(
            f"EXAMPLE {i}\nJUSTIFICATION:\n{texts[unit_id]}\n"
            f"CODING:\n{json.dumps(answer, ensure_ascii=False)}"
        )
    if not blocks:
        return ""
    return (
        "## WORKED EXAMPLES (human-coded and adjudicated)\n\n"
        + "\n\n".join(blocks)
        + "\n\nFollow the same style: exact code names, verbatim quotes, no extra codes."
    )


def build_system_prompt(
    codebook: Codebook,
    variant: PromptVariant = "codebook_only",
    few_shot_block: str = "",
    include_codebook_examples: bool = True,
) -> str:
    sections = [
        TASK_BRIEF,
        "## CODEBOOK\n\n" + codebook.render(include_examples=include_codebook_examples),
    ]
    if variant == "few_shot" and few_shot_block:
        sections.append(few_shot_block)
    sections.append(OUTPUT_CONTRACT)
    return "\n\n".join(sections)


def build_user_prompt(unit_text: str) -> str:
    return f"JUSTIFICATION:\n{unit_text}\n\nCode this justification now."


def build_messages(system_prompt: str, unit_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(unit_text)},
    ]
