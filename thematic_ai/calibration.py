"""Teaching a model how the humans actually applied the codebook.

A codebook says what a code *means*. It does not say how often coders reach for
it, how many codes they put on one justification, or which codes they treat as
alternatives rather than companions. Left to infer that, a model over-codes: on
this dataset GPT-5.1 applied 2.68 codes per unit against a human 1.74, and its
false positives were overwhelmingly extra codes rather than wrong ones.

Everything here is derived from the training split at run time, so it stays
correct if the codebook or the coding changes, and never sees the test split.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .codebook import Codebook
from .coding_rules import COMPETING_GROUPS, matched_codes

TOKEN = re.compile(r"[a-z']+")


def _tokenise(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


@dataclass
class CodingStats:
    """Descriptive statistics of how the humans used the codebook."""

    n_units: int
    codes_per_unit: pd.Series          # distribution, index = count
    mean_codes: float
    base_rates: pd.DataFrame           # code, n_units, rate
    never_used: list[str]
    exclusive_pairs: list[tuple[str, str]]
    rarely_together: list[tuple[str, str, float]]
    companion_pairs: list[tuple[str, str, float]]

    def render(self, codebook: Codebook, max_pairs: int = 14) -> str:
        lines = [
            "## HOW THE HUMAN CODERS ACTUALLY APPLIED THIS CODEBOOK",
            "",
            f"Across {self.n_units} justifications they coded, the average was "
            f"{self.mean_codes:.2f} codes per justification. The distribution:",
        ]
        for count, n in self.codes_per_unit.items():
            share = n / max(self.n_units, 1)
            lines.append(f"  {count} code(s): {share:.0%} of justifications")
        lines += [
            "",
            "Be sparing with extras, but do not drop a specific code whose wording "
            "is in the text. Assigning more than 4 codes is almost always wrong.",
            "",
            "How often each code was applied, as a share of justifications:",
        ]
        for row in self.base_rates.itertuples():
            marker = "  RARE - " if row.rate < 0.05 else "  "
            lines.append(f"{marker}{row.code}: {row.rate:.0%}")

        if self.never_used:
            # Deliberately softer than "never use these". An earlier, blunter
            # wording zeroed out codes that are rare rather than unused, and
            # cost more in recall than it gained in precision.
            lines += [
                "",
                "These codes did not come up in the sample above. They are rare, not "
                "forbidden: apply one where the text is a clear instance, but do not "
                "reach for it as a way of adding detail.",
                "  " + "; ".join(self.never_used),
            ]

        if self.exclusive_pairs:
            lines += [
                "",
                "The coders treated these pairs as alternatives and never applied both "
                "to the same justification. Choose whichever fits better, not both:",
            ]
            for a, b in self.exclusive_pairs[:max_pairs]:
                lines.append(f"  {a}  OR  {b}")

        if self.rarely_together:
            lines += [
                "",
                "These pairs overlap in meaning and the coders usually picked just one. "
                "Apply both only where the text makes two genuinely separate points:",
            ]
            for a, b, rate in self.rarely_together[:max_pairs]:
                lines.append(
                    f"  {a} + {b}: both applied in only {rate:.0%} of the "
                    f"justifications carrying either"
                )

        if self.companion_pairs:
            lines += [
                "",
                "These specific codes usually appear together with a broader one. If the "
                "wording for the specific code is present, apply it as well as the broader "
                "code — do not swallow the specific into the general:",
            ]
            for specific, general, rate in self.companion_pairs[:max_pairs]:
                lines.append(
                    f"  {specific} (with {general} in {rate:.0%} of its uses)"
                )
        return "\n".join(lines)


def compute_coding_stats(
    train_gold: pd.DataFrame,
    codebook: Codebook,
    min_support_for_exclusivity: int = 3,
) -> CodingStats:
    """Summarise the training split's coding behaviour."""
    label_sets = [set(codes) for codes in train_gold["gold_codes"]]
    n_units = len(label_sets)

    counts = pd.Series([len(s) for s in label_sets]).value_counts().sort_index()
    applied = Counter(code for s in label_sets for code in s)

    base_rates = pd.DataFrame(
        [{"code": c, "n_units": applied.get(c, 0), "rate": applied.get(c, 0) / max(n_units, 1)}
         for c in codebook.names]
    ).sort_values("rate", ascending=False).reset_index(drop=True)

    never_used = base_rates.loc[base_rates["n_units"] == 0, "code"].tolist()

    co_occurrence: Counter[tuple[str, str]] = Counter()
    for s in label_sets:
        for a in s:
            for b in s:
                if a < b:
                    co_occurrence[(a, b)] += 1

    competing: set[tuple[str, str]] = set()
    for group in COMPETING_GROUPS:
        members = [c for c in group if c in codebook]
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                competing.add((a, b) if a < b else (b, a))

    exclusive: list[tuple[str, str]] = []
    rarely_together: list[tuple[str, str, float]] = []
    for a, b in competing:
        if applied.get(a, 0) < min_support_for_exclusivity and applied.get(b, 0) < min_support_for_exclusivity:
            continue
        key = (a, b) if a < b else (b, a)
        both = co_occurrence[key]
        if both == 0:
            exclusive.append((a, b))
            continue
        either = applied[a] + applied[b] - both
        rate = both / either if either else 0.0
        if rate < 0.34:
            rarely_together.append((a, b, rate))
    exclusive.sort(key=lambda pair: -(applied[pair[0]] + applied[pair[1]]))
    rarely_together.sort(key=lambda triple: -(applied[triple[0]] + applied[triple[1]]))

    # Rare code that usually rides along with a more common one. Jaccard is
    # low (the common code is used alone a lot) but P(common|rare) is high, so
    # the model must not treat them as alternatives.
    companion_pairs: list[tuple[str, str, float]] = []
    frequent = [c for c in codebook.names if applied.get(c, 0) >= min_support_for_exclusivity]
    for rare in frequent:
        for common in frequent:
            if rare == common or applied[rare] > applied[common]:
                continue
            key = (rare, common) if rare < common else (common, rare)
            both = co_occurrence[key]
            if both < 2:
                continue
            p_common_given_rare = both / applied[rare]
            if p_common_given_rare >= 0.4:
                companion_pairs.append((rare, common, p_common_given_rare))
    companion_pairs.sort(key=lambda t: (-t[2], -applied[t[0]]))

    return CodingStats(
        n_units=n_units,
        codes_per_unit=counts,
        mean_codes=float(np.mean([len(s) for s in label_sets])) if n_units else 0.0,
        base_rates=base_rates,
        never_used=never_used,
        exclusive_pairs=exclusive,
        rarely_together=rarely_together,
        companion_pairs=companion_pairs,
    )


class NeighbourRetriever:
    """TF-IDF nearest neighbours over the training justifications.

    Showing the model the handful of *already coded* justifications that most
    resemble the one in front of it is far more informative than a fixed set of
    examples: for near-duplicate phrasings — and this corpus has many — it turns
    the task into "copy what the humans did here".
    """

    def __init__(self, train_gold: pd.DataFrame):
        self.unit_ids = list(train_gold["unit_id"])
        self.texts = list(train_gold["unit_text"])
        self.label_sets = {
            row.unit_id: list(row.gold_codes) for row in train_gold.itertuples()
        }

        documents = [_tokenise(t) for t in self.texts]
        document_frequency = Counter(term for doc in documents for term in set(doc))
        n = max(len(documents), 1)
        self.idf = {
            term: np.log((n + 1) / (df + 1)) + 1.0 for term, df in document_frequency.items()
        }
        self.vectors = [self._vectorise(doc) for doc in documents]

    def _vectorise(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        vector = {t: (c / len(tokens)) * self.idf.get(t, 0.0) for t, c in counts.items()}
        norm = np.sqrt(sum(v * v for v in vector.values()))
        return {t: v / norm for t, v in vector.items()} if norm else {}

    def retrieve(
        self, text: str, k: int = 10, exclude: set[str] | None = None
    ) -> list[tuple[str, float]]:
        query = self._vectorise(_tokenise(text))
        if not query:
            return []
        skip = exclude or set()
        scored = []
        for unit_id, vector in zip(self.unit_ids, self.vectors):
            if unit_id in skip:
                continue
            shared = query.keys() & vector.keys()
            if shared:
                scored.append((unit_id, sum(query[t] * vector[t] for t in shared)))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]

    def render_context(
        self,
        text: str,
        k: int = 10,
        min_similarity: float = 0.03,
        exclude: set[str] | None = None,
    ) -> str:
        """The retrieved neighbours, formatted for the user message."""
        neighbours = [
            (u, s) for u, s in self.retrieve(text, k, exclude=exclude) if s >= min_similarity
        ]
        if not neighbours:
            return ""
        texts = dict(zip(self.unit_ids, self.texts))
        lines = [
            "Here are the most similar justifications from the human-coded set, with the "
            "codes the coders actually assigned. Match both their restraint and their "
            "specific codes — if a neighbour with similar wording carries a specific "
            "code (Death wish, racial remark, Consistent offensive behavior, Calm and "
            "thoughtful, …), prefer that coding grain.",
            "",
        ]
        for unit_id, _ in neighbours:
            codes = self.label_sets.get(unit_id, [])
            snippet = texts[unit_id]
            if len(snippet) > 280:
                snippet = snippet[:277] + "..."
            lines.append(f'  "{snippet}"')
            lines.append(f"    -> {'; '.join(codes) if codes else '(no codes)'}")
        return "\n".join(lines)


class CueExampleRetriever:
    """Gold examples for codes whose wording cues fire in the query.

    Neighbour retrieval surfaces similar *phrasing*, which is dominated by the
    common codes. Cue examples surface the rare specific codes the model keeps
    swallowing (Death wish, racial remark, Calm and thoughtful, …).
    """

    def __init__(self, train_gold: pd.DataFrame):
        self._examples: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {}
        for row in train_gold.itertuples():
            text = str(row.unit_text or "")
            codes = tuple(row.gold_codes)
            for code in codes:
                self._examples.setdefault(code, []).append((row.unit_id, text, codes))
        for code, items in self._examples.items():
            items.sort(key=lambda item: len(item[1]))

    def render_context(
        self,
        text: str,
        exclude: set[str] | None = None,
        per_code: int = 1,
        max_codes: int = 8,
    ) -> str:
        skip = exclude or set()
        blocks: list[str] = []
        for code in matched_codes(text)[:max_codes]:
            shown = 0
            for unit_id, example, codes in self._examples.get(code, []):
                if unit_id in skip:
                    continue
                snippet = example if len(example) <= 220 else example[:217] + "..."
                blocks.append(f'  [{code}] "{snippet}"')
                blocks.append(f"    -> {'; '.join(codes) if codes else '(no codes)'}")
                shown += 1
                if shown >= per_code:
                    break
        if not blocks:
            return ""
        return (
            "Human-coded examples of specific codes whose wording appears in this "
            "justification. Copy the coding grain; do not copy a code the current "
            "text does not actually support.\n\n" + "\n".join(blocks)
        )
