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
            "Be sparing. Assign a code only where the participant makes that point "
            "explicitly. Assigning more than 3 codes is almost always wrong.",
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

    # Pairs that are semantically adjacent (same theme) and reasonably common,
    # yet never co-occur. Those are the choices the model keeps getting wrong by
    # hedging and applying both.
    exclusive: list[tuple[str, str]] = []
    rarely_together: list[tuple[str, str, float]] = []
    co_occurrence = Counter()
    for s in label_sets:
        for a in s:
            for b in s:
                if a < b:
                    co_occurrence[(a, b)] += 1
    frequent = [c for c in codebook.names if applied.get(c, 0) >= min_support_for_exclusivity]
    for i, a in enumerate(frequent):
        for b in frequent[i + 1 :]:
            key = (a, b) if a < b else (b, a)
            if codebook.theme_of(a) != codebook.theme_of(b):
                continue
            both = co_occurrence[key]
            if both == 0:
                exclusive.append((a, b))
                continue
            # Jaccard: how often both were applied, out of the units carrying
            # either. Low means the coders saw them as competing descriptions.
            either = applied[a] + applied[b] - both
            rate = both / either if either else 0.0
            if rate < 0.34:
                rarely_together.append((a, b, rate))
    exclusive.sort(key=lambda pair: -(applied[pair[0]] + applied[pair[1]]))
    rarely_together.sort(key=lambda triple: -(applied[triple[0]] + applied[triple[1]]))

    return CodingStats(
        n_units=n_units,
        codes_per_unit=counts,
        mean_codes=float(np.mean([len(s) for s in label_sets])) if n_units else 0.0,
        base_rates=base_rates,
        never_used=never_used,
        exclusive_pairs=exclusive,
        rarely_together=rarely_together,
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

    def retrieve(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        query = self._vectorise(_tokenise(text))
        if not query:
            return []
        scored = []
        for unit_id, vector in zip(self.unit_ids, self.vectors):
            shared = query.keys() & vector.keys()
            if shared:
                scored.append((unit_id, sum(query[t] * vector[t] for t in shared)))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]

    def render_context(self, text: str, k: int = 10, min_similarity: float = 0.03) -> str:
        """The retrieved neighbours, formatted for the user message."""
        neighbours = [(u, s) for u, s in self.retrieve(text, k) if s >= min_similarity]
        if not neighbours:
            return ""
        texts = dict(zip(self.unit_ids, self.texts))
        lines = [
            "Here are the most similar justifications from the human-coded set, with the "
            "codes the coders actually assigned. Match their level of restraint.",
            "",
        ]
        for unit_id, _ in neighbours:
            codes = self.label_sets.get(unit_id, [])
            lines.append(f'  "{texts[unit_id]}"')
            lines.append(f"    -> {'; '.join(codes) if codes else '(no codes)'}")
        return "\n".join(lines)
