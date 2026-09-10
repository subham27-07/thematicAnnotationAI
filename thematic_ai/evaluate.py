"""Scoring a model run against human gold labels.

Coding a justification is a multi-label problem: a unit can carry zero, one or
six codes. Plain accuracy is therefore misleading (a model that predicts
nothing is ~95% "accurate" per cell), so the headline numbers here are micro /
macro F1 per code, plus Cohen's kappa so a model can be placed on the same
scale as the human-human reliability table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .codebook import Codebook
from .data import label_matrix


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _cohen_kappa(gold: np.ndarray, pred: np.ndarray) -> float:
    """Kappa for two binary vectors; 1.0 when both are constant and identical."""
    n = len(gold)
    if n == 0:
        return float("nan")
    observed = float((gold == pred).mean())
    p_gold, p_pred = float(gold.mean()), float(pred.mean())
    expected = p_gold * p_pred + (1 - p_gold) * (1 - p_pred)
    if np.isclose(expected, 1.0):
        return 1.0 if np.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)


def code_level_metrics(gold: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Per-code precision / recall / F1 / kappa over the aligned matrices."""
    rows = []
    for code in gold.columns:
        g = gold[code].to_numpy()
        p = pred[code].to_numpy()
        tp = int(((g == 1) & (p == 1)).sum())
        fp = int(((g == 0) & (p == 1)).sum())
        fn = int(((g == 1) & (p == 0)).sum())
        tn = int(((g == 0) & (p == 0)).sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        rows.append(
            {
                "code": code,
                "support_gold": int(g.sum()),
                "support_pred": int(p.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": _safe_div(2 * precision * recall, precision + recall),
                "percent_agreement": _safe_div(tp + tn, len(g)),
                "cohens_kappa": _cohen_kappa(g, p),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["support_gold", "f1"], ascending=[False, False]).reset_index(drop=True)


def overall_metrics(gold: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    """Corpus-level summary of a multi-label run."""
    g = gold.to_numpy()
    p = pred.to_numpy()
    tp = int(((g == 1) & (p == 1)).sum())
    fp = int(((g == 0) & (p == 1)).sum())
    fn = int(((g == 1) & (p == 0)).sum())

    micro_p = _safe_div(tp, tp + fp)
    micro_r = _safe_div(tp, tp + fn)

    per_code = code_level_metrics(gold, pred)
    present = per_code[per_code["support_gold"] > 0]

    intersection = ((g == 1) & (p == 1)).sum(axis=1)
    union = ((g == 1) | (p == 1)).sum(axis=1)
    jaccard = np.where(union > 0, intersection / np.maximum(union, 1), 1.0)

    return {
        "n_units": int(len(gold)),
        "n_codes_evaluated": int(gold.shape[1]),
        "gold_labels": int(g.sum()),
        "pred_labels": int(p.sum()),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": _safe_div(2 * micro_p * micro_r, micro_p + micro_r),
        "macro_f1_present_codes": float(present["f1"].mean()) if len(present) else 0.0,
        "macro_kappa_present_codes": float(present["cohens_kappa"].mean()) if len(present) else 0.0,
        "weighted_f1": _safe_div(
            float((present["f1"] * present["support_gold"]).sum()),
            float(present["support_gold"].sum()),
        ),
        "exact_set_match": float((g == p).all(axis=1).mean()),
        "mean_jaccard": float(jaccard.mean()),
        "hamming_loss": float((g != p).mean()),
        "at_least_one_correct": float(((intersection > 0) | ((g.sum(axis=1) == 0) & (p.sum(axis=1) == 0))).mean()),
    }


def theme_level_metrics(
    gold: pd.DataFrame, pred: pd.DataFrame, codebook: Codebook
) -> pd.DataFrame:
    """Same scores rolled up to themes, where models agree with humans more."""
    themes = sorted({codebook.theme_of(c) or "Uncategorised" for c in gold.columns})
    gold_theme = pd.DataFrame(index=gold.index)
    pred_theme = pd.DataFrame(index=pred.index)
    for theme in themes:
        members = [c for c in gold.columns if (codebook.theme_of(c) or "Uncategorised") == theme]
        gold_theme[theme] = (gold[members].sum(axis=1) > 0).astype(int)
        pred_theme[theme] = (pred[members].sum(axis=1) > 0).astype(int)
    frame = code_level_metrics(gold_theme, pred_theme)
    return frame.rename(columns={"code": "theme"})


@dataclass
class EvaluationResult:
    overall: dict[str, float]
    per_code: pd.DataFrame
    per_theme: pd.DataFrame
    errors: pd.DataFrame
    gold_matrix: pd.DataFrame
    pred_matrix: pd.DataFrame

    def summary_row(self, **extra: Any) -> pd.Series:
        return pd.Series({**extra, **self.overall})


def error_table(
    gold_matrix: pd.DataFrame,
    pred_matrix: pd.DataFrame,
    unit_texts: dict[str, str],
) -> pd.DataFrame:
    """Unit-by-unit misses and false positives, for reading actual mistakes."""
    rows = []
    for unit_id in gold_matrix.index:
        gold_codes = set(gold_matrix.columns[gold_matrix.loc[unit_id] == 1])
        pred_codes = set(pred_matrix.columns[pred_matrix.loc[unit_id] == 1])
        if gold_codes == pred_codes:
            continue
        rows.append(
            {
                "unit_id": unit_id,
                "unit_text": unit_texts.get(unit_id, ""),
                "gold_codes": "; ".join(sorted(gold_codes)),
                "pred_codes": "; ".join(sorted(pred_codes)),
                "missed": "; ".join(sorted(gold_codes - pred_codes)),
                "spurious": "; ".join(sorted(pred_codes - gold_codes)),
                "n_agreed": len(gold_codes & pred_codes),
            }
        )
    return pd.DataFrame(rows)


def frequent_codes(train_gold: pd.DataFrame, min_support: int = 8) -> list[str]:
    """Codes that appear often enough in the training split to score stably.

    Rare codes (one or two gold rows) can only land on 0, 0.5 or 1.0 F1, so
    they drown the headline number without saying much about the model.
    """
    counts = pd.Series([code for codes in train_gold["gold_codes"] for code in codes]).value_counts()
    return [code for code, n in counts.items() if n >= min_support]


def project_labels(labels: dict[str, set[str]], codes: list[str]) -> dict[str, set[str]]:
    """Keep only `codes` in every unit's set."""
    allowed = set(codes)
    return {unit: (assigned & allowed) for unit, assigned in labels.items()}


def evaluate_run(
    run: Any,
    gold_labels: dict[str, set[str]],
    codebook: Codebook,
    restrict_to_gold_codes: bool = False,
    codes: list[str] | None = None,
) -> EvaluationResult:
    """Score an `AnnotationRun` against gold label sets.

    Only units present in both the run and the gold set are scored, so an
    interrupted run still yields valid (if smaller) numbers.
    """
    predicted = run.label_sets
    unit_ids = [u for u in run.units["unit_id"] if u in gold_labels]
    if not unit_ids:
        raise ValueError("no overlap between the run's units and the gold set")

    if codes is not None:
        codes = [c for c in codes if c in set(codebook.names)]
    elif restrict_to_gold_codes:
        codes = sorted({c for u in unit_ids for c in gold_labels[u]})
    else:
        codes = codebook.names
    gold_matrix = label_matrix(gold_labels, unit_ids, codes)
    pred_matrix = label_matrix(predicted, unit_ids, codes)

    texts = run.units.set_index("unit_id")["unit_text"].to_dict()
    return EvaluationResult(
        overall=overall_metrics(gold_matrix, pred_matrix),
        per_code=code_level_metrics(gold_matrix, pred_matrix),
        per_theme=theme_level_metrics(gold_matrix, pred_matrix, codebook),
        errors=error_table(gold_matrix, pred_matrix, texts),
        gold_matrix=gold_matrix,
        pred_matrix=pred_matrix,
    )


def human_baseline(
    annotations: pd.DataFrame | None,
    codebook: Codebook,
    gold_labels: dict[str, set[str]],
    unit_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Score each human coder against the adjudicated gold, as a ceiling.

    Returns an empty frame when the per-coder annotations are not available,
    since the pipeline's required inputs are the codebook and the adjudicated
    file alone.
    """
    if annotations is None or annotations.empty:
        return pd.DataFrame()
    rows = []
    for coder, group in annotations.groupby("coder"):
        coder_labels = {u: set(g["code"]) for u, g in group.groupby("unit_id")}
        scored = [u for u in (unit_ids or list(gold_labels)) if u in coder_labels]
        if not scored:
            continue
        gold_matrix = label_matrix(gold_labels, scored, codebook.names)
        pred_matrix = label_matrix(coder_labels, scored, codebook.names)
        rows.append({"annotator": coder, "kind": "human", **overall_metrics(gold_matrix, pred_matrix)})
    return pd.DataFrame(rows)


def confidence_sweep(
    run: Any,
    gold_labels: dict[str, set[str]],
    codebook: Codebook,
    thresholds: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> pd.DataFrame:
    """Re-score an existing run while discarding low-confidence codes.

    Costs nothing — it re-reads predictions already made. Both models here lean
    towards over-coding (recall well above precision), so trimming the tail of
    hedged codes is the cheapest available accuracy gain. Pick the threshold on
    the test split, then pass it as `min_confidence` on the full run.
    """
    unit_ids = [u for u in run.units["unit_id"] if u in gold_labels]
    gold_matrix = label_matrix(gold_labels, unit_ids, codebook.names)
    annotations = run.annotations[run.annotations["unit_id"].isin(unit_ids)]

    rows = []
    for threshold in thresholds:
        kept = annotations[annotations["confidence"] >= threshold]
        predicted = {unit: set(g["code"]) for unit, g in kept.groupby("unit_id")}
        pred_matrix = label_matrix(predicted, unit_ids, codebook.names)
        rows.append({"min_confidence": threshold, **overall_metrics(gold_matrix, pred_matrix)})
    return pd.DataFrame(rows)


def agreement_with_each_coder(
    run: Any,
    annotations: pd.DataFrame | None,
    codebook: Codebook,
    unit_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Score the model against each individual coder, not just the gold.

    Worth running whenever adjudication tracked one coder closely: agreement
    with the adjudicated labels then largely measures agreement with that one
    person, and the per-coder numbers show how much of the score is specific to
    them.
    """
    if annotations is None or annotations.empty:
        return pd.DataFrame()
    predicted = run.label_sets
    rows = []
    for coder, group in annotations.groupby("coder"):
        coder_labels = {u: set(g["code"]) for u, g in group.groupby("unit_id")}
        scored = [u for u in (unit_ids or list(predicted)) if u in coder_labels and u in predicted]
        if not scored:
            continue
        reference = label_matrix(coder_labels, scored, codebook.names)
        prediction = label_matrix(predicted, scored, codebook.names)
        rows.append({"reference": coder, "kind": "single_coder", **overall_metrics(reference, prediction)})
    return pd.DataFrame(rows)


def ceiling_analysis(
    per_code: pd.DataFrame,
    reliability: str | Path | pd.DataFrame,
    round_number: int = 1,
    min_support: int = 3,
    agreement_floor: float = 0.6,
    disagreement_ceiling: float = 0.35,
) -> dict[str, Any]:
    """Split the model's per-code scores by how well the humans agreed.

    The decisive diagnostic when a model's headline F1 looks disappointing. If
    the model does well exactly where the coders agreed and badly exactly where
    they did not, the remaining error is the codebook's inconsistency rather
    than the model's comprehension, and better prompting will not move it.
    """
    merged = compare_to_human_reliability(per_code, reliability, round_number)
    merged = merged[merged["support_gold"] >= min_support].dropna(subset=["human_human_kappa"])
    agreed = merged[merged["human_human_kappa"] >= agreement_floor]
    disputed = merged[merged["human_human_kappa"] < disagreement_ceiling]
    return {
        "n_codes": int(len(merged)),
        "f1_where_humans_agreed": float(agreed["f1"].mean()) if len(agreed) else float("nan"),
        "n_codes_humans_agreed": int(len(agreed)),
        "f1_where_humans_disagreed": float(disputed["f1"].mean()) if len(disputed) else float("nan"),
        "n_codes_humans_disagreed": int(len(disputed)),
        "kappa_f1_correlation": float(merged["human_human_kappa"].corr(merged["f1"])),
        "detail": merged[["code", "support_gold", "human_human_kappa", "f1"]].sort_values(
            "human_human_kappa", ascending=False
        ),
    }


def human_reliability(
    annotations: pd.DataFrame,
    codebook: Codebook,
    coverage: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Per-code Cohen's kappa between the two coders on doubly-coded units.

    Computed from the per-coder annotations rather than read from a
    pre-exported reliability table, so it stays in step with the data. Scored
    over the units both coders finished: a code absent from one coder's rows
    for such a unit is a genuine negative, not a missing observation.
    """
    coders = sorted(annotations["coder"].dropna().unique())
    if len(coders) != 2:
        return pd.DataFrame()

    if coverage:
        shared = coverage.get(coders[0], set()) & coverage.get(coders[1], set())
    else:
        shared = set.intersection(
            *(set(g["unit_id"]) for _, g in annotations.groupby("coder"))
        )
    units = sorted(shared)
    if not units:
        return pd.DataFrame()

    matrices = [
        label_matrix(
            annotations[annotations["coder"] == coder]
            .groupby("unit_id")["code"]
            .apply(set)
            .to_dict(),
            units,
            codebook.names,
        )
        for coder in coders
    ]

    rows = []
    for code in codebook.names:
        a = matrices[0][code].to_numpy()
        b = matrices[1][code].to_numpy()
        rows.append(
            {
                "code": code,
                "n_units": len(units),
                "n_coder_a": int(a.sum()),
                "n_coder_b": int(b.sum()),
                "percent_agreement": float((a == b).mean()),
                "cohens_kappa": _cohen_kappa(a, b),
            }
        )
    return pd.DataFrame(rows)


def compare_to_human_reliability(
    per_code: pd.DataFrame,
    reliability: str | Path | pd.DataFrame,
    round_number: int = 1,
) -> pd.DataFrame:
    """Put model-vs-gold kappa beside the human-human kappa for each code.

    Accepts either a reliability table computed by `human_reliability` or a
    path to the tool's reliability export. When neither is available the
    model's own kappa is returned unchanged, with the comparison columns empty.
    """
    if isinstance(reliability, pd.DataFrame):
        table = reliability.copy()
    elif Path(reliability).exists():
        table = pd.read_csv(reliability)
        table = table[table["round_number"] == round_number]
        table["cohens_kappa"] = pd.to_numeric(
            table["cohens_kappa"].astype(str).str.lstrip("'"), errors="coerce"
        )
    else:
        table = None

    if table is None or table.empty:
        out = per_code.copy()
        out["human_human_kappa"] = np.nan
        out["human_human_agreement"] = np.nan
        out["kappa_gap"] = np.nan
        return out

    merged = per_code.merge(
        table[["code", "cohens_kappa", "percent_agreement"]].rename(
            columns={"cohens_kappa": "human_human_kappa", "percent_agreement": "human_human_agreement"}
        ),
        on="code",
        how="left",
    )
    merged["kappa_gap"] = merged["cohens_kappa"] - merged["human_human_kappa"]
    return merged
