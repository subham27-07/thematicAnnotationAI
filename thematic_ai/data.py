"""Loading the corpus, the human annotations and the adjudicated gold set.

A *unit* is one account-level justification written by one participant. Its id
is ``f"{user_id}::{account}"``, which is the same key the human annotation tool
exported, so the corpus and the human files join on it directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .codebook import Codebook

UNIT_META_COLUMNS = [
    "user_id",
    "account",
    "condition",
    "user_finish_id",
    "accountsOrder",
    "aor",
    "num_messages",
    "account_suspension",
    "account_confidence",
    "account_duration",
]


def make_unit_id(user_id: object, account: object) -> str:
    account_str = str(account)
    if account_str.endswith(".0"):
        account_str = account_str[:-2]
    return f"{user_id}::{account_str}"


def load_units(path: str | Path, text_column: str = "justification") -> pd.DataFrame:
    """Return one row per annotatable unit, with `unit_id` and `unit_text`."""
    frame = pd.read_csv(path)
    if text_column not in frame.columns:
        raise ValueError(f"{path} has no {text_column!r} column")

    frame = frame.copy()
    frame["unit_id"] = [
        make_unit_id(u, a) for u, a in zip(frame["user_id"], frame["account"])
    ]
    frame["unit_text"] = frame[text_column].fillna("").astype(str).str.strip()

    duplicated = frame["unit_id"].duplicated().sum()
    if duplicated:
        raise ValueError(f"{duplicated} duplicate unit_ids in {path}")

    keep = ["unit_id", "unit_text"] + [c for c in UNIT_META_COLUMNS if c in frame.columns]
    return frame[keep].reset_index(drop=True)


def load_human_annotations(path: str | Path, codebook: Codebook | None = None) -> pd.DataFrame:
    """Per-coder annotations in long format (one row per unit x code x span)."""
    frame = pd.read_csv(path)
    frame["code"] = frame["code"].astype(str).str.strip()
    if codebook is not None:
        resolved = frame["code"].map(lambda c: codebook.resolve(c))
        unknown = frame.loc[resolved.isna(), "code"].unique()
        if len(unknown):
            raise ValueError(f"annotations reference codes absent from the codebook: {list(unknown)}")
        frame["code"] = resolved
    return frame


def load_adjudicated(path: str | Path, codebook: Codebook | None = None) -> pd.DataFrame:
    """Adjudicated (gold) annotations in long format."""
    frame = pd.read_csv(path)
    if "adjudication_status" in frame.columns:
        frame = frame[frame["adjudication_status"].astype(str).str.lower() == "resolved"]
    frame = frame.copy()
    frame["code"] = frame["code"].astype(str).str.strip()
    if codebook is not None:
        resolved = frame["code"].map(lambda c: codebook.resolve(c))
        unknown = frame.loc[resolved.isna(), "code"].unique()
        if len(unknown):
            raise ValueError(f"adjudicated file references unknown codes: {list(unknown)}")
        frame["code"] = resolved
    return frame.reset_index(drop=True)


def adjudicated_unit_ids(project_export_path: str | Path) -> list[str]:
    """Units a human marked as adjudicated, including any resolved to no codes.

    The adjudicated CSV only lists positive labels, so a unit that was reviewed
    and legitimately received zero codes is invisible there. Recovering it from
    the project export keeps the evaluation denominator honest.
    """
    path = Path(project_export_path)
    if not path.exists():
        return []
    export = json.loads(path.read_text())
    external = {u["id"]: u.get("external_id") for u in export.get("units", [])}
    out: list[str] = []
    for adj in export.get("adjudications", []):
        if str(adj.get("status", "")).lower() not in {"resolved", "adjudicated", "done"}:
            continue
        ext = external.get(adj.get("unit_id"))
        if ext:
            out.append(ext)
    return sorted(set(out))


def gold_label_sets(
    adjudicated: pd.DataFrame,
    annotations: pd.DataFrame | None = None,
    source: str = "adjudicated",
) -> dict[str, set[str]]:
    """Map unit_id -> set of gold codes.

    source:
      adjudicated  - codes agreed after disagreement resolution (recommended)
      union        - every code any human coder applied
      intersection - codes every coder who saw the unit applied
    """
    if source == "adjudicated":
        return {
            unit: set(group["code"])
            for unit, group in adjudicated.groupby("unit_id")
        }

    if annotations is None:
        raise ValueError(f"source={source!r} needs the per-coder annotations frame")

    if source == "union":
        return {
            unit: set(group["code"])
            for unit, group in annotations.groupby("unit_id")
        }

    if source == "intersection":
        out: dict[str, set[str]] = {}
        for unit, group in annotations.groupby("unit_id"):
            coders = group["coder"].unique()
            per_coder = [set(g["code"]) for _, g in group.groupby("coder")]
            shared = set.intersection(*per_coder) if len(coders) > 1 else per_coder[0]
            out[unit] = shared
        return out

    raise ValueError(f"unknown gold source {source!r}")


def build_gold_frame(
    units: pd.DataFrame,
    adjudicated: pd.DataFrame,
    annotations: pd.DataFrame | None = None,
    source: str = "adjudicated",
    extra_unit_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Units that carry a gold label, with their gold code set attached."""
    labels = gold_label_sets(adjudicated, annotations, source)
    for unit_id in extra_unit_ids or []:
        labels.setdefault(unit_id, set())

    gold = units[units["unit_id"].isin(labels)].copy()
    gold["gold_codes"] = gold["unit_id"].map(lambda u: sorted(labels[u]))
    gold["n_gold_codes"] = gold["gold_codes"].map(len)
    missing = set(labels) - set(gold["unit_id"])
    if missing:
        raise ValueError(f"{len(missing)} gold units are absent from the corpus, e.g. {sorted(missing)[:3]}")
    return gold.reset_index(drop=True)


def train_test_split_units(
    gold: pd.DataFrame,
    test_size: float = 0.5,
    seed: int = 20260909,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split gold units, stratified on how many codes the unit carries.

    Stratifying on label count keeps the rare multi-code units from all landing
    on one side, which otherwise makes the two splits look like different tasks.
    """
    rng = np.random.default_rng(seed)
    strata = gold["n_gold_codes"].clip(upper=3)
    test_idx: list[int] = []
    for _, group in gold.groupby(strata):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        n_test = min(max(n_test, 1 if len(idx) > 1 else 0), len(idx) - 1 if len(idx) > 1 else 0)
        test_idx.extend(idx[:n_test].tolist())

    test_mask = gold.index.isin(test_idx)
    train = gold[~test_mask].reset_index(drop=True)
    test = gold[test_mask].reset_index(drop=True)
    return train, test


def label_matrix(
    labels: dict[str, set[str]],
    unit_ids: list[str],
    codes: list[str],
) -> pd.DataFrame:
    """Binary unit x code indicator matrix, 0/1, aligned to `unit_ids`."""
    data = np.zeros((len(unit_ids), len(codes)), dtype=np.int8)
    code_pos = {c: i for i, c in enumerate(codes)}
    for row, unit in enumerate(unit_ids):
        for code in labels.get(unit, ()):  # unseen unit -> all zeros
            col = code_pos.get(code)
            if col is not None:
                data[row, col] = 1
    return pd.DataFrame(data, index=pd.Index(unit_ids, name="unit_id"), columns=codes)
