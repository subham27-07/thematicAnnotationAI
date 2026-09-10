"""High-level wiring shared by the CLI scripts and the notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .calibration import NeighbourRetriever, compute_coding_stats
from .codebook import Codebook, load_codebook
from .config import RunConfig
from .data import (
    adjudicated_unit_ids,
    build_gold_frame,
    coder_coverage,
    gold_label_sets,
    load_adjudicated,
    load_human_annotations,
    load_units,
    train_test_split_units,
)
from .evaluate import human_reliability
from .prompts import build_system_prompt, render_few_shot, select_few_shot_units


@dataclass
class Workspace:
    """Corpus, codebook and adjudicated labels, split into train and test.

    `annotations` and `reliability` hold the per-coder rows and the derived
    human-human kappas. Both are optional and feed only the human comparisons;
    the pipeline itself needs only the codebook and the adjudicated file.
    """

    config: RunConfig
    codebook: Codebook
    units: pd.DataFrame
    annotations: pd.DataFrame | None
    reliability: pd.DataFrame | None
    adjudicated: pd.DataFrame
    gold: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    gold_labels: dict[str, set[str]]

    @property
    def unlabelled(self) -> pd.DataFrame:
        """Corpus units no human has coded — the ones the run is really for."""
        return self.units[~self.units["unit_id"].isin(self.gold_labels)].reset_index(drop=True)

    def describe(self) -> pd.Series:
        return pd.Series(
            {
                "codes_in_codebook": len(self.codebook),
                "units_in_corpus": len(self.units),
                "units_human_coded": (
                    self.annotations["unit_id"].nunique() if self.annotations is not None else 0
                ),
                "units_adjudicated": len(self.gold),
                "train_units": len(self.train),
                "test_units": len(self.test),
                "unlabelled_units": len(self.unlabelled),
                "gold_labels_total": int(self.gold["n_gold_codes"].sum()),
                "mean_codes_per_gold_unit": round(float(self.gold["n_gold_codes"].mean()), 3),
            }
        )


def load_workspace(
    config: RunConfig,
    gold_source: str = "adjudicated",
    test_size: float = 0.5,
    include_zero_code_units: bool = True,
) -> Workspace:
    """Load the inputs and build the train/test split once.

    Required: the codebook and the adjudicated annotations. Everything else is
    read if present and skipped if not.
    """
    codebook = load_codebook(config.codebook_path)
    units = load_units(config.dataset_path)
    adjudicated = load_adjudicated(config.adjudicated_path, codebook)
    # Per-coder rows are not part of the gold evaluation. The file is gone and
    # is never required; load it only if someone still has a copy on disk.
    annotations = (
        load_human_annotations(config.annotations_path, codebook)
        if Path(config.annotations_path).exists()
        else None
    )

    export = config.adjudicated_path.parent / "project-export.json"
    reliability = (
        human_reliability(annotations, codebook, coder_coverage(export))
        if annotations is not None
        else None
    )

    extra: list[str] = []
    if include_zero_code_units:
        extra = [u for u in adjudicated_unit_ids(export) if u in set(units["unit_id"])]

    gold = build_gold_frame(units, adjudicated, annotations, gold_source, extra_unit_ids=extra)
    train, test = train_test_split_units(gold, test_size=test_size, seed=config.seed)
    labels = {row.unit_id: set(row.gold_codes) for row in gold.itertuples()}

    return Workspace(
        config=config,
        codebook=codebook,
        units=units,
        annotations=annotations,
        reliability=reliability,
        adjudicated=adjudicated,
        gold=gold,
        train=train,
        test=test,
        gold_labels=labels,
    )


def make_system_prompt(workspace: Workspace, config: RunConfig | None = None) -> str:
    """Build the system prompt for the configured variant.

    Few-shot demonstrations and the calibration statistics are drawn from the
    training split only, so test-split scores are not inflated by the model
    having seen the answers.
    """
    config = config or workspace.config
    few_shot_block = ""
    calibration_block = ""

    if config.prompt_variant in {"few_shot", "calibrated"}:
        chosen = select_few_shot_units(workspace.train, config.few_shot_k, seed=config.seed)
        few_shot_block = render_few_shot(chosen, workspace.gold)

    if config.prompt_variant == "calibrated":
        stats = compute_coding_stats(workspace.train, workspace.codebook)
        calibration_block = stats.render(workspace.codebook)

    return build_system_prompt(
        workspace.codebook,
        variant=config.prompt_variant,
        few_shot_block=few_shot_block,
        calibration_block=calibration_block,
        include_codebook_examples=config.include_codebook_examples,
    )


def make_context_builder(
    workspace: Workspace, config: RunConfig | None = None
) -> Callable[[dict], str] | None:
    """Per-unit retrieved neighbours, for the `calibrated` variant.

    Returns None for the other variants, which use one fixed prompt for every
    unit.
    """
    config = config or workspace.config
    if config.prompt_variant != "calibrated" or config.retrieved_neighbours <= 0:
        return None

    retriever = NeighbourRetriever(workspace.train)
    k = config.retrieved_neighbours

    def build(record: dict) -> str:
        return retriever.render_context(str(record.get("unit_text") or ""), k=k)

    return build
