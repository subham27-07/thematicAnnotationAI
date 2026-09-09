"""High-level wiring shared by the CLI scripts and the notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .codebook import Codebook, load_codebook
from .config import RunConfig
from .data import (
    adjudicated_unit_ids,
    build_gold_frame,
    gold_label_sets,
    load_adjudicated,
    load_human_annotations,
    load_units,
    train_test_split_units,
)
from .prompts import build_system_prompt, render_few_shot, select_few_shot_units


@dataclass
class Workspace:
    """Corpus, codebook and human labels, split into train and test."""

    config: RunConfig
    codebook: Codebook
    units: pd.DataFrame
    annotations: pd.DataFrame
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
                "units_human_coded": self.annotations["unit_id"].nunique(),
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
    """Load every input file and build the train/test split once."""
    codebook = load_codebook(config.codebook_path)
    units = load_units(config.dataset_path)
    annotations = load_human_annotations(config.annotations_path, codebook)
    adjudicated = load_adjudicated(config.adjudicated_path, codebook)

    extra: list[str] = []
    if include_zero_code_units:
        export = config.adjudicated_path.parent / "project-export.json"
        extra = [u for u in adjudicated_unit_ids(export) if u in set(units["unit_id"])]

    gold = build_gold_frame(units, adjudicated, annotations, gold_source, extra_unit_ids=extra)
    train, test = train_test_split_units(gold, test_size=test_size, seed=config.seed)
    labels = {row.unit_id: set(row.gold_codes) for row in gold.itertuples()}

    return Workspace(
        config=config,
        codebook=codebook,
        units=units,
        annotations=annotations,
        adjudicated=adjudicated,
        gold=gold,
        train=train,
        test=test,
        gold_labels=labels,
    )


def make_system_prompt(workspace: Workspace, config: RunConfig | None = None) -> str:
    """Build the system prompt for the configured variant.

    Few-shot demonstrations are drawn from the training split only, so the
    test-split scores are not inflated by the model having seen the answers.
    """
    config = config or workspace.config
    few_shot_block = ""
    if config.prompt_variant == "few_shot":
        chosen = select_few_shot_units(
            workspace.train, workspace.adjudicated, config.few_shot_k, seed=config.seed
        )
        few_shot_block = render_few_shot(chosen, workspace.gold, workspace.adjudicated)
    return build_system_prompt(
        workspace.codebook,
        variant=config.prompt_variant,
        few_shot_block=few_shot_block,
        include_codebook_examples=config.include_codebook_examples,
    )
