#!/usr/bin/env python3
"""Evaluate one or more model + prompt configurations on the adjudicated gold.

By default every adjudicated unit is scored. A unit never sees its own gold
labels; similar examples come from the other units. Pass ``--test-size 0.5``
for a hold-out split.

    python scripts/run_experiment.py --backend ollama --prompt-variant codebook_only few_shot
    python scripts/run_experiment.py --backend openai --model gpt-5.1 --prompt-variant few_shot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thematic_ai import RunConfig, annotate_units, evaluate_run, refine_run  # noqa: E402
from thematic_ai.evaluate import (  # noqa: E402
    agreement_with_each_coder,
    ceiling_analysis,
    compare_to_human_reliability,
    confidence_sweep,
    human_baseline,
)
from thematic_ai.pipeline import (  # noqa: E402
    load_workspace,
    make_context_builder,
    make_system_prompt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("openai", "ollama"), default="ollama")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--prompt-variant",
        nargs="+",
        choices=("codebook_only", "few_shot", "calibrated"),
        default=["codebook_only", "few_shot", "calibrated"],
    )
    parser.add_argument("--few-shot-k", type=int, default=12)
    parser.add_argument("--test-size", type=float, default=0.0, help="0 = score every adjudicated unit")
    parser.add_argument("--limit", type=int, default=0, help="cap the test set, for a quick check")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"), default="medium")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output-dir", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent / "outputs"
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    base = RunConfig(backend=args.backend, model=args.model)
    workspace = load_workspace(base, test_size=args.test_size)
    print(workspace.describe().to_string(), file=sys.stderr)

    test = workspace.test.head(args.limit) if args.limit else workspace.test
    test_ids = list(test["unit_id"])

    summaries: list[pd.Series] = []

    baseline = human_baseline(workspace.annotations, workspace.codebook, workspace.gold_labels, test_ids)
    for _, row in baseline.iterrows():  # empty unless the per-coder export is present
        summaries.append(row)
        print(
            f"\nhuman {row['annotator']}: micro-F1 {row['micro_f1']:.3f} "
            f"| exact-set {row['exact_set_match']:.3f} (n={int(row['n_units'])})",
            file=sys.stderr,
        )

    for variant in args.prompt_variant:
        config = RunConfig(
            backend=args.backend,
            model=args.model,
            prompt_variant=variant,
            few_shot_k=args.few_shot_k,
            reasoning_effort=args.reasoning_effort,
            think=args.think,
            max_workers=args.workers,
            use_cache=not args.no_cache,
            run_label="all_gold" if args.test_size <= 0 else "test_split",
            output_dir=output_dir,
        )
        system_prompt = make_system_prompt(workspace, config)
        print(
            f"\n=== {config.backend}:{config.model} | {variant} | {len(test)} test units ===",
            file=sys.stderr,
        )

        started = time.perf_counter()

        def progress(done: int, total: int) -> None:
            if done % 10 == 0 or done == total:
                print(f"  {done}/{total}", file=sys.stderr)

        run = annotate_units(
            test,
            workspace.codebook,
            config,
            system_prompt,
            progress=progress,
            context_builder=make_context_builder(workspace, config),
        )
        run.save(output_dir, prefix="eval")
        if variant == "calibrated":
            print("  adjudicating second pass...", file=sys.stderr)
            run = refine_run(run, workspace, progress=progress)
            run.save(output_dir, prefix="eval_refined")
        elapsed = time.perf_counter() - started

        result = evaluate_run(run, workspace.gold_labels, workspace.codebook)
        tag = f"{config.backend}__{config.model_slug}__{variant}"
        result.per_code.to_csv(eval_dir / f"per_code__{tag}.csv", index=False)
        result.per_theme.to_csv(eval_dir / f"per_theme__{tag}.csv", index=False)
        result.errors.to_csv(eval_dir / f"errors__{tag}.csv", index=False)

        # The human comparisons need the per-coder rows, which come from the
        # CSV export or from project-export.json. Skipped when neither exists.
        if workspace.reliability is not None and not workspace.reliability.empty:
            compare_to_human_reliability(result.per_code, workspace.reliability).to_csv(
                eval_dir / f"vs_human_reliability__{tag}.csv", index=False
            )
            ceiling = ceiling_analysis(result.per_code, workspace.reliability)
            ceiling.pop("detail").to_csv(eval_dir / f"ceiling__{tag}.csv", index=False)
        else:
            ceiling = None
        per_coder = agreement_with_each_coder(run, workspace.annotations, workspace.codebook, test_ids)
        if not per_coder.empty:
            per_coder.to_csv(eval_dir / f"vs_each_coder__{tag}.csv", index=False)

        sweep = confidence_sweep(run, workspace.gold_labels, workspace.codebook)
        sweep.to_csv(eval_dir / f"confidence_sweep__{tag}.csv", index=False)
        best = sweep.loc[sweep["micro_f1"].idxmax()]

        summaries.append(
            result.summary_row(
                annotator=f"{config.backend}:{config.model}",
                kind="model",
                prompt_variant=variant,
                minutes=round(elapsed / 60, 2),
                failed_units=run.n_failed,
                invalid_codes_dropped=len(run.dropped_codes),
            )
        )
        print(
            f"  micro-F1 {result.overall['micro_f1']:.3f} | macro-F1 {result.overall['macro_f1_present_codes']:.3f} "
            f"| exact-set {result.overall['exact_set_match']:.3f} | kappa {result.overall['macro_kappa_present_codes']:.3f}",
            file=sys.stderr,
        )
        if ceiling:
            print(
                f"  on codes the two humans agreed about (kappa>=0.6, n={ceiling['n_codes_humans_agreed']}): "
                f"F1 {ceiling['f1_where_humans_agreed']:.3f}; on codes they did not "
                f"(kappa<0.35, n={ceiling['n_codes_humans_disagreed']}): F1 {ceiling['f1_where_humans_disagreed']:.3f}",
                file=sys.stderr,
            )
        if best["min_confidence"] > 0:
            print(
                f"  discarding codes below confidence {best['min_confidence']:.2f} would give "
                f"micro-F1 {best['micro_f1']:.3f} (--min-confidence {best['min_confidence']:.2f})",
                file=sys.stderr,
            )

    summary = pd.DataFrame(summaries)
    lead = [c for c in ("annotator", "kind", "prompt_variant") if c in summary.columns]
    summary = summary[lead + [c for c in summary.columns if c not in lead]]
    summary_path = eval_dir / f"summary__{args.backend}__{base.model_slug}.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + summary.to_string(index=False), file=sys.stderr)
    print(f"\nwrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
