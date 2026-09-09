#!/usr/bin/env python3
"""Annotate the justification corpus with GPT-5.1 or a local Qwen via Ollama.

Both backends write the identical schema; only the filename and the
`backend`/`model` columns differ.

    # OpenAI
    export OPENAI_API_KEY=sk-...
    python scripts/run_annotate.py --backend openai --model gpt-5.1

    # Local Qwen through Ollama
    ollama serve &
    python scripts/run_annotate.py --backend ollama --model qwen3.5:latest

    # Few-shot prompt, unlabelled units only, first 50 for a dry run
    python scripts/run_annotate.py --backend ollama --prompt-variant few_shot \
        --units unlabelled --limit 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thematic_ai import RunConfig, annotate_units, evaluate_run  # noqa: E402
from thematic_ai.pipeline import (  # noqa: E402
    load_workspace,
    make_context_builder,
    make_system_prompt,
)

UNIT_SELECTIONS = ("all", "unlabelled", "gold", "train", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("openai", "ollama"), default="ollama")
    parser.add_argument("--model", default="", help="defaults to gpt-5.1 / qwen3.5:latest")
    parser.add_argument("--prompt-variant", choices=("codebook_only", "few_shot", "calibrated"), default="calibrated")
    parser.add_argument("--units", choices=UNIT_SELECTIONS, default="all")
    parser.add_argument("--limit", type=int, default=0, help="annotate only the first N units")
    parser.add_argument("--few-shot-k", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="medium",
        help="GPT-5.1 only",
    )
    parser.add_argument("--think", action="store_true", help="let an Ollama thinking model reason first")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--label", default="", help="suffix for the output filename")
    parser.add_argument("--output-dir", default="", help="defaults to outputs/")
    parser.add_argument("--score", action="store_true", help="also score against gold where units overlap")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = RunConfig(
        backend=args.backend,
        model=args.model,
        prompt_variant=args.prompt_variant,
        few_shot_k=args.few_shot_k,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        think=args.think,
        max_workers=args.workers,
        min_confidence=args.min_confidence,
        use_cache=not args.no_cache,
        run_label=args.label or (args.units if args.units != "all" else ""),
    )
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    workspace = load_workspace(config)
    print(workspace.describe().to_string(), file=sys.stderr)

    selection = {
        "all": workspace.units,
        "unlabelled": workspace.unlabelled,
        "gold": workspace.gold,
        "train": workspace.train,
        "test": workspace.test,
    }[args.units]
    if args.limit:
        selection = selection.head(args.limit)

    system_prompt = make_system_prompt(workspace, config)
    print(
        f"\n{config.backend}:{config.model} | {config.prompt_variant} | "
        f"{len(selection)} units | prompt {len(system_prompt):,} chars",
        file=sys.stderr,
    )

    started = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            rate = done / max(time.perf_counter() - started, 1e-9)
            eta = (total - done) / rate if rate else 0
            print(f"  {done}/{total} units  {rate:.2f}/s  eta {eta/60:.1f} min", file=sys.stderr)

    run = annotate_units(
        selection,
        workspace.codebook,
        config,
        system_prompt,
        progress=progress,
        context_builder=make_context_builder(workspace, config),
    )
    paths = run.save()

    print(f"\ndone in {(time.perf_counter() - started)/60:.1f} min", file=sys.stderr)
    print(
        f"{len(run.annotations)} code assignments over {len(run.units)} units "
        f"({run.n_failed} failed, {len(run.dropped_codes)} invalid codes dropped)",
        file=sys.stderr,
    )
    for name, path in paths.items():
        print(f"  {name}: {path}", file=sys.stderr)

    if args.score:
        scored = [u for u in run.units["unit_id"] if u in workspace.gold_labels]
        if scored:
            result = evaluate_run(run, workspace.gold_labels, workspace.codebook, workspace.adjudicated)
            print("\nagainst adjudicated gold:", file=sys.stderr)
            for key, value in result.overall.items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}", file=sys.stderr)
        else:
            print("\nno gold overlap to score", file=sys.stderr)

    return 1 if run.n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
