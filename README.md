# thematicAnnotationAI

LLM-assisted deductive thematic coding of content-moderation justifications, with a
train/test evaluation against human coders.

Participants in the study reviewed a suspicious social media account, decided whether to
suspend it, and wrote a short free-text justification. `Qual Analysis - accounts_tidy.csv`
holds 1,450 of those justifications. A subset was coded by two human annotators against a
38-code codebook and then adjudicated. This repository applies the same codebook with a
language model, and measures how close the model gets to the humans.

## One pipeline, two backends

The backend is a single setting. **OpenAI GPT-5.1** and a **local Qwen served by Ollama**
share the same codebook, prompt, JSON schema, validation, span alignment and output columns.
There are no `gpt_*` or `qwen_*` fields; the two runs differ only in their filename and in the
`backend` / `model` columns, so their outputs concatenate without any renaming.

```bash
pip install -r requirements.txt

# OpenAI — put `export OPENAI_API_KEY=sk-...` in .env (already gitignored),
# or set it in the shell; the pipeline reads both.
python scripts/run_annotate.py --backend openai --model gpt-5.1 --units all

# Local Qwen
ollama serve &
ollama pull qwen3.5:latest
python scripts/run_annotate.py --backend ollama --model qwen3.5:latest --units all
```

Rehearse on a handful of units first — `--units test --limit 5 --score` codes five held-out
units and prints how it did.

## Measuring accuracy

The 202 adjudicated units are split in half, stratified on how many codes each unit carries.
Few-shot demonstrations may only be drawn from the train half; every model is scored on the
same held-out test half, next to the two human coders scored against the same gold labels.

```bash
python scripts/run_experiment.py --backend ollama --model qwen3.5:latest
python scripts/run_experiment.py --backend openai --model gpt-5.1
```

This writes per-code, per-theme, error and summary tables to `outputs/evaluation/`, including
a comparison of the model's Cohen's kappa against the human-human kappa recorded in
`data/reliability-all-rounds.csv`.

Coding here is multi-label — a unit carries zero to six of the 38 codes — so per-cell accuracy
is meaningless (predict nothing and you score ~95%). The headline numbers are micro-F1,
macro-F1 over codes that actually occur, exact set match, and chance-corrected kappa.

Both models over-code: recall runs well above precision. The experiment therefore also sweeps
a confidence threshold over the predictions already made, which costs no extra API calls. On
the GPT-5.1 few-shot run, discarding codes below 0.9 confidence raised micro-F1 from 0.633 to
0.687 and exact set match from 0.25 to 0.33. Pick the threshold on the test split, then pass
it to the full run as `--min-confidence 0.9`.

## Notebooks

| notebook | purpose |
|---|---|
| `notebooks/01_annotation_pipeline.ipynb` | the full pipeline: load, configure a backend, inspect the prompt, dry-run, annotate the corpus, describe the output against the human distribution |
| `notebooks/02_codebook_only_and_evaluation.ipynb` | the codebook-only prompt (codebook plus its own examples, no human annotations shown), the train/test split, and the accuracy comparison between models, prompt variants and human coders |

## Prompt variants

`codebook_only`
: The codebook and nothing else — every code with its definition, inclusion criteria,
  exclusion criteria and the example the codebook itself carries. This is the honest test of
  whether the codebook as written is enough to apply it, and doubles as a diagnostic of the
  codebook: codes with thin definitions tend to be both the model's worst codes and the ones
  the humans disagreed on.

`few_shot`
: The same codebook plus *k* worked examples from adjudicated human coding. Examples are
  chosen greedily to cover as many distinct codes as possible, and only ever from the training
  split, so test scores are not inflated by the model having seen the answers.

## Output files

Each run writes four files prefixed with its run key (`{backend}__{model}__{variant}`):

| file | contents |
|---|---|
| `annotations__<key>.csv` | long format, one row per unit × code, mirroring the human annotation export |
| `annotations__<key>__units.csv` | one row per unit: codes, note, status, latency, tokens |
| `annotations__<key>__wide.csv` | binary unit × code matrix |
| `annotations__<key>__manifest.json` | the exact config and system prompt that produced the run |

The long file carries `unit_id, unit_text, theme, code, scope, start_offset, end_offset,
quote` — the same columns as `data/annotations-all-rounds.csv` — plus `quote_match`,
`confidence`, `reason` and run metadata.

## How it works

**Units.** A unit is one account-level justification, keyed `user_id::account`. That is the
key the human annotation tool exported, so the corpus and the human files join directly.

**Constrained decoding.** Both backends are given the same JSON schema, whose `code` field is
an enum of the 38 codebook codes. A model cannot return a code that does not exist. Anything
that still slips through — a duplicate, a sub-threshold confidence — is dropped and recorded
in the manifest rather than written out.

**Span alignment.** Each code comes with a verbatim quote, which the pipeline locates back in
the justification to recover character offsets: exact match, then case-insensitive, then
whitespace-normalised, then approximate. `quote_match` records which succeeded. A quote that
cannot be located becomes `scope="unit"` with no offsets — exactly how the human tool
represents a code that applies to the whole justification.

**Caching.** Every answer is cached under `outputs/cache/`, keyed by model, prompt and
settings. Re-running is free and an interrupted run resumes. Failures are not cached, so they
are retried next time.

**Gold denominator.** The adjudicated set includes units a human reviewed and resolved to *no*
codes. Those exist only in `data/project-export.json` — the CSV lists positive labels only —
and are recovered from it, because dropping them would quietly reward a model for over-coding.

## Cost and runtime

GPT-5.1 at `--reasoning-effort low` and 8 workers runs at roughly 0.5 units/second, so the
full 1,450 units takes about an hour. A local Qwen runs at roughly 15–25 s per unit on an
Apple laptop, so the same job is an overnight run. The cache makes either safe to stop and
resume. `--reasoning-effort none` is markedly faster and cheaper on GPT-5.1 and is worth
evaluating against `low` on the test split before committing to a full run.

## Layout

```
thematic_ai/
  config.py      run configuration and cache keys
  codebook.py    codebook loading, prompt rendering, code-name resolution
  data.py        corpus, human annotations, gold labels, train/test split
  prompts.py     system prompt, JSON schema, few-shot selection
  backends.py    OpenAI and Ollama clients behind one interface
  spans.py       quote-to-offset alignment
  annotate.py    the annotation loop, validation, output assembly
  evaluate.py    multi-label metrics, kappa, span IoU, human baseline
  pipeline.py    the wiring shared by the scripts and the notebooks
scripts/
  run_annotate.py     annotate the corpus with either backend
  run_experiment.py   train/test evaluation and model comparison
```
