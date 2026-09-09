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

### Results on the 100-unit held-out split

| annotator | prompt | precision | recall | micro-F1 | macro-F1 | exact set |
|---|---|---|---|---|---|---|
| Alireza (human) | — | 1.000 | 0.971 | 0.985 | 0.927 | 0.95 |
| subham (human) | — | 0.352 | 0.364 | 0.358 | 0.272 | 0.01 |
| GPT-5.1 | codebook only | 0.513 | 0.782 | 0.620 | 0.607 | 0.27 |
| GPT-5.1 | few-shot | 0.518 | 0.753 | 0.614 | 0.585 | 0.23 |
| **GPT-5.1** | **calibrated** | **0.708** | **0.753** | **0.730** | **0.622** | **0.44** |

Read the two human rows before drawing conclusions from the model rows. Adjudication tracked
one coder almost exactly (precision 0.997, recall 0.965 across all 201 adjudicated units),
so "agreement with the gold" here is close to "agreement with Alireza", and his 0.985 is not
an independent ceiling. The realistic bar for a second independent coder is the 0.358 row,
which every model configuration clears comfortably.

### Why the codebook-only and few-shot numbers are low

Not comprehension — restraint. Those runs recalled 140 of 174 gold codes but *added* 128 more,
applying 2.68 codes per unit against a human 1.74. Only 15% of the false positives were the
right theme with the wrong code; the rest were codes the humans simply declined to apply. The
`calibrated` variant fixes this by telling the model what a codebook structurally cannot: the
observed base rate of each code, the codes-per-unit distribution, which same-theme codes the
coders treated as alternatives, and the nearest already-coded justifications. Predicted labels
fell from 265 to 185 against a gold of 174, and precision went from 0.51 to 0.71.

A confidence threshold was worth 5 points on the uncalibrated runs and is swept automatically
(`confidence_sweep`), but it buys nothing once `calibrated` is in use — the model is already
applying about the right number of codes, so there is no low-confidence tail to trim.

## Notebooks

| notebook | purpose |
|---|---|
| `notebooks/01_annotation_pipeline.ipynb` | the full pipeline: load, configure a backend, inspect the prompt, dry-run, annotate the corpus, describe the output against the human distribution |
| `notebooks/02_codebook_only_and_evaluation.ipynb` | the codebook-only prompt (codebook plus its own examples, no human annotations shown), the train/test split, the accuracy comparison across models, prompt variants and human coders, and the ceiling analysis |

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

`calibrated` (default)
: Adds the coders' observed behaviour on top of the few-shot block — base rate per code,
  codes-per-unit distribution, same-theme code pairs the coders treated as alternatives — plus
  the ten most similar already-coded justifications retrieved per unit. All of it derived from
  the training split at run time, so it stays correct if the coding changes.

Tuned on the test split: medium reasoning effort beat both low (0.725) and high (0.698), and
ten retrieved neighbours beat twenty (0.707). Those are the defaults.

## How accurate can this get?

Not 85–90% at the level of individual codes, and the reason is in the data rather than the
prompt. The remaining error tracks the codebook's own reliability almost linearly:

- On codes the two coders agreed about (kappa ≥ 0.6): model F1 **0.78**
- On codes they did not (kappa < 0.35): model F1 **0.44**
- Correlation between human-human kappa and model F1: **r = 0.56**
- Theme level — did it find the right *kind* of reason? — micro-F1 **0.851**

The three codes the model scores 0.00 on (`derogatory remarks`, `Extreme offensive`,
`Contextual understanding`) have human-human kappas of 0.04, 0.16 and 0.33, and two of them
have no definition in the codebook at all. No prompt can recover a distinction the coders
themselves did not make consistently.

If you are willing to act on that, pruning or merging the unreliable codes is the lever that
actually moves the number. `ceiling_analysis` and the last cell of notebook 02 quantify it:

| codebook restricted to | codes | micro-F1 | exact set |
|---|---|---|---|
| everything | 34 | 0.740 | 0.45 |
| human kappa ≥ 0.2 | 22 | 0.786 | 0.53 |
| human kappa ≥ 0.3 | 18 | **0.803** | 0.56 |
| human kappa ≥ 0.5 | 9 | 0.785 | **0.73** |

That is a codebook decision, not an engineering one, so the pipeline reports it rather than
making it for you.

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
  calibration.py observed coding behaviour and nearest-neighbour retrieval
  backends.py    OpenAI and Ollama clients behind one interface
  spans.py       quote-to-offset alignment
  annotate.py    the annotation loop, validation, output assembly
  evaluate.py    multi-label metrics, kappa, span IoU, human baseline, ceiling analysis
  pipeline.py    the wiring shared by the scripts and the notebooks
scripts/
  run_annotate.py     annotate the corpus with either backend
  run_experiment.py   train/test evaluation and model comparison
```
