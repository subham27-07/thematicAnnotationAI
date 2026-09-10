# thematicAnnotationAI

LLM-assisted deductive thematic coding of content-moderation justifications, with a
train/test evaluation against human coders.

Participants in the study reviewed a suspicious social media account, decided whether to
suspend it, and wrote a short free-text justification. `Qual Analysis - accounts_tidy.csv`
holds 1,450 of those justifications. A subset was coded by two human annotators against a
45-code codebook and then adjudicated. This repository applies the same codebook with a
language model, and measures how close the model gets to the humans.

## Row-level coding

The unit of analysis is the whole justification. Each row receives the set of codebook codes
that apply to it — no spans, offsets or quotes are produced, and none are asked of the model.
Two files define the task:

- `data/codebook.csv` — 45 codes with definitions, inclusion and exclusion criteria
- `data/adjudicated-all-rounds.csv` — the gold standard, 376 units resolved after the two
  coders' disagreements were settled

Those are the only required inputs. The adjudicated export carries one row per highlighted
span, so a code applied to two phrases of the same justification appears twice; the loader
collapses those to one row per unit × code before anything is scored.

`data/project-export.json` is read when present, for two things the CSV cannot supply: the
units a coder resolved to *no* codes, and the per-coder rows behind the human comparisons.
Nothing breaks without it.

## One pipeline, two backends

The backend is a single setting. **OpenAI GPT-5.1** and a **local Qwen served by Ollama**
share the same codebook, prompt, JSON schema, validation and output columns. There are no
`gpt_*` or `qwen_*` fields; the two runs differ only in their filename and in the `backend` /
`model` columns, so their outputs concatenate without any renaming.

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

The 376 adjudicated units are split in half, stratified on how many codes each unit carries:
189 train, 187 test. Few-shot demonstrations and the calibration statistics may only be drawn
from the train half; every model is scored on the same held-out test half, next to the two
human coders scored against the same gold labels.

```bash
python scripts/run_experiment.py --backend ollama --model qwen3.5:latest
python scripts/run_experiment.py --backend openai --model gpt-5.1
```

This writes per-code, per-theme, error and summary tables to `outputs/evaluation/`, including
a comparison of the model's Cohen's kappa against the human-human kappa on each code.

Coding here is multi-label — a unit carries zero to seven of the 45 codes — so per-cell
accuracy is meaningless (predict nothing and you score ~96%). The headline numbers are
micro-F1, macro-F1 over codes that actually occur, exact set match, and chance-corrected kappa.

### Results on the 187-unit held-out split

325 gold labels. Every row scored against `data/adjudicated-all-rounds.csv`.

| annotator | prompt | precision | recall | micro-F1 | macro-F1 | exact set | kappa |
|---|---|---|---|---|---|---|---|
| Alireza (human) | — | 0.994 | 0.966 | 0.980 | 0.949 | 0.94 | 0.948 |
| subham (human) | — | 0.396 | 0.448 | 0.420 | 0.229 | 0.01 | 0.203 |
| GPT-5.1 | codebook only | 0.439 | 0.800 | 0.567 | 0.537 | 0.16 | 0.514 |
| GPT-5.1 | few-shot | 0.476 | 0.797 | 0.596 | 0.546 | 0.20 | 0.524 |
| **GPT-5.1** | **calibrated** | **0.712** | **0.723** | **0.718** | **0.570** | **0.40** | **0.554** |

Read the two human rows before drawing conclusions from the model rows. Adjudication tracked
one coder almost exactly, so "agreement with the gold" here is close to "agreement with
Alireza", and his 0.980 is not an independent ceiling. The realistic bar for a second
independent coder is the 0.420 row, which every model configuration clears comfortably.

### Why the codebook-only and few-shot numbers are low

Not comprehension — restraint. The codebook-only run recalled 260 of 325 gold codes but
*added* 332 more, applying 3.17 codes per unit against a human 1.74. Most of those false
positives were not the wrong code for the right idea; they were codes the humans simply
declined to apply.

The `calibrated` variant fixes this by telling the model what a codebook structurally cannot:
the observed base rate of each code, the codes-per-unit distribution, which same-theme codes
the coders treated as alternatives, and the nearest already-coded justifications. Predicted
labels fell from 592 to 330 against a gold of 325, and precision went from 0.44 to 0.71 while
recall dropped only 8 points.

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

Tuned on the test split: medium reasoning effort beat both low and high, and ten retrieved
neighbours beat twenty. Those are the defaults.

## How accurate can this get?

Not 85–90% at the level of individual codes, and the reason is in the data rather than the
prompt. The remaining error tracks the codebook's own reliability:

- On codes the two coders agreed about (kappa ≥ 0.6): model F1 **0.72**
- On codes they did not (kappa < 0.35): model F1 **0.49**
- Theme level — did it find the right *kind* of reason? — micro-F1 **0.825**

The codebook's own numbers set that limit. Across the 500 doubly-coded units the two coders
reached kappa ≥ 0.6 on only 5 of 45 codes, and 22 codes have a kappa below 0.2. `derogatory
remarks` was applied once by one coder and 67 times by the other (kappa 0.03); `Permanent ban`
6 times against 76 (kappa 0.10); `Extreme offensive` 5 against 40 (kappa 0.12). No prompt can
recover a distinction the coders themselves did not make consistently, and the adjudicated
labels for those codes are one person's judgement rather than an agreed standard.

Pruning or merging the unreliable codes is the lever that actually moves the number.
`ceiling_analysis` and the last cell of notebook 02 quantify it on the same held-out split:

| codebook restricted to | codes | gold labels | micro-F1 | exact set |
|---|---|---|---|---|
| everything | 42 | 321 | 0.727 | 0.42 |
| human kappa ≥ 0.2 | 23 | 278 | 0.750 | 0.49 |
| human kappa ≥ 0.3 | 19 | 256 | 0.778 | 0.55 |
| human kappa ≥ 0.5 | 9 | 127 | **0.808** | **0.76** |

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

The long file carries `unit_id, user_id, account, unit_text, theme, code, confidence, reason`
plus run metadata. One row per unit × code, with no span columns: `confidence` and `reason`
are there to be read during validation, not to locate text.

## How it works

**Units.** A unit is one account-level justification, keyed `user_id::account`. That is the
key the human annotation tool exported, so the corpus and the human files join directly.

**Constrained decoding.** Both backends are given the same JSON schema, whose `code` field is
an enum of the 45 codebook codes. A model cannot return a code that does not exist. Anything
that still slips through — a duplicate, a sub-threshold confidence — is dropped and recorded
in the manifest rather than written out.

**Caching.** Every answer is cached under `outputs/cache/`, keyed by model, prompt and
settings. Re-running is free and an interrupted run resumes. Failures are not cached, so they
are retried next time.

**Gold denominator.** The adjudicated set includes units a human reviewed and resolved to *no*
codes. Those exist only in `data/project-export.json` — the CSV lists positive labels only —
and are recovered from it, because dropping them would quietly reward a model for over-coding.

**Human comparisons.** The per-coder rows and the human-human kappas are derived from
`project-export.json` when `data/annotations-all-rounds.csv` is absent, so the baseline and
the ceiling analysis stay available without any extra exports. Kappa is computed over the 500
units both coders finished, so a code one coder omitted counts as a real disagreement rather
than a missing observation.

## Cost and runtime

GPT-5.1 at the default medium reasoning effort and 8 workers runs at roughly 1.5 units/second,
so the full 1,450 units takes about 15 minutes. A local Qwen runs at roughly 15–25 s per unit
on an Apple laptop, so the same job is an overnight run. The cache makes either safe to stop
and resume. Lower reasoning efforts are cheaper but scored worse on the test split, so measure
before trading them in.

## Layout

```
thematic_ai/
  config.py      run configuration and cache keys
  codebook.py    codebook loading, prompt rendering, code-name resolution
  data.py        corpus, gold labels, train/test split, optional per-coder rows
  prompts.py     system prompt, JSON schema, few-shot selection
  calibration.py observed coding behaviour and nearest-neighbour retrieval
  backends.py    OpenAI and Ollama clients behind one interface
  annotate.py    the annotation loop, validation, output assembly
  evaluate.py    multi-label metrics, kappa, human baseline and reliability, ceiling analysis
  pipeline.py    the wiring shared by the scripts and the notebooks
scripts/
  run_annotate.py     annotate the corpus with either backend
  run_experiment.py   train/test evaluation and model comparison
```
