# flamai-multilingual-eval-framework

Evaluation pipeline for the Falcon Language Support project. Compares 17 regional
LLMs against Gemma-4 26B A4B IT across translation and instruction-following tasks
in 17 languages, using automated metrics plus an LLM-as-judge pairwise comparison,
and classifies each regional model A–E against the baseline.

Runs entirely on [Modal](https://modal.com) — a FastAPI model registry, GPU
inference workers, CPU/GPU metric workers, and a judge worker, all wired together
by one state machine.

---

## Table of contents

- [Architecture](#architecture)
- [The pipeline state machine](#the-pipeline-state-machine)
- [Metrics catalog](#metrics-catalog)
- [Regional models](#regional-models)
- [Classification scheme](#classification-scheme)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Registry API](#registry-api)
- [Repository layout](#repository-layout)
- [Testing](#testing)
- [Known issues](#known-issues)

---

## Architecture

Two subsystems share one Modal app and one set of volumes:

```
┌─────────────────────────────────────────────────────────────┐
│  Modal app  (modal_app.py)                                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Registry service — hexagonal architecture, FastAPI │    │
│  │                                                     │    │
│  │  api/routes.py → RegistryService → RegistryStore    │    │
│  │  (port: Protocol)                                   │    │
│  │       ├─ VolumeRegistryStore   (production)         │    │
│  │       └─ InMemoryRegistryStore (tests)              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
<<<<<<< Updated upstream
│  ─────────── shared Modal volumes ──────────                │
=======
│  ─────────── shared Modal volumes ───────────               │
>>>>>>> Stashed changes
│  registry · weights · benchmarks · outputs                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Evaluation pipeline — one state machine            │    │
│  │                                                     │    │
│  │  run_pipeline()                                     │    │
│  │    → state_machine.advance()                        │    │
│  │      PENDING → INFERENCE → LIGHT_METRICS            │    │
│  │              → MODEL_METRICS → JUDGE → REPORT → DONE│    │
<<<<<<< Updated upstream
│  │                          ↘ FAILED (from any state   │    │
=======
│  │                          ↘ FAILED (from any state)  │    │
>>>>>>> Stashed changes
│  │                                                     │    │
│  │  Workers: VLLMWorker · LightMetricWorker ·          │    │
│  │  ModelMetricWorker · JudgeWorker · ReportGenerator  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Registry service.** `RegistryStore` is a `typing.Protocol`, not a base class —
the service layer (`RegistryService`) depends only on the protocol, never on a
concrete adapter. Swapping `VolumeRegistryStore` for `InMemoryRegistryStore` is
one environment variable (`PERSISTENCE_BACKEND=memory|volume`). The test suite
forces `memory` so every test runs against a fresh in-memory store with zero
Modal dependency. Domain objects (`ModelConfig`, `HardwareConfig`, `MetricConfig`,
`TaskConfig`) are plain dataclasses with no framework imports.

**Evaluation pipeline.** There is one state machine, not three separate
orchestrators. `run_pipeline()` is the single entrypoint — it takes a list of
`LanguageSpec`s (1 or 17) and a `stop_at` state, and drives every spec through
`advance()`. Running 1 language vs 17 only changes whether the work happens
inline in the current process or fanned out to parallel Modal containers; the
manifest writing, resume detection, and report rendering are identical code
either way.

---

## The pipeline state machine

```
PENDING → INFERENCE → LIGHT_METRICS → MODEL_METRICS → JUDGE → REPORT → DONE
                                                                          ↘ FAILED
```

| State | What happens | Worker |
|---|---|---|
| `INFERENCE` | Runs the regional model and (once, shared) Gemma-4 on the benchmark samples | `VLLMWorker` |
| `LIGHT_METRICS` | Runs all CPU-tier metrics (BLEU, chrF, instruction-following checks, number-verbalization checks) | `LightMetricWorker` |
| `MODEL_METRICS` | Runs GPU-tier metrics (BERTScore, back-translation; COMET intended but not currently wired — see [Known issues](#known-issues)) | `ModelMetricWorker` |
| `JUDGE` | Pairwise LLM-as-judge comparison, regional vs Gemma-4, with position-swapped repeats | `JudgeWorker` |
| `REPORT` | Aggregates everything into `final_report.json` and assigns an A–E classification | `ReportGenerator` |

Any state can transition to `FAILED`; the error is recorded on the
`LanguageRun` and surfaced in the final summary rather than crashing the run.

A precondition step, `ensure_gemma4_baseline`, runs once before any language
reaches `INFERENCE` — this is what prevents N parallel containers from each
spawning their own Gemma-4 inference run when evaluating multiple languages
at once.

### One entrypoint, not three

```bash
# Single language, full pipeline through judging + report
modal run modal_app.py::run_pipeline --slug greek

# Single language, inference + light metrics only (skip judge/report)
modal run modal_app.py::run_pipeline --slug greek --stop-at light_metrics

# A few languages in parallel
modal run modal_app.py::run_pipeline --slug arabic,hebrew,amharic

# All 17 regional models — prints a cost estimate, confirms before launching
modal run modal_app.py::run_pipeline --slug all

# Resume an existing run — already-completed languages are skipped
modal run modal_app.py::run_pipeline --slug all --run-id <existing-run-id>
```

| Flag | Default | Meaning |
|---|---|---|
| `--slug` | `tamil` | one slug, comma-separated list, or `all` |
| `--stop-at` | `report` | `light_metrics` or `report` |
| `--task` | `translation` | `translation` or `instructions` |
| `--limit` | `200` | prompts per language |
| `--judge-model` | `gemini-3.5-flash` | `gemini-*` or `claude-*` |
| `--judge-limit` | `50` | prompts judged per language (0 = all) |
| `--swap-runs` | `2` | position-swap repeats per judged prompt |
| `--skip-model-metrics` | `True` | skip BERTScore/back-translation (cost control) |
| `--run-id` | _(generated)_ | resume an existing run |

---

## Metrics catalog

Metrics are auto-discovered by scanning `src/metrics/{translation,instruction,number_verbalization}/`
for `BaseMetric` subclasses — adding a metric is dropping a file in the right
subpackage, no registration step. Each metric declares a `compute_tier`
(`LIGHT` → `LightMetricWorker`, CPU; `MODEL` → `ModelMetricWorker`, GPU) and,
for instruction metrics, a `category` that the worker uses to filter samples
before scoring.

### Translation (T1–T5)

| | Metric | Tier | Notes |
|---|---|---|---|
| T1 | BLEU | LIGHT | `sacrebleu`, both `en→target` and `target→en` |
| T2 | chrF | LIGHT | `sacrebleu` |
| T3 | COMET | MODEL | **Spec'd, not currently implemented** — see [Known issues](#known-issues) |
| T4 | BERTScore | MODEL | `microsoft/mdeberta-v3-base` |
| T5 | Back-translation consistency | MODEL | MarianMT round-trip + BLEU; several languages explicitly skipped where no reliable checkpoint exists rather than guessing one |

### Instruction-following (I1–I5)

| | Metric | Tier | Category | Notes |
|---|---|---|---|---|
| I1 | Language adherence | LIGHT | (all) | `langdetect`; Māori/Tok Pisin use a vocabulary-marker fallback since `langdetect` doesn't support either |
| I2 | Length constraint accuracy | LIGHT | `length_constraint` | rule-based, word/sentence counts |
| I3 | Format compliance | LIGHT | `structured_output` | regex against expected format |
| I4 | Topic boundary respect | LIGHT | `topic_boundary` | `sentence-transformers` cosine similarity, threshold 0.3 |
| I5 | Tone/register detection | LIGHT | `tone_style` | rule-based honorific/register markers per language — deliberately not a classifier (see docstring in `tone_register.py`); treat as directional |

### Number verbalization (N1–N8)

All LIGHT tier, all scoped to the `number_verbalization` category: digit-by-digit
compliance, word-form compliance, digit leakage, digit preservation, currency
unit accuracy, number-type classification, language of number words,
mixed-sentence consistency.

### LLM-as-judge

Independent of the metrics above. `JudgeWorker` makes pairwise calls (Gemini or
Anthropic) comparing the regional model's output against Gemma-4's for the same
prompt, across 3 dimensions (`fluency`/`adequacy`/`overall` for translation;
analogous dimensions for instructions), each judged twice with the A/B order
swapped to control for position bias. Verdicts feed `regional_win_rate`, which
is the primary signal for classification.

---

## Regional models

17 languages, each with a regional model evaluated against the shared Gemma-4
baseline:

| Language | Model | GPU | Notes |
|---|---|---|---|
| Tamil | `tamil-mistral-7b` | L4 | |
| Marathi | `mahamarathi-7b` | L4 | base model — no instruct variant |
| Kannada | `ambari-7b` | L4 | base model — no instruct variant |
| Gujarati | `gujju-llama-7b` | L4 | base model — no instruct variant |
| Arabic | `jais-2-8b` | L4 | |
| Hebrew | `dictalm-2-7b` | L4 | |
| Korean | `polyglot-ko-12b` | L40S | base model — no instruct variant |
| Malay | `mallam-5b` | L4 | |
| Swahili | `swahili-gemma-7b` | L4 | |
| Amharic | `walia-llm-7b` | L4 | |
| French | `lucie-7b` | L4 | |
| Swedish | `viking-7b` | L4 | base model — no instruct variant |
| Czech | `csmpt-7b` | L4 | base model — no instruct variant |
| Greek | `meltemi-7b` | L4 | |
| Brazilian Portuguese | `tucano-2b4` | L4 | |
| Māori | `goldfish-mri-39m` | T4 | base model — no instruct variant |
| Tok Pisin | `goldfish-tpi-125m` | T4 | base model — no instruct variant |

Base models receive plain-text prompts (no chat template) since they have no
instruct variant. Their instruction-following results should be read with that
in mind — `format_compliance`, `tone_register`, etc. are measuring continuation
behavior, not instruction-following, for these 8.

---

## Classification scheme

Each language's `final_report.json` entry gets an A–E grade based on
`regional_win_rate` from the judge (primary signal), refined by other metrics
where available:

| Grade | Meaning | Win rate |
|---|---|---|
| A | Regional superior | > 60% |
| B | Regional preferred | 50–60% |
| C | Comparable | 40–60% |
| D | Gemma-4 preferred | 30–40% |
| E | Gemma-4 superior | < 30% |

---

## Setup

### Prerequisites

- A [Modal](https://modal.com) account, `modal` CLI installed and authenticated
- A Gemini API key (free tier works for small `--judge-limit` runs; paid tier
  recommended for full runs — free tier is rate-limited to ~15 requests/minute)

### Modal secrets

```bash
modal secret create phase2a-registry-url REGISTRY_URL=<your-registry-url> JWT_TOKEN=<token>
modal secret create phase2a-auth-secrets JWT_SECRET=<32+ byte secret> HF_TOKEN=<huggingface-token>
modal secret create phase2a-judge GEMINI_API_KEY=<your-key>
```

`phase2a-judge` can also carry `ANTHROPIC_API_KEY` if you want to use a
`claude-*` judge model instead of `gemini-*` — both can coexist in the same
secret.

### Modal volumes

Created automatically on first use:

| Volume | Mount | Contents |
|---|---|---|
| `phase2a-registry` | `/data/registry` | model/hardware/metric/task configs |
| `phase2a-weights` | `/data/weights` | HuggingFace model cache |
| `phase2a-benchmarks` | `/data/benchmarks` | FLORES-200 + instruction-following datasets |
| `phase2a-outputs` | `/data/outputs` | inference outputs, metrics, judge verdicts, reports |

### Local install (registry tests only — the pipeline runs entirely on Modal)

No `requirements.txt` exists yet in this repo. To run the test suite locally:

```bash
pip install fastapi uvicorn pytest pyjwt httpx pydantic sacrebleu langdetect
PERSISTENCE_BACKEND=memory pytest tests/ -q
```

Worth adding a `requirements.txt` or `pyproject.toml` pinning these — none
currently exists, so versions aren't locked anywhere.

### Deploy

```bash
modal deploy modal_app.py
```

This deploys the registry service and exposes its URL — note it down, you'll
need it for the seed/template scripts below.

---

## Running the pipeline

`scripts/seed_registry.py` and `scripts/fetch_chat_templates.py` are plain
Python scripts that talk to the deployed registry over HTTP — they are not
run with `modal run`.

```bash
# 1. Generate a write-scoped JWT for the registry
export JWT_TOKEN=$(python scripts/jwt_token_generator.py --secret $JWT_SECRET --scopes registry:write)
export REGISTRY_URL=https://<your-deployed-registry-url>.modal.run

# 2. Seed the registry with the 17 model configs + hardware + metrics + tasks
python scripts/seed_registry.py

# 3. Fetch chat templates for instruct-variant models
#    (registry needs HF_TOKEN in phase2a-auth-secrets for gated repos)
python scripts/fetch_chat_templates.py

# 4. Try one language first
modal run modal_app.py::run_pipeline --slug greek --judge-limit 5

# 5. Once that looks right, run a small subset
modal run modal_app.py::run_pipeline --slug arabic,hebrew,amharic

# 6. Full run — prints a cost estimate and waits for confirmation
modal run modal_app.py::run_pipeline --slug all
```

Outputs land in `/data/outputs/runs/{run_id}/`:

```
runs/{run_id}/
  gemma4/{task}_outputs.jsonl
  regional/{slug}_{task}_outputs.jsonl
  metrics/{metric}_{slug}.jsonl
  judge/{slug}_{task}_verdicts.jsonl
  reports/
    final_report.json        # all languages run so far, this run_id
    {slug}_summary.md         # single-language run
    phase5_summary.md         # multi-language run
  run_manifest.json
```

---

## Registry API

FastAPI service deployed via Modal, exposing:

| Resource | Endpoints |
|---|---|
| Models | `GET/POST /models`, `GET/PATCH /models/{id}`, `PATCH /models/{id}/{enable,disable,deprecate}`, `POST /models/{id}/fetch-chat-template` |
| Hardware | `GET/POST /hardware`, `GET/PATCH /hardware/{id}` |
| Metrics | `GET/POST /metrics`, `GET/PATCH /metrics/{name}` |
| Tasks | `GET /tasks`, `GET/PATCH /tasks/{name}` |
| Runs | `GET/POST /runs`, `GET /runs/{id}` |
| Health | `GET /health` |

All mutating endpoints require a JWT bearer token with the appropriate scope
(`registry:read` / `registry:write`). Model state transitions (`active` →
`disabled`/`deprecated`) are validated by `src/core/services/state_machine.py`
— a separate, smaller state machine from the evaluation pipeline's; `deprecated`
is terminal.

---

## Repository layout

```
modal_app.py              Modal app — worker registrations, run_pipeline entrypoint
modal_common.py            Volumes, images, GPU presets

src/
  main.py                  FastAPI app
  deps.py                  PERSISTENCE_BACKEND dependency wiring

  core/
    domain/                ModelConfig, HardwareConfig, MetricConfig, TaskConfig
    ports/                 RegistryStore protocol
    services/
      registry_service.py  business logic, depends only on the port
      state_machine.py      model lifecycle transitions (active/disabled/deprecated)

  adapters/persistence/
    volume_store.py         production: Modal volume-backed
    memory_store.py         tests: in-memory

  api/
    routes.py                all registry endpoints
    auth.py                  JWT verification
    schemas.py               Pydantic request/response models

  pipeline/
    state_machine.py         the evaluation pipeline state machine
    entrypoints.py           run_pipeline() — the one entrypoint
    report_render.py         markdown report renderer
    loader.py                 benchmark sample loading
    run.py                    path helpers, manifest I/O

  workers/
    inference.py              VLLMWorker (no src/ imports — see note below)
    light_metrics.py           LightMetricWorker, auto-discovery
    model_metrics.py           ModelMetricWorker, auto-discovery
    judge.py                   JudgeWorker — Gemini/Anthropic pairwise judge
    reporter.py                ReportGenerator, classification logic

  metrics/
    translation/               T1–T5
    instruction/                I1–I5
    number_verbalization/       N1–N8
    base.py                     BaseMetric, ComputeTier, MetricStage

scripts/
  seed_registry.py             populates the 17 model configs
  fetch_chat_templates.py      pulls chat templates for instruct models
  jwt_token_generator.py       generates JWTs for manual API testing

tests/                         136 tests, run with PERSISTENCE_BACKEND=memory
```

`src/workers/inference.py` deliberately does not import anything from the rest
of `src/` — the vLLM Modal image doesn't have the registry package installed,
so it re-declares the small subset of `ModelConfig` fields it needs as a local
`_ModelInfo` dataclass and talks to the registry over plain HTTP.

---

## Testing

```bash
PERSISTENCE_BACKEND=memory pytest tests/ -q
```

<<<<<<< Updated upstream
136 tests across registry CRUD, auth/JWT scopes, model state transitions, path
helpers, manifest I/O, the evaluation state machine, and metric correctness
(including the `applies_to()` category-filtering contract — see below).

Light-tier metric tests run with the real `sacrebleu`/`langdetect` packages
installed; GPU-tier metrics (BERTScore, back-translation) and `sentence-transformers`-based
metrics (topic boundary) are not covered by the test suite — they require
real model downloads and are exercised manually against live runs instead.
=======
148 tests across registry CRUD, auth/JWT scopes, model state transitions, path
helpers, manifest I/O, the evaluation state machine, metric correctness
(including the `applies_to()` category-filtering contract and exact-set
discovery checks for both LIGHT and MODEL tiers), and `JudgeWorker` structural
integrity (`tests/test_judge.py` — see Known issues below for why this exists).

Light-tier metric tests run with the real `sacrebleu`/`langdetect` packages
installed and exercise `compute()` against real sample data. GPU-tier metrics
(COMET, BERTScore, back-translation) and `sentence-transformers`-based metrics
(topic boundary) are covered by discovery and tier-classification tests only
— `test_discover_finds_all_3_model_metrics` confirms all three MODEL-tier
classes are found, but their `compute()` methods require real model
downloads and aren't exercised by the automated suite; numerical correctness
against real checkpoints is verified manually against live runs instead.
>>>>>>> Stashed changes

---

## Known issues

<<<<<<< Updated upstream
- **COMET (T3) has no working implementation.** `src/metrics/translation/comet.py`
  is a duplicate of `src/workers/model_metrics.py`'s content at the wrong path
  — it defines `ModelMetricWorker` and discovery logic again, not a
  `COMETMetric` class. `ModelMetricWorker`'s auto-discovery will find
  `BERTScoreMetric` and `BackTranslationMetric` but never a COMET metric.
  Needs a real `COMETMetric(BaseMetric)` written and placed in that file.
- **`modal_common.py` packaging has been inconsistent across exports** — verify
  before deploying that it actually contains volume mounts, image definitions,
  and `build_registry_config`/`env_config`, not a duplicate of `modal_app.py`.
- **`src/workers/judge.py` has had duplicate class definitions appear in past
  versions** of this file (two full `JudgeWorker` classes in one file, the
  second silently shadowing the first with stale defaults). Worth a one-time
  check (`grep -c "class JudgeWorker" src/workers/judge.py` should print `1`)
  before relying on judge behavior matching what's documented here.
- **Back-translation (T5) has no checkpoint for Malay, Swahili, Brazilian
  Portuguese, or Tok Pisin.** The metric explicitly skips these and records
  why in `MetricResult.notes`, rather than guessing a multilingual fallback
  model that could produce a misleadingly low score. Korean's mapped
  checkpoint (`opus-mt-ko-en`) was not independently verified at
  implementation time.
- **Tone/register detection (I5) is a rule-based heuristic**, not a trained
  classifier, and coverage is asymmetric across languages — some have
  well-documented formal/informal lexical splits, others have weak or no
  reliable markers. Treat I5 scores as directional.
- **`test_discover_finds_all_13_light_metrics`** in `tests/test_metrics.py`
  predates I4/I5 and only asserts a subset relationship (`missing = expected - names`),
  so it would not catch a regression in `topic_boundary` or `tone_register`
  discovery. Worth adding explicit assertions for both.
=======
These have been fixed and now have regression tests guarding them — listed
here so the history and the test coverage are visible, not because they're
currently broken:

Still open:

- **`modal_common.py` packaging has been inconsistent across exports** — at
  least one export of this repo had `modal_common.py` containing a near-copy
  of `modal_app.py`'s content instead of the volume mounts, image
  definitions, GPU presets, and `build_registry_config`/`env_config` it's
  supposed to contain. There's no automated check for this — verify before
  deploying that `vllm_image`, `model_metrics_image`, `judge_image`,
  `VOLUME_MOUNTS`, `build_registry_config`, and `env_config` are all actually
  *defined* in `modal_common.py`, not just imported successfully (an import
  succeeding doesn't mean the names resolve to the right objects if the
  file itself was swapped). Worth adding a `tests/test_modal_common.py`
  that imports these names and asserts their types.
- **Back-translation (T5) has no checkpoint for Malay, Swahili, Brazilian
  Portuguese, or Tok Pisin.** The metric explicitly skips these and records
  why in `MetricResult.notes`, rather than guessing a multilingual fallback
  model that could produce a misleadingly low score.
- **Tone/register detection (I5) is a rule-based heuristic**, not a trained
  classifier, and coverage is asymmetric across languages — some have
  well-documented formal/informal lexical splits, others have weak or no
  reliable markers. Treat I5 scores as directional. This is a deliberate
  design choice (see the docstring in `tone_register.py`), not a bug.
>>>>>>> Stashed changes
