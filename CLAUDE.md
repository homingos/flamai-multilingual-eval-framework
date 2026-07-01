# Falcon Language Support — Claude Project Instructions

## What This Project Is

AI/ML research and implementation project for adding robust multilingual support to Flam AI's Talking Avatar product. The work involves evaluating regional language models, fine-tuning Whisper (audio encoder), and establishing cross-language validation methodology.

**Current LLM in production:** Gemma-4 26B A4B IT (multilingual pre-trained)
**Project status:** Experimentation
**Notion task page:** https://app.notion.com/p/38078ca0ce5280dcb7d1e12e0c397e45

---

## Current Status (as of 2026-06-26)

### Direction change: European Language Expansion
**Pivoting to pan-European multilingual challenger models across 30 European languages.**
- Sarvam-M-24B qualitative runs for Kannada, Marathi, Gujarati: **CANCELLED**
- Those languages remain on Gemma-4 for now
- Next step: tokenizer test on 8 European challenger models × 30 languages

### Task 1 — Tokenizer Evaluation (original 17 languages): ✅ Complete
17 regional models selected across 17 languages. Results in `data/results.csv`, report in `docs/llm-evaluation.md`.

### Task 1 — Tokenizer Evaluation (European expansion): ✅ Complete (2026-06-26)
Ran 8 European challenger models' tokenizers on FLORES-200 for 30 European languages.
Decision rule: `fertility < Gemma-4 AND vocab_coverage ≥ 80% AND roundtrip ≥ 95%`
Results: `data/european_results.csv`, summary: `data/european_summary.json`
Script: `experiments/european_tokenizer_test.py`

**Gate results by model (final — all 30 languages with baselines):**
| Model | Passes | Languages |
|---|---|---|
| `utter-project/EuroLLM-22B-Instruct-2512` | **24/30** | DE, IT, PT, NL, PL, RO, UK, SV, CS, EL, RU, DA, FI, HU, HR, SK, SL, BG, LT, LV, ET, GA, NB, MT |
| `CohereLabs/aya-vision-32b` | **13/30** | FR, DE, ES, IT, PT, NL, PL, RO, UK, CS, EL, RU, TR |
| `openGPT-X/Teuken-7B-instruct-v0.6` | 1/30 | BG only (fertility 1.335 vs G4 2.055) |
| `meta-llama/Llama-3.3-70B-Instruct` | 1/30 | CS only |
| `VAGOsolutions/Llama-3.1-SauerkrautLM-70b-Instruct` | 1/30 | CS only (identical tokenizer to Llama-3.3) |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | **0/30** | roundtrip 0% on all languages (Tekken tokenizer) |
| `BramVanroy/GEITje-7B-ultra` | **0/30** | fertility too high + vocab gaps across most languages |
| `TildeAI/TildeOpen-30b` | **0/30** | roundtrip 0% on all languages |

**Official EU language coverage (24 official; English not tested):**
- EuroLLM-22B: **21/23** — fails only French (fert 1.525 vs G4 1.490) and Spanish (fert 1.383 vs G4 1.347), both by < 3%
- Aya-Vision-32B: **13/23**
- Gemma-4 still best for: Serbian, Icelandic, Albanian (no challenger wins), French + Spanish (EuroLLM barely misses)

**Key findings:**
- EuroLLM-22B dominates Central/Eastern/Northern Europe — sweeps all Baltic, Nordic, Slavic, and Celtic languages
- EuroLLM fails French and Spanish only because Gemma-4's tokenizer is already extremely efficient on those high-resource Western languages (delta < 0.04 tokens/word)
- Teuken-7B surprise win on Bulgarian: fertility 1.335 vs G4 2.055 (35% fewer tokens)
- Aya-Vision-32B wins on Arabic-script/Cyrillic/Turkish where EuroLLM is weaker; near-perfect vocab coverage (98–100%)
- Gemma-4 wins Serbia, Iceland, Albania — no challenger has sufficient vocab coverage or fertility advantage

**Eliminated models (do not proceed to qualitative):** Mistral-Small-3.2, GEITje-7B, TildeOpen-30B

**40 (model × language) pairs queued for qualitative eval:**
- EuroLLM-22B × 24 languages (DE, IT, PT, NL, PL, RO, UK, SV, CS, EL, RU, DA, FI, HU, HR, SK, SL, BG, LT, LV, ET, GA, NB, MT)
- Aya-Vision-32B × 13 languages (FR, DE, ES, IT, PT, NL, PL, RO, UK, CS, EL, RU, TR)
- Llama-3.3-70B × CS — 1 pair
- SauerkrautLM-70B × CS — 1 pair
- Teuken-7B × BG — 1 pair

**8 Challenger models (for reference):**
| Role | HuggingFace Model ID | Params |
|---|---|---|
| General multilingual | `meta-llama/Llama-3.3-70B-Instruct` | 70B |
| European multilingual | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 24B |
| EU sovereign | `openGPT-X/Teuken-7B-instruct-v0.6` | 7B |
| European benchmark | `utter-project/EuroLLM-22B-Instruct-2512` | 22B |
| German specialist | `VAGOsolutions/Llama-3.1-SauerkrautLM-70b-Instruct` | 70B |
| Dutch specialist | `BramVanroy/GEITje-7B-ultra` | 7B |
| Nordic/underrepresented | `TildeAI/TildeOpen-30b` | 30B (base; use `martinsu/tildeopen-30b-mu-instruct` for Phase 2) |
| Multilingual instruction | `CohereLabs/aya-vision-32b` | 32B |

**30 target languages:** French, German, Spanish, Italian, Portuguese, Dutch, Polish, Romanian, Ukrainian, Swedish, Czech, Greek, Russian, Danish, Finnish, Hungarian, Turkish, Croatian, Slovak, Slovenian, Bulgarian, Lithuanian, Latvian, Estonian, Irish, Norwegian, Maltese, Serbian, Icelandic, Albanian *(Welsh excluded — no model support)*

### Task 1B — Qualitative LLM Validation (original languages): 🔄 Partial

Two evaluation tasks per language: **Translation** and **Instruction Following**.
Pipeline runs on Modal (`modal_app.py`). All infrastructure bugs have been resolved (see bug fix log below).

#### Translation task
- **Greek** (Meltemi-7B): ✅ Grade B — win rate 53%, BLEU 27.38, BERTScore F1 0.8545 vs 0.8474 — run ID `2026-06-19_091321_3c0719`
- **Tamil** (Tamil-Mistral-7B): ✅ Grade E — Gemma-4 strongly preferred (win rate ~0%) — run ID `2026-06-23_085618_29e74c`
- **Marathi** (MahaMarathi-7B): ✅ Grade E — Gemma-4 strongly preferred (win rate 0%) — run ID `2026-06-23_085823_bf781f`
- **Kannada** (Ambari-7B): ✅ Grade E — Gemma-4 strongly preferred (win rate 0%, BLEU 2.07 vs 15.41) — run ID `2026-06-23_105532_c67e18`
- **Gujarati** (Gujju-Llama-7B): ✅ Grade E — Gemma-4 strongly preferred (win rate 0%, BLEU 0.0 vs 20.02, chrF 1.27 vs 50.56) — run ID `2026-06-23_131533_bad6cb`
- **Arabic** (Jais-2-8B): ✅ Grade E — Gemma-4 strongly preferred (win rate 13%, BLEU 27.0 vs 25.87, chrF 54.93 vs 55.62) — run ID `2026-06-24_073508_1b374d`
- **Korean** (Polyglot-Ko-12B): ✅ Grade E — Gemma-4 strongly preferred (win rate 0%, BLEU 0.02 vs 15.53, chrF 0.64 vs 37.42) — run ID `2026-06-24_101013_5ad649`
- **Hebrew** (DictaLM-3.0-Nemotron-12B): ✅ Grade E — Gemma-4 strongly preferred (win rate 0%, BLEU 0.04 vs 31.39, chrF 0.91 vs 58.64) — run ID `2026-06-24_114351_159a77`
- All other 9 languages: Pending

#### Instruction Following task
- **Greek** (Meltemi-7B): ✅ Grade E — Gemma-4 strongly preferred (win rate 2%) — run ID `2026-06-19_115923_1b890f`
- **Tamil** (Tamil-Mistral-7B): ✅ Grade E — Gemma-4 strongly preferred (win rate 3%) — run ID `2026-06-23_072240_70be14`
- **Marathi** (MahaMarathi-7B): ✅ Grade E — Gemma-4 strongly preferred (win rate 0%) — run ID `2026-06-23_084607_6b35b3`
- **Kannada** (Ambari-7B): ⚠️ Skipped — GPU inference crash (208/1200 samples, pipeline failed at LIGHT_METRICS on partial data). Translation Grade E confirms nothing to distill. Failed run ID `2026-06-23_121912_8dc209`.
- **Gujarati** (Gujju-Llama-7B): ✅ Grade E — Gemma-4 strongly preferred (win rate 0%) — run ID `2026-06-23_131533_46c68c`
- **Arabic** (Jais-2-8B): ✅ Grade D — Gemma-4 preferred (win rate 24%) — run ID `2026-06-24_073405_e6e60d`
- **Korean** (Polyglot-Ko-12B): ✅ Grade E — Gemma-4 strongly preferred (win rate 0%) — run ID `2026-06-24_101013_9ce150`
- **Hebrew** (DictaLM-3.0-Nemotron-12B): ✅ Grade E — Gemma-4 strongly preferred (win rate 0%) — run ID `2026-06-24_114214_47d7ab`
- All other 9 languages: Pending (see teammate assignment below)

---

## Bug Fix Log (resolved 2026-06-23)

Three bugs were found and fixed during the first parallel run attempt. Do not revert these.

### 1. Gemma-4 baseline ID mismatch (critical — affected all languages)
**File:** `src/pipeline/state_machine.py`, `src/pipeline/entrypoints.py`, `modal_app.py`
**Problem:** `ensure_gemma4_baseline` generated baseline outputs using only the first language's sample IDs (e.g. `inst_amharic_001`). Each language has its own IDs (`inst_arabic_001`, `inst_tamil_001`, etc.), so the judge found 0 matching pairs for every language except the first → 0 verdicts across the board.
**Fix:** Changed `any_slug: str` → `all_slugs: list`. Now loops over every slug in the run, generating Gemma-4 outputs for each language's sample IDs before regional inference starts.

### 2. French model 404
**File:** Registry volume (`phase2a-registry/models.json`)
**Problem:** `OpenLLM-France/Lucie-7B-Instruct` was made private on HuggingFace.
**Fix:** Updated to `OpenLLM-France/Lucie-7B-Instruct-v1.1` in the registry. Already uploaded.

### 3. Goldfish models max_model_len
**File:** Registry volume (`phase2a-registry/models.json`)
**Problem:** Māori (Goldfish-mri-39M) and Tok Pisin (Goldfish-tpi-125M) had `max_model_len=2048` in registry but models only support 512 max positions → vLLM crash.
**Fix:** Updated to `max_model_len=512` in registry. Already uploaded.

### 4. Czech stale cache (pre-step required before running)
**File:** `phase2a-weights` Modal volume
**Problem:** `flash_attn_triton.py` missing from cached weights for `BUT-FIT/csmpt7b` — stale file from a different HF commit.
**Fix:** A `clear_csmpt_cache` utility has been added to `modal_app.py`. Run this ONCE before running Czech:
```bash
modal run modal_app.py::clear_csmpt_cache
```

---

## EuroLLM-22B Per-Language Checklist

Run this exact sequence for each of the 23 remaining EU languages. Do not start the next language until both tasks are done and `/viz` is updated.

**Step 1 — Pre-translate instruction dataset**
```bash
modal run --detach experiments/translate_instruction_dataset.py --slug <lang>
```
Wait for it to complete (check Modal dashboard or logs). This generates the translated prompts that the instructions eval pipeline reads.

**Step 2 — Translation eval**
```bash
echo "" | modal run --detach modal_app.py::run_pipeline \
  --model-id eurollm-22b --slug <lang> --task translation --n-samples 200
```
Save the printed run ID. Check when done: `modal run modal_app.py::check_run --run-id <run_id>`

**Step 3 — Instructions eval**
```bash
echo "" | modal run --detach modal_app.py::run_pipeline \
  --model-id eurollm-22b --slug <lang> --task instructions --n-samples 200
```
Save the printed run ID. Check when done: `modal run modal_app.py::check_run --run-id <run_id>`

**Step 4 — Update dashboard + CLAUDE.md + Notion**
```
/viz add <Language> EuroLLM-22B translation run_id=<id> and instructions run_id=<id>
```
This updates `qualitative_results.json`, pre-generates review HTML, updates CLAUDE.md tables (Ram's Queue, Instruction Metrics, Translation Metrics, full eval runs), updates both Notion tables, commits, and deploys.

**Remaining languages in order:**
`italian`, `portuguese`, `dutch`, `polish`, `romanian`, `ukrainian`, `swedish`, `czech`, `greek`, `russian`, `danish`, `finnish`, `hungarian`, `croatian`, `slovak`, `slovenian`, `bulgarian`, `lithuanian`, `latvian`, `estonian`, `irish`, `norwegian`, `maltese`

**Note:** Steps 2 and 3 can run in parallel (translation and instructions are independent). Step 1 must complete before Step 3 (instructions needs translated prompts). Step 2 has no dependency on Step 1.

---

## EuroLLM-22B Evaluation — Key Run IDs

### Smoke test runs (2026-06-29, limit=20, stop_at=light_metrics)
| Run ID | Task | Notes |
|---|---|---|
| `2026-06-29_095132_b337bf` | translation | 24 EU languages × 20 samples — PASSED |
| `2026-06-29_101435_3b48dd` | instructions | 24 EU languages × 20 samples — PASSED |

### Full eval runs (limit=1000, stop_at=report)
| Run ID | Task | Language | Grade | Notes |
|---|---|---|---|---|
| `2026-06-29_105640_e350db` | translation | German | ❌ E | BLEU 42.36 vs 41.35, chrF 65.72 vs 65.09 — judge prefers Gemma-4 (76% win) |
| `2026-07-01_070156_f721ae` | instructions | German | ❌ E | Lang adh 99%, format 100%, but judge strongly prefers Gemma-4 (84% win) |
| `2026-07-01_094424_d04bf7` | translation | Italian | ❌ E | BLEU 32.61 vs 33.58, chrF 58.76 vs 59.19 — judge prefers Gemma-4 (69% win) |
| `2026-07-01_103347_5e75d1` | instructions | Italian | ❌ E | Lang adh 99.8%, format 100%, but judge strongly prefers Gemma-4 (87% win) |

---

## How to Run a Language

Each language needs **both tasks** completed before a final verdict can be made. Run instructions and translation sequentially for each language — do not move to the next language until both tasks are done.

**Run instructions task:**
```bash
echo "" | modal run --detach modal_app.py::run_pipeline --slug <slug> --task instructions --n-samples 200
```

**Run translation task:**
```bash
echo "" | modal run --detach modal_app.py::run_pipeline --slug <slug> --task translation --n-samples 200
```

The `--detach` flag keeps the run alive after your terminal disconnects. The run ID is printed at startup — save it.

**Check status of a run:**
```bash
modal run modal_app.py::check_run --run-id <run_id>
```

**Available slugs (original 17 languages):**
`tamil`, `marathi`, `kannada`, `gujarati`, `arabic`, `hebrew`, `korean`, `malay`, `swahili`, `amharic`, `french`, `swedish`, `czech`, `brazilian_portuguese`, `maori`, `tok_pisin`

**EuroLLM-22B slugs (24 EU languages — always pass `--model-id utter-project/EuroLLM-22B-Instruct-2512`):**
`german`, `italian`, `portuguese`, `dutch`, `polish`, `romanian`, `ukrainian`, `swedish`, `czech`, `greek`, `russian`, `danish`, `finnish`, `hungarian`, `croatian`, `slovak`, `slovenian`, `bulgarian`, `lithuanian`, `latvian`, `estonian`, `irish`, `norwegian`, `maltese`

**Estimated cost per language:** ~$0.75 (both tasks: 2 × ~$0.38) — v2 uses 200 samples and no Gemma-4 baseline
**Estimated time per language:** ~30 minutes (two sequential ~15-min runs)

**After each task completes**, check the report:
```bash
modal run modal_app.py::check_run --run-id <run_id>
```
A valid result has a Grade (A–D) with an `avg_score` (0–1). If it shows "?" or no judge verdicts, something went wrong — check the logs.

---

## After Each Run: Update Notion

**Notion page ID:** `38078ca0-ce52-80dc-b7d1-e12e0c397e45`
**URL:** https://app.notion.com/p/38078ca0ce5280dcb7d1e12e0c397e45

After every completed run, pull the report JSON from the Modal volume and update the Notion tables:

```bash
# Download to /tmp for reading metrics
modal volume get phase2a-outputs runs/<run_id>/reports/<slug>_summary.json /tmp/<slug>_report.json

# Also save to repo (filename includes run ID for traceability)
modal volume get phase2a-outputs runs/<run_id>/reports/<slug>_summary.json data/reports/<slug>_<task>_<run_id>.json
```

Then read the JSON and update the relevant Notion table row:

**Translation task table** (v2 pointwise — score-based):
- `avg_score` → Avg Score (0–1)
- `fluency_avg` → Fluency avg
- `adequacy_avg` → Adequacy avg
- `overall_avg` → Overall avg
- `sample_count` → Samples judged
- `classification` → Grade (prefix ✅ for A/B, ⚠️ for C, ❌ for D)

**Instruction following task table** (v2 pointwise — score-based):
- `avg_score` → Avg Score (0–1)
- `language_compliance_avg` → Language compliance avg
- `instruction_following_avg` → Instruction following avg
- `helpfulness_avg` → Helpfulness avg
- `overall_avg` → Overall avg
- `sample_count` → Samples judged
- `classification` → Grade (prefix ✅ for A/B, ⚠️ for C, ❌ for D)
- Add a short Notes entry describing the primary failure mode

**Grade mapping (v2):** A ≥ 0.75, B ≥ 0.50, C ≥ 0.25, D < 0.25. Only A and B grades move forward to Whisper fine-tuning.

Also update the CLAUDE.md progress table (Ram's Queue section) after each run.

---

## Model Selection Principle

**Always use the largest/most up-to-date version of a regional model family when available.**

Task 1 tokenizer evaluation used smaller models (e.g. Jais-2-8B) as representatives — the tokenizer is shared across model sizes in the same family, so the tokenizer score is identical regardless of size. But for Task 1B qualitative validation, a larger model may meaningfully outperform Gemma-4 even when the smaller version doesn't.

**Decision (2026-06-24):** After Korean + Hebrew are complete, re-run Arabic using **Jais-2-70B-Chat** (`inceptionai/Jais-2-70B-Chat`) instead of Jais-2-8B. If Jais-2-70B beats Gemma-4 26B, the plan is to distill that knowledge into Gemma rather than serving the 70B directly (cost/latency constraint for production).

Apply this principle going forward: before finalising a Grade E/D verdict for any language, check if a significantly larger model exists in the same family. If yes, run it.

### Upgrade candidates identified (2026-06-24)

| Language | Model Used | Upgrade | Notes |
|---|---|---|---|
| Arabic | Jais-2-8B | **Jais-2-70B-Chat** (`inceptionai/Jais-2-70B-Chat`) | Planned — run after Korean + Hebrew |
| Hebrew | DictaLM-2.0-7B | **DictaLM-3.0-Nemotron-12B** (`dicta-il/DictaLM-3.0-Nemotron-12B-Instruct`) | Registry updated — clean instruct variant (24B-Thinking outputs `<think>` blocks that corrupt judge) |
| Korean | Polyglot-Ko-12B | **EXAONE-3.5-32B-Instruct** (`LGAI-EXAONE/EXAONE-3.5-32B-Instruct`) | Polyglot-Ko is a base model (no instruct tuning, 2022) — re-run with EXAONE after current run |
| Greek | Meltemi-7B | **Krikri-8B-Instruct** (`ilsp/Llama-Krikri-8B-Instruct`) | Same ILSP lab, Llama 3.1 base, May 2025 — re-run both tasks |
| Tamil | Tamil-Mistral-7B | **Sarvam-M (24B)** (`sarvamai/sarvam-m`) | Single model covers all 4 Indic langs; +23% MMLU-IN vs Mistral Small; comparable to Llama-3.3 70B |
| Kannada | Ambari-7B | **Sarvam-M (24B)** (`sarvamai/sarvam-m`) | Same model — Kannada is in Sarvam-M's 11-language list |
| Marathi | MahaMarathi-7B | **Sarvam-M (24B)** (`sarvamai/sarvam-m`) | Same model — Marathi explicitly supported |
| Gujarati | Gujju-Llama-7B | **Sarvam-M (24B)** (`sarvamai/sarvam-m`) | Same model — Gujarati explicitly supported |

**IMPORTANT — do not overwrite old results:** When re-running a language with a better model, always ADD a new row in both the Notion tables and the CLAUDE.md Ram's Queue table. Keep the original row (e.g. Tamil-Mistral-7B Grade E) intact for comparison. Each row = one model evaluation, not one language.

**Indic re-run escalation path (all 4 languages: Tamil, Kannada, Marathi, Gujarati):**
1. **Sarvam-M 24B** (`sarvamai/sarvam-m`) — single L40S, drop-in. Run first.
2. **Sarvam-30B** (`sarvamai/sarvam-30b`) — single A100 80GB (or L40S in FP8). Run if M is inconclusive (Grade C).
3. **Sarvam-105B** (`sarvamai/sarvam-105b`) — requires multi-GPU Modal worker (tensor_parallel_size=8 per model card). Only build this if 30B is still inconclusive. High cost.

**Registry changes needed before Indic re-runs:** Update slugs `tamil`, `kannada`, `marathi`, `gujarati` to point to `sarvamai/sarvam-m` (gpu_preset: `l40s`, 24B, chat_template: null). Disable thinking mode at inference by passing `enable_thinking=False` in the chat template call.

---

## Language Assignments

**Ram's languages:** Tamil, Marathi, Kannada, Gujarati, Arabic, Korean, Hebrew
**Teammate's languages:** Swedish, Malay, Amharic, French, Czech, Māori, Tok Pisin, Swahili, Brazilian Portuguese

Do NOT run languages assigned to the other person — each person runs their own set end-to-end (both tasks).

---

## Ram's Queue

### Current progress
| Language | Model | Instructions | Translation |
|---|---|---|---|
| Greek | Meltemi-7B | ✅ Grade E (win rate 2%) · `2026-06-19_115923_1b890f` | ✅ Grade B (win rate 53%) · `2026-06-19_091321_3c0719` |
| Tamil | Tamil-Mistral-7B | ✅ Grade E (win rate 3%) · `2026-06-23_072240_70be14` | ✅ Grade E (win rate 0%) · `2026-06-23_085618_29e74c` |
| Marathi | MahaMarathi-7B | ✅ Grade E (win rate 0%) · `2026-06-23_084607_6b35b3` | ✅ Grade E (win rate 0%) · `2026-06-23_085823_bf781f` |
| Kannada | Ambari-7B | ⚠️ skipped — inference crash · `2026-06-23_121912_8dc209` | ✅ Grade E (win rate 0%) · `2026-06-23_105532_c67e18` |
| Gujarati | Gujju-Llama-7B | ✅ Grade E (win rate 0%) · `2026-06-23_131533_46c68c` | ✅ Grade E (win rate 0%) · `2026-06-23_131533_bad6cb` |
| Arabic | Jais-2-8B | ✅ Grade D (win rate 24%) · `2026-06-24_073405_e6e60d` | ✅ Grade E (win rate 13%) · `2026-06-24_073508_1b374d` |
| Korean | Polyglot-Ko-12B | ✅ Grade E (win rate 0%) · `2026-06-24_101013_9ce150` | ✅ Grade E (win rate 0%) · `2026-06-24_101013_5ad649` |
| Hebrew | DictaLM-3.0-Nemotron-12B | ✅ Grade E (win rate 0%) · `2026-06-24_114214_47d7ab` | ✅ Grade E (win rate 0%) · `2026-06-24_114351_159a77` |
| Tamil | Sarvam-M-24B | ✅ Grade E (win rate 5%) · `2026-06-25_081543_85cde3` | ✅ Grade E (win rate 6.7%) · `2026-06-25_094523_dace81` |
| German | EuroLLM-22B | ✅ Grade E (win rate 15%) · `2026-07-01_070156_f721ae` | ✅ Grade E (win rate 24%) · `2026-06-29_105640_e350db` |
| Italian | EuroLLM-22B | ❌ Grade E (win rate 12.67%) · `2026-07-01_103347_5e75d1` | ❌ Grade E (win rate 18.67%) · `2026-07-01_094424_d04bf7` |

### Instruction Metrics Detail
(R = Regional model, G = Gemma-4 baseline; all values as %)

| Language | Model | Lang Adh (R) | Lang Adh (G) | Format (R) | Format (G) | Length (R) | Length (G) | Tone (R) | Tone (G) | Judge Win |
|---|---|---|---|---|---|---|---|---|---|---|
| Greek | Meltemi-7B | 2 | — | 100 | 100 | 39 | 88 | — | — | 2 |
| Tamil | Tamil-Mistral-7B | 100 | 100 | 0 | 100 | 33 | 88 | 48 | 55 | 3 |
| Marathi | MahaMarathi-7B | 90 | 100 | 50 | 100 | 48 | 88 | 50 | 75 | 0 |
| Kannada | Ambari-7B | — | — | — | — | — | — | — | — | skipped |
| Gujarati | Gujju-Llama-7B | 65 | 100 | — | 100 | 88 | 88 | — | 51 | 0 |
| Arabic | Jais-2-8B | 98 | 100 | 100 | 100 | 63 | 90 | 53 | 100 | 24 |
| Hebrew | DictaLM-3.0-Nemotron-12B | 0 | 99 | 0 | 100 | 58 | 90 | — | 61 | 0 |
| Korean | Polyglot-Ko-12B | 40 | 98 | 0 | 100 | 33 | 88 | 62 | 100 | 0 |
| Tamil | Sarvam-M-24B | 96 | 100 | 100 | 100 | 86 | 88 | 36 | 55 | 5 |
| German | EuroLLM-22B | 99 | 98 | 100 | 100 | 89.5 | 90 | — | — | 15 |
| Italian | EuroLLM-22B | 99.8 | 98 | 100 | 100 | 88.5 | 90 | — | — | 12.67 |

### Translation Metrics Detail

| Language | Model | BLEU (R) | BLEU (G) | chrF (R) | chrF (G) | Judge Win |
|---|---|---|---|---|---|---|
| Greek | Meltemi-7B | 27.38 | 26.04 | 51.27 | 50.55 | 53 |
| Tamil | Tamil-Mistral-7B | 0.01 | 13.25 | 4.63 | 50.96 | 0 |
| Marathi | MahaMarathi-7B | 0.02 | 15.48 | 6.55 | 47.85 | 0 |
| Kannada | Ambari-7B | 2.07 | 15.41 | 15.17 | 50.20 | 0 |
| Gujarati | Gujju-Llama-7B | 0.0 | 20.02 | 1.27 | 50.56 | 0 |
| Arabic | Jais-2-8B | 27.0 | 25.87 | 54.93 | 55.62 | 13 |
| Hebrew | DictaLM-3.0-Nemotron-12B | 0.04 | 31.39 | 0.91 | 58.64 | 0 |
| Korean | Polyglot-Ko-12B | 0.02 | 15.53 | 0.64 | 37.42 | 0 |
| Tamil | Sarvam-M-24B | 1.61 | 13.25 | 25.10 | 50.95 | 6.7 |
| German | EuroLLM-22B | 42.36 | 41.35 | 65.72 | 65.09 | 24 |
| Italian | EuroLLM-22B | 32.61 | 33.58 | 58.76 | 59.19 | 18.67 |

### Next up
**European tokenizer expansion** — run tokenizer tests on the 8 challenger models above across all 30 European languages. Then qualitative eval for (model, language) pairs that beat Gemma-4's tokenizer.

*Sarvam-M-24B Indic re-runs (Tamil, Kannada, Marathi, Gujarati): CANCELLED as of 2026-06-26.*
*Korean re-run (EXAONE), Greek re-run (Krikri): on hold pending European phase results.*

---

## Teammate's Queue

> **For the teammate's Claude Code session:** Run the languages below one at a time — both tasks (instructions + translation) per language before moving to the next. Check that each report shows a real Grade with a win rate percentage before starting the next run.

### Current progress
| Language | Model | Instructions | Translation |
|---|---|---|---|
| Swedish | Viking-7B | Pending | Pending |
| Malay | MaLLaM-5B | Pending | Pending |
| Amharic | Walia-LLM-7B | Pending | Pending |
| French | Lucie-7B-Instruct-v1.1 | Pending | Pending |
| Czech | CSMPT-7B | Pending | Pending |
| Māori | Goldfish-mri-39M | Pending | Pending |
| Tok Pisin | Goldfish-tpi-125M | Pending | Pending |
| Swahili | Swahili-Gemma-7B | Pending | Pending |
| Brazilian Portuguese | Tucano-2b4 | Pending | Pending |

### Run order and commands

**1. Swedish** — no pre-steps needed
```bash
echo "" | modal run --detach modal_app.py::run_pipeline --slug swedish --task instructions --limit 1000
# wait ~45 min, check report, then:
echo "" | modal run --detach modal_app.py::run_pipeline --slug swedish --task translation --limit 1000
```

**2. Malay** — no pre-steps needed
```bash
echo "" | modal run --detach modal_app.py::run_pipeline --slug malay --task instructions --limit 1000
echo "" | modal run --detach modal_app.py::run_pipeline --slug malay --task translation --limit 1000
```

**3. Amharic** — no pre-steps needed
```bash
echo "" | modal run --detach modal_app.py::run_pipeline --slug amharic --task instructions --limit 1000
echo "" | modal run --detach modal_app.py::run_pipeline --slug amharic --task translation --limit 1000
```

**4. French** — registry already updated to `Lucie-7B-Instruct-v1.1`, no other pre-steps
```bash
echo "" | modal run --detach modal_app.py::run_pipeline --slug french --task instructions --limit 1000
echo "" | modal run --detach modal_app.py::run_pipeline --slug french --task translation --limit 1000
```

**5. Czech** — clear stale weights cache FIRST, then run both tasks
```bash
modal run modal_app.py::clear_csmpt_cache
echo "" | modal run --detach modal_app.py::run_pipeline --slug czech --task instructions --limit 1000
echo "" | modal run --detach modal_app.py::run_pipeline --slug czech --task translation --limit 1000
```

**6. Māori** — max_model_len already fixed in registry (512), no other pre-steps
```bash
echo "" | modal run --detach modal_app.py::run_pipeline --slug maori --task instructions --limit 1000
echo "" | modal run --detach modal_app.py::run_pipeline --slug maori --task translation --limit 1000
```

**7. Tok Pisin** — max_model_len already fixed in registry (512), no other pre-steps
```bash
echo "" | modal run --detach modal_app.py::run_pipeline --slug tok_pisin --task instructions --limit 1000
echo "" | modal run --detach modal_app.py::run_pipeline --slug tok_pisin --task translation --limit 1000
```

**8. Swahili** — had unknown failure in a previous parallel run; run and check error logs if it fails again
```bash
echo "" | modal run --detach modal_app.py::run_pipeline --slug swahili --task instructions --limit 1000
echo "" | modal run --detach modal_app.py::run_pipeline --slug swahili --task translation --limit 1000
```

**9. Brazilian Portuguese** — had unknown failure in a previous parallel run; run and check error logs if it fails again
```bash
echo "" | modal run --detach modal_app.py::run_pipeline --slug brazilian_portuguese --task instructions --limit 1000
echo "" | modal run --detach modal_app.py::run_pipeline --slug brazilian_portuguese --task translation --limit 1000
```

---

## Slash Commands

- `/concept <term>` — explains the term clearly (beginner-friendly) and automatically adds it to the "Concepts & Terminology" section on the Notion page

---

## Research Areas

### Regional LLM Evaluation (complete)
Find the best language model per target language/region via tokenizer benchmarking against Gemma-4.
Results in `data/results.csv`, report in `docs/llm-evaluation.md`, visualisation in `docs/viz/`.

### Qualitative LLM Validation (in progress — see Current Status above)
Validate the 17 tokenizer winners qualitatively before committing to fine-tuning.
Two tasks per language: **Translation** (English ↔ target, ~1000 samples) and **Instruction Following** (Talking Avatar domain, ~1000 samples).
Evaluation via LLM-as-judge framework + automated metrics.
Output: per-language verdict (confirmed winner / revert to Gemma-4). See `docs/plans/task1b-qualitative-llm-validation.md`.

### Whisper Fine-tuning per Language
Improve speech-to-text accuracy for user questions in each target language.

**Dataset needed:** (audio clip → correct transcript) pairs per language
- Synthetic is fine to start: generate audio via TTS on known text, use as training pairs
- Target: 1–10 hours of audio per language minimum

**Fine-tuning stack:**
- Base model: `openai/whisper-large-v3`
- Libraries: HuggingFace `transformers`, `datasets`
- Environment: Google Colab (free GPU) to start

### Cross-language Validation
Ensure fine-tuning on language A doesn't silently degrade performance on languages B, C, D (catastrophic forgetting).

**Approach:**
- Maintain a fixed validation set per supported language (never used for training)
- After every fine-tuning run, evaluate on all language validation sets
- Track Word Error Rate (WER) per language over time — lower is better
- Flag and investigate any regression before deploying

---

## Pipeline Architecture (quick reference)

The evaluation pipeline is a Modal app (`modal_app.py`). Key components:

- **`RegistryService`** — FastAPI web endpoint created as part of `modal run`. **Inference workers do NOT call it** — they read model configs directly from `/data/registry/models.json` (the `phase2a-registry` volume mounted in every container). If a model config needs changing, update `phase2a-registry/models.json` locally and re-upload via `modal volume put --force phase2a-registry models.json`.
- **`VLLMWorkerL4/L40S/A100/T4`** — GPU inference workers. Model loads once at container startup (no memory snapshots — removed due to vLLM 0.9.x crash).
- **`LightMetricWorkerModal`** — CPU worker for fast metrics (language adherence, tone, format).
- **`JudgeWorkerModal`** — CPU worker making Gemini API calls. 300 calls per language (50 prompts × 2 swap runs × 3 dimensions).
- **`phase2a-outputs`** volume — all inference outputs, metrics, verdicts, reports stored here. Outputs are checkpointed every 50 prompts — safe to resume after container crash.
- **`phase2a-weights`** volume — HuggingFace model cache. Persists across runs.

**State machine:** `PENDING → INFERENCE → LIGHT_METRICS → MODEL_METRICS → JUDGE → REPORT → DONE`

---

## Directory Structure

```
falcon-language/
├── modal_app.py            # Modal entrypoints + worker class registrations
├── modal_common.py         # Shared volumes, images, GPU presets
├── src/
│   ├── pipeline/
│   │   ├── state_machine.py   # Pipeline state machine + ensure_gemma4_baseline
│   │   ├── entrypoints.py     # run_pipeline() orchestrator
│   │   ├── loader.py          # Dataset loader (load_samples)
│   │   └── run.py             # Output path helpers + volume I/O
│   ├── workers/
│   │   ├── inference.py       # VLLMWorker base class
│   │   └── judge.py           # JudgeWorkerModal
│   └── metrics/               # Auto-discovered metric classes
├── experiments/               # Evaluation scripts, tokenizer tests
├── data/
│   ├── results.csv            # Task 1 tokenizer evaluation results
│   └── datasets/
│       ├── translation/       # FLORES-200 samples per language
│       └── instructions/      # Custom Talking Avatar prompts per language
├── docs/
│   ├── llm-evaluation.md
│   ├── viz/
│   └── plans/
└── scripts/
```

**Note:** `.claude/` is excluded via `.gitignore` — session-local data used only by the developer's Claude Code instance.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM (production) | Gemma-4 26B A4B IT |
| LLM inference (eval) | vLLM ≥0.9.0 on Modal |
| Judge model | Gemini via API (`gemini-3.5-flash`) |
| GPU cloud | Modal (serverless) |
| Audio encoder | OpenAI Whisper (`whisper-large-v3`) |
| ML framework | HuggingFace `transformers` + `datasets` |
| Model hub | HuggingFace |

---

## Working Conventions

- Ram is new to AI model training and dataset creation — explain concepts before implementing
- One language at a time during evaluation — run, verify report is valid, then next
- Commit per meaningful milestone, push at end of session
- Do not run Modal containers without the user's explicit permission
