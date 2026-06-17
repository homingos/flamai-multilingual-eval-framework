# Task 1 Plan: Regional LLM Evaluation

## Goal
Identify the best open-weight language model per target language by running tokenizer efficiency tests against Gemma-4 26B A4B IT as the baseline.

## Reference Research
- `docs/pdfs/Regional LLMs by Continent.pdf` — ChatGPT-generated report ranking top 10 LLMs per continent by regional language fit. Use as a starting shortlist; our tokenizer tests verify their claims independently.
- `docs/pdfs/Tokenizer evaluation — detailed report - document_pdf.pdf` — Indic tokenizer evaluation report by the team. **This is the target format** for our global report: 7 metrics, pivot tables per metric, per-language × per-tokenizer results, aggregate summary.

---

## License Categories (important filter)

| Category | Meaning | Usable? |
|---|---|---|
| **Open-weight / Apache 2.0** | Weights downloadable, free commercial use | Yes — priority |
| **Open-weight / CC-BY-NC** | Downloadable but non-commercial only | Yes for research/eval; check before production |
| **Open-weight / custom licence** | Downloadable, commercial terms vary | Check case-by-case |
| **Proprietary / API-only** | Cannot download tokenizer; cloud only | No — can't run tokenizer test locally |
| **Closed / private** | Weights not released | No — find open-weight alternative |

Models that are API-only (GPT-4.1, Claude, Gemini, HyperCLOVA X, Sabiá-4, Mansa) are excluded from the tokenizer test pipeline. If they appear as the best regional option, note them but find the best open-weight alternative for comparison.

---

## Language List (~72 languages)

### Indic (priority group)
Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Odia, Assamese, Urdu, Nepali, Sinhala, Maithili

### Middle East & West Asia
Arabic, Persian (Farsi), Turkish, Hebrew, Kurdish, Azerbaijani, Uzbek, Kazakh

### East & Southeast Asia
Mandarin Chinese, Japanese, Korean, Vietnamese, Thai, Indonesian, Malay, Tagalog, Burmese, Khmer

### Africa
Swahili, Amharic, Hausa, Yoruba, Igbo, Zulu, Xhosa, Somali, Wolof, Shona

### Europe
French, German, Spanish, Portuguese, Italian, Dutch, Polish, Russian, Ukrainian, Romanian, Swedish, Czech, Greek

### Americas
Spanish (Latin American), Portuguese (Brazilian), Quechua, Nahuatl, Haitian Creole

### Oceania / Pacific
Māori, Samoan, Hawaiian, Tok Pisin

---

## Phases

### Phase 0 — Language List ✅
Compiled above. ~72 languages filtered to those with active NLP communities.

---

### Phase 1 — Parallel Agent Research
One agent per language. Each agent researches and returns:
- Best 1–2 regional LLM candidates
- HuggingFace URL (required)
- Parameter count (e.g. 7B, 13B)
- Licence type (Apache 2.0 / CC-BY-NC / custom / proprietary)
- Whether tokenizer is loadable locally
- Any known benchmark scores

Agents that find no viable regional candidate return: `"no candidate — Gemma-4 default"`

**Known starting points from `docs/pdfs/Regional LLMs by Continent.pdf`:**

| Region | Model | Licence | HF URL |
|---|---|---|---|
| Indic (Hindi) | Krutrim-2 12B | Open/self-host | krutrim-ai-labs/Krutrim-2-instruct |
| Indic (Hindi) | OpenHathi-7B | Open | sarvamai/OpenHathi-7B-Hi-v0.1-Base |
| Indic (Hindi) | Llama-3 Nanda 10B | Open | MBZUAI/Llama-3-Nanda-10B-Chat |
| Arabic | Jais 30B Chat | Open (commercial restrictions) | inceptionai/jais-30b-v3 |
| Arabic | SILMA 9B | Open | silma-ai/SILMA-9B-Instruct-v1.0 |
| SEA (multi) | SEA-LION v3 8B | Open | aisingapore/sea-lion-v3 |
| SEA (multi) | Sailor2 20B | Open | sea-ai-lab/Sailor2-20B |
| Japanese | CyberAgent CALM3 22B | Open | cyberagent/calm3-22b-chat |
| African (multi) | InkubaLM 0.4B | Open | Lelapa-AI/InkubaLM |
| African (multi) | AfroLlama_V1 8B | Open | Jacaranda/AfroLlama_V1 |
| African (Swahili) | UlizaLlama3 8B | Open | Jacaranda/UlizaLlama3 |
| African (Yoruba) | YorubaLlama 8B | Open | Jacaranda/YorubaLlama |
| African (Xhosa/Zulu) | Xhosa_ZuluLlama3_v1 8B | Open | Jacaranda/Xhosa_ZuluLlama3_v1 |
| European (multi) | EuroLLM | Open | EuroLLM/EuroLLM-9B |
| European (French) | Lucie-7B | Open/OSI | OpenLLM-France/Lucie-7B-Instruct-v1.1 |
| European (French) | CroissantLLM | Open | croissantllm/CroissantLLMBase |
| European (Finnish) | Poro 34B | Apache 2.0 | LumiOpen/Poro-34B |
| European (Bulgarian) | BgGPT 3.0 | Open (HF) | INSAIT-Institute/BgGPT-Gemma-3-27B-IT |
| South America | Latam-GPT | Open/regional | (to be confirmed by agent) |
| South America (PT) | Tucano 2 | Apache 2.0 | Polygl0t/tucano-2 |
| Broad multilingual | Qwen3 32B | Apache 2.0 | Qwen/Qwen3-32B |
| Broad multilingual | Aya Expanse 32B | CC-BY-NC | CoherForAI/aya-expanse-32b |

---

### Phase 2 — Shortlist Testable Candidates
Filter Phase 1 results to keep only:
- Open-weight models with a HuggingFace tokenizer available
- Models where `AutoTokenizer.from_pretrained()` works without full model download

Result: `language → candidate HF model ID` pairs.

---

### Phase 3 — Environment Setup
```bash
pip install transformers datasets
export HF_TOKEN=<your_token>   # needed for some gated models
```
Test sentences sourced from FLORES-200 dataset (professionally translated, 200+ languages).

---

### Phase 4 — Tokenizer Evaluation Script
Script: `experiments/tokenizer_test.py`

**Tokenizers to test per language:**
- `Gemma-4` — production baseline (always included)
- Regional candidate model from the shortlist (`docs/plans/task1-phase2-shortlist.md`)
- `BLOOM` — multilingual baseline (for cross-comparison context)
- `mT5` — multilingual baseline (for cross-comparison context)

**7 metrics computed per tokenizer × language:**

| Metric | Formula | Direction | Target |
|---|---|---|---|
| Fertility | `total_tokens / total_words` | Lower is better | 1.0–2.5 |
| Compression ratio | `total_chars / total_tokens` | Higher is better | >3.0 |
| Byte fallback rate (%) | % tokens that are raw byte fallbacks (`<0xNN>`) | Lower is better | 0% |
| UNK rate (%) | % tokens equal to `unk_token_id` | Lower is better | 0% |
| Vocabulary coverage (%) | % unique chars that map to exactly 1 token | Higher is better | >80% |
| Roundtrip fidelity (%) | % segments where encode→decode = original | Higher is better | 100% |
| Avg tokens / segment | `total_tokens / total_sentences` | Lower is better | same as fertility |

**Output files:**

`data/results.csv` — one row per tokenizer × language combination:
```
tokenizer_name, language, region, fertility, compression_ratio, byte_fallback_rate,
unknown_rate, vocab_coverage, roundtrip_pass_rate, avg_tokens_per_sent,
total_tokens, total_words, total_chars, total_sentences
```

`data/summary.json` — per-tokenizer aggregates (unweighted + character-weighted averages across all languages tested).

---

### Phase 5 — Generate Detailed Report
Script: `experiments/detailed_report.py`

Produces `docs/llm-evaluation.md` matching the format in `docs/pdfs/Tokenizer evaluation — detailed report - document_pdf.pdf`:

1. **Metric glossary** — definitions + direction table
2. **Regional aggregates** — unweighted and character-weighted averages per tokenizer, grouped by continent
3. **Pivot tables** — one table per metric, rows = languages, columns = tokenizers (Gemma-4 + regional candidate + BLOOM + mT5)
4. **Full summary table** — all aggregate columns in one view
5. **Complete per-language results** — every row from `data/results.csv`

Update Notion page "Task 1" section with key findings and link to the report.

---

## Decision Rule (7-metric)

A regional candidate is **preferred over Gemma-4** for a language when it wins on the majority of these signals:

| Signal | Prefer candidate when… |
|---|---|
| Fertility | candidate fertility < Gemma-4 fertility |
| Compression ratio | candidate compression > Gemma-4 compression |
| Byte fallback rate | candidate rate ≈ 0% (Gemma-4 may already be 0%) |
| UNK rate | candidate rate = 0% |
| Vocabulary coverage | candidate coverage > Gemma-4 coverage |
| Roundtrip fidelity | both should be ~100%; flag any that aren't |
| Avg tokens / segment | candidate avg < Gemma-4 avg |

**Primary signals:** Fertility and vocabulary coverage (most diagnostic of native language support).
**Secondary signals:** Compression ratio, byte fallback, roundtrip fidelity.
**Tiebreaker:** If metrics are roughly equal, prefer the model with better licence (Apache 2.0 > CC-BY-NC > custom).
