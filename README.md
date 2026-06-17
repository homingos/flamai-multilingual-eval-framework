# Falcon Tokenizer Evaluation Framework

> **Which regional LLM tokenizer works best for each language?**
> A systematic benchmark comparing Gemma-4 (production baseline) against 55 regional language models across 63 languages and 245 tokenizer runs.

---

## Overview

Flam AI's Talking Avatar uses **Gemma-4 26B A4B IT** as its production LLM. This framework evaluates whether specialist regional models tokenize each language more efficiently — fewer tokens means faster inference, lower cost, and better language fidelity for TTS pipelines.

Evaluation uses the **FLORES-200** corpus (~1,012 sentences per language) and measures 7 tokenizer metrics per model per language.

---

## Key Findings

| Stat | Value |
|------|-------|
| Languages evaluated | 63 |
| Regional candidates tested | 55 |
| Baselines | Gemma-4, BLOOM, mT5 |
| Total tokenizer runs | 245 |
| **Regional winners (beat Gemma-4)** | **17** |
| Countries mapped on visualization | 163 |

### Regional Winners (17 languages where specialist model beats Gemma-4)

| Language | Winner Model | Fertility improvement |
|----------|-------------|----------------------|
| Tamil | Tamil-Mistral-7B | 2.37 → 1.73 (−27%) |
| Marathi | MahaMarathi-7B | 1.99 → 1.64 (−18%) |
| Kannada | Ambari-7B | 3.24 → 2.74 (−15%) |
| Gujarati | Gujju-Llama-7B | 2.42 → 2.03 (−16%) |
| Arabic | Jais-2-8B | 2.03 → 1.46 (−28%) |
| Hebrew | DictaLM-2.0-7B | 2.71 → 2.64 (−3%) |
| Korean | Polyglot-Ko-12B | 2.42 → 2.20 (−9%) |
| Malay | MaLLaM-5B | 1.63 → 1.42 (−13%) |
| Swahili | Swahili-Gemma-7B | 2.09 → 2.05 (−2%) |
| Amharic | Walia-LLM-7B | 3.03 → 1.62 (−47%) |
| French | Lucie-7B | 1.49 → 1.43 (−4%) |
| Swedish | Viking-7B | 1.84 → 1.47 (−20%) |
| Czech | CSMPT-7B | 2.16 → 1.41 (−35%) |
| Greek | Meltemi-7B | 2.47 → 1.40 (−43%) |
| Brazilian Portuguese | Tucano-2b4 | 1.45 → 1.25 (−14%) |
| Māori | Goldfish-mri-39M | 1.83 → 1.22 (−33%) |
| Tok Pisin | Goldfish-tpi-125M | 2.02 → 1.82 (−10%) |

---

## Evaluation Metrics

Each tokenizer run measures 7 metrics:

| Metric | What it measures | Better |
|--------|-----------------|--------|
| **Fertility** | Tokens produced per whitespace word | Lower |
| **Compression ratio** | Chars per token | Higher |
| **Vocab coverage %** | % of language characters in the model's vocabulary | Higher |
| **Roundtrip fidelity %** | Encode → decode → exact match rate | Higher |
| **Byte fallback rate %** | % tokens that are raw byte fallbacks (tokenizer failure) | Lower |
| **UNK rate %** | % unknown tokens produced | Lower |
| **Avg tokens / sentence** | Raw throughput measure | Lower |

**Win condition:** a regional candidate beats Gemma-4 if: `fertility < Gemma-4 AND vocab_coverage ≥ 80% AND roundtrip_fidelity ≥ 95%`

---

## Repository Structure

```
falcon-language/
├── data/
│   ├── results.csv                      # 245 tokenizer runs (63 langs × 3 baselines + 56 regional)
│   ├── summary.json                     # Per-model aggregates (unweighted + char-weighted)
│   └── datasets/
│       ├── instructions/                # 17,000 instruction-following samples (1000/language × 17)
│       │   ├── meta.json
│       │   └── <language>/
│       │       ├── samples.jsonl        # 1000 samples: 200 per category
│       │       └── meta.json
│       └── translation/                 # ~34,000 translation samples (pending eng_Latn download)
│           ├── meta.json
│           └── <language>/
│               ├── samples.jsonl        # ~2024 samples: 1012 en→target + 1012 target→en
│               └── meta.json
├── experiments/
│   ├── tokenizer_test.py                # Main tokenizer evaluation runner
│   ├── detailed_report.py               # Generates docs/llm-evaluation.md from results.csv
│   ├── language_map.py                  # Interactive Plotly world map → docs/viz/language-map.html
│   ├── generate_pdf.py                  # PDF report generator → docs/reports/
│   ├── create_instruction_dataset.py    # Generates data/datasets/instructions/
│   └── create_translation_dataset.py    # Generates data/datasets/translation/ (needs internet)
├── docs/
│   ├── viz/                 # language-map.html, world-map.png
│   ├── reports/             # Generated PDF reports
│   ├── llm-evaluation.md    # Full per-language report
│   └── llm-research-raw.md  # Raw research notes
└── scripts/                 # Fine-tuning scripts (future)
```

---

## Quick Start

```bash
# Install dependencies
pip install transformers datasets sentencepiece plotly kaleido

# Run tokenizer evaluation (append mode — skips completed pairs)
python experiments/tokenizer_test.py --append

# Skip baseline models (Gemma-4/BLOOM/mT5), test regional candidates only
python experiments/tokenizer_test.py --skip-baselines

# Evaluate a specific language only
python experiments/tokenizer_test.py --language Tamil --append

# Regenerate the markdown report
python experiments/detailed_report.py

# Regenerate the interactive world map
python experiments/language_map.py
# → open docs/language-map.html in a browser
```

---

## Interactive World Map

Open `docs/viz/language-map.html` in any browser for the full interactive visualization:

- **Choropleth map** — 163 countries colored by best LLM for their primary language
- **India state layer** — 24 states colored by state language (Tamil Nadu, Karnataka, etc.)
- **Click any country** → sidebar with 7-metric comparison table and delta %
- **Region filters** — Indic, Middle East, East Asia, SEA, Africa, Europe, Americas, Oceania
- **Stats popovers** — click "17 Candidates Beat Gemma-4" or "64 Languages Tested" to see full lists

---

## Adding a New Language / Candidate

1. Add an entry to `CANDIDATES` in `experiments/tokenizer_test.py`:
   ```python
   "YourLanguage": ("org/model-name-on-huggingface", False),
   ```
2. Add the language → country mapping in `experiments/language_map.py` (`LANG_COUNTRIES` dict)
3. Run: `python experiments/tokenizer_test.py --language YourLanguage --append`
4. Regenerate the report and map

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Corpus | FLORES-200 (`facebook/flores`, devtest split) |
| Baseline LLM | Gemma-4 26B A4B IT (`google/gemma-4-27b-it` tokenizer) |
| Secondary baselines | BLOOM (`bigscience/bloom`), mT5 (`google/mt5-base`) |
| ML framework | HuggingFace `transformers` + `datasets` |
| Visualization | Plotly (choropleth + GeoJSON) |
| Map image export | Kaleido |

---

## Project Context

Part of the Falcon Language Support initiative for Flam AI's Talking Avatar product. The goal is to identify the best LLM per target language for the full pipeline: STT (Whisper) → LLM → TTS.

### Research Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Tokenizer Evaluation | ✅ Complete | Benchmark 63 languages — 17 regional candidates beat Gemma-4 |
| Qualitative Validation | 🔜 Next | Validate the 17 winners on translation + instruction following (1000 samples/language/task) |
| Whisper Fine-tuning | Planned | Improve STT accuracy per language using confirmed winning LLMs |
| Cross-language Validation | Planned | Prevent catastrophic forgetting across languages during fine-tuning |

### Qualitative Validation Methodology

For each of the 17 tokenizer winners, two tasks are evaluated head-to-head against Gemma-4 26B A4B IT:

**Translation** (English ↔ target language, ~1000 samples from FLORES-200)
- Automated: BLEU, chrF, COMET, BERTScore, back-translation consistency
- LLM-as-judge: head-to-head blind comparison

**Instruction Following** (~1000 Talking Avatar domain prompts)
- Categories: tone/style, length constraints, language compliance, topic boundaries, structured output
- Automated: language adherence rate, format compliance, length accuracy, keyword constraints
- LLM-as-judge: head-to-head blind comparison

See `docs/plans/task1b-qualitative-llm-validation.md` for full dataset spec and metric definitions.
