# Task 1 Plan: Regional LLM Evaluation

## Goal
Identify the best open-weight language model per target language by running tokenizer efficiency tests against Gemma-4 36B as the baseline.

## Reference Research
- `docs/Regional LLMs by Continent.pdf` — ChatGPT-generated report ranking top 10 LLMs per continent by regional language fit. Use as a starting shortlist; our tokenizer tests verify their claims independently.

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

**Known starting points from the ChatGPT PDF:**

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

### Phase 4 — Tokenizer Test Script
Script: `experiments/tokenizer_test.py`

For each shortlisted candidate:
1. Load candidate tokenizer via `AutoTokenizer.from_pretrained(model_id)`
2. Load Gemma-4 tokenizer as baseline
3. Run FLORES-200 sentences for the target language through both
4. Record token count per sentence
5. Compute average token count (candidate vs Gemma-4)
6. Write results to `data/tokenizer_results.csv`

Output columns: `language, candidate_model, candidate_avg_tokens, gemma4_avg_tokens, delta, delta_pct`

---

### Phase 5 — Document Results
- CSV → `data/tokenizer_results.csv`
- Summary → `docs/llm-evaluation.md` (per-language table + recommendation)
- Update Notion page "Task 1" section with findings

---

## Decision Rule
- `delta < 0` (candidate uses fewer tokens): candidate is more efficient → **preferred over Gemma-4**
- `delta ≈ 0`: no tokenizer advantage → evaluate on other criteria (benchmark scores, model size, licence)
- `delta > 0` (candidate uses more tokens): Gemma-4 handles that language better → **stick with Gemma-4**
