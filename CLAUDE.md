# Falcon Language Support — Claude Project Instructions

## What This Project Is

AI/ML research and implementation project for adding robust multilingual support to Flam AI's Talking Avatar product. The work involves evaluating regional language models, fine-tuning Whisper (audio encoder), and establishing cross-language validation methodology.

**Current LLM in production:** Gemma-4 36B (multilingual pre-trained)
**Project status:** Experimentation

## Key References

- **Notion task page:** https://app.notion.com/p/38078ca0ce5280dcb7d1e12e0c397e45

GitHub repos will be cloned into this directory as sub-folders once available.

## Slash Commands

- `/concept <term>` — explains the term clearly (beginner-friendly) and automatically adds it to the "Concepts & Terminology" section on the Notion page

## Priority Tasks (work in this order)

### 1. Regional LLM Evaluation
Find the best language model per target language/region (e.g. Sarvam for Indian languages).

**Approach — Tokenizer Test:**
- Run representative sentences in the target language through the candidate model's tokenizer
- Count tokens produced vs Gemma-4 baseline on the same sentences
- Fewer tokens = better native language support
- Document: model name, size, token count vs baseline, license, API availability

**Starting points:** Sarvam (Hindi/Indian), AceGPT (Arabic), SeaLLM (Southeast Asian), BLOOM (multilingual)
**Tool:** Hugging Face (huggingface.co), run tokenizers via `transformers` library

### 2. Whisper Fine-tuning per Language
Improve speech-to-text accuracy for user questions in each target language.

**Dataset needed:** (audio clip → correct transcript) pairs per language
- Synthetic is fine to start: generate audio via TTS on known text, use as training pairs
- Target: 1–10 hours of audio per language minimum

**Fine-tuning stack:**
- Base model: `openai/whisper-large-v3`
- Libraries: HuggingFace `transformers`, `datasets`
- Environment: Google Colab (free GPU) to start

### 3. Cross-language Validation
Ensure fine-tuning on language A doesn't silently degrade performance on languages B, C, D (catastrophic forgetting).

**Approach:**
- Maintain a fixed validation set per supported language (never used for training)
- After every fine-tuning run, evaluate on all language validation sets
- Track Word Error Rate (WER) per language over time — lower is better
- Flag and investigate any regression before deploying

## Feature Requirements (separate from training tasks)

These are product features to implement once the AI pipeline is stable:

1. **Natural Language Responses** — responses are too textbookish; fix via prompt engineering
2. **Artifact in Response** — add `artifact` (URL) field to response structure alongside `context` and `answer`; toggled per project
3. **Questioning Behavior** — model should ask clarifying questions to guide users through instructions based on PDF content
4. **Text-TTS Normalization** — digits/symbols must normalize before TTS (e.g. `5` → `"five"`); verify current behavior first before fixing

## Directory Structure

```
falcon-language/
├── experiments/     # tokenizer tests, evaluation scripts, notebooks
├── data/            # datasets (audio clips, transcripts, validation sets)
├── scripts/         # training and fine-tuning scripts
├── docs/            # notes, findings, language research
└── <repos>/         # GitHub repos cloned here as sub-folders
```

## Tech Stack

| Component | Tool |
|---|---|
| LLM (production) | Gemma-4 36B |
| Audio encoder | OpenAI Whisper (`whisper-large-v3`) |
| ML framework | HuggingFace `transformers` + `datasets` |
| GPU environment | Google Colab (start) |
| Model hub | Hugging Face (huggingface.co) |

## Working Conventions

- Ram is new to AI model training and dataset creation — explain concepts before implementing
- One task at a time: implement, test, then move to next
- Commit per meaningful milestone, push at end of session
- `docs/` for findings and research notes during experimentation
