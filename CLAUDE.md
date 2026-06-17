# Falcon Language Support — Claude Project Instructions

## What This Project Is

AI/ML research and implementation project for adding robust multilingual support to Flam AI's Talking Avatar product. The work involves evaluating regional language models, fine-tuning Whisper (audio encoder), and establishing cross-language validation methodology.

**Current LLM in production:** Gemma-4 26B A4B IT (multilingual pre-trained)
**Project status:** Experimentation

## Key References

- **Notion task page:** https://app.notion.com/p/38078ca0ce5280dcb7d1e12e0c397e45

GitHub repos will be cloned into this directory as sub-folders once available.

## Slash Commands

- `/concept <term>` — explains the term clearly (beginner-friendly) and automatically adds it to the "Concepts & Terminology" section on the Notion page

## Research Areas

### Regional LLM Evaluation (complete)
Find the best language model per target language/region via tokenizer benchmarking against Gemma-4.
Results in `data/results.csv`, report in `docs/llm-evaluation.md`, visualisation in `docs/viz/`.

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

## Feature Requirements (separate from training tasks)

These are product features to implement once the AI pipeline is stable:

1. **Natural Language Responses** — responses are too textbookish; fix via prompt engineering
2. **Artifact in Response** — add `artifact` (URL) field to response structure alongside `context` and `answer`; toggled per project
3. **Questioning Behavior** — model should ask clarifying questions to guide users through instructions based on PDF content
4. **Text-TTS Normalization** — digits/symbols must normalize before TTS (e.g. `5` → `"five"`); verify current behavior first before fixing

## Directory Structure

```
falcon-language/
├── experiments/        # evaluation scripts, tokenizer tests, notebooks
├── data/               # results.csv, summary.json, datasets
├── scripts/            # training and fine-tuning scripts
├── docs/
│   ├── viz/            # interactive map (language-map.html) and map image
│   ├── reports/        # generated PDF reports
│   ├── plans/          # internal planning docs (not for repo consumers)
│   ├── llm-evaluation.md
│   └── llm-research-raw.md
└── <repos>/            # GitHub repos cloned here as sub-folders
```

**Note:** `.claude/` is excluded via `.gitignore` — it contains session-local data (memory, task state) used only by the developer's Claude Code instance.

## Tech Stack

| Component | Tool |
|---|---|
| LLM (production) | Gemma-4 26B A4B IT |
| Audio encoder | OpenAI Whisper (`whisper-large-v3`) |
| ML framework | HuggingFace `transformers` + `datasets` |
| GPU environment | Google Colab (start) |
| Model hub | Hugging Face (huggingface.co) |

## Working Conventions

- Ram is new to AI model training and dataset creation — explain concepts before implementing
- One area at a time: implement, test, then move to next
- Commit per meaningful milestone, push at end of session
- `docs/` for findings and research notes during experimentation
