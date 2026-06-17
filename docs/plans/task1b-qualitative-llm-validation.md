# Qualitative LLM Validation Plan

**Status:** Planned  
**Scope:** 17 languages where a regional candidate beat Gemma-4 in the tokenizer evaluation  
**Goal:** Confirm that the tokenizer winners are also qualitatively better than Gemma-4 on real-world tasks before committing to Whisper fine-tuning

---

## Background

The tokenizer benchmark (Task 1) measures *efficiency* — how compactly a model encodes a language. It does not measure *quality* — whether the model actually produces better responses. This phase validates the winners on two dimensions critical to the Talking Avatar pipeline: translation and instruction following.

---

## Languages in Scope (17)

| Language | Winning Model |
|----------|--------------|
| Tamil | Tamil-Mistral-7B |
| Marathi | MahaMarathi-7B |
| Kannada | Ambari-7B |
| Gujarati | Gujju-Llama-7B |
| Arabic | Jais-2-8B |
| Hebrew | DictaLM-2.0-7B |
| Korean | Polyglot-Ko-12B |
| Malay | MaLLaM-5B |
| Swahili | Swahili-Gemma-7B |
| Amharic | Walia-LLM-7B |
| French | Lucie-7B |
| Swedish | Viking-7B |
| Czech | CSMPT-7B |
| Greek | Meltemi-7B |
| Brazilian Portuguese | Tucano-2b4 |
| Māori | Goldfish-mri-39M |
| Tok Pisin | Goldfish-tpi-125M |

---

## Step 1: Dataset Creation

**1000 samples per language per task** (2000 total per language)

### 1a. Translation Dataset (~1000 samples/language)

Source: FLORES-200 devtest corpus (~1012 sentences already available in `data/`) — reuse directly.

Each sample:
```
{
  "id": "flores_1234",
  "source_en": "The patient needs to take the medication twice a day.",
  "reference_translation": "<gold translation in target language>",
  "direction": "en→target"  // and separately "target→en"
}
```

**Directions:** Both
- `en → target` (English to regional language)
- `target → en` (regional language back to English)

Reference translations come from FLORES-200 gold annotations (already available).

### 1b. Instruction Following Dataset (~1200 samples/language)

Domain: Talking Avatar use cases. Categories and sample distribution:

| Category | Count | Description |
|----------|-------|-------------|
| Tone/style | 200 | "Respond in a warm, friendly tone", "Use formal language", "Sound empathetic" |
| Length constraint | 200 | "Answer in exactly 2 sentences", "Keep it under 30 words" |
| Language compliance | 200 | "Reply only in [language]", "Do not mix English words" |
| Topic boundaries | 200 | "Only answer based on the provided context", "Do not speculate" |
| Structured output | 200 | "Respond as a numbered list", "Start with a greeting, end with a question" |
| Number verbalization | 200 | TTS normalization: postal codes/phone/OTP → digit-by-digit; money/measurements/percentages/dates → word form. Covers 8 number types: postal codes, phone numbers, OTPs/PINs, currency, decimal measurements, percentages, large quantities, dates + mixed sentences. |

Each sample:
```
{
  "id": "inst_0042",
  "system_instruction": "You are a helpful avatar assistant. Always respond in Tamil. Keep responses under 3 sentences.",
  "user_prompt": "What are the steps to reset my password?",
  "expected_constraints": ["language=Tamil", "max_sentences=3"],
  "category": "language_compliance"
}
```

Generation: Use GPT-4o or Gemma-4 itself to generate diverse prompts; manually review a sample per category.

---

## Step 2: Evaluation Metrics

### A. LLM-as-Judge (primary)

Third-party LLM evaluates candidate vs Gemma-4 head-to-head, providing:
- Winner per sample
- Reasoning
- Confidence score

*(Framework details to be provided — integrate once available)*

### B. Translation Metrics (automated, no human/LLM needed)

| Metric | What it measures | Tool |
|--------|-----------------|------|
| **BLEU** | N-gram precision vs reference | `sacrebleu` |
| **chrF** | Character-level F-score — better for Arabic/Indic/morphological languages | `sacrebleu` |
| **COMET** | Neural MT quality, trained on human judgements — best correlation with human eval | `unbabel-comet` |
| **BERTScore** | Semantic similarity using multilingual embeddings | `bert-score` |
| **Back-translation consistency** | Translate E→T→E using fixed MT model; BLEU vs original English | `sacrebleu` + MarianMT |

### C. Instruction Following Metrics (automated)

| Metric | Applies to | What it measures | Method |
|--------|-----------|-----------------|--------|
| **Language adherence rate** | All categories | % of responses in the correct target language | `langdetect` or `lingua` |
| **Length constraint accuracy** | length_constraint | % of responses within specified sentence/word bounds | Rule-based parser |
| **Format compliance** | structured_output | Structural adherence (lists, greeting/closing, numbered steps) | Regex + rule-based |
| **Topic boundary respect** | topic_boundary | Does the model stay on-topic when instructed? | Embedding cosine similarity to context |
| **Keyword constraint** | All categories | Inclusion/exclusion of specified words | Exact match |
| **Register/tone classifier** | tone_style | Formal vs informal tone detection | Pre-trained multilingual classifier |

### D. Number Verbalization Metrics (automated, rule-based)

Specific to the `number_verbalization` category. These are fully automated — no LLM-as-judge needed for most.

| Metric | What it measures | Method | Target |
|--------|-----------------|--------|--------|
| **Digit-by-digit compliance** | For postal codes, phone numbers, OTPs, PINs: does the model read each digit individually (not as a composite number)? | Check that original digit string does not appear literally in output AND each digit maps to a word | 100% on identifier-type prompts |
| **Word-form compliance** | For money, measurements, percentages, dates: does the model use word form with no raw digits remaining? | Regex: zero digit characters should remain in verbalized portion of output | 100% on value-type prompts |
| **No literal digit leakage** | Overall: do any raw digit characters appear in the output where they should have been verbalized? | Count raw digit characters in output that correspond to input numbers | Lower is better; 0 = perfect |
| **Digit preservation accuracy** | Are all digits from the input number accounted for in the output (none dropped or hallucinated)? | Compare digit count in input number vs digit-word count in output | 100% |
| **Currency unit accuracy** | When a currency symbol is present ($, ₹, €, £, ¥), is the correct currency word included in the output? | Check for expected currency word (dollars, rupees, euros…) in output | 100% on currency prompts |
| **Number type classification accuracy** | Did the model apply the correct rule for the number type? (postal code read digit-by-digit, not as "five lakh sixty thousand") | Label-based check: given `number_type` in the prompt, verify the rule applied matches expected | Higher is better |
| **Language of number words** | Are the verbalized number words in the target language (not English)? | Language detection on the number-word spans; check against known digit words per language | 100% target language |
| **Mixed-sentence consistency** | For prompts containing multiple number types, are ALL numbers handled correctly (right rule per type)? | Composite of digit-by-digit + word-form checks across all numbers in the sentence | Strictest test |

---

## Step 3: Comparison Framework Integration

Once dataset is ready and metrics are defined:

1. Run both models (candidate + Gemma-4) on all samples
2. Compute automated metrics for both
3. Feed pairs into LLM-as-judge framework
4. Aggregate: per-language win rate, per-category breakdown, metric deltas

---

## Step 4: Output

Per-language report covering:
- **Translation:** BLEU/chrF/COMET delta (candidate vs Gemma-4)
- **Instruction following:** constraint satisfaction rate, LLM-judge win rate
- **Verdict:** Confirmed winner / Revert to Gemma-4 / Inconclusive (needs manual review)

Final output feeds directly into the Whisper fine-tuning language list — only confirmed winners proceed.

---

## Dependencies / Blockers

- [ ] LLM-as-judge framework details (endpoint, input format, auth) — to be provided
- [ ] Decide on instruction dataset generation method (GPT-4o vs manual vs template)
- [ ] HuggingFace access to all 17 candidate models for inference (not just tokenizer)
