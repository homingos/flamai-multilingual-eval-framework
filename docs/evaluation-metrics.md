# Evaluation Metrics — Falcon Multilingual LLM Benchmark

This document defines all evaluation metrics for the qualitative LLM comparison framework. For each language, two models are compared head-to-head: the **regional candidate** vs **Gemma-4 26B A4B IT** (baseline).

---

## Dataset Overview

Two datasets per language (17 languages total):

| Dataset | Samples/language | Location |
|---------|-----------------|----------|
| Translation | ~2,024 | `data/datasets/translation/<lang>/samples.jsonl` |
| Instruction Following | 1,200 | `data/datasets/instructions/<lang>/samples.jsonl` |

### Translation sample schema
```json
{
  "id": "trans_tamil_en_tgt_0001",
  "language": "Tamil",
  "direction": "en→target",
  "source_lang": "English",
  "target_lang": "Tamil",
  "source": "We now have 4-month-old mice that are non-diabetic.",
  "reference": "<gold translation from FLORES-200>",
  "flores_id": 0
}
```

### Instruction following sample schema
```json
{
  "id": "inst_tamil_tone_style_001",
  "language": "Tamil",
  "category": "tone_style",
  "system_instruction": "You are a helpful avatar assistant. Respond in Tamil with a warm, friendly tone.",
  "user_prompt": "How do I reset my password?",
  "expected_constraints": { "tone": "friendly" }
}
```

The `category` field identifies which metric group applies. The `expected_constraints` field is what the evaluator checks against.

**Instruction following categories:**
- `tone_style` — tone/register compliance
- `length_constraint` — word/sentence count bounds
- `language_compliance` — no language mixing
- `topic_boundary` — stays on context, no speculation
- `structured_output` — format adherence (lists, headers, Q&A)
- `number_verbalization` — TTS normalization rules for numbers

---

## 1. Translation Metrics

Run on both directions: `en→target` and `target→en`.

For each sample, both models generate a translation. Metrics compare each model's output against the `reference` field (gold FLORES-200 human translation).

### 1.1 BLEU
- **What:** N-gram precision between model output and reference translation.
- **Library:** `sacrebleu`
- **Score range:** 0–100, higher is better.
- **Notes:** Use corpus-level BLEU (aggregate over all samples), not sentence-level. Weak for morphologically rich languages (Arabic, Indic) — use chrF there.

```python
import sacrebleu
bleu = sacrebleu.corpus_bleu(hypotheses, [references])
print(bleu.score)
```

### 1.2 chrF
- **What:** Character-level F-score between output and reference. More reliable than BLEU for Arabic, Indic, and other morphologically complex languages.
- **Library:** `sacrebleu`
- **Score range:** 0–100, higher is better.

```python
chrf = sacrebleu.corpus_chrf(hypotheses, [references])
print(chrf.score)
```

### 1.3 COMET
- **What:** Neural MT quality metric trained on human judgements. Best correlation with human evaluation among all automated metrics.
- **Library:** `unbabel-comet` (`pip install unbabel-comet`)
- **Model:** `Unbabel/wmt22-comet-da`
- **Score range:** ~0–1, higher is better.
- **Input:** requires `source`, `hypothesis`, `reference` — all three.

```python
from comet import download_model, load_from_checkpoint
model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

data = [{"src": source, "mt": hypothesis, "ref": reference} for ...]
scores = model.predict(data, batch_size=8)
print(scores.system_score)
```

### 1.4 BERTScore
- **What:** Semantic similarity using multilingual embeddings. Catches paraphrases and meaning-equivalent translations that BLEU misses.
- **Library:** `bert-score` (`pip install bert-score`)
- **Model:** `bert-base-multilingual-cased` (works across all 17 languages)
- **Score range:** 0–1 (F1), higher is better.

```python
from bert_score import score
P, R, F1 = score(hypotheses, references, lang="multilingual", model_type="bert-base-multilingual-cased")
print(F1.mean().item())
```

### 1.5 Back-translation Consistency
- **What:** Translates model output back to English using a fixed MT model (MarianMT), then computes BLEU against the original English source. Tests whether meaning is preserved through the full round-trip.
- **Library:** `transformers` (MarianMT) + `sacrebleu`
- **Score range:** 0–100 BLEU, higher is better.
- **Only applies to:** `en→target` direction samples.

```python
from transformers import MarianMTModel, MarianTokenizer
# Load target→en MarianMT model, translate hypothesis back to English
# Then compute sacrebleu.corpus_bleu(back_translated, [original_english_sources])
```

---

## 2. Instruction Following Metrics

Run on all 1,200 instruction samples. Both models receive the same `system_instruction` + `user_prompt`. Metrics evaluate whether the model output satisfies `expected_constraints`.

### 2.1 Language Adherence Rate
- **Applies to:** All categories
- **What:** % of responses written in the correct target language.
- **Library:** `langdetect` (`pip install langdetect`) or `lingua-language-detector`
- **Method:** Detect language of model output, compare to expected `language` field.
- **Target:** ≥ 95%

```python
from langdetect import detect
detected = detect(model_output)
correct = (detected == expected_lang_code)
```

### 2.2 Length Constraint Accuracy
- **Applies to:** `length_constraint` category
- **What:** % of responses satisfying the word or sentence count constraint in `expected_constraints`.
- **Method:** Rule-based — count words (`.split()`) or sentences (split on `.!?`).
- **Constraints in dataset:** `max_sentences`, `min_sentences`, `max_words`

```python
def check_length(output, constraints):
    if "max_sentences" in constraints:
        sentences = [s for s in output.split('.') if s.strip()]
        return len(sentences) <= constraints["max_sentences"]
    if "max_words" in constraints:
        return len(output.split()) <= constraints["max_words"]
```

### 2.3 Format Compliance
- **Applies to:** `structured_output` category
- **What:** Whether the response follows the specified structure.
- **Method:** Regex + rule-based checks per format type.

| Format constraint | Check |
|------------------|-------|
| `numbered_list` | Output contains `1.`, `2.`, `3.` pattern |
| `greeting_answer_offer` | Starts with greeting token, ends with offer phrase |
| `ends_with_offer` | Last sentence contains offer-to-help phrase |
| `bullet_points_for_lists` | Contains `•` or `-` list markers |
| `header_then_details` | First line is shorter than subsequent lines (header pattern) |
| `qa_restatement` | First sentence ends with `?` (restated question) |

### 2.4 Topic Boundary Respect
- **Applies to:** `topic_boundary` category
- **What:** Does the model stay on-topic and avoid speculation?
- **Method:** Embedding cosine similarity between model output and the user prompt. A response that goes off-topic will have low similarity to the original question.
- **Library:** `sentence-transformers` with `paraphrase-multilingual-MiniLM-L12-v2`
- **Threshold:** cosine similarity < 0.3 flags a potential boundary violation.

### 2.5 Register / Tone Detection
- **Applies to:** `tone_style` category
- **What:** Is the output in the expected register (formal vs informal, empathetic vs direct)?
- **Method:** Pre-trained multilingual tone/sentiment classifier, or prompt a lightweight LLM to score formality on a 1–5 scale.
- **Library:** `transformers` with a multilingual classifier, or rule-based proxies (e.g. presence of formal honorifics per language).

---

## 3. Number Verbalization Metrics

Applies exclusively to the `number_verbalization` category (200 samples/language).

Each sample's `expected_constraints` contains:
- `number_type` — one of: `postal_code`, `phone_number`, `otp_pin`, `currency`, `decimal_measurement`, `percentage`, `quantity`, `date`, `mixed`
- `rule` — `digit_by_digit` | `word_form` | `mixed`
- `numbers_in_prompt` — list of the original number strings present in the input
- `digit_by_digit_types` — list of number types that must be read digit-by-digit
- `word_form_types` — list of number types that must be converted to words

### 3.1 Digit-by-Digit Compliance
- **What:** For `postal_code`, `phone_number`, `otp_pin`: does the model read each digit individually (not as a composite number)?
- **Check:** The original digit string must NOT appear literally in the output. Each digit of the number should appear as an individual word.
- **Method:** Strip non-digit characters from `numbers_in_prompt`, confirm none of the resulting digit sequences appear as a continuous run in the output.

```python
import re
def check_digit_by_digit(output, number_str):
    digits = re.sub(r'\D', '', number_str)
    # Fail if the full digit sequence appears unbroken in output
    return digits not in re.sub(r'\D', '', output)
```

### 3.2 Word-Form Compliance
- **What:** For `currency`, `decimal_measurement`, `percentage`, `quantity`, `date`: does the model use word form with no raw digits remaining in the verbalized portion?
- **Check:** After removing expected non-numeric characters, no digit characters (`0-9`) should remain in the section of output that corresponds to the number.
- **Method:** Regex scan on model output for digit characters.

```python
def check_word_form(output, number_str):
    digits = re.sub(r'\D', '', number_str)
    # Each digit in the original number should not appear as a raw digit in output
    for d in set(digits):
        if d in output:
            return False
    return True
```

### 3.3 No Literal Digit Leakage
- **What:** Overall check — do any raw digit characters remain in the output where a number should have been verbalized?
- **Method:** Count raw digit characters in the output. Lower is better; 0 = perfect verbalization (for word_form types).
- **Score:** `digit_leakage_count` per response. Report mean across samples.

### 3.4 Digit Preservation Accuracy
- **What:** Are all digits from the input number present in the output (none dropped or hallucinated)?
- **Method:** Count digits in `numbers_in_prompt` entry → count digit-word equivalents in output (map each digit to its word form in the target language). Compare counts.
- **Score:** % of samples where digit count matches.

### 3.5 Currency Unit Accuracy
- **What:** When a currency symbol is present (`$`, `₹`, `€`, `£`, `¥`), does the output include the correct currency unit word?
- **Method:** Map currency symbol → expected word per language (e.g. `$` → "dollars" in English output context, "டாலர்" in Tamil). Check for presence in output.
- **Applies to:** `currency` type samples only.

```python
CURRENCY_WORDS = {
    "$": ["dollar", "dollars"],
    "₹": ["rupee", "rupees"],
    "€": ["euro", "euros"],
    "£": ["pound", "pounds"],
    "¥": ["yen"],
    # Add target-language equivalents per language
}
```

### 3.6 Number Type Classification Accuracy
- **What:** Did the model apply the correct rule for the number type? (e.g. did it read a postal code digit-by-digit rather than as a large integer?)
- **Method:** Given `number_type` and `rule` from `expected_constraints`, verify the applied rule matches. This combines checks 3.1 and 3.2: if `rule == digit_by_digit` and check 3.1 passes → correct. If `rule == word_form` and check 3.2 passes → correct.
- **Score:** % of samples where the correct rule was applied.

### 3.7 Language of Number Words
- **What:** Are the verbalized number words in the target language, not English?
- **Method:** Extract the spans in the output that replaced numbers. Run `langdetect` on those spans. Flag any that detect as English when the target language is not English.
- **Score:** % of samples where number words are in the correct target language.

### 3.8 Mixed-Sentence Consistency
- **What:** For `mixed` type samples (sentences containing multiple number types), are ALL numbers handled correctly — each with the right rule applied?
- **Applies to:** samples where `number_type == "mixed"` (10 prompts × cycling = 40 samples/language in the dataset)
- **Method:** Composite of checks 3.1 + 3.2 across all entries in `numbers_in_prompt`. All must pass for the sample to be counted correct.
- **Score:** Strictest metric — % of mixed samples where every number is correctly verbalized.

---

## Summary Table

| # | Metric | Category | Automated? | Library |
|---|--------|----------|-----------|---------|
| T1 | BLEU | Translation | ✅ | `sacrebleu` |
| T2 | chrF | Translation | ✅ | `sacrebleu` |
| T3 | COMET | Translation | ✅ | `unbabel-comet` |
| T4 | BERTScore | Translation | ✅ | `bert-score` |
| T5 | Back-translation consistency | Translation (en→target only) | ✅ | `transformers` + `sacrebleu` |
| I1 | Language adherence rate | All instruction categories | ✅ | `langdetect` |
| I2 | Length constraint accuracy | `length_constraint` | ✅ | Rule-based |
| I3 | Format compliance | `structured_output` | ✅ | Regex |
| I4 | Topic boundary respect | `topic_boundary` | ✅ | `sentence-transformers` |
| I5 | Register / tone detection | `tone_style` | ⚠️ Partial | Classifier or LLM-proxy |
| N1 | Digit-by-digit compliance | `number_verbalization` | ✅ | Regex |
| N2 | Word-form compliance | `number_verbalization` | ✅ | Regex |
| N3 | No literal digit leakage | `number_verbalization` | ✅ | Regex |
| N4 | Digit preservation accuracy | `number_verbalization` | ✅ | Rule-based |
| N5 | Currency unit accuracy | `number_verbalization` | ✅ | Lookup table |
| N6 | Number type classification accuracy | `number_verbalization` | ✅ | Composite of N1+N2 |
| N7 | Language of number words | `number_verbalization` | ✅ | `langdetect` |
| N8 | Mixed-sentence consistency | `number_verbalization` (mixed type) | ✅ | Composite of N1+N2 |

14 of 18 metrics are fully automated (no LLM or human needed). I5 (tone) and T3 (COMET) require a model call but no human review.

---

## Dependencies

```
pip install sacrebleu unbabel-comet bert-score langdetect sentence-transformers transformers
```

---

## Output Format (suggested)

Per language, per model, report:

```json
{
  "language": "Tamil",
  "model": "Tamil-Mistral-7B",
  "translation": {
    "bleu_en_to_target": 32.4,
    "bleu_target_to_en": 28.1,
    "chrf_en_to_target": 51.2,
    "comet_en_to_target": 0.84,
    "bertscore_f1": 0.79,
    "back_translation_bleu": 41.3
  },
  "instruction_following": {
    "language_adherence_rate": 0.97,
    "length_constraint_accuracy": 0.88,
    "format_compliance": 0.91,
    "topic_boundary_respect": 0.85
  },
  "number_verbalization": {
    "digit_by_digit_compliance": 0.93,
    "word_form_compliance": 0.89,
    "no_literal_digit_leakage_mean": 0.2,
    "digit_preservation_accuracy": 0.95,
    "currency_unit_accuracy": 0.91,
    "number_type_classification_accuracy": 0.88,
    "language_of_number_words": 0.96,
    "mixed_sentence_consistency": 0.81
  }
}
```
