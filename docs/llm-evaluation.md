# Global Tokenizer Evaluation — Detailed Report

**Generated:** 2026-06-16 10:41 UTC  ·  **Source:** `data/results.csv`  ·  **Corpus:** FLORES-200 devtest (~1012 sentences/language)

> **Caveats.** Fertility uses whitespace "words", which is imperfect for languages without whitespace
> word boundaries (Japanese, Thai, Burmese, Khmer). For those languages, use `avg_tokens_per_sent`
> as the primary metric instead of fertility. Vocabulary coverage is computed from characters present
> in the FLORES-200 corpus only.

---

## 1. Metric glossary

All metrics computed over the same FLORES-200 devtest segments per language.

### 1.1 Quick reference: which direction is "better"?

| Metric | Better | Target |
| --- | --- | --- |
| Fertility | **Lower** | 1.0–2.5 for script-based languages |
| Compression ratio | **Higher** | > 3.0 chars/token |
| Byte fallback rate (%) | **Lower** | 0% ideal; any > 0 = unhandled script |
| UNK rate (%) | **Lower** | 0%; any UNK = vocab miss |
| Vocabulary coverage (%) | **Higher** | > 80% of unique chars as single tokens |
| Roundtrip fidelity (%) | **Higher** | 100% = lossless encode→decode |
| Avg tokens / segment | **Lower** | same story as fertility (inference cost) |

### 1.2 Fertility
**What it is:** Tokens per whitespace-separated 'word': `total_tokens / total_words`.
**Direction:** Lower is better. High fertility = same document uses more tokens (higher API cost, less room in context).

### 1.3 Compression ratio
**What it is:** Characters per token: `total_chars / total_tokens`.
**Direction:** Higher is better. Low compression often means byte-level or very fine splitting.

### 1.4 Byte fallback rate
**What it is:** Percentage of output tokens that represent raw byte fallback, not normal subwords.
**Direction:** Lower is better. 0% means no detected byte-fallback tokens.

### 1.5 UNK rate
**What it is:** Percentage of token ids equal to the tokenizer's `unk_token_id`.
**Direction:** Lower is better; 0% is the target.

### 1.6 Vocabulary coverage
**What it is:** Among all unique characters in that language's test text, the fraction that encode to exactly **one** token when passed alone to the tokenizer.
**Direction:** Higher is better. High coverage means many script characters are first-class tokens.

### 1.7 Roundtrip fidelity
**What it is:** Per segment: after `encode` then `decode`, does the text match the original?
**Direction:** Higher is better; 100% means no lossy tokenization on this test set.

### 1.8 Avg tokens per segment
**What it is:** `total_tokens / total_sentences`.
**Direction:** Lower is better for cost/latency. Use this as the primary efficiency metric for non-whitespace languages (Japanese, Thai, Burmese, Khmer).

---

## 2. Aggregate summary (all languages)

**Unweighted** averages treat each language equally. **Character-weighted** averages weight by `total_chars` so languages with more text influence the score more.

| Tokenizer | Languages tested | Avg fertility | Avg compression | Avg byte fallback % | Avg vocab coverage % | Avg roundtrip % |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma-4 | 63 | 3.217 | 3.027 | 0.298 | 99.672 | 100.0 |
| BLOOM | 63 | 4.548 | 2.961 | 2.698 | 93.277 | 100.0 |

---

## 3. Regional candidate vs Gemma-4 — verdict per language

Primary signals: fertility and vocab coverage. Secondary: byte fallback rate and roundtrip fidelity.

| Language | Region | Candidate | Fertility G4 | Fertility cand | Vcov G4 % | Vcov cand % | BFR cand % | RT cand % | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bengali | Indic | BanglaLLama-3.1-8B | 1.683 | 8.021 | 100.0 | 58.02 | 16.93 | 100.0 | ❌ Gemma-4 wins |
| Tamil | Indic | Tamil-Mistral-7B | 2.374 | 1.734 | 100.0 | 86.61 | 0.0 | 100.0 | ✅ Candidate wins |
| Telugu | Indic | Telugu-Llama2-7B | 2.843 | 19.59 | 99.35 | 50.0 | 90.6 | 100.0 | ❌ Gemma-4 wins |
| Malayalam | Indic | MalayaLLM-Gemma-9B | 3.357 | 5.875 | 100.0 | 100.0 | 0.0 | 100.0 | ⚠️ Mixed |
| Gujarati | Indic | Gujju-Llama-7B | 2.415 | 2.033 | 99.29 | 82.98 | 0.0 | 100.0 | ✅ Candidate wins |
| Punjabi | Indic | Dhee-Qwen3-Punjabi-2B | 2.82 | 7.758 | 99.33 | 67.33 | 19.43 | 43.38 | ❌ Gemma-4 wins |
| Urdu | Indic | Qalb-1.0-8B | 1.489 | 3.02 | 100.0 | 91.11 | 17.51 | 100.0 | ❌ Gemma-4 wins |
| Nepali | Indic | NEPALI-LLM-9B | 2.216 | 2.48 | 99.35 | 100.0 | 0.0 | 100.0 | ⚠️ Mixed |
| Sinhala | Indic | llama3-sinhala-8B | 2.997 | 11.324 | 98.64 | 52.38 | 28.31 | 100.0 | ❌ Gemma-4 wins |
| Persian | Middle East | Maral-7B | 1.679 | 5.138 | 100.0 | 53.49 | 0.04 | 100.0 | ⚠️ Mixed |
| Turkish | Middle East | Trendyol-8B | 2.109 | 2.547 | 100.0 | 100.0 | 0.6 | 100.0 | ⚠️ Mixed |
| Hebrew | Middle East | DictaLM-2.0-7B | 2.706 | 2.642 | 100.0 | 83.33 | 0.0 | 100.0 | ✅ Candidate wins |
| Kurdish | Middle East | Mistral-Nemo-Kurdish | 2.358 | 2.486 | 100.0 | 96.19 | 0.51 | 100.0 | ⚠️ Mixed |
| Uzbek | Middle East | Mistral-7B-Uz | 2.839 | 3.515 | 97.85 | 100.0 | 0.0 | 100.0 | ⚠️ Mixed |
| Japanese | East Asia | LLM-jp-3-13B | 30.432 | 25.627 | 100.0 | 71.3 | 0.0 | 100.0 | ⚠️ Mixed |
| Korean | East Asia | Polyglot-Ko-12B | 2.415 | 2.199 | 99.35 | 93.26 | 0.13 | 100.0 | ✅ Candidate wins |
| Vietnamese | SEA | Arcee-VyLinh-3B | 1.21 | 1.29 | 100.0 | 100.0 | 0.89 | 100.0 | ⚠️ Mixed |
| Thai | SEA | Typhoon2-7B | 10.084 | 16.315 | 100.0 | 98.77 | 2.35 | 100.0 | ❌ Gemma-4 wins |
| Indonesian | SEA | Nusantara-7B | 1.575 | 2.146 | 100.0 | 100.0 | 0.62 | 100.0 | ⚠️ Mixed |
| Malay | SEA | MaLLaM-5B | 1.632 | 1.417 | 100.0 | 84.76 | 0.04 | 100.0 | ✅ Candidate wins |
| Burmese | SEA | Burmese-GPT-1B | 6.136 | 10.719 | 100.0 | 91.5 | 0.6 | 100.0 | ⚠️ Mixed |
| Swahili | Africa | Swahili-Gemma-7B | 2.087 | 2.047 | 100.0 | 100.0 | 0.0 | 100.0 | ✅ Candidate wins |
| Igbo | Africa | Kakugo-3B-Igbo | 2.358 | 2.688 | 100.0 | 95.0 | 4.25 | 100.0 | ❌ Gemma-4 wins |
| Wolof | Africa | Wolof-Qwen-1.5B | 1.921 | 2.103 | 99.19 | 100.0 | 0.53 | 100.0 | ⚠️ Mixed |
| French | Europe | Lucie-7B | 1.49 | 1.427 | 98.31 | 82.2 | 0.0 | 100.0 | ✅ Candidate wins |
| German | Europe | LeoLM-7B | 1.655 | 2.18 | 100.0 | 81.48 | 0.01 | 100.0 | ⚠️ Mixed |
| Spanish | Europe | Salamandra-7B | 1.347 | 1.354 | 100.0 | 85.58 | 0.03 | 100.0 | ⚠️ Mixed |
| Portuguese | Europe | Gervasio-8B | 1.453 | 1.709 | 100.0 | 100.0 | 0.74 | 100.0 | ⚠️ Mixed |
| Italian | Europe | LLaMAntino-3-8B | 1.535 | 1.831 | 100.0 | 100.0 | 0.58 | 100.0 | ⚠️ Mixed |
| Dutch | Europe | Fietje-2 | 1.63 | 2.325 | 100.0 | 94.34 | 0.1 | 100.0 | ⚠️ Mixed |
| Polish | Europe | Bielik-11B | 2.096 | 2.924 | 100.0 | 81.13 | 3.58 | 100.0 | ❌ Gemma-4 wins |
| Russian | Europe | Vikhr-Nemo-12B | 1.884 | 2.13 | 99.33 | 98.67 | 0.71 | 100.0 | ⚠️ Mixed |
| Ukrainian | Europe | MamayLM-12B | 2.273 | 2.273 | 100.0 | 100.0 | 0.0 | 100.0 | ⚠️ Mixed |
| Romanian | Europe | LLMic-3B | 1.8 | 2.192 | 100.0 | 77.68 | 16.17 | 4.15 | ❌ Gemma-4 wins |
| Swedish | Europe | Viking-7B | 1.841 | 1.474 | 100.0 | 100.0 | 0.94 | 100.0 | ✅ Candidate wins |
| Czech | Europe | CSMPT-7B | 2.157 | 1.407 | 99.13 | 96.52 | 0.03 | 100.0 | ✅ Candidate wins |
| Greek | Europe | Meltemi-7B | 2.472 | 1.397 | 100.0 | 95.42 | 0.04 | 100.0 | ✅ Candidate wins |
| Lat.Am. Spanish | Americas | LatamGPT-70B | 1.347 | 1.604 | 100.0 | 100.0 | 0.65 | 100.0 | ⚠️ Mixed |
| Brazilian Portuguese | Americas | Tucano-2b4 | 1.453 | 1.252 | 100.0 | 86.84 | 0.01 | 99.31 | ✅ Candidate wins |

---

## 4. Comparison across metrics (pivot tables)

Rows = languages. Columns = tokenizers. Scan across a row to compare models on one language; scan down a column to see one model across languages.

### Fertility (tokens / whitespace word)
*Lower is better.*

**Indic**

| language | Gemma-4 | BLOOM | mT5 | BanglaLLama-3.1-8B | Tamil-Mistral-7B | Telugu-Llama2-7B | MalayaLLM-Gemma-9B | Gujju-Llama-7B | Dhee-Qwen3-Punjabi-2B | Qalb-1.0-8B | NEPALI-LLM-9B | llama3-sinhala-8B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hindi | 1.385 | 1.37 | — | — | — | — | — | — | — | — | — | — |
| Bengali | 1.683 | 1.652 | — | 8.021 | — | — | — | — | — | — | — | — |
| Tamil | 2.374 | 2.099 | — | — | 1.734 | — | — | — | — | — | — | — |
| Telugu | 2.843 | 2.15 | — | — | — | 19.59 | — | — | — | — | — | — |
| Kannada | 3.241 | 2.217 | — | — | — | — | — | — | — | — | — | — |
| Malayalam | 3.357 | 2.54 | — | — | — | — | 5.875 | — | — | — | — | — |
| Marathi | 1.987 | 1.754 | — | — | — | — | — | — | — | — | — | — |
| Gujarati | 2.415 | 1.776 | — | — | — | — | — | 2.033 | — | — | — | — |
| Punjabi | 2.82 | 1.528 | — | — | — | — | — | — | 7.758 | — | — | — |
| Odia | 4.852 | 1.912 | — | — | — | — | — | — | — | — | — | — |
| Assamese | 2.819 | 1.999 | — | — | — | — | — | — | — | — | — | — |
| Urdu | 1.489 | 1.344 | — | — | — | — | — | — | — | 3.02 | — | — |
| Nepali | 2.216 | 1.723 | — | — | — | — | — | — | — | — | 2.48 | — |
| Sinhala | 2.997 | 10.689 | — | — | — | — | — | — | — | — | — | 11.324 |
| Maithili | 1.806 | 1.734 | — | — | — | — | — | — | — | — | — | — |

**Middle East**

| language | Gemma-4 | BLOOM | mT5 | Maral-7B | Trendyol-8B | DictaLM-2.0-7B | Mistral-Nemo-Kurdish | Mistral-7B-Uz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arabic | 2.031 | 1.603 | — | — | — | — | — | — |
| Persian | 1.679 | 2.055 | — | 5.138 | — | — | — | — |
| Turkish | 2.109 | 3.051 | — | — | 2.547 | — | — | — |
| Hebrew | 2.706 | 4.568 | — | — | — | 2.642 | — | — |
| Kurdish | 2.358 | 2.446 | — | — | — | — | 2.486 | — |
| Azerbaijani | 2.873 | 3.296 | — | — | — | — | — | — |
| Uzbek | 2.839 | 2.974 | — | — | — | — | — | 3.515 |
| Kazakh | 3.184 | 5.089 | — | — | — | — | — | — |

**East Asia**

| language | Gemma-4 | BLOOM | mT5 | LLM-jp-3-13B | Polyglot-Ko-12B |
| --- | --- | --- | --- | --- | --- |
| Chinese | 14.132 | 12.117 | — | — | — |
| Japanese | 30.432 | 44.828 | — | 25.627 | — |
| Korean | 2.415 | 4.887 | — | — | 2.199 |

**SEA**

| language | Gemma-4 | BLOOM | mT5 | Arcee-VyLinh-3B | Typhoon2-7B | Nusantara-7B | MaLLaM-5B | Burmese-GPT-1B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vietnamese | 1.21 | 1.128 | — | 1.29 | — | — | — | — |
| Thai | 10.084 | 29.119 | — | — | 16.315 | — | — | — |
| Indonesian | 1.575 | 1.333 | — | — | — | 2.146 | — | — |
| Malay | 1.632 | 1.444 | — | — | — | — | 1.417 | — |
| Tagalog | 1.84 | 1.895 | — | — | — | — | — | — |
| Burmese | 6.136 | 25.746 | — | — | — | — | — | 10.719 |
| Khmer | 12.747 | 29.492 | — | — | — | — | — | — |

**Africa**

| language | Gemma-4 | BLOOM | mT5 | Swahili-Gemma-7B | Kakugo-3B-Igbo | Wolof-Qwen-1.5B |
| --- | --- | --- | --- | --- | --- | --- |
| Swahili | 2.087 | 1.602 | — | 2.047 | — | — |
| Amharic | 3.032 | 7.905 | — | — | — | — |
| Hausa | 1.864 | 1.924 | — | — | — | — |
| Yoruba | 2.588 | 1.759 | — | — | — | — |
| Igbo | 2.358 | 1.892 | — | — | 2.688 | — |
| Zulu | 3.384 | 3.043 | — | — | — | — |
| Xhosa | 3.302 | 2.933 | — | — | — | — |
| Somali | 2.246 | 2.402 | — | — | — | — |
| Wolof | 1.921 | 1.845 | — | — | — | 2.103 |
| Shona | 2.927 | 2.848 | — | — | — | — |

**Europe**

| language | Gemma-4 | BLOOM | mT5 | Lucie-7B | LeoLM-7B | Salamandra-7B | Gervasio-8B | LLaMAntino-3-8B | Fietje-2 | Bielik-11B | Vikhr-Nemo-12B | MamayLM-12B | LLMic-3B | Viking-7B | CSMPT-7B | Meltemi-7B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| French | 1.49 | 1.289 | — | 1.427 | — | — | — | — | — | — | — | — | — | — | — | — |
| German | 1.655 | 2.12 | — | — | 2.18 | — | — | — | — | — | — | — | — | — | — | — |
| Spanish | 1.347 | 1.269 | — | — | — | 1.354 | — | — | — | — | — | — | — | — | — | — |
| Portuguese | 1.453 | 1.312 | — | — | — | — | 1.709 | — | — | — | — | — | — | — | — | — |
| Italian | 1.535 | 1.828 | — | — | — | — | — | 1.831 | — | — | — | — | — | — | — | — |
| Dutch | 1.63 | 2.04 | — | — | — | — | — | — | 2.325 | — | — | — | — | — | — | — |
| Polish | 2.096 | 2.99 | — | — | — | — | — | — | — | 2.924 | — | — | — | — | — | — |
| Russian | 1.884 | 3.425 | — | — | — | — | — | — | — | — | 2.13 | — | — | — | — | — |
| Ukrainian | 2.273 | 3.899 | — | — | — | — | — | — | — | — | — | 2.273 | — | — | — | — |
| Romanian | 1.8 | 2.205 | — | — | — | — | — | — | — | — | — | — | 2.192 | — | — | — |
| Swedish | 1.841 | 2.222 | — | — | — | — | — | — | — | — | — | — | — | 1.474 | — | — |
| Czech | 2.157 | 2.885 | — | — | — | — | — | — | — | — | — | — | — | — | 1.407 | — |
| Greek | 2.472 | 4.341 | — | — | — | — | — | — | — | — | — | — | — | — | — | 1.397 |

**Americas**

| language | Gemma-4 | BLOOM | mT5 | LatamGPT-70B | Tucano-2b4 |
| --- | --- | --- | --- | --- | --- |
| Lat.Am. Spanish | 1.347 | 1.269 | — | 1.604 | — |
| Brazilian Portuguese | 1.453 | 1.312 | — | — | 1.252 |
| Quechua | 3.147 | 3.158 | — | — | — |
| Haitian Creole | 1.796 | 1.85 | — | — | — |

**Oceania**

| language | Gemma-4 | BLOOM | mT5 |
| --- | --- | --- | --- |
| Māori | 1.825 | 1.91 | — |
| Samoan | 1.821 | 1.823 | — |
| Tok Pisin | 1.644 | 1.668 | — |

### Compression ratio (chars / token)
*Higher is better.*

**Indic**

| language | Gemma-4 | BLOOM | mT5 | BanglaLLama-3.1-8B | Tamil-Mistral-7B | Telugu-Llama2-7B | MalayaLLM-Gemma-9B | Gujju-Llama-7B | Dhee-Qwen3-Punjabi-2B | Qalb-1.0-8B | NEPALI-LLM-9B | llama3-sinhala-8B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hindi | 3.69 | 3.73 | — | — | — | — | — | — | — | — | — | — |
| Bengali | 3.93 | 4.004 | — | 0.825 | — | — | — | — | — | — | — | — |
| Tamil | 3.87 | 4.378 | — | — | 5.297 | — | — | — | — | — | — | — |
| Telugu | 2.751 | 3.637 | — | — | — | 0.399 | — | — | — | — | — | — |
| Kannada | 2.645 | 3.868 | — | — | — | — | — | — | — | — | — | — |
| Malayalam | 2.98 | 3.937 | — | — | — | — | 1.703 | — | — | — | — | — |
| Marathi | 3.517 | 3.982 | — | — | — | — | — | — | — | — | — | — |
| Gujarati | 2.506 | 3.407 | — | — | — | — | — | 2.977 | — | — | — | — |
| Punjabi | 1.827 | 3.372 | — | — | — | — | — | — | 0.664 | — | — | — |
| Odia | 1.416 | 3.595 | — | — | — | — | — | — | — | — | — | — |
| Assamese | 2.316 | 3.266 | — | — | — | — | — | — | — | — | — | — |
| Urdu | 3.156 | 3.496 | — | — | — | — | — | — | — | 1.556 | — | — |
| Nepali | 3.062 | 3.938 | — | — | — | — | — | — | — | — | 2.736 | — |
| Sinhala | 2.112 | 0.592 | — | — | — | — | — | — | — | — | — | 0.559 |
| Maithili | 2.912 | 3.032 | — | — | — | — | — | — | — | — | — | — |

**Middle East**

| language | Gemma-4 | BLOOM | mT5 | Maral-7B | Trendyol-8B | DictaLM-2.0-7B | Mistral-Nemo-Kurdish | Mistral-7B-Uz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arabic | 2.917 | 3.695 | — | — | — | — | — | — |
| Persian | 3.135 | 2.56 | — | 1.024 | — | — | — | — |
| Turkish | 3.655 | 2.527 | — | — | 3.027 | — | — | — |
| Hebrew | 2.172 | 1.286 | — | — | — | 2.224 | — | — |
| Kurdish | 2.438 | 2.35 | — | — | — | — | 2.313 | — |
| Azerbaijani | 2.619 | 2.283 | — | — | — | — | — | — |
| Uzbek | 2.888 | 2.757 | — | — | — | — | — | 2.333 |
| Kazakh | 2.452 | 1.534 | — | — | — | — | — | — |

**East Asia**

| language | Gemma-4 | BLOOM | mT5 | LLM-jp-3-13B | Polyglot-Ko-12B |
| --- | --- | --- | --- | --- | --- |
| Chinese | 1.479 | 1.725 | — | — | — |
| Japanese | 1.731 | 1.175 | — | 2.055 | — |
| Korean | 1.77 | 0.875 | — | — | 1.944 |

**SEA**

| language | Gemma-4 | BLOOM | mT5 | Arcee-VyLinh-3B | Typhoon2-7B | Nusantara-7B | MaLLaM-5B | Burmese-GPT-1B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vietnamese | 3.727 | 3.998 | — | 3.496 | — | — | — | — |
| Thai | 2.906 | 1.006 | — | — | 1.796 | — | — | — |
| Indonesian | 4.562 | 5.39 | — | — | — | 3.347 | — | — |
| Malay | 4.437 | 5.014 | — | — | — | — | 5.109 | — |
| Tagalog | 3.374 | 3.275 | — | — | — | — | — | — |
| Burmese | 2.499 | 0.596 | — | — | — | — | — | 1.43 |
| Khmer | 2.067 | 0.893 | — | — | — | — | — | — |

**Africa**

| language | Gemma-4 | BLOOM | mT5 | Swahili-Gemma-7B | Kakugo-3B-Igbo | Wolof-Qwen-1.5B |
| --- | --- | --- | --- | --- | --- | --- |
| Swahili | 3.108 | 4.048 | — | 3.169 | — | — |
| Amharic | 1.658 | 0.636 | — | — | — | — |
| Hausa | 2.987 | 2.893 | — | — | — | — |
| Yoruba | 1.944 | 2.862 | — | — | — | — |
| Igbo | 2.29 | 2.854 | — | — | 2.009 | — |
| Zulu | 2.77 | 3.081 | — | — | — | — |
| Xhosa | 2.717 | 3.059 | — | — | — | — |
| Somali | 2.896 | 2.707 | — | — | — | — |
| Wolof | 2.643 | 2.751 | — | — | — | 2.414 |
| Shona | 2.907 | 2.987 | — | — | — | — |

**Europe**

| language | Gemma-4 | BLOOM | mT5 | Lucie-7B | LeoLM-7B | Salamandra-7B | Gervasio-8B | LLaMAntino-3-8B | Fietje-2 | Bielik-11B | Vikhr-Nemo-12B | MamayLM-12B | LLMic-3B | Viking-7B | CSMPT-7B | Meltemi-7B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| French | 4.129 | 4.774 | — | 4.311 | — | — | — | — | — | — | — | — | — | — | — | — |
| German | 4.27 | 3.333 | — | — | 3.242 | — | — | — | — | — | — | — | — | — | — | — |
| Spanish | 4.47 | 4.745 | — | — | — | 4.447 | — | — | — | — | — | — | — | — | — | — |
| Portuguese | 4.215 | 4.669 | — | — | — | — | 3.585 | — | — | — | — | — | — | — | — | — |
| Italian | 4.185 | 3.514 | — | — | — | — | — | 3.508 | — | — | — | — | — | — | — | — |
| Dutch | 3.935 | 3.144 | — | — | — | — | — | — | 2.759 | — | — | — | — | — | — | — |
| Polish | 3.415 | 2.394 | — | — | — | — | — | — | — | 2.447 | — | — | — | — | — | — |
| Russian | 3.838 | 2.111 | — | — | — | — | — | — | — | — | 3.395 | — | — | — | — | — |
| Ukrainian | 3.083 | 1.798 | — | — | — | — | — | — | — | — | — | 3.083 | — | — | — | — |
| Romanian | 3.479 | 2.84 | — | — | — | — | — | — | — | — | — | — | 2.856 | — | — | — |
| Swedish | 3.543 | 2.935 | — | — | — | — | — | — | — | — | — | — | — | 4.423 | — | — |
| Czech | 3.077 | 2.3 | — | — | — | — | — | — | — | — | — | — | — | — | 4.717 | — |
| Greek | 2.666 | 1.518 | — | — | — | — | — | — | — | — | — | — | — | — | — | 4.72 |

**Americas**

| language | Gemma-4 | BLOOM | mT5 | LatamGPT-70B | Tucano-2b4 |
| --- | --- | --- | --- | --- | --- |
| Lat.Am. Spanish | 4.47 | 4.745 | — | 3.752 | — |
| Brazilian Portuguese | 4.215 | 4.669 | — | — | 4.893 |
| Quechua | 2.811 | 2.801 | — | — | — |
| Haitian Creole | 2.918 | 2.831 | — | — | — |

**Oceania**

| language | Gemma-4 | BLOOM | mT5 |
| --- | --- | --- | --- |
| Māori | 2.647 | 2.529 | — |
| Samoan | 2.635 | 2.632 | — |
| Tok Pisin | 3.247 | 3.201 | — |

### Byte fallback rate (%)
*Lower is better.*

**Indic**

| language | Gemma-4 | BLOOM | mT5 | BanglaLLama-3.1-8B | Tamil-Mistral-7B | Telugu-Llama2-7B | MalayaLLM-Gemma-9B | Gujju-Llama-7B | Dhee-Qwen3-Punjabi-2B | Qalb-1.0-8B | NEPALI-LLM-9B | llama3-sinhala-8B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hindi | 0.0 | 0.03 | — | — | — | — | — | — | — | — | — | — |
| Bengali | 0.0 | 0.27 | — | 16.93 | — | — | — | — | — | — | — | — |
| Tamil | 0.0 | 0.24 | — | — | 0.0 | — | — | — | — | — | — | — |
| Telugu | 1.0 | 0.62 | — | — | — | 90.6 | — | — | — | — | — | — |
| Kannada | 0.0 | 0.86 | — | — | — | — | — | — | — | — | — | — |
| Malayalam | 0.0 | 0.17 | — | — | — | — | 0.0 | — | — | — | — | — |
| Marathi | 0.0 | 0.23 | — | — | — | — | — | — | — | — | — | — |
| Gujarati | 0.01 | 0.34 | — | — | — | — | — | 0.0 | — | — | — | — |
| Punjabi | 0.01 | 0.07 | — | — | — | — | — | — | 19.43 | — | — | — |
| Odia | 1.06 | 0.22 | — | — | — | — | — | — | — | — | — | — |
| Assamese | 0.01 | 0.26 | — | — | — | — | — | — | — | — | — | — |
| Urdu | 0.0 | 0.04 | — | — | — | — | — | — | — | 17.51 | — | — |
| Nepali | 0.0 | 0.04 | — | — | — | — | — | — | — | — | 0.0 | — |
| Sinhala | 0.02 | 30.3 | — | — | — | — | — | — | — | — | — | 28.31 |
| Maithili | 0.0 | 0.01 | — | — | — | — | — | — | — | — | — | — |

**Middle East**

| language | Gemma-4 | BLOOM | mT5 | Maral-7B | Trendyol-8B | DictaLM-2.0-7B | Mistral-Nemo-Kurdish | Mistral-7B-Uz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arabic | 0.0 | 0.0 | — | — | — | — | — | — |
| Persian | 0.0 | 0.13 | — | 0.04 | — | — | — | — |
| Turkish | 0.96 | 0.12 | — | — | 0.6 | — | — | — |
| Hebrew | 0.0 | 7.34 | — | — | — | 0.0 | — | — |
| Kurdish | 0.0 | 0.09 | — | — | — | — | 0.51 | — |
| Azerbaijani | 0.91 | 0.47 | — | — | — | — | — | — |
| Uzbek | 0.01 | 0.0 | — | — | — | — | — | 0.0 |
| Kazakh | 0.0 | 2.79 | — | — | — | — | — | — |

**East Asia**

| language | Gemma-4 | BLOOM | mT5 | LLM-jp-3-13B | Polyglot-Ko-12B |
| --- | --- | --- | --- | --- | --- |
| Chinese | 0.05 | 0.79 | — | — | — |
| Japanese | 0.0 | 2.32 | — | 0.0 | — |
| Korean | 0.07 | 22.1 | — | — | 0.13 |

**SEA**

| language | Gemma-4 | BLOOM | mT5 | Arcee-VyLinh-3B | Typhoon2-7B | Nusantara-7B | MaLLaM-5B | Burmese-GPT-1B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vietnamese | 0.18 | 0.02 | — | 0.89 | — | — | — | — |
| Thai | 0.0 | 11.86 | — | — | 2.35 | — | — | — |
| Indonesian | 0.0 | 0.0 | — | — | — | 0.62 | — | — |
| Malay | 0.0 | 0.0 | — | — | — | — | 0.04 | — |
| Tagalog | 0.0 | 0.02 | — | — | — | — | — | — |
| Burmese | 0.0 | 27.36 | — | — | — | — | — | 0.6 |
| Khmer | 0.01 | 26.35 | — | — | — | — | — | — |

**Africa**

| language | Gemma-4 | BLOOM | mT5 | Swahili-Gemma-7B | Kakugo-3B-Igbo | Wolof-Qwen-1.5B |
| --- | --- | --- | --- | --- | --- | --- |
| Swahili | 0.0 | 0.05 | — | 0.0 | — | — |
| Amharic | 0.02 | 26.31 | — | — | — | — |
| Hausa | 0.0 | 1.5 | — | — | — | — |
| Yoruba | 0.01 | 0.11 | — | — | — | — |
| Igbo | 0.0 | 0.15 | — | — | 4.25 | — |
| Zulu | 0.0 | 0.01 | — | — | — | — |
| Xhosa | 0.01 | 0.03 | — | — | — | — |
| Somali | 0.0 | 0.4 | — | — | — | — |
| Wolof | 0.02 | 0.01 | — | — | — | 0.53 |
| Shona | 0.0 | 0.02 | — | — | — | — |

**Europe**

| language | Gemma-4 | BLOOM | mT5 | Lucie-7B | LeoLM-7B | Salamandra-7B | Gervasio-8B | LLaMAntino-3-8B | Fietje-2 | Bielik-11B | Vikhr-Nemo-12B | MamayLM-12B | LLMic-3B | Viking-7B | CSMPT-7B | Meltemi-7B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| French | 2.04 | 0.0 | — | 0.0 | — | — | — | — | — | — | — | — | — | — | — | — |
| German | 0.01 | 0.05 | — | — | 0.01 | — | — | — | — | — | — | — | — | — | — | — |
| Spanish | 0.0 | 0.0 | — | — | — | 0.03 | — | — | — | — | — | — | — | — | — | — |
| Portuguese | 0.0 | 0.04 | — | — | — | — | 0.74 | — | — | — | — | — | — | — | — | — |
| Italian | 0.0 | 0.01 | — | — | — | — | — | 0.58 | — | — | — | — | — | — | — | — |
| Dutch | 0.01 | 0.0 | — | — | — | — | — | — | 0.1 | — | — | — | — | — | — | — |
| Polish | 1.94 | 0.03 | — | — | — | — | — | — | — | 3.58 | — | — | — | — | — | — |
| Russian | 0.01 | 0.24 | — | — | — | — | — | — | — | — | 0.71 | — | — | — | — | — |
| Ukrainian | 0.0 | 2.11 | — | — | — | — | — | — | — | — | — | 0.0 | — | — | — | — |
| Romanian | 2.4 | 1.54 | — | — | — | — | — | — | — | — | — | — | 16.17 | — | — | — |
| Swedish | 0.0 | 0.06 | — | — | — | — | — | — | — | — | — | — | — | 0.94 | — | — |
| Czech | 1.51 | 0.25 | — | — | — | — | — | — | — | — | — | — | — | — | 0.03 | — |
| Greek | 0.0 | 1.45 | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.04 |

**Americas**

| language | Gemma-4 | BLOOM | mT5 | LatamGPT-70B | Tucano-2b4 |
| --- | --- | --- | --- | --- | --- |
| Lat.Am. Spanish | 0.0 | 0.0 | — | 0.65 | — |
| Brazilian Portuguese | 0.0 | 0.04 | — | — | 0.01 |
| Quechua | 0.0 | 0.0 | — | — | — |
| Haitian Creole | 0.0 | 0.05 | — | — | — |

**Oceania**

| language | Gemma-4 | BLOOM | mT5 |
| --- | --- | --- | --- |
| Māori | 5.77 | 0.02 | — |
| Samoan | 0.72 | 0.02 | — |
| Tok Pisin | 0.0 | 0.0 | — |

### UNK rate (%)
*Lower is better.*

**Indic**

| language | Gemma-4 | BLOOM | mT5 | BanglaLLama-3.1-8B | Tamil-Mistral-7B | Telugu-Llama2-7B | MalayaLLM-Gemma-9B | Gujju-Llama-7B | Dhee-Qwen3-Punjabi-2B | Qalb-1.0-8B | NEPALI-LLM-9B | llama3-sinhala-8B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hindi | 0.0 | 0.0 | — | — | — | — | — | — | — | — | — | — |
| Bengali | 0.0 | 0.0 | — | 0.0 | — | — | — | — | — | — | — | — |
| Tamil | 0.0 | 0.0 | — | — | 0.0 | — | — | — | — | — | — | — |
| Telugu | 0.0 | 0.0 | — | — | — | 0.0 | — | — | — | — | — | — |
| Kannada | 0.0 | 0.0 | — | — | — | — | — | — | — | — | — | — |
| Malayalam | 0.0 | 0.0 | — | — | — | — | 0.0 | — | — | — | — | — |
| Marathi | 0.0 | 0.0 | — | — | — | — | — | — | — | — | — | — |
| Gujarati | 0.0 | 0.0 | — | — | — | — | — | 0.0 | — | — | — | — |
| Punjabi | 0.0 | 0.0 | — | — | — | — | — | — | 0.0 | — | — | — |
| Odia | 0.0 | 0.0 | — | — | — | — | — | — | — | — | — | — |
| Assamese | 0.0 | 0.0 | — | — | — | — | — | — | — | — | — | — |
| Urdu | 0.0 | 0.0 | — | — | — | — | — | — | — | 0.0 | — | — |
| Nepali | 0.0 | 0.0 | — | — | — | — | — | — | — | — | 0.0 | — |
| Sinhala | 0.0 | 0.0 | — | — | — | — | — | — | — | — | — | 0.0 |
| Maithili | 0.0 | 0.0 | — | — | — | — | — | — | — | — | — | — |

**Middle East**

| language | Gemma-4 | BLOOM | mT5 | Maral-7B | Trendyol-8B | DictaLM-2.0-7B | Mistral-Nemo-Kurdish | Mistral-7B-Uz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arabic | 0.0 | 0.0 | — | — | — | — | — | — |
| Persian | 0.0 | 0.0 | — | 0.0 | — | — | — | — |
| Turkish | 0.0 | 0.0 | — | — | 0.0 | — | — | — |
| Hebrew | 0.0 | 0.0 | — | — | — | 0.0 | — | — |
| Kurdish | 0.0 | 0.0 | — | — | — | — | 0.0 | — |
| Azerbaijani | 0.0 | 0.0 | — | — | — | — | — | — |
| Uzbek | 0.0 | 0.0 | — | — | — | — | — | 0.0 |
| Kazakh | 0.0 | 0.0 | — | — | — | — | — | — |

**East Asia**

| language | Gemma-4 | BLOOM | mT5 | LLM-jp-3-13B | Polyglot-Ko-12B |
| --- | --- | --- | --- | --- | --- |
| Chinese | 0.0 | 0.0 | — | — | — |
| Japanese | 0.0 | 0.0 | — | 0.0 | — |
| Korean | 0.0 | 0.0 | — | — | 0.0 |

**SEA**

| language | Gemma-4 | BLOOM | mT5 | Arcee-VyLinh-3B | Typhoon2-7B | Nusantara-7B | MaLLaM-5B | Burmese-GPT-1B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vietnamese | 0.0 | 0.0 | — | 0.0 | — | — | — | — |
| Thai | 0.0 | 0.0 | — | — | 0.0 | — | — | — |
| Indonesian | 0.0 | 0.0 | — | — | — | 0.0 | — | — |
| Malay | 0.0 | 0.0 | — | — | — | — | 0.0 | — |
| Tagalog | 0.0 | 0.0 | — | — | — | — | — | — |
| Burmese | 0.0 | 0.0 | — | — | — | — | — | 0.0 |
| Khmer | 0.0 | 0.0 | — | — | — | — | — | — |

**Africa**

| language | Gemma-4 | BLOOM | mT5 | Swahili-Gemma-7B | Kakugo-3B-Igbo | Wolof-Qwen-1.5B |
| --- | --- | --- | --- | --- | --- | --- |
| Swahili | 0.0 | 0.0 | — | 0.0 | — | — |
| Amharic | 0.0 | 0.0 | — | — | — | — |
| Hausa | 0.0 | 0.0 | — | — | — | — |
| Yoruba | 0.0 | 0.0 | — | — | — | — |
| Igbo | 0.0 | 0.0 | — | — | 0.0 | — |
| Zulu | 0.0 | 0.0 | — | — | — | — |
| Xhosa | 0.0 | 0.0 | — | — | — | — |
| Somali | 0.0 | 0.0 | — | — | — | — |
| Wolof | 0.0 | 0.0 | — | — | — | 0.0 |
| Shona | 0.0 | 0.0 | — | — | — | — |

**Europe**

| language | Gemma-4 | BLOOM | mT5 | Lucie-7B | LeoLM-7B | Salamandra-7B | Gervasio-8B | LLaMAntino-3-8B | Fietje-2 | Bielik-11B | Vikhr-Nemo-12B | MamayLM-12B | LLMic-3B | Viking-7B | CSMPT-7B | Meltemi-7B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| French | 0.0 | 0.0 | — | 0.0 | — | — | — | — | — | — | — | — | — | — | — | — |
| German | 0.0 | 0.0 | — | — | 0.0 | — | — | — | — | — | — | — | — | — | — | — |
| Spanish | 0.0 | 0.0 | — | — | — | 0.0 | — | — | — | — | — | — | — | — | — | — |
| Portuguese | 0.0 | 0.0 | — | — | — | — | 0.0 | — | — | — | — | — | — | — | — | — |
| Italian | 0.0 | 0.0 | — | — | — | — | — | 0.0 | — | — | — | — | — | — | — | — |
| Dutch | 0.0 | 0.0 | — | — | — | — | — | — | 0.0 | — | — | — | — | — | — | — |
| Polish | 0.0 | 0.0 | — | — | — | — | — | — | — | 0.0 | — | — | — | — | — | — |
| Russian | 0.0 | 0.0 | — | — | — | — | — | — | — | — | 0.0 | — | — | — | — | — |
| Ukrainian | 0.0 | 0.0 | — | — | — | — | — | — | — | — | — | 0.0 | — | — | — | — |
| Romanian | 0.0 | 0.0 | — | — | — | — | — | — | — | — | — | — | 0.0 | — | — | — |
| Swedish | 0.0 | 0.0 | — | — | — | — | — | — | — | — | — | — | — | 0.0 | — | — |
| Czech | 0.0 | 0.0 | — | — | — | — | — | — | — | — | — | — | — | — | 0.0 | — |
| Greek | 0.0 | 0.0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.0 |

**Americas**

| language | Gemma-4 | BLOOM | mT5 | LatamGPT-70B | Tucano-2b4 |
| --- | --- | --- | --- | --- | --- |
| Lat.Am. Spanish | 0.0 | 0.0 | — | 0.0 | — |
| Brazilian Portuguese | 0.0 | 0.0 | — | — | 0.0 |
| Quechua | 0.0 | 0.0 | — | — | — |
| Haitian Creole | 0.0 | 0.0 | — | — | — |

**Oceania**

| language | Gemma-4 | BLOOM | mT5 |
| --- | --- | --- | --- |
| Māori | 0.0 | 0.0 | — |
| Samoan | 0.0 | 0.0 | — |
| Tok Pisin | 0.0 | 0.0 | — |

### Vocabulary coverage (%)
*Higher is better.*

**Indic**

| language | Gemma-4 | BLOOM | mT5 | BanglaLLama-3.1-8B | Tamil-Mistral-7B | Telugu-Llama2-7B | MalayaLLM-Gemma-9B | Gujju-Llama-7B | Dhee-Qwen3-Punjabi-2B | Qalb-1.0-8B | NEPALI-LLM-9B | llama3-sinhala-8B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hindi | 100.0 | 98.65 | — | — | — | — | — | — | — | — | — | — |
| Bengali | 100.0 | 99.38 | — | 58.02 | — | — | — | — | — | — | — | — |
| Tamil | 100.0 | 100.0 | — | — | 86.61 | — | — | — | — | — | — | — |
| Telugu | 99.35 | 99.35 | — | — | — | 50.0 | — | — | — | — | — | — |
| Kannada | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | — |
| Malayalam | 100.0 | 98.57 | — | — | — | — | 100.0 | — | — | — | — | — |
| Marathi | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | — |
| Gujarati | 99.29 | 95.74 | — | — | — | — | — | 82.98 | — | — | — | — |
| Punjabi | 99.33 | 99.33 | — | — | — | — | — | — | 67.33 | — | — | — |
| Odia | 96.32 | 96.32 | — | — | — | — | — | — | — | — | — | — |
| Assamese | 99.38 | 98.76 | — | — | — | — | — | — | — | — | — | — |
| Urdu | 100.0 | 98.52 | — | — | — | — | — | — | — | 91.11 | — | — |
| Nepali | 99.35 | 100.0 | — | — | — | — | — | — | — | — | 100.0 | — |
| Sinhala | 98.64 | 52.38 | — | — | — | — | — | — | — | — | — | 52.38 |
| Maithili | 100.0 | 99.34 | — | — | — | — | — | — | — | — | — | — |

**Middle East**

| language | Gemma-4 | BLOOM | mT5 | Maral-7B | Trendyol-8B | DictaLM-2.0-7B | Mistral-Nemo-Kurdish | Mistral-7B-Uz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arabic | 100.0 | 100.0 | — | — | — | — | — | — |
| Persian | 100.0 | 99.22 | — | 53.49 | — | — | — | — |
| Turkish | 100.0 | 98.15 | — | — | 100.0 | — | — | — |
| Hebrew | 100.0 | 96.3 | — | — | — | 83.33 | — | — |
| Kurdish | 100.0 | 99.05 | — | — | — | — | 96.19 | — |
| Azerbaijani | 100.0 | 97.09 | — | — | — | — | — | — |
| Uzbek | 97.85 | 98.92 | — | — | — | — | — | 100.0 |
| Kazakh | 100.0 | 86.96 | — | — | — | — | — | — |

**East Asia**

| language | Gemma-4 | BLOOM | mT5 | LLM-jp-3-13B | Polyglot-Ko-12B |
| --- | --- | --- | --- | --- | --- |
| Chinese | 99.86 | 98.06 | — | — | — |
| Japanese | 100.0 | 87.24 | — | 71.3 | — |
| Korean | 99.35 | 19.36 | — | — | 93.26 |

**SEA**

| language | Gemma-4 | BLOOM | mT5 | Arcee-VyLinh-3B | Typhoon2-7B | Nusantara-7B | MaLLaM-5B | Burmese-GPT-1B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vietnamese | 100.0 | 100.0 | — | 100.0 | — | — | — | — |
| Thai | 100.0 | 81.48 | — | — | 98.77 | — | — | — |
| Indonesian | 100.0 | 99.03 | — | — | — | 100.0 | — | — |
| Malay | 100.0 | 100.0 | — | — | — | — | 84.76 | — |
| Tagalog | 99.07 | 98.15 | — | — | — | — | — | — |
| Burmese | 100.0 | 53.59 | — | — | — | — | — | 91.5 |
| Khmer | 98.38 | 63.78 | — | — | — | — | — | — |

**Africa**

| language | Gemma-4 | BLOOM | mT5 | Swahili-Gemma-7B | Kakugo-3B-Igbo | Wolof-Qwen-1.5B |
| --- | --- | --- | --- | --- | --- | --- |
| Swahili | 100.0 | 100.0 | — | 100.0 | — | — |
| Amharic | 99.08 | 28.0 | — | — | — | — |
| Hausa | 99.07 | 93.46 | — | — | — | — |
| Yoruba | 100.0 | 99.22 | — | — | — | — |
| Igbo | 100.0 | 100.0 | — | — | 95.0 | — |
| Zulu | 100.0 | 99.09 | — | — | — | — |
| Xhosa | 99.03 | 99.03 | — | — | — | — |
| Somali | 100.0 | 100.0 | — | — | — | — |
| Wolof | 99.19 | 100.0 | — | — | — | 100.0 |
| Shona | 100.0 | 100.0 | — | — | — | — |

**Europe**

| language | Gemma-4 | BLOOM | mT5 | Lucie-7B | LeoLM-7B | Salamandra-7B | Gervasio-8B | LLaMAntino-3-8B | Fietje-2 | Bielik-11B | Vikhr-Nemo-12B | MamayLM-12B | LLMic-3B | Viking-7B | CSMPT-7B | Meltemi-7B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| French | 98.31 | 100.0 | — | 82.2 | — | — | — | — | — | — | — | — | — | — | — | — |
| German | 100.0 | 100.0 | — | — | 81.48 | — | — | — | — | — | — | — | — | — | — | — |
| Spanish | 100.0 | 100.0 | — | — | — | 85.58 | — | — | — | — | — | — | — | — | — | — |
| Portuguese | 100.0 | 100.0 | — | — | — | — | 100.0 | — | — | — | — | — | — | — | — | — |
| Italian | 100.0 | 100.0 | — | — | — | — | — | 100.0 | — | — | — | — | — | — | — | — |
| Dutch | 100.0 | 98.11 | — | — | — | — | — | — | 94.34 | — | — | — | — | — | — | — |
| Polish | 100.0 | 97.17 | — | — | — | — | — | — | — | 81.13 | — | — | — | — | — | — |
| Russian | 99.33 | 92.67 | — | — | — | — | — | — | — | — | 98.67 | — | — | — | — | — |
| Ukrainian | 100.0 | 88.44 | — | — | — | — | — | — | — | — | — | 100.0 | — | — | — | — |
| Romanian | 100.0 | 96.43 | — | — | — | — | — | — | — | — | — | — | 77.68 | — | — | — |
| Swedish | 100.0 | 99.05 | — | — | — | — | — | — | — | — | — | — | — | 100.0 | — | — |
| Czech | 99.13 | 95.65 | — | — | — | — | — | — | — | — | — | — | — | — | 96.52 | — |
| Greek | 100.0 | 86.27 | — | — | — | — | — | — | — | — | — | — | — | — | — | 95.42 |

**Americas**

| language | Gemma-4 | BLOOM | mT5 | LatamGPT-70B | Tucano-2b4 |
| --- | --- | --- | --- | --- | --- |
| Lat.Am. Spanish | 100.0 | 100.0 | — | 100.0 | — |
| Brazilian Portuguese | 100.0 | 100.0 | — | — | 86.84 |
| Quechua | 100.0 | 100.0 | — | — | — |
| Haitian Creole | 100.0 | 99.13 | — | — | — |

**Oceania**

| language | Gemma-4 | BLOOM | mT5 |
| --- | --- | --- | --- |
| Māori | 100.0 | 94.62 | — |
| Samoan | 100.0 | 97.37 | — |
| Tok Pisin | 100.0 | 100.0 | — |

### Roundtrip fidelity (%)
*Higher is better.*

**Indic**

| language | Gemma-4 | BLOOM | mT5 | BanglaLLama-3.1-8B | Tamil-Mistral-7B | Telugu-Llama2-7B | MalayaLLM-Gemma-9B | Gujju-Llama-7B | Dhee-Qwen3-Punjabi-2B | Qalb-1.0-8B | NEPALI-LLM-9B | llama3-sinhala-8B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hindi | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | — |
| Bengali | 100.0 | 100.0 | — | 100.0 | — | — | — | — | — | — | — | — |
| Tamil | 100.0 | 100.0 | — | — | 100.0 | — | — | — | — | — | — | — |
| Telugu | 100.0 | 100.0 | — | — | — | 100.0 | — | — | — | — | — | — |
| Kannada | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | — |
| Malayalam | 100.0 | 100.0 | — | — | — | — | 100.0 | — | — | — | — | — |
| Marathi | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | — |
| Gujarati | 100.0 | 100.0 | — | — | — | — | — | 100.0 | — | — | — | — |
| Punjabi | 100.0 | 100.0 | — | — | — | — | — | — | 43.38 | — | — | — |
| Odia | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | — |
| Assamese | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | — |
| Urdu | 100.0 | 100.0 | — | — | — | — | — | — | — | 100.0 | — | — |
| Nepali | 100.0 | 100.0 | — | — | — | — | — | — | — | — | 100.0 | — |
| Sinhala | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | 100.0 |
| Maithili | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | — |

**Middle East**

| language | Gemma-4 | BLOOM | mT5 | Maral-7B | Trendyol-8B | DictaLM-2.0-7B | Mistral-Nemo-Kurdish | Mistral-7B-Uz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arabic | 100.0 | 100.0 | — | — | — | — | — | — |
| Persian | 100.0 | 100.0 | — | 100.0 | — | — | — | — |
| Turkish | 100.0 | 100.0 | — | — | 100.0 | — | — | — |
| Hebrew | 100.0 | 100.0 | — | — | — | 100.0 | — | — |
| Kurdish | 100.0 | 100.0 | — | — | — | — | 100.0 | — |
| Azerbaijani | 100.0 | 100.0 | — | — | — | — | — | — |
| Uzbek | 100.0 | 100.0 | — | — | — | — | — | 100.0 |
| Kazakh | 100.0 | 100.0 | — | — | — | — | — | — |

**East Asia**

| language | Gemma-4 | BLOOM | mT5 | LLM-jp-3-13B | Polyglot-Ko-12B |
| --- | --- | --- | --- | --- | --- |
| Chinese | 100.0 | 100.0 | — | — | — |
| Japanese | 100.0 | 100.0 | — | 100.0 | — |
| Korean | 100.0 | 100.0 | — | — | 100.0 |

**SEA**

| language | Gemma-4 | BLOOM | mT5 | Arcee-VyLinh-3B | Typhoon2-7B | Nusantara-7B | MaLLaM-5B | Burmese-GPT-1B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vietnamese | 100.0 | 100.0 | — | 100.0 | — | — | — | — |
| Thai | 100.0 | 100.0 | — | — | 100.0 | — | — | — |
| Indonesian | 100.0 | 100.0 | — | — | — | 100.0 | — | — |
| Malay | 100.0 | 100.0 | — | — | — | — | 100.0 | — |
| Tagalog | 100.0 | 100.0 | — | — | — | — | — | — |
| Burmese | 100.0 | 100.0 | — | — | — | — | — | 100.0 |
| Khmer | 100.0 | 100.0 | — | — | — | — | — | — |

**Africa**

| language | Gemma-4 | BLOOM | mT5 | Swahili-Gemma-7B | Kakugo-3B-Igbo | Wolof-Qwen-1.5B |
| --- | --- | --- | --- | --- | --- | --- |
| Swahili | 100.0 | 100.0 | — | 100.0 | — | — |
| Amharic | 100.0 | 100.0 | — | — | — | — |
| Hausa | 100.0 | 100.0 | — | — | — | — |
| Yoruba | 100.0 | 100.0 | — | — | — | — |
| Igbo | 100.0 | 100.0 | — | — | 100.0 | — |
| Zulu | 100.0 | 100.0 | — | — | — | — |
| Xhosa | 100.0 | 100.0 | — | — | — | — |
| Somali | 100.0 | 100.0 | — | — | — | — |
| Wolof | 100.0 | 100.0 | — | — | — | 100.0 |
| Shona | 100.0 | 100.0 | — | — | — | — |

**Europe**

| language | Gemma-4 | BLOOM | mT5 | Lucie-7B | LeoLM-7B | Salamandra-7B | Gervasio-8B | LLaMAntino-3-8B | Fietje-2 | Bielik-11B | Vikhr-Nemo-12B | MamayLM-12B | LLMic-3B | Viking-7B | CSMPT-7B | Meltemi-7B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| French | 100.0 | 100.0 | — | 100.0 | — | — | — | — | — | — | — | — | — | — | — | — |
| German | 100.0 | 100.0 | — | — | 100.0 | — | — | — | — | — | — | — | — | — | — | — |
| Spanish | 100.0 | 100.0 | — | — | — | 100.0 | — | — | — | — | — | — | — | — | — | — |
| Portuguese | 100.0 | 100.0 | — | — | — | — | 100.0 | — | — | — | — | — | — | — | — | — |
| Italian | 100.0 | 100.0 | — | — | — | — | — | 100.0 | — | — | — | — | — | — | — | — |
| Dutch | 100.0 | 100.0 | — | — | — | — | — | — | 100.0 | — | — | — | — | — | — | — |
| Polish | 100.0 | 100.0 | — | — | — | — | — | — | — | 100.0 | — | — | — | — | — | — |
| Russian | 100.0 | 100.0 | — | — | — | — | — | — | — | — | 100.0 | — | — | — | — | — |
| Ukrainian | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | 100.0 | — | — | — | — |
| Romanian | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | — | 4.15 | — | — | — |
| Swedish | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | — | — | 100.0 | — | — |
| Czech | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | — | — | — | 100.0 | — |
| Greek | 100.0 | 100.0 | — | — | — | — | — | — | — | — | — | — | — | — | — | 100.0 |

**Americas**

| language | Gemma-4 | BLOOM | mT5 | LatamGPT-70B | Tucano-2b4 |
| --- | --- | --- | --- | --- | --- |
| Lat.Am. Spanish | 100.0 | 100.0 | — | 100.0 | — |
| Brazilian Portuguese | 100.0 | 100.0 | — | — | 99.31 |
| Quechua | 100.0 | 100.0 | — | — | — |
| Haitian Creole | 100.0 | 100.0 | — | — | — |

**Oceania**

| language | Gemma-4 | BLOOM | mT5 |
| --- | --- | --- | --- |
| Māori | 100.0 | 100.0 | — |
| Samoan | 100.0 | 100.0 | — |
| Tok Pisin | 100.0 | 100.0 | — |

### Avg tokens / segment
*Lower is better.*

**Indic**

| language | Gemma-4 | BLOOM | mT5 | BanglaLLama-3.1-8B | Tamil-Mistral-7B | Telugu-Llama2-7B | MalayaLLM-Gemma-9B | Gujju-Llama-7B | Dhee-Qwen3-Punjabi-2B | Qalb-1.0-8B | NEPALI-LLM-9B | llama3-sinhala-8B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hindi | 35.1 | 34.72 | — | — | — | — | — | — | — | — | — | — |
| Bengali | 32.45 | 31.85 | — | 154.6 | — | — | — | — | — | — | — | — |
| Tamil | 39.36 | 34.79 | — | — | 28.75 | — | — | — | — | — | — | — |
| Telugu | 47.58 | 35.99 | — | — | — | 327.89 | — | — | — | — | — | — |
| Kannada | 51.56 | 35.27 | — | — | — | — | — | — | — | — | — | — |
| Malayalam | 49.52 | 37.48 | — | — | — | — | 86.67 | — | — | — | — | — |
| Marathi | 37.39 | 33.02 | — | — | — | — | — | — | — | — | — | — |
| Gujarati | 49.84 | 36.66 | — | — | — | — | — | 41.96 | — | — | — | — |
| Punjabi | 72.18 | 39.1 | — | — | — | — | — | — | 198.6 | — | — | — |
| Odia | 94.07 | 37.07 | — | — | — | — | — | — | — | — | — | — |
| Assamese | 53.73 | 38.11 | — | — | — | — | — | — | — | — | — | — |
| Urdu | 40.67 | 36.72 | — | — | — | — | — | — | — | 82.52 | — | — |
| Nepali | 40.96 | 31.84 | — | — | — | — | — | — | — | — | 45.83 | — |
| Sinhala | 61.33 | 218.73 | — | — | — | — | — | — | — | — | — | 231.72 |
| Maithili | 43.92 | 42.18 | — | — | — | — | — | — | — | — | — | — |

**Middle East**

| language | Gemma-4 | BLOOM | mT5 | Maral-7B | Trendyol-8B | DictaLM-2.0-7B | Mistral-Nemo-Kurdish | Mistral-7B-Uz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arabic | 39.4 | 31.11 | — | — | — | — | — | — |
| Persian | 39.04 | 47.79 | — | 119.46 | — | — | — | — |
| Turkish | 36.7 | 53.08 | — | — | 44.31 | — | — | — |
| Hebrew | 46.44 | 78.4 | — | — | — | 45.35 | — | — |
| Kurdish | 52.84 | 54.81 | — | — | — | — | 55.7 | — |
| Azerbaijani | 54.24 | 62.23 | — | — | — | — | — | — |
| Uzbek | 51.09 | 53.51 | — | — | — | — | — | 63.25 |
| Kazakh | 54.58 | 87.25 | — | — | — | — | — | — |

**East Asia**

| language | Gemma-4 | BLOOM | mT5 | LLM-jp-3-13B | Polyglot-Ko-12B |
| --- | --- | --- | --- | --- | --- |
| Chinese | 28.89 | 24.77 | — | — | — |
| Japanese | 32.51 | 47.88 | — | 27.37 | — |
| Korean | 36.83 | 74.52 | — | — | 33.53 |

**SEA**

| language | Gemma-4 | BLOOM | mT5 | Arcee-VyLinh-3B | Typhoon2-7B | Nusantara-7B | MaLLaM-5B | Burmese-GPT-1B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vietnamese | 36.75 | 34.26 | — | 39.17 | — | — | — | — |
| Thai | 42.88 | 123.81 | — | — | 69.37 | — | — | — |
| Indonesian | 30.83 | 26.09 | — | — | — | 42.02 | — | — |
| Malay | 32.56 | 28.81 | — | — | — | — | 28.28 | — |
| Tagalog | 48.66 | 50.13 | — | — | — | — | — | — |
| Burmese | 64.52 | 270.74 | — | — | — | — | — | 112.72 |
| Khmer | 73.25 | 169.46 | — | — | — | — | — | — |

**Africa**

| language | Gemma-4 | BLOOM | mT5 | Swahili-Gemma-7B | Kakugo-3B-Igbo | Wolof-Qwen-1.5B |
| --- | --- | --- | --- | --- | --- | --- |
| Swahili | 43.91 | 33.71 | — | 43.06 | — | — |
| Amharic | 51.93 | 135.37 | — | — | — | — |
| Hausa | 46.33 | 47.84 | — | — | — | — |
| Yoruba | 64.46 | 43.8 | — | — | — | — |
| Igbo | 58.05 | 46.58 | — | — | 66.19 | — |
| Zulu | 52.92 | 47.59 | — | — | — | — |
| Xhosa | 50.6 | 44.93 | — | — | — | — |
| Somali | 51.75 | 55.36 | — | — | — | — |
| Wolof | 47.54 | 45.68 | — | — | — | 52.06 |
| Shona | 50.18 | 48.83 | — | — | — | — |

**Europe**

| language | Gemma-4 | BLOOM | mT5 | Lucie-7B | LeoLM-7B | Salamandra-7B | Gervasio-8B | LLaMAntino-3-8B | Fietje-2 | Bielik-11B | Vikhr-Nemo-12B | MamayLM-12B | LLMic-3B | Viking-7B | CSMPT-7B | Meltemi-7B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| French | 37.73 | 32.63 | — | 36.13 | — | — | — | — | — | — | — | — | — | — | — | — |
| German | 35.59 | 45.59 | — | — | 46.88 | — | — | — | — | — | — | — | — | — | — | — |
| Spanish | 34.71 | 32.69 | — | — | — | 34.88 | — | — | — | — | — | — | — | — | — | — |
| Portuguese | 33.62 | 30.36 | — | — | — | — | 39.53 | — | — | — | — | — | — | — | — | — |
| Italian | 36.87 | 43.91 | — | — | — | — | — | 43.98 | — | — | — | — | — | — | — | — |
| Dutch | 37.05 | 46.37 | — | — | — | — | — | — | 52.83 | — | — | — | — | — | — | — |
| Polish | 40.37 | 57.58 | — | — | — | — | — | — | — | 56.32 | — | — | — | — | — | — |
| Russian | 36.58 | 66.52 | — | — | — | — | — | — | — | — | 41.36 | — | — | — | — | — |
| Ukrainian | 43.09 | 73.9 | — | — | — | — | — | — | — | — | — | 43.09 | — | — | — | — |
| Romanian | 42.2 | 51.69 | — | — | — | — | — | — | — | — | — | — | 51.41 | — | — | — |
| Swedish | 36.86 | 44.5 | — | — | — | — | — | — | — | — | — | — | — | 29.53 | — | — |
| Czech | 40.87 | 54.67 | — | — | — | — | — | — | — | — | — | — | — | — | 26.66 | — |
| Greek | 58.39 | 102.52 | — | — | — | — | — | — | — | — | — | — | — | — | — | 32.98 |

**Americas**

| language | Gemma-4 | BLOOM | mT5 | LatamGPT-70B | Tucano-2b4 |
| --- | --- | --- | --- | --- | --- |
| Lat.Am. Spanish | 34.71 | 32.69 | — | 41.35 | — |
| Brazilian Portuguese | 33.62 | 30.36 | — | — | 28.96 |
| Quechua | 49.58 | 49.74 | — | — | — |
| Haitian Creole | 41.08 | 42.33 | — | — | — |

**Oceania**

| language | Gemma-4 | BLOOM | mT5 |
| --- | --- | --- | --- |
| Māori | 54.7 | 57.26 | — |
| Samoan | 57.27 | 57.33 | — |
| Tok Pisin | 51.44 | 52.18 | — |

---

## 5. Complete per-language results

Every tokenizer × language combination from `data/results.csv`.

| tokenizer | language | region | fertility | compression | byte_fallback% | unk% | vcov% | roundtrip% | avg_tok/sent | total_tokens | total_chars | sentences |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-4 | Hindi | Indic | 1.385 | 3.69 | 0.0 | 0.0 | 100.0 | 100.0 | 35.1 | 35521 | 131058 | 1012 |
| BLOOM | Hindi | Indic | 1.37 | 3.73 | 0.03 | 0.0 | 98.65 | 100.0 | 34.72 | 35135 | 131058 | 1012 |
| Gemma-4 | Bengali | Indic | 1.683 | 3.93 | 0.0 | 0.0 | 100.0 | 100.0 | 32.45 | 32836 | 129042 | 1012 |
| BLOOM | Bengali | Indic | 1.652 | 4.004 | 0.27 | 0.0 | 99.38 | 100.0 | 31.85 | 32228 | 129042 | 1012 |
| BanglaLLama-3.1-8B | Bengali | Indic | 8.021 | 0.825 | 16.93 | 0.0 | 58.02 | 100.0 | 154.6 | 156453 | 129042 | 1012 |
| Gemma-4 | Tamil | Indic | 2.374 | 3.87 | 0.0 | 0.0 | 100.0 | 100.0 | 39.36 | 39830 | 154133 | 1012 |
| BLOOM | Tamil | Indic | 2.099 | 4.378 | 0.24 | 0.0 | 100.0 | 100.0 | 34.79 | 35208 | 154133 | 1012 |
| Tamil-Mistral-7B | Tamil | Indic | 1.734 | 5.297 | 0.0 | 0.0 | 86.61 | 100.0 | 28.75 | 29096 | 154133 | 1012 |
| Gemma-4 | Telugu | Indic | 2.843 | 2.751 | 1.0 | 0.0 | 99.35 | 100.0 | 47.58 | 48148 | 132470 | 1012 |
| BLOOM | Telugu | Indic | 2.15 | 3.637 | 0.62 | 0.0 | 99.35 | 100.0 | 35.99 | 36421 | 132470 | 1012 |
| Telugu-Llama2-7B | Telugu | Indic | 19.59 | 0.399 | 90.6 | 0.0 | 50.0 | 100.0 | 327.89 | 331821 | 132470 | 1012 |
| Gemma-4 | Kannada | Indic | 3.241 | 2.645 | 0.0 | 0.0 | 100.0 | 100.0 | 51.56 | 52180 | 138040 | 1012 |
| BLOOM | Kannada | Indic | 2.217 | 3.868 | 0.86 | 0.0 | 100.0 | 100.0 | 35.27 | 35689 | 138040 | 1012 |
| Gemma-4 | Malayalam | Indic | 3.357 | 2.98 | 0.0 | 0.0 | 100.0 | 100.0 | 49.52 | 50113 | 149336 | 1012 |
| BLOOM | Malayalam | Indic | 2.54 | 3.937 | 0.17 | 0.0 | 98.57 | 100.0 | 37.48 | 37928 | 149336 | 1012 |
| MalayaLLM-Gemma-9B | Malayalam | Indic | 5.875 | 1.703 | 0.0 | 0.0 | 100.0 | 100.0 | 86.67 | 87715 | 149336 | 1012 |
| Gemma-4 | Marathi | Indic | 1.987 | 3.517 | 0.0 | 0.0 | 100.0 | 100.0 | 37.39 | 37835 | 133051 | 1012 |
| BLOOM | Marathi | Indic | 1.754 | 3.982 | 0.23 | 0.0 | 100.0 | 100.0 | 33.02 | 33415 | 133051 | 1012 |
| Gemma-4 | Gujarati | Indic | 2.415 | 2.506 | 0.01 | 0.0 | 99.29 | 100.0 | 49.84 | 50443 | 126428 | 1012 |
| BLOOM | Gujarati | Indic | 1.776 | 3.407 | 0.34 | 0.0 | 95.74 | 100.0 | 36.66 | 37103 | 126428 | 1012 |
| Gujju-Llama-7B | Gujarati | Indic | 2.033 | 2.977 | 0.0 | 0.0 | 82.98 | 100.0 | 41.96 | 42463 | 126428 | 1012 |
| Gemma-4 | Punjabi | Indic | 2.82 | 1.827 | 0.01 | 0.0 | 99.33 | 100.0 | 72.18 | 73045 | 133425 | 1012 |
| BLOOM | Punjabi | Indic | 1.528 | 3.372 | 0.07 | 0.0 | 99.33 | 100.0 | 39.1 | 39572 | 133425 | 1012 |
| Dhee-Qwen3-Punjabi-2B | Punjabi | Indic | 7.758 | 0.664 | 19.43 | 0.0 | 67.33 | 43.38 | 198.6 | 200987 | 133425 | 1012 |
| Gemma-4 | Odia | Indic | 4.852 | 1.416 | 1.06 | 0.0 | 96.32 | 100.0 | 94.07 | 95199 | 134839 | 1012 |
| BLOOM | Odia | Indic | 1.912 | 3.595 | 0.22 | 0.0 | 96.32 | 100.0 | 37.07 | 37512 | 134839 | 1012 |
| Gemma-4 | Assamese | Indic | 2.819 | 2.316 | 0.01 | 0.0 | 99.38 | 100.0 | 53.73 | 54371 | 125936 | 1012 |
| BLOOM | Assamese | Indic | 1.999 | 3.266 | 0.26 | 0.0 | 98.76 | 100.0 | 38.11 | 38564 | 125936 | 1012 |
| Gemma-4 | Urdu | Indic | 1.489 | 3.156 | 0.0 | 0.0 | 100.0 | 100.0 | 40.67 | 41163 | 129919 | 1012 |
| BLOOM | Urdu | Indic | 1.344 | 3.496 | 0.04 | 0.0 | 98.52 | 100.0 | 36.72 | 37160 | 129919 | 1012 |
| Qalb-1.0-8B | Urdu | Indic | 3.02 | 1.556 | 17.51 | 0.0 | 91.11 | 100.0 | 82.52 | 83511 | 129919 | 1012 |
| Gemma-4 | Nepali | Indic | 2.216 | 3.062 | 0.0 | 0.0 | 99.35 | 100.0 | 40.96 | 41449 | 126903 | 1012 |
| BLOOM | Nepali | Indic | 1.723 | 3.938 | 0.04 | 0.0 | 100.0 | 100.0 | 31.84 | 32222 | 126903 | 1012 |
| NEPALI-LLM-9B | Nepali | Indic | 2.48 | 2.736 | 0.0 | 0.0 | 100.0 | 100.0 | 45.83 | 46375 | 126903 | 1012 |
| Gemma-4 | Sinhala | Indic | 2.997 | 2.112 | 0.02 | 0.0 | 98.64 | 100.0 | 61.33 | 62066 | 131080 | 1012 |
| BLOOM | Sinhala | Indic | 10.689 | 0.592 | 30.3 | 0.0 | 52.38 | 100.0 | 218.73 | 221351 | 131080 | 1012 |
| llama3-sinhala-8B | Sinhala | Indic | 11.324 | 0.559 | 28.31 | 0.0 | 52.38 | 100.0 | 231.72 | 234504 | 131080 | 1012 |
| Gemma-4 | Maithili | Indic | 1.806 | 2.912 | 0.0 | 0.0 | 100.0 | 100.0 | 43.92 | 44446 | 129408 | 1012 |
| BLOOM | Maithili | Indic | 1.734 | 3.032 | 0.01 | 0.0 | 99.34 | 100.0 | 42.18 | 42683 | 129408 | 1012 |
| Gemma-4 | Arabic | Middle East | 2.031 | 2.917 | 0.0 | 0.0 | 100.0 | 100.0 | 39.4 | 39874 | 116307 | 1012 |
| BLOOM | Arabic | Middle East | 1.603 | 3.695 | 0.0 | 0.0 | 100.0 | 100.0 | 31.11 | 31481 | 116307 | 1012 |
| Gemma-4 | Persian | Middle East | 1.679 | 3.135 | 0.0 | 0.0 | 100.0 | 100.0 | 39.04 | 39504 | 123827 | 1012 |
| BLOOM | Persian | Middle East | 2.055 | 2.56 | 0.13 | 0.0 | 99.22 | 100.0 | 47.79 | 48361 | 123827 | 1012 |
| Maral-7B | Persian | Middle East | 5.138 | 1.024 | 0.04 | 0.0 | 53.49 | 100.0 | 119.46 | 120894 | 123827 | 1012 |
| Gemma-4 | Turkish | Middle East | 2.109 | 3.655 | 0.96 | 0.0 | 100.0 | 100.0 | 36.7 | 37142 | 135761 | 1012 |
| BLOOM | Turkish | Middle East | 3.051 | 2.527 | 0.12 | 0.0 | 98.15 | 100.0 | 53.08 | 53719 | 135761 | 1012 |
| Trendyol-8B | Turkish | Middle East | 2.547 | 3.027 | 0.6 | 0.0 | 100.0 | 100.0 | 44.31 | 44846 | 135761 | 1012 |
| Gemma-4 | Hebrew | Middle East | 2.706 | 2.172 | 0.0 | 0.0 | 100.0 | 100.0 | 46.44 | 46995 | 102054 | 1012 |
| BLOOM | Hebrew | Middle East | 4.568 | 1.286 | 7.34 | 0.0 | 96.3 | 100.0 | 78.4 | 79336 | 102054 | 1012 |
| DictaLM-2.0-7B | Hebrew | Middle East | 2.642 | 2.224 | 0.0 | 0.0 | 83.33 | 100.0 | 45.35 | 45891 | 102054 | 1012 |
| Gemma-4 | Kurdish | Middle East | 2.358 | 2.438 | 0.0 | 0.0 | 100.0 | 100.0 | 52.84 | 53472 | 130372 | 1012 |
| BLOOM | Kurdish | Middle East | 2.446 | 2.35 | 0.09 | 0.0 | 99.05 | 100.0 | 54.81 | 55472 | 130372 | 1012 |
| Mistral-Nemo-Kurdish | Kurdish | Middle East | 2.486 | 2.313 | 0.51 | 0.0 | 96.19 | 100.0 | 55.7 | 56366 | 130372 | 1012 |
| Gemma-4 | Azerbaijani | Middle East | 2.873 | 2.619 | 0.91 | 0.0 | 100.0 | 100.0 | 54.24 | 54891 | 143739 | 1012 |
| BLOOM | Azerbaijani | Middle East | 3.296 | 2.283 | 0.47 | 0.0 | 97.09 | 100.0 | 62.23 | 62972 | 143739 | 1012 |
| Gemma-4 | Uzbek | Middle East | 2.839 | 2.888 | 0.01 | 0.0 | 97.85 | 100.0 | 51.09 | 51704 | 149323 | 1012 |
| BLOOM | Uzbek | Middle East | 2.974 | 2.757 | 0.0 | 0.0 | 98.92 | 100.0 | 53.51 | 54156 | 149323 | 1012 |
| Mistral-7B-Uz | Uzbek | Middle East | 3.515 | 2.333 | 0.0 | 0.0 | 100.0 | 100.0 | 63.25 | 64005 | 149323 | 1012 |
| Gemma-4 | Kazakh | Middle East | 3.184 | 2.452 | 0.0 | 0.0 | 100.0 | 100.0 | 54.58 | 55231 | 135404 | 1012 |
| BLOOM | Kazakh | Middle East | 5.089 | 1.534 | 2.79 | 0.0 | 86.96 | 100.0 | 87.25 | 88293 | 135404 | 1012 |
| Gemma-4 | Chinese | East Asia | 14.132 | 1.479 | 0.05 | 0.0 | 99.86 | 100.0 | 28.89 | 29240 | 43248 | 1012 |
| BLOOM | Chinese | East Asia | 12.117 | 1.725 | 0.79 | 0.0 | 98.06 | 100.0 | 24.77 | 25070 | 43248 | 1012 |
| Gemma-4 | Japanese | East Asia | 30.432 | 1.731 | 0.0 | 0.0 | 100.0 | 100.0 | 32.51 | 32897 | 56943 | 1012 |
| BLOOM | Japanese | East Asia | 44.828 | 1.175 | 2.32 | 0.0 | 87.24 | 100.0 | 47.88 | 48459 | 56943 | 1012 |
| LLM-jp-3-13B | Japanese | East Asia | 25.627 | 2.055 | 0.0 | 0.0 | 71.3 | 100.0 | 27.37 | 27703 | 56943 | 1012 |
| Gemma-4 | Korean | East Asia | 2.415 | 1.77 | 0.07 | 0.0 | 99.35 | 100.0 | 36.83 | 37268 | 65965 | 1012 |
| BLOOM | Korean | East Asia | 4.887 | 0.875 | 22.1 | 0.0 | 19.36 | 100.0 | 74.52 | 75412 | 65965 | 1012 |
| Polyglot-Ko-12B | Korean | East Asia | 2.199 | 1.944 | 0.13 | 0.0 | 93.26 | 100.0 | 33.53 | 33933 | 65965 | 1012 |
| Gemma-4 | Vietnamese | SEA | 1.21 | 3.727 | 0.18 | 0.0 | 100.0 | 100.0 | 36.75 | 37189 | 138595 | 1012 |
| BLOOM | Vietnamese | SEA | 1.128 | 3.998 | 0.02 | 0.0 | 100.0 | 100.0 | 34.26 | 34670 | 138595 | 1012 |
| Arcee-VyLinh-3B | Vietnamese | SEA | 1.29 | 3.496 | 0.89 | 0.0 | 100.0 | 100.0 | 39.17 | 39643 | 138595 | 1012 |
| Gemma-4 | Thai | SEA | 10.084 | 2.906 | 0.0 | 0.0 | 100.0 | 100.0 | 42.88 | 43393 | 126109 | 1012 |
| BLOOM | Thai | SEA | 29.119 | 1.006 | 11.86 | 0.0 | 81.48 | 100.0 | 123.81 | 125297 | 126109 | 1012 |
| Typhoon2-7B | Thai | SEA | 16.315 | 1.796 | 2.35 | 0.0 | 98.77 | 100.0 | 69.37 | 70202 | 126109 | 1012 |
| Gemma-4 | Indonesian | SEA | 1.575 | 4.562 | 0.0 | 0.0 | 100.0 | 100.0 | 30.83 | 31196 | 142318 | 1012 |
| BLOOM | Indonesian | SEA | 1.333 | 5.39 | 0.0 | 0.0 | 99.03 | 100.0 | 26.09 | 26404 | 142318 | 1012 |
| Nusantara-7B | Indonesian | SEA | 2.146 | 3.347 | 0.62 | 0.0 | 100.0 | 100.0 | 42.02 | 42520 | 142318 | 1012 |
| Gemma-4 | Malay | SEA | 1.632 | 4.437 | 0.0 | 0.0 | 100.0 | 100.0 | 32.56 | 32954 | 146201 | 1012 |
| BLOOM | Malay | SEA | 1.444 | 5.014 | 0.0 | 0.0 | 100.0 | 100.0 | 28.81 | 29160 | 146201 | 1012 |
| MaLLaM-5B | Malay | SEA | 1.417 | 5.109 | 0.04 | 0.0 | 84.76 | 100.0 | 28.28 | 28616 | 146201 | 1012 |
| Gemma-4 | Tagalog | SEA | 1.84 | 3.374 | 0.0 | 0.0 | 99.07 | 100.0 | 48.66 | 49245 | 166169 | 1012 |
| BLOOM | Tagalog | SEA | 1.895 | 3.275 | 0.02 | 0.0 | 98.15 | 100.0 | 50.13 | 50736 | 166169 | 1012 |
| Gemma-4 | Burmese | SEA | 6.136 | 2.499 | 0.0 | 0.0 | 100.0 | 100.0 | 64.52 | 65295 | 163169 | 1012 |
| BLOOM | Burmese | SEA | 25.746 | 0.596 | 27.36 | 0.0 | 53.59 | 100.0 | 270.74 | 273985 | 163169 | 1012 |
| Burmese-GPT-1B | Burmese | SEA | 10.719 | 1.43 | 0.6 | 0.0 | 91.5 | 100.0 | 112.72 | 114076 | 163169 | 1012 |
| Gemma-4 | Khmer | SEA | 12.747 | 2.067 | 0.01 | 0.0 | 98.38 | 100.0 | 73.25 | 74124 | 153212 | 1012 |
| BLOOM | Khmer | SEA | 29.492 | 0.893 | 26.35 | 0.0 | 63.78 | 100.0 | 169.46 | 171497 | 153212 | 1012 |
| Gemma-4 | Swahili | Africa | 2.087 | 3.108 | 0.0 | 0.0 | 100.0 | 100.0 | 43.91 | 44433 | 138104 | 1012 |
| BLOOM | Swahili | Africa | 1.602 | 4.048 | 0.05 | 0.0 | 100.0 | 100.0 | 33.71 | 34118 | 138104 | 1012 |
| Swahili-Gemma-7B | Swahili | Africa | 2.047 | 3.169 | 0.0 | 0.0 | 100.0 | 100.0 | 43.06 | 43578 | 138104 | 1012 |
| Gemma-4 | Amharic | Africa | 3.032 | 1.658 | 0.02 | 0.0 | 99.08 | 100.0 | 51.93 | 52552 | 87132 | 1012 |
| BLOOM | Amharic | Africa | 7.905 | 0.636 | 26.31 | 0.0 | 28.0 | 100.0 | 135.37 | 136993 | 87132 | 1012 |
| Gemma-4 | Hausa | Africa | 1.864 | 2.987 | 0.0 | 0.0 | 99.07 | 100.0 | 46.33 | 46889 | 140067 | 1012 |
| BLOOM | Hausa | Africa | 1.924 | 2.893 | 1.5 | 0.0 | 93.46 | 100.0 | 47.84 | 48410 | 140067 | 1012 |
| Gemma-4 | Yoruba | Africa | 2.588 | 1.944 | 0.01 | 0.0 | 100.0 | 100.0 | 64.46 | 65233 | 126844 | 1012 |
| BLOOM | Yoruba | Africa | 1.759 | 2.862 | 0.11 | 0.0 | 99.22 | 100.0 | 43.8 | 44325 | 126844 | 1012 |
| Gemma-4 | Igbo | Africa | 2.358 | 2.29 | 0.0 | 0.0 | 100.0 | 100.0 | 58.05 | 58748 | 134560 | 1012 |
| BLOOM | Igbo | Africa | 1.892 | 2.854 | 0.15 | 0.0 | 100.0 | 100.0 | 46.58 | 47144 | 134560 | 1012 |
| Kakugo-3B-Igbo | Igbo | Africa | 2.688 | 2.009 | 4.25 | 0.0 | 95.0 | 100.0 | 66.19 | 66987 | 134560 | 1012 |
| Gemma-4 | Zulu | Africa | 3.384 | 2.77 | 0.0 | 0.0 | 100.0 | 100.0 | 52.92 | 53557 | 148362 | 1012 |
| BLOOM | Zulu | Africa | 3.043 | 3.081 | 0.01 | 0.0 | 99.09 | 100.0 | 47.59 | 48158 | 148362 | 1012 |
| Gemma-4 | Xhosa | Africa | 3.302 | 2.717 | 0.01 | 0.0 | 99.03 | 100.0 | 50.6 | 51204 | 139123 | 1012 |
| BLOOM | Xhosa | Africa | 2.933 | 3.059 | 0.03 | 0.0 | 99.03 | 100.0 | 44.93 | 45474 | 139123 | 1012 |
| Gemma-4 | Somali | Africa | 2.246 | 2.896 | 0.0 | 0.0 | 100.0 | 100.0 | 51.75 | 52374 | 151662 | 1012 |
| BLOOM | Somali | Africa | 2.402 | 2.707 | 0.4 | 0.0 | 100.0 | 100.0 | 55.36 | 56023 | 151662 | 1012 |
| Gemma-4 | Wolof | Africa | 1.921 | 2.643 | 0.02 | 0.0 | 99.19 | 100.0 | 47.54 | 48114 | 127173 | 1012 |
| BLOOM | Wolof | Africa | 1.845 | 2.751 | 0.01 | 0.0 | 100.0 | 100.0 | 45.68 | 46231 | 127173 | 1012 |
| Wolof-Qwen-1.5B | Wolof | Africa | 2.103 | 2.414 | 0.53 | 0.0 | 100.0 | 100.0 | 52.06 | 52687 | 127173 | 1012 |
| Gemma-4 | Shona | Africa | 2.927 | 2.907 | 0.0 | 0.0 | 100.0 | 100.0 | 50.18 | 50778 | 147603 | 1012 |
| BLOOM | Shona | Africa | 2.848 | 2.987 | 0.02 | 0.0 | 100.0 | 100.0 | 48.83 | 49412 | 147603 | 1012 |
| Gemma-4 | French | Europe | 1.49 | 4.129 | 2.04 | 0.0 | 98.31 | 100.0 | 37.73 | 38180 | 157638 | 1012 |
| BLOOM | French | Europe | 1.289 | 4.774 | 0.0 | 0.0 | 100.0 | 100.0 | 32.63 | 33020 | 157638 | 1012 |
| Lucie-7B | French | Europe | 1.427 | 4.311 | 0.0 | 0.0 | 82.2 | 100.0 | 36.13 | 36565 | 157638 | 1012 |
| Gemma-4 | German | Europe | 1.655 | 4.27 | 0.01 | 0.0 | 100.0 | 100.0 | 35.59 | 36020 | 153809 | 1012 |
| BLOOM | German | Europe | 2.12 | 3.333 | 0.05 | 0.0 | 100.0 | 100.0 | 45.59 | 46141 | 153809 | 1012 |
| LeoLM-7B | German | Europe | 2.18 | 3.242 | 0.01 | 0.0 | 81.48 | 100.0 | 46.88 | 47442 | 153809 | 1012 |
| Gemma-4 | Spanish | Europe | 1.347 | 4.47 | 0.0 | 0.0 | 100.0 | 100.0 | 34.71 | 35125 | 156997 | 1012 |
| BLOOM | Spanish | Europe | 1.269 | 4.745 | 0.0 | 0.0 | 100.0 | 100.0 | 32.69 | 33084 | 156997 | 1012 |
| Salamandra-7B | Spanish | Europe | 1.354 | 4.447 | 0.03 | 0.0 | 85.58 | 100.0 | 34.88 | 35303 | 156997 | 1012 |
| Gemma-4 | Portuguese | Europe | 1.453 | 4.215 | 0.0 | 0.0 | 100.0 | 100.0 | 33.62 | 34022 | 143418 | 1012 |
| BLOOM | Portuguese | Europe | 1.312 | 4.669 | 0.04 | 0.0 | 100.0 | 100.0 | 30.36 | 30720 | 143418 | 1012 |
| Gervasio-8B | Portuguese | Europe | 1.709 | 3.585 | 0.74 | 0.0 | 100.0 | 100.0 | 39.53 | 40005 | 143418 | 1012 |
| Gemma-4 | Italian | Europe | 1.535 | 4.185 | 0.0 | 0.0 | 100.0 | 100.0 | 36.87 | 37313 | 156138 | 1012 |
| BLOOM | Italian | Europe | 1.828 | 3.514 | 0.01 | 0.0 | 100.0 | 100.0 | 43.91 | 44436 | 156138 | 1012 |
| LLaMAntino-3-8B | Italian | Europe | 1.831 | 3.508 | 0.58 | 0.0 | 100.0 | 100.0 | 43.98 | 44511 | 156138 | 1012 |
| Gemma-4 | Dutch | Europe | 1.63 | 3.935 | 0.01 | 0.0 | 100.0 | 100.0 | 37.05 | 37490 | 147507 | 1012 |
| BLOOM | Dutch | Europe | 2.04 | 3.144 | 0.0 | 0.0 | 98.11 | 100.0 | 46.37 | 46923 | 147507 | 1012 |
| Fietje-2 | Dutch | Europe | 2.325 | 2.759 | 0.1 | 0.0 | 94.34 | 100.0 | 52.83 | 53462 | 147507 | 1012 |
| Gemma-4 | Polish | Europe | 2.096 | 3.415 | 1.94 | 0.0 | 100.0 | 100.0 | 40.37 | 40850 | 139483 | 1012 |
| BLOOM | Polish | Europe | 2.99 | 2.394 | 0.03 | 0.0 | 97.17 | 100.0 | 57.58 | 58275 | 139483 | 1012 |
| Bielik-11B | Polish | Europe | 2.924 | 2.447 | 3.58 | 0.0 | 81.13 | 100.0 | 56.32 | 56991 | 139483 | 1012 |
| Gemma-4 | Russian | Europe | 1.884 | 3.838 | 0.01 | 0.0 | 99.33 | 100.0 | 36.58 | 37023 | 142096 | 1012 |
| BLOOM | Russian | Europe | 3.425 | 2.111 | 0.24 | 0.0 | 92.67 | 100.0 | 66.52 | 67317 | 142096 | 1012 |
| Vikhr-Nemo-12B | Russian | Europe | 2.13 | 3.395 | 0.71 | 0.0 | 98.67 | 100.0 | 41.36 | 41857 | 142096 | 1012 |
| Gemma-4 | Ukrainian | Europe | 2.273 | 3.083 | 0.0 | 0.0 | 100.0 | 100.0 | 43.09 | 43605 | 134446 | 1012 |
| BLOOM | Ukrainian | Europe | 3.899 | 1.798 | 2.11 | 0.0 | 88.44 | 100.0 | 73.9 | 74789 | 134446 | 1012 |
| MamayLM-12B | Ukrainian | Europe | 2.273 | 3.083 | 0.0 | 0.0 | 100.0 | 100.0 | 43.09 | 43605 | 134446 | 1012 |
| Gemma-4 | Romanian | Europe | 1.8 | 3.479 | 2.4 | 0.0 | 100.0 | 100.0 | 42.2 | 42702 | 148571 | 1012 |
| BLOOM | Romanian | Europe | 2.205 | 2.84 | 1.54 | 0.0 | 96.43 | 100.0 | 51.69 | 52315 | 148571 | 1012 |
| LLMic-3B | Romanian | Europe | 2.192 | 2.856 | 16.17 | 0.0 | 77.68 | 4.15 | 51.41 | 52025 | 148571 | 1012 |
| Gemma-4 | Swedish | Europe | 1.841 | 3.543 | 0.0 | 0.0 | 100.0 | 100.0 | 36.86 | 37303 | 132173 | 1012 |
| BLOOM | Swedish | Europe | 2.222 | 2.935 | 0.06 | 0.0 | 99.05 | 100.0 | 44.5 | 45032 | 132173 | 1012 |
| Viking-7B | Swedish | Europe | 1.474 | 4.423 | 0.94 | 0.0 | 100.0 | 100.0 | 29.53 | 29881 | 132173 | 1012 |
| Gemma-4 | Czech | Europe | 2.157 | 3.077 | 1.51 | 0.0 | 99.13 | 100.0 | 40.87 | 41359 | 127257 | 1012 |
| BLOOM | Czech | Europe | 2.885 | 2.3 | 0.25 | 0.0 | 95.65 | 100.0 | 54.67 | 55327 | 127257 | 1012 |
| CSMPT-7B | Czech | Europe | 1.407 | 4.717 | 0.03 | 0.0 | 96.52 | 100.0 | 26.66 | 26979 | 127257 | 1012 |
| Gemma-4 | Greek | Europe | 2.472 | 2.666 | 0.0 | 0.0 | 100.0 | 100.0 | 58.39 | 59088 | 157539 | 1012 |
| BLOOM | Greek | Europe | 4.341 | 1.518 | 1.45 | 0.0 | 86.27 | 100.0 | 102.52 | 103749 | 157539 | 1012 |
| Meltemi-7B | Greek | Europe | 1.397 | 4.72 | 0.04 | 0.0 | 95.42 | 100.0 | 32.98 | 33378 | 157539 | 1012 |
| Gemma-4 | Lat.Am. Spanish | Americas | 1.347 | 4.47 | 0.0 | 0.0 | 100.0 | 100.0 | 34.71 | 35125 | 156997 | 1012 |
| BLOOM | Lat.Am. Spanish | Americas | 1.269 | 4.745 | 0.0 | 0.0 | 100.0 | 100.0 | 32.69 | 33084 | 156997 | 1012 |
| LatamGPT-70B | Lat.Am. Spanish | Americas | 1.604 | 3.752 | 0.65 | 0.0 | 100.0 | 100.0 | 41.35 | 41843 | 156997 | 1012 |
| Gemma-4 | Brazilian Portuguese | Americas | 1.453 | 4.215 | 0.0 | 0.0 | 100.0 | 100.0 | 33.62 | 34022 | 143418 | 1012 |
| BLOOM | Brazilian Portuguese | Americas | 1.312 | 4.669 | 0.04 | 0.0 | 100.0 | 100.0 | 30.36 | 30720 | 143418 | 1012 |
| Tucano-2b4 | Brazilian Portuguese | Americas | 1.252 | 4.893 | 0.01 | 0.0 | 86.84 | 99.31 | 28.96 | 29308 | 143418 | 1012 |
| Gemma-4 | Quechua | Americas | 3.147 | 2.811 | 0.0 | 0.0 | 100.0 | 100.0 | 49.58 | 50177 | 141027 | 1012 |
| BLOOM | Quechua | Americas | 3.158 | 2.801 | 0.0 | 0.0 | 100.0 | 100.0 | 49.74 | 50341 | 141027 | 1012 |
| Gemma-4 | Haitian Creole | Americas | 1.796 | 2.918 | 0.0 | 0.0 | 100.0 | 100.0 | 41.08 | 41569 | 121280 | 1012 |
| BLOOM | Haitian Creole | Americas | 1.85 | 2.831 | 0.05 | 0.0 | 99.13 | 100.0 | 42.33 | 42833 | 121280 | 1012 |
| Gemma-4 | Māori | Oceania | 1.825 | 2.647 | 5.77 | 0.0 | 100.0 | 100.0 | 54.7 | 55352 | 146527 | 1012 |
| BLOOM | Māori | Oceania | 1.91 | 2.529 | 0.02 | 0.0 | 94.62 | 100.0 | 57.26 | 57950 | 146527 | 1012 |
| Gemma-4 | Samoan | Oceania | 1.821 | 2.635 | 0.72 | 0.0 | 100.0 | 100.0 | 57.27 | 57953 | 152724 | 1012 |
| BLOOM | Samoan | Oceania | 1.823 | 2.632 | 0.02 | 0.0 | 97.37 | 100.0 | 57.33 | 58022 | 152724 | 1012 |
| Gemma-4 | Tok Pisin | Oceania | 1.644 | 3.247 | 0.0 | 0.0 | 100.0 | 100.0 | 51.44 | 52053 | 169011 | 1012 |
| BLOOM | Tok Pisin | Oceania | 1.668 | 3.201 | 0.0 | 0.0 | 100.0 | 100.0 | 52.18 | 52803 | 169011 | 1012 |

---

## 6. Raw files

- `data/results.csv` — machine-readable detail (one row per tokenizer × language)
- `data/summary.json` — per-tokenizer aggregates (unweighted + character-weighted)

*Regenerate this report:* `python experiments/detailed_report.py`