# Task 1 — Phase 2: Tokenizer Test Shortlist

Filtered from `docs/llm-research-raw.md`. Keeps only models with downloadable weights and accessible tokenizers.

## Summary
| Category | Count |
|---|---|
| Priority 1 — test immediately (Apache 2.0 / MIT / BSD) | 37 |
| Priority 2 — test for eval only (CC-BY-NC; check before production) | 7 |
| Custom / check terms before production use | 8 |
| Skipped — Gemma-4 is better (no viable regional model) | 7 |
| Skipped — no candidate found | 2 |

---

## Priority 1 — Fully Permissive Licences (Apache 2.0 / MIT / BSD)
> These can proceed to tokenizer test immediately. Safe for production.

| Language | Best Candidate | HF ID | Params | Licence |
|---|---|---|---|---|
| **Tamil** | Tamil-Mistral-7B | `Hemanth-thunder/Tamil-Mistral-7B-Instruct-v0.1` | 7B | Apache 2.0 |
| **Telugu** | Telugu-Llama2-7B | `Telugu-LLM-Labs/Telugu-Llama2-7B-v0-Base` | 7B | MIT |
| **Kannada** | Ambari-7B-Base | `Cognitive-Lab/Ambari-7B-base-v0.1` | 7B | MIT |
| **Malayalam** | MalayaLLM Gemma 2 9B | `VishnuPJ/MalayaLLM_Gemma_2_9B_Instruct_V1.0` | 9B | MIT |
| **Marathi** | MahaMarathi-7B | `marathi-llm/MahaMarathi-7B-v24.01-Base` | 7B | MIT |
| **Punjabi** | Dhee-NxtGen-Qwen3 | `dheeyantra/dhee-nxtgen-qwen3-punjabi-v2` | 2B | Apache 2.0 |
| **Odia** | Qwen 1.5 Odia 7B | `OdiaGenAI-LLM/qwen_1.5_odia_7b` | 7B | Apache 2.0 |
| **Assamese** | Goldfish ASM | `goldfish-models/asm_beng_full` | 125M | Apache 2.0 |
| **Urdu** | Qalb-1.0-8B | `enstazao/Qalb-1.0-8B-Instruct` | 8B | Apache 2.0 |
| **Nepali** | NEPALI-LLM | `shivam9980/NEPALI-LLM` | 9B | Apache 2.0 |
| **Sinhala** | llama3-sinhala | `ihalage/llama3-sinhala` | 8B | Apache 2.0 |
| **Persian (Farsi)** | Maral-7B | `MaralGPT/Maral-7B-alpha-1` | 7B | MIT |
| **Arabic** | Jais-2-8B-Chat | `inceptionai/Jais-2-8B-Chat` | 8B | Apache 2.0 |
| **Turkish** | Trendyol-LLM-8B | `Trendyol/Trendyol-LLM-8B-T1` | 8B | Apache 2.0 |
| **Hebrew** | DictaLM 2.0 | `dicta-il/dictalm2.0-instruct` | 7B | Apache 2.0 |
| **Kurdish** | Mistral-Nemo-Kurdish | `nazimali/Mistral-Nemo-Kurdish-Instruct` | 12B | Apache 2.0 |
| **Azerbaijani** | AzQ-1.7B | `karabakh-nlp/AzQ-1.7B` | 1.7B | Apache 2.0 |
| **Uzbek** | Mistral-7B-Uz | `behbudiy/Mistral-7B-Instruct-Uz` | 7B | Apache 2.0 |
| **Japanese** | LLM-jp-3 13B | `llm-jp/llm-jp-3-13b-instruct` | 13B | Apache 2.0 |
| **Korean** | Polyglot-Ko 12.8B | `EleutherAI/polyglot-ko-12.8b` | 12.8B | Apache 2.0 |
| **Vietnamese** | Arcee-VyLinh | `arcee-ai/Arcee-VyLinh` | 3B | Apache 2.0 |
| **Thai** | Typhoon2-Qwen2.5-7B | `scb10x/typhoon2-qwen2.5-7b-instruct` | 7B | Apache 2.0 |
| **Burmese** | Burmese-GPT | `WYNN747/Burmese-GPT` | 1B | MIT |
| **Swahili** | Swahili_Gemma | `Mollel/Swahili_Gemma` | 7B | Apache 2.0 |
| **Amharic** | Walia-LLM | `israel/LLAMA-Walia-II` | 7B | Apache 2.0 |
| **Igbo** | Kakugo-3B-Igbo | `ptrdvn/kakugo-3B-ibo` | 3B | Apache 2.0 |
| **French** | Lucie-7B | `OpenLLM-France/Lucie-7B` | 7B | Apache 2.0 |
| **German** | LeoLM | `LeoLM/leo-mistral-hessianai-7b-chat` | 7B | Apache 2.0 |
| **Spanish** | Salamandra-7B | `BSC-LT/salamandra-7b` | 7B | Apache 2.0 |
| **Portuguese (EU)** | Gervásio-8B | `PORTULAN/gervasio-8b-portuguese-ptpt-decoder` | 8B | MIT |
| **Dutch** | Fietje-2 | `BramVanroy/fietje-2` | 2.7B | MIT |
| **Polish** | Bielik-11B | `speakleash/Bielik-11B-v2.3-Instruct` | 11B | Apache 2.0 |
| **Russian** | Vikhr-Nemo-12B | `Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24` | 12B | Apache 2.0 |
| **Romanian** | LLMic | `faur-ai/LLMic` | 3B | Apache 2.0 |
| **Swedish** | Viking-7B | `LumiOpen/Viking-7B` | 7B | Apache 2.0 |
| **Czech** | CSMPT-7B | `BUT-FIT/csmpt7b` | 7B | Apache 2.0 |
| **Greek** | Meltemi-7B | `ilsp/Meltemi-7B-Instruct-v1.5` | 7B | Apache 2.0 |
| **Brazilian Portuguese** | Tucano-2b4-Instruct | `TucanoBR/Tucano-2b4-Instruct` | 2.4B | Apache 2.0 |
| **Māori** | Goldfish mri | `goldfish-models/mri_latn_10mb` | 39M | Apache 2.0 |
| **Hawaiian** | Goldfish haw | `goldfish-models/haw_latn_5mb` | 39M | Apache 2.0 |
| **Tok Pisin** | Goldfish tpi | `goldfish-models/tpi_latn_full` | 125M | Apache 2.0 |

---

## Priority 2 — Non-Commercial Only (CC-BY-NC)
> Tokenizer test is fine (research). Confirm with Jacaranda/MBZUAI before production use.

| Language | Best Candidate | HF ID | Params | Contact for commercial |
|---|---|---|---|---|
| **Kazakh** | LLama-3.1-KazLLM | `issai/LLama-3.1-KazLLM-1.0-8B` | 8B | Nazarbayev University ISSAI |
| **Hausa** | HausaLlama | `Jacaranda/HausaLlama` | 8B | Jacaranda AI |
| **Yoruba** | YorubaLlama | `Jacaranda/YorubaLlama` | 8B | Jacaranda AI |
| **Zulu** | Xhosa-ZuluLlama3 | `Jacaranda/Xhosa_ZuluLlama3_v1` | 8B | Jacaranda AI |
| **Xhosa** | Xhosa-ZuluLlama3 | `Jacaranda/Xhosa_ZuluLlama3_v1` | 8B | Jacaranda AI (same model as Zulu) |
| **Korean** (alt) | EXAONE 3.5 | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` | 7.8B | LG AI Research |
| **Wolof** | wolof-qwen-1.5b | `ciskoM/wolof-qwen-1.5b` | 1.5B | individual researcher |

---

## Custom / Restricted Licence — Check Before Production
> Tokenizer test is fine. These use Llama 2/3, Gemma, or other community licences that restrict commercial use above certain thresholds.

| Language | Best Candidate | HF ID | Params | Base Licence |
|---|---|---|---|---|
| **Hindi** | LLama3-Gaja-Hindi | `Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1` | 8B | Llama 3 Community |
| **Bengali** | BanglaLLama-3.1-8B | `BanglaLLM/BanglaLLama-3.1-8b-bangla-alpaca-orca-instruct-v0.0.1` | 8B | Llama 3.1 Community |
| **Gujarati** | Gujju-Llama | `sampoorna42/gujju-llama-base-v1.0` | 7B | GPL-3.0 (copyleft) |
| **Mandarin Chinese** | ChatGLM3-6B | `THUDM/chatglm3-6b` | 6B | ChatGLM3 Custom |
| **Indonesian** | Nusantara-7b | `kalisai/Nusantara-7b-Indo-Chat` | 7B | Qwen1.5 + Apache mix |
| **Malay** | MaLLaM-5B | `mesolitica/mallam-5B-4096` | 5B | No licence declared |
| **Italian** | LLaMAntino-3-ANITA | `swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA` | 8B | Llama 3 Community |
| **Ukrainian** | MamayLM-Gemma-3-12B | `INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0` | 12B | Gemma Terms |

---

## Skipped — Use Gemma-4 (No Competitive Regional Candidate)

| Language | Reason |
|---|---|
| **Maithili** | Only encoder-only BERT model found |
| **Tagalog** | Experimental LoRA adapter only; Gemma-4 likely stronger |
| **Somali** | No open-weight generative model found |
| **Shona** | Only 39M research model |
| **Quechua** | Only encoder + translation models |
| **Haitian Creole** | Models found but too low quality for production |
| **Samoan** | No model found |

---

## Next: Run Tokenizer Test
1. Set up environment (see `docs/plans/task1-regional-llm-evaluation.md` Phase 3)
2. Run `experiments/tokenizer_test.py` starting with Priority 1 models
3. Results → `data/tokenizer_results.csv`
4. Summary → `docs/llm-evaluation.md`

### Suggested test order (highest-value languages first)
1. Indic group: Tamil, Telugu, Kannada, Malayalam, Marathi, Punjabi, Urdu, Hindi, Bengali
2. Middle East: Arabic, Turkish, Persian, Hebrew
3. East Asia: Japanese, Korean, Chinese, Vietnamese, Thai
4. Europe: French, German, Spanish, Polish, Russian
5. Africa: Swahili, Amharic, Igbo
6. Americas + Oceania: Brazilian Portuguese, rest
