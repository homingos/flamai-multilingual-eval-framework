# Regional LLM Research — Raw Findings
**Phase 1 output. 65 languages researched in parallel.**
Cross-reference: `docs/pdfs/Regional LLMs by Continent.pdf` (ChatGPT pre-research)

## Stats
| Metric | Count |
|---|---|
| Languages researched | 65 |
| With language-specific candidates | 63 |
| No viable candidate found | 2 (Samoan, Somali) |
| Gemma-4 likely best (no competitive regional model) | 7 |

## License Key
| Symbol | Category | Meaning |
|---|---|---|
| ✅ | `open_apache` | Apache 2.0, MIT, BSD — fully permissive, commercial-friendly |
| ⚠️ | `open_cc_bync` | CC-BY-NC — non-commercial only; needs negotiation for production |
| 🔶 | `open_custom` | Open weights but custom/restrictive licence — check terms case-by-case |
| ❌ | `proprietary_api` | API-only, tokenizer not downloadable — excluded from tokenizer tests |

---

## Indic Languages

| Language | Model | HF ID | Params | Licence | ✓ Tokenizer | Notes |
|---|---|---|---|---|---|---|
| Hindi | Airavata | `ai4bharat/Airavata` | 7B | 🔶 Llama 2 Community | ✅ | Hindi instruction model by AI4Bharat on IndicInstruct (404k examples) |
| Hindi | LLama3-Gaja-Hindi | `Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1` | 8B | 🔶 Llama 3 Community | ✅ | Llama 3 bilingual English-Hindi fine-tune by Cognitive-Lab |
| Bengali | BanglaLLama-3.1-8B | `BanglaLLM/BanglaLLama-3.1-8b-bangla-alpaca-orca-instruct-v0.0.1` | 8B | 🔶 Llama 3.1 Community | ✅ | Bengali fine-tune on 172k instruction pairs |
| Bengali | LilTii-v0.2 | `Polygl0t/LilTii-v0.2` | 0.6B | 🔶 Unconfirmed permissive | ✅ | Bengali-only model trained from scratch (Mar 2026) |
| Tamil | Tamil-Mistral-7B | `Hemanth-thunder/Tamil-Mistral-7B-Instruct-v0.1` | 7B | ✅ Apache 2.0 | ✅ | Best-licensed Tamil model; Mistral-7B fine-tune |
| Tamil | Tamil-LLaMA-7B | `abhinand/tamil-llama-7b-instruct-v0.2` | 7B | 🔶 Llama 2 Community | ✅ | 16k added Tamil tokens; Llama 2 base |
| Tamil | **Sarvam-M-24B** ⭐ | `sarvamai/sarvam-m` | 24B | ✅ Apache 2.0 | ✅ | **Task 1B upgrade.** Single model covers Tamil, Kannada, Marathi, Gujarati + 7 more Indic langs; +23% MMLU-IN vs Mistral Small; comparable to Llama-3.3 70B. Tokenizer fertility=3.614 (higher than Gemma-4 2.374, but inference quality far superior) |
| Telugu | Telugu-Llama2-7B | `Telugu-LLM-Labs/Telugu-Llama2-7B-v0-Base` | 7B | ✅ MIT | ✅ | MIT-licensed Telugu-specific base model |
| Telugu | Telugu LLaMA Instruct | `abhinand/telugu-llama-7b-instruct-v0.1` | 7B | 🔶 GPL-3.0 | ✅ | 16k added Telugu tokens; GPL may restrict commercial use |
| Kannada | Ambari-7B-Base | `Cognitive-Lab/Ambari-7B-base-v0.1` | 7B | ✅ MIT | ✅ | First Kannada-English bilingual LLM; ~500M Kannada tokens |
| Kannada | Ambari-7B-Instruct | `Cognitive-Lab/Ambari-7B-Instruct-v0.1` | 7B | 🔶 Llama 2 Community | ✅ | Instruction-tuned variant of Ambari |
| Malayalam | MalayaLLM Gemma 2 9B | `VishnuPJ/MalayaLLM_Gemma_2_9B_Instruct_V1.0` | 9B | ✅ MIT | ✅ | Gemma-2 9B fine-tuned on Aya dataset for Malayalam |
| Malayalam | Malayalam LLaMA 7B | `abhinand/malayalam-llama-7b-instruct-v0.1` | 7B | 🔶 Llama 2 Community | ✅ | 16k Malayalam vocab expansion + 500k samples |
| Marathi | MahaMarathi-7B | `marathi-llm/MahaMarathi-7B-v24.01-Base` | 7B | ✅ MIT | ✅ | Continually pretrained and instruction fine-tuned Marathi LLM |
| Marathi | Misal-7B | `smallstepai/Misal-7B-instruct-v0.1` | 7B | 🔶 Llama 2 Community | ✅ | Instruction-tuned exclusively for Marathi |
| Gujarati | Gujju-Llama Base | `sampoorna42/gujju-llama-base-v1.0` | 7B | 🔶 GPL-3.0 | ✅ | LLaMA-2 fine-tuned on CulturaX Gujarati; GPL copyleft |
| Gujarati | GujaratiBERT | `l3cube-pune/gujarati-bert` | 110M | ✅ CC-BY-4.0 | ✅ | Encoder-only; good for classification/embeddings, not generative |
| Punjabi | Dhee-NxtGen-Qwen3-Punjabi | `dheeyantra/dhee-nxtgen-qwen3-punjabi-v2` | 2B | ✅ Apache 2.0 | ✅ | Qwen3-2B fine-tuned for Punjabi by DheeYantra (2025) |
| Punjabi | BharatLLM Punjabi 7B | `FoundryAILabs/bharat-punjabi-7b-lora` | 7B (LoRA) | ✅ Apache 2.0 | ✅ | LoRA adapter on Mistral-7B for Punjabi educational content |
| Odia | Llama3 8B Odia | `OdiaGenAI-LLM/Llama3_8B_Odia_Unsloth` | 8B | ⚠️ CC-BY-NC-SA 4.0 | ✅ | 171k Odia instruction dataset; non-commercial only |
| Odia | Qwen 1.5 Odia 7B | `OdiaGenAI-LLM/qwen_1.5_odia_7b` | 7B | ✅ Apache 2.0 | ✅ | Permissive-licensed Odia base; needs instruction-tuning |
| Assamese | Goldfish ASM | `goldfish-models/asm_beng_full` | 125M | ✅ Apache 2.0 | ✅ | GPT-2 trained on 348MB of Assamese text |
| Assamese | Assamese SLM | `prthmgoyl/slm-indic-assamese-santali-pretrained` | 1B | 🔶 Unspecified | ✅ | Larger option but no licence declared — use with caution |
| Urdu | Qalb-1.0-8B-Instruct | `enstazao/Qalb-1.0-8B-Instruct` | 8B | ✅ Apache 2.0 | ✅ | 1.97B Urdu tokens + instruction tuning; 90.34% benchmark score |
| Urdu | Lughaat-1.0-8B | `muhammadnoman76/Lughaat-1.0-8B-Instruct` | 8B | ✅ Apache 2.0 | ✅ | Largest Urdu instruction dataset; 91.4% average benchmark |
| Nepali | NepBERTa | `NepBERTa/NepBERTa` | 125M | ⚠️ CC-BY-NC-SA 4.0 | ✅ | Encoder-only (NLU tasks); not generative |
| Nepali | NEPALI-LLM | `shivam9980/NEPALI-LLM` | 9B | ✅ Apache 2.0 | ✅ | Gemma-2 9B fine-tuned on 150K Nepali samples |
| Sinhala | llama3-sinhala | `ihalage/llama3-sinhala` | 8B | ✅ Apache 2.0 | ✅ | LLaMA-3 8B instruction fine-tuned on Sinhala datasets |
| Sinhala | SinLlama_v01 | `polyglots/SinLlama_v01` | 8B | 🔶 Llama 3 Community | ✅ | Extended Sinhala tokenizer; PEFT adapter only |
| Maithili | maiBERT_TF | `rockerritesh/maiBERT_TF` | 110M | ✅ MIT | ✅ | **Encoder-only (BERT).** No generative Maithili LLM exists. Gemma-4 is recommended for generation. |

---

## Middle East & West Asia

| Language | Model | HF ID | Params | Licence | ✓ Tokenizer | Notes |
|---|---|---|---|---|---|---|
| Arabic | Jais-2-8B-Chat | `inceptionai/Jais-2-8B-Chat` | 8B | ✅ Apache 2.0 | ✅ | Trained from scratch by MBZUAI/Inception; custom Arabic vocab; Dec 2025 |
| Arabic | SILMA-9B-Instruct | `silma-ai/SILMA-9B-Instruct-v1.0` | 9B | 🔶 Gemma Licence | ✅ | Strong Arabic benchmark results; commercial use allowed with conditions |
| Persian (Farsi) | Dorna2-Llama3.1-8B | `PartAI/Dorna2-Llama3.1-8B-Instruct` | 8B | 🔶 Llama 3.1 Community | ✅ | Persian fine-tune by PartAI (Open Persian LLM Leaderboard maintainers) |
| Persian (Farsi) | Maral-7B-alpha-1 | `MaralGPT/Maral-7B-alpha-1` | 7B | ✅ MIT | ✅ | Mistral-7B Persian instruction model; MIT licensed |
| Turkish | Trendyol-LLM-8B | `Trendyol/Trendyol-LLM-8B-T1` | 8B | ✅ Apache 2.0 | ✅ | Qwen3-8B fine-tuned on Turkish by Trendyol (2025) |
| Turkish | Turkcell-LLM-7b | `TURKCELL/Turkcell-LLM-7b-v1` | 7B | ✅ Apache 2.0 | ✅ | Mistral fine-tuned on 5B Turkish tokens by Turkcell |
| Hebrew | DictaLM 2.0 Instruct | `dicta-il/dictalm2.0-instruct` | 7B | ✅ Apache 2.0 | ✅ | Extended Hebrew tokenizer; ~190B tokens; halves token count vs base Mistral |
| Hebrew | Hebrew-Gemma-11B | `yam-peleg/Hebrew-Gemma-11B` | 11B | 🔶 Gemma Terms | ✅ | Larger option; restricted by Google Gemma custom licence |
| Hebrew | **DictaLM-3.0-Nemotron-12B** ⭐ | `dicta-il/DictaLM-3.0-Nemotron-12B-Instruct` | 12B | ✅ Apache 2.0 | ✅ | **Task 1B upgrade.** Same Dicta lab; Nemotron-12B base; cleaner instruct format (avoids `<think>` blocks of 3.0-24B-Thinking). Note: Nemotron tokenizer has poor Hebrew support (fertility=0.181, roundtrip=0%) — evaluated on inference quality only |
| Kurdish | Mistral-Nemo-Kurdish | `nazimali/Mistral-Nemo-Kurdish-Instruct` | 12B | ✅ Apache 2.0 | ✅ | Only Kurdish-specific open-weight LLM; Kurmanji dialect |
| Azerbaijani | AzQ-1.7B | `karabakh-nlp/AzQ-1.7B` | 1.7B | ✅ Apache 2.0 | ✅ | Qwen3-1.7B fine-tuned on Azerbaijani; expert-curated dataset |
| Azerbaijani | mGPT-1.3B-azerbaijan | `ai-forever/mGPT-1.3B-azerbaijan` | 1.3B | ✅ MIT | ✅ | mGPT-XL continued on Azerbaijani Wikipedia + C4 |
| Uzbek | Mistral-7B-Instruct-Uz | `behbudiy/Mistral-7B-Instruct-Uz` | 7B | ✅ Apache 2.0 | ✅ | Mistral continued + instruction-tuned on Uzbek |
| Uzbek | mGPT-1.3B-uzbek | `ai-forever/mGPT-1.3B-uzbek` | 1.3B | ✅ MIT | ✅ | mGPT base continued on Uzbek |
| Kazakh | Llama-3.1-Sherkala-8B | `inceptionai/Llama-3.1-Sherkala-8B-Chat` | 8B | ⚠️ CC-BY-NC-SA 4.0 | ✅ | 45B token Kazakh training; 25% vocab expansion; MBZUAI-backed |
| Kazakh | LLama-3.1-KazLLM | `issai/LLama-3.1-KazLLM-1.0-8B` | 8B | ⚠️ CC-BY-NC 4.0 | ✅ | Nazarbayev University; Dec 2024; quantized variants available |

---

## East & Southeast Asia

| Language | Model | HF ID | Params | Licence | ✓ Tokenizer | Notes |
|---|---|---|---|---|---|---|
| Mandarin Chinese | ChatGLM3-6B | `THUDM/chatglm3-6b` | 6B | 🔶 ChatGLM3 Custom | ✅ | Tsinghua/Zhipu AI; requires trust_remote_code=True |
| Mandarin Chinese | Baichuan2-7B-Chat | `baichuan-inc/Baichuan2-7B-Chat` | 7B | 🔶 Baichuan 2 Custom | ✅ | 2.6T tokens; commercial approval needed for DAU > 1M |
| Japanese | LLM-jp-3 13B Instruct | `llm-jp/llm-jp-3-13b-instruct` | 13B | ✅ Apache 2.0 | ✅ | Built from scratch by NII (Japan's national AI institute); Sep 2024 |
| Japanese | CyberAgentLM3 22B | `cyberagent/calm3-22b-chat` | 22B | ✅ Apache 2.0 | ✅ | 2T tokens from scratch; larger capacity option |
| Korean | Polyglot-Ko 12.8B | `EleutherAI/polyglot-ko-12.8b` | 12.8B | ✅ Apache 2.0 | ✅ | Trained purely on 863 GB of Korean text; EleutherAI |
| Korean | EXAONE 3.5 7.8B | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` | 7.8B | ⚠️ EXAONE NC Licence | ✅ | LG AI Research; strong Korean benchmarks; non-commercial only |
| Korean | **EXAONE-3.5-32B-Instruct** ⭐ | `LGAI-EXAONE/EXAONE-3.5-32B-Instruct` | 32B | ⚠️ EXAONE NC Licence | ✅ | **Task 1B upgrade.** Polyglot-Ko is a 2022 base-only model (no instruct tuning). EXAONE-3.5-32B is LG AI's flagship instruct model; fertility=2.199 (same as Polyglot), vcov=94.57% (slightly better). Requires trust_remote_code=True |
| Vietnamese | Arcee-VyLinh | `arcee-ai/Arcee-VyLinh` | 3B | ✅ Apache 2.0 | ✅ | Qwen2.5-3B fine-tuned for Vietnamese; 32k context |
| Vietnamese | PhoGPT-7B5-Instruct | `vinai/PhoGPT-7B5-Instruct` | 7.5B | ✅ BSD-3-Clause | ✅ | Pre-trained from scratch on 41GB Vietnamese by VinAI |
| Thai | Typhoon2-Qwen2.5-7B | `scb10x/typhoon2-qwen2.5-7b-instruct` | 7B | ✅ Apache 2.0 | ✅ | SCB 10X; Thai-primary; 128k context; Dec 2024 |
| Thai | OpenThaiGPT1.5-7B | `openthaigpt/openthaigpt1.5-7b-instruct` | 7B | 🔶 Qwen Custom | ✅ | 2M+ Thai instruction pairs; 100M MAU ceiling on licence |
| Indonesian | Nusantara-7b-Indo-Chat | `kalisai/Nusantara-7b-Indo-Chat` | 7B | 🔶 Qwen1.5 + Apache | ✅ | Qwen1.5-7B fine-tuned on Bahasa Indonesia |
| Indonesian | Cendol-LLaMA2-7B | `indonlp/cendol-llama2-7b-inst` | 7B | 🔶 Llama 2 Community | ✅ | IndoNLP research group; broad Indonesian NLP task coverage |
| Malay | MaLLaM 5B | `mesolitica/mallam-5B-4096` | 5B | 🔶 Unspecified (MIT codebase) | ✅ | Pre-trained from scratch on 90B Malaysian tokens; top Malay lab |
| Malay | Malaysian-Qwen2.5-7B | `mesolitica/Malaysian-Qwen2.5-7B-Instruct` | 7B | 🔶 Unspecified | ✅ | LoRA fine-tune on 1.5B Malaysian instruction tokens; Mar 2025 |
| Tagalog | Tagalog-SeaLLM-7B | `922-Narra/tagalog-seallm-7b-v1` | 7B | ✅ Apache 2.0 | ✅ | Experimental LoRA adapter; limited training data. **Gemma-4 likely better for production.** |
| Burmese | Burmese-GPT | `WYNN747/Burmese-GPT` | 1B | ✅ MIT | ✅ | Burmese corpus: literature, news, Wikipedia |
| Burmese | Myanmarsar-GPT | `simbolo-ai/Myanmarsar-GPT` | 1B | ✅ MIT | ✅ | 20k Burmese texts; Jan 2024 |
| Khmer | PrahokBART | `nict-astrec-att/prahokbart_base` | 62M | ✅ MIT | ✅ | Seq2seq trained from scratch on Khmer; COLING 2025 |
| Khmer | SurMuy v1 | `AingHongsin/SurMuy_v1_512512201` | 9B | 🔶 Unspecified | ✅ | SeaLLM fine-tune for Khmer QA; verify licence before use |

---

## Africa

| Language | Model | HF ID | Params | Licence | ✓ Tokenizer | Notes |
|---|---|---|---|---|---|---|
| Swahili | Swahili_Gemma | `Mollel/Swahili_Gemma` | 7B | ✅ Apache 2.0 | ✅ | Gemma-7B fine-tuned for Swahili; best for commercial use |
| Swahili | UlizaLlama3 | `Jacaranda/UlizaLlama3` | 8B | ⚠️ CC-BY-NC-SA 4.0 | ✅ | Stronger technically (11.3GB Swahili + 66k instruction pairs); non-commercial |
| Amharic | Walia-LLM | `israel/LLAMA-Walia-II` | 7B | ✅ Apache 2.0 | ✅ | LLaMA-2 fine-tuned for Amharic; EMNLP 2024; Apache 2.0 |
| Amharic | EthioNLP/Amharic-LLAMA | `EthioNLP/Amharic-LLAMA-all-data` | 7B | 🔶 Llama 2 inherited | ✅ | Same training as Walia; no explicit licence declared |
| Hausa | HausaLlama | `Jacaranda/HausaLlama` | 8B | ⚠️ CC-BY-NC-SA 4.0 | ✅ | Only dedicated Hausa LLM; 8.4GB Hausa + 66k pairs; contact Jacaranda for commercial use |
| Hausa | AfroLlama_V1 | `Jacaranda/AfroLlama_V1` | 8B | ⚠️ CC-BY-NC-SA 4.0 | ✅ | 6-language African model including Hausa; non-commercial |
| Yoruba | YorubaLlama | `Jacaranda/YorubaLlama` | 8B | ⚠️ CC-BY-NC-SA 4.0 | ✅ | Only dedicated Yoruba LLM; 8.1GB text + 66k pairs |
| Yoruba | InkubaLM-0.4B | `lelapa/InkubaLM-0.4B` | 0.4B | ⚠️ CC-BY-NC 4.0 | ✅ | Small African multilingual model; Yoruba is 1 of 7 languages |
| Igbo | Kakugo 3B Igbo | `ptrdvn/kakugo-3B-ibo` | 3B | ✅ Apache 2.0 | ✅ | IBM Granite fine-tuned for Igbo; Jan 2026 |
| Igbo | N-ATLaS LLM | `NCAIR1/N-ATLaS` | 8B | 🔶 OSRI Licence | ✅ | Nigerian trilingual (Igbo + Hausa + Yoruba); NCAIR/Awarri |
| Zulu | Xhosa-ZuluLlama3 | `Jacaranda/Xhosa_ZuluLlama3_v1` | 8B | ⚠️ CC-BY-NC-SA 4.0 | ✅ | Purpose-built for isiZulu + isiXhosa; Jacaranda |
| Zulu | InkubaLM-0.4B | `lelapa/InkubaLM-0.4B` | 0.4B | ⚠️ CC-BY-NC 4.0 | ✅ | Includes Zulu among 5 African languages |
| Xhosa | Xhosa-ZuluLlama3 | `Jacaranda/Xhosa_ZuluLlama3_v1` | 8B | ⚠️ CC-BY-NC-SA 4.0 | ✅ | Same model as Zulu; covers isiXhosa and isiZulu together |
| Xhosa | InkubaLM-0.4B | `lelapa/InkubaLM-0.4B` | 0.4B | ⚠️ CC-BY-NC 4.0 | ✅ | Smaller African multilingual alternative |
| Somali | — | — | — | — | ❌ | **No viable open-weight generative LLM for Somali.** AfriBERTa covers Somali but is encoder-only. Gemma-4 is the recommended fallback. |
| Wolof | wolof-qwen-1.5b | `ciskoM/wolof-qwen-1.5b` | 1.5B | ⚠️ CC-BY-NC 4.0 | ✅ | Qwen2.5 fine-tuned on 120k Wolof pairs; non-commercial |
| Wolof | nllb-200-wo-fr-600M | `cifope/nllb-200-wo-fr-distilled-600M` | 600M | ✅ MIT | ✅ | NLLB-200 fine-tuned for Wolof-French translation; MIT |
| Shona | Goldfish Shona | `goldfish-models/sna_latn_10mb` | 39M | ✅ Apache 2.0 | ✅ | Very small (39M); research use only. **Gemma-4 recommended for production.** |

---

## Europe

| Language | Model | HF ID | Params | Licence | ✓ Tokenizer | Notes |
|---|---|---|---|---|---|---|
| French | Lucie-7B | `OpenLLM-France/Lucie-7B` | 7B | ✅ Apache 2.0 | ✅ | OSI-compliant French-primary by OpenLLM-France/LINAGORA |
| French | CroissantLLM Base | `croissantllm/CroissantLLMBase` | 1.3B | ✅ MIT | ✅ | Truly bilingual French-English; 1:1 ratio; custom French tokenizer |
| German | LeoLM-leo-mistral-7b | `LeoLM/leo-mistral-hessianai-7b-chat` | 7B | ✅ Apache 2.0 | ✅ | First open commercial German foundation LLM; Mistral base |
| German | LLäMmlein 1B | `LSX-UniWue/LLaMmlein_1B` | 1B | 🔶 RAIL-M (research only) | ✅ | Trained from scratch on German; research-only licence |
| Spanish | Salamandra 7B | `BSC-LT/salamandra-7b` | 7B | ✅ Apache 2.0 | ✅ | Barcelona Supercomputing Center; 35 EU languages, Spanish upsampled |
| Spanish | LINCE-ZERO | `clibrain/lince-zero` | 7B | ✅ Apache 2.0 | ✅ | Spanish instruction model on Falcon-7B; 80k Spanish examples |
| Portuguese | Gervásio 8B PT-PT | `PORTULAN/gervasio-8b-portuguese-ptpt-decoder` | 8B | ✅ MIT | ✅ | Fine-tuned for European Portuguese specifically |
| Portuguese | Tucano2-qwen-3.7B | `Polygl0t/Tucano2-qwen-3.7B-Instruct` | 3.7B | ✅ Apache 2.0 | ✅ | Covers both PT-PT and PT-BR; Qwen2.5 base; Mar 2026 |
| Italian | LLaMAntino-3-ANITA-8B | `swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA` | 8B | 🔶 Llama 3 Community | ✅ | Highest-performing Italian LLM on benchmarks; DPO-aligned |
| Italian | Llama-3.1-8b-ITA | `DeepMount00/Llama-3.1-8b-ITA` | 8B | 🔶 Llama 3 Community | ✅ | Newer Llama 3.1 base; regularly updated |
| Dutch | Fietje-2 | `BramVanroy/fietje-2` | 2.7B | ✅ MIT | ✅ | Phi-2 fine-tuned on 28B Dutch tokens; Dec 2024 |
| Dutch | Tweety-7B-Dutch | `Tweeties/tweety-7b-dutch-v24a` | 7B | ✅ Apache 2.0 | ✅ | Native Dutch 50k-token vocabulary; Mistral architecture |
| Polish | Bielik 11B v2.3 | `speakleash/Bielik-11B-v2.3-Instruct` | 11B | ✅ Apache 2.0 | ✅ | Most widely adopted Polish open LLM; SpeakLeash |
| Polish | PLLuM 12B Instruct | `CYFRAGOVPL/PLLuM-12B-instruct-2412` | 12B | ✅ Apache 2.0 | ✅ | Polish government-backed; 18B Polish tokens |
| Russian | Vikhr-Nemo-12B | `Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24` | 12B | ✅ Apache 2.0 | ✅ | Best Russian open-weight model; 1M context; Sep 2024 |
| Russian | saiga_llama3_8b | `IlyaGusev/saiga_llama3_8b` | 8B | 🔶 Llama 3 Community | ✅ | SFT + KTO aligned Russian chatbot on Llama-3-8B |
| Ukrainian | MamayLM-Gemma-3-12B | `INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0` | 12B | 🔶 Gemma Terms | ✅ | 75B Ukrainian + English tokens; INSAIT; Apr 2025 |
| Ukrainian | Lapa LLM v0.1.2 | `lapa-llm/lapa-v0.1.2-instruct` | 12B | 🔶 Gemma Terms | ✅ | Custom Ukrainian tokenizer; 1.5x token reduction; UCU/KPI |
| Romanian | LLMic | `faur-ai/LLMic` | 3B | ✅ Apache 2.0 | ✅ | Romanian-English pre-trained from scratch; Jan 2025 |
| Romanian | RoMistral-7b-Instruct | `OpenLLM-Ro/RoMistral-7b-Instruct` | 7B | ⚠️ CC-BY-NC 4.0 | ✅ | Stronger instruction model; non-commercial only |
| Swedish | Viking-7B | `LumiOpen/Viking-7B` | 7B | ✅ Apache 2.0 | ✅ | Nordic specialist; 2T tokens incl. Swedish, Finnish, Danish, Norwegian |
| Swedish | Viking-13B | `LumiOpen/Viking-13B` | 13B | ✅ Apache 2.0 | ✅ | Same training; larger capacity |
| Czech | CSMPT-7B | `BUT-FIT/csmpt7b` | 7B | ✅ Apache 2.0 | ✅ | First Czech-only 7B LLM; 67B Czech tokens; Brno Univ. |
| Czech | CSTinyLlama-1.2B | `BUT-FIT/CSTinyLlama-1.2B` | 1.2B | ✅ Apache 2.0 | ✅ | Czech-specific 64k-vocab tokenizer; lighter option |
| Greek | Meltemi-7B-Instruct-v1.5 | `ilsp/Meltemi-7B-Instruct-v1.5` | 7B | ✅ Apache 2.0 | ✅ | Only Greek-specific LLM; extended Greek tokenizer (1.52 vs 6.80 t/w); ORPO fine-tuned |
| Greek | Meltemi-7B-v1 | `ilsp/Meltemi-7B-v1` | 7B | ✅ Apache 2.0 | ✅ | Base version; 40B Greek tokens |
| Greek | **Krikri-8B-Instruct** ⭐ | `ilsp/Llama-Krikri-8B-Instruct` | 8B | ✅ Apache 2.0 | ✅ | **Task 1B upgrade.** Same ILSP lab; Llama 3.1 base (May 2025); fertility=1.487, vcov=100% (perfect vocab coverage vs Meltemi's 95.42%); DPO-tuned instruct model |

---

## Americas

| Language | Model | HF ID | Params | Licence | ✓ Tokenizer | Notes |
|---|---|---|---|---|---|---|
| Spanish (Latin American) | LatamGPT-70B-SFT | `latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0` | 70B | 🔶 Llama 3.1 Community | ✅ | 297B Latin American tokens + SFT; 75+ institutions; Chile's CENIA |
| Portuguese (Brazilian) | Tucano-2b4-Instruct | `TucanoBR/Tucano-2b4-Instruct` | 2.4B | ✅ Apache 2.0 | ✅ | Natively pre-trained on GigaVerbo (200B PT tokens); Nov 2024 |
| Portuguese (Brazilian) | TeenyTinyLlama-460m | `nicholasKluge/TeenyTinyLlama-460m` | 460M | ✅ Apache 2.0 | ✅ | Early Brazilian PT native-pretrained model; very small |
| Quechua | QuBERTa | `Llamacha/QuBERTa` | 125M | ✅ Apache 2.0 | ✅ | **Encoder-only (RoBERTa).** No generative Quechua LLM exists. Gemma-4 recommended. |
| Nahuatl | t5-small-spanish-nahuatl | `hackathon-pln-es/t5-small-spanish-nahuatl` | 60M | ✅ Apache 2.0 | ✅ | Spanish-Nahuatl translation (seq2seq); Somos NLP 2022 |
| Nahuatl | mt5-large-es-nah | `luisarmando/mt5-large-es-nah` | 1B | 🔶 MPL-2.0 | ✅ | Larger Spanish-Nahuatl translation model; 2024 |
| Haitian Creole | Makandal-v2 | `jsbeaudry/makandal-v2` | 0.6B | ✅ MIT | ✅ | Experimental; high hallucination rate acknowledged. **Gemma-4 recommended for production.** |
| Haitian Creole | Kreyol-MT | `jhu-clsp/kreyol-mt` | 600M | ✅ MIT | ✅ | Translation-only (mBART); NAACL 2024 |

---

## Oceania / Pacific

| Language | Model | HF ID | Params | Licence | ✓ Tokenizer | Notes |
|---|---|---|---|---|---|---|
| Māori | Goldfish mri_latn_10mb | `goldfish-models/mri_latn_10mb` | 39M | ✅ Apache 2.0 | ✅ | Only credible Māori-specific open LLM; GPT-2; Apache 2.0 |
| Samoan | — | — | — | — | ❌ | **No open-weight Samoan LLM found (2022+).** Gemma-4 is the fallback. |
| Hawaiian | Goldfish haw_latn_5mb | `goldfish-models/haw_latn_5mb` | 39M | ✅ Apache 2.0 | ✅ | GPT-2 trained on Hawaiian text; research use; Apache 2.0 |
| Tok Pisin | Goldfish tpi_latn_full | `goldfish-models/tpi_latn_full` | 125M | ✅ Apache 2.0 | ✅ | Trained on full 16MB Tok Pisin corpus; Aug 2024 |

---

## Languages Where Gemma-4 Is Recommended (No Strong Regional Candidate)

| Language | Reason |
|---|---|
| Maithili | Only encoder BERT exists; no open-weight generative LLM |
| Tagalog | Only experimental LoRA adapter; limited training data |
| Somali | No viable open-weight generative LLM found |
| Shona | Only 39M research model; insufficient for production |
| Quechua | Only encoder (RoBERTa) and translation (T5) models |
| Haitian Creole | Models exist but quality too low for production use |
| Samoan | No open-weight model found at all |

---

## Next Step: Phase 2 — Shortlist Testable Candidates
Filter this list to models where `tokenizer_accessible = true` and license is not a blocker.
Run: `experiments/tokenizer_test.py`
Reference baseline: Gemma-4 tokenizer
Test sentences: FLORES-200 dataset
