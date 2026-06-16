"""
Tokenizer evaluation script — global multilingual sweep.

For each language, tests 3 tokenizers:
  - Gemma-4      : production baseline
  - BLOOM        : multilingual baseline
  - mT5          : multilingual baseline
  - Regional candidate (if one exists for that language)

Corpus: FLORES-200 devtest split (~1012 sentences per language).

7 metrics computed per tokenizer × language:
  fertility, compression_ratio, byte_fallback_rate, unknown_rate,
  vocab_coverage, roundtrip_pass_rate, avg_tokens_per_sent

Outputs:
  data/results.csv   — one row per tokenizer × language
  data/summary.json  — per-tokenizer aggregates (unweighted + char-weighted)

Usage:
  python experiments/tokenizer_test.py
  python experiments/tokenizer_test.py --region Indic
  python experiments/tokenizer_test.py --language Tamil --language Hindi
  python experiments/tokenizer_test.py --skip-baselines   # regional candidates only
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

RESULTS_CSV  = DATA_DIR / "results.csv"
SUMMARY_JSON = DATA_DIR / "summary.json"

# ── Tokenizer IDs ─────────────────────────────────────────────────────────────
# Confirm GEMMA4_HF_ID with the team — check https://huggingface.co/google
GEMMA4_HF_ID = "google/gemma-4-27b-it"

BASELINES = {
    "Gemma-4": GEMMA4_HF_ID,
    "BLOOM":   "bigscience/bloom",
    "mT5":     "google/mt5-base",
}

# ── Language config ───────────────────────────────────────────────────────────
# flores_code : FLORES-200 config name (None = language not in FLORES-200, skip)
# candidate   : (display_name, hf_model_id) or None if Gemma-4 is the fallback
LANGUAGES = [
    # ── Indic ────────────────────────────────────────────────────────────────
    {"name": "Hindi",             "region": "Indic",       "flores_code": "hin_Deva", "candidate": ("Airavata-7B",          "ai4bharat/Airavata")},
    {"name": "Bengali",           "region": "Indic",       "flores_code": "ben_Beng", "candidate": ("BanglaLLama-3.1-8B",   "BanglaLLM/BanglaLLama-3.1-8b-bangla-alpaca-orca-instruct-v0.0.1")},
    {"name": "Tamil",             "region": "Indic",       "flores_code": "tam_Taml", "candidate": ("Tamil-Mistral-7B",     "Hemanth-thunder/Tamil-Mistral-7B-Instruct-v0.1")},
    {"name": "Telugu",            "region": "Indic",       "flores_code": "tel_Telu", "candidate": ("Telugu-Llama2-7B",     "Telugu-LLM-Labs/Telugu-Llama2-7B-v0-Base")},
    {"name": "Kannada",           "region": "Indic",       "flores_code": "kan_Knda", "candidate": ("Ambari-7B",            "Cognitive-Lab/Ambari-7B-base-v0.1")},
    {"name": "Malayalam",         "region": "Indic",       "flores_code": "mal_Mlym", "candidate": ("MalayaLLM-Gemma-9B",   "VishnuPJ/MalayaLLM_Gemma_2_9B_Instruct_V1.0")},
    {"name": "Marathi",           "region": "Indic",       "flores_code": "mar_Deva", "candidate": ("MahaMarathi-7B",       "marathi-llm/MahaMarathi-7B-v24.01-Base")},
    {"name": "Gujarati",          "region": "Indic",       "flores_code": "guj_Gujr", "candidate": ("Gujju-Llama-7B",       "sampoorna42/gujju-llama-base-v1.0")},
    {"name": "Punjabi",           "region": "Indic",       "flores_code": "pan_Guru", "candidate": ("Dhee-Qwen3-Punjabi-2B","dheeyantra/dhee-nxtgen-qwen3-punjabi-v2")},
    {"name": "Odia",              "region": "Indic",       "flores_code": "ory_Orya", "candidate": ("Qwen-Odia-7B",         "OdiaGenAI-LLM/qwen_1.5_odia_7b")},
    {"name": "Assamese",          "region": "Indic",       "flores_code": "asm_Beng", "candidate": ("Goldfish-ASM-125M",    "goldfish-models/asm_beng_full")},
    {"name": "Urdu",              "region": "Indic",       "flores_code": "urd_Arab", "candidate": ("Qalb-1.0-8B",          "enstazao/Qalb-1.0-8B-Instruct")},
    {"name": "Nepali",            "region": "Indic",       "flores_code": "npi_Deva", "candidate": ("NEPALI-LLM-9B",        "shivam9980/NEPALI-LLM")},
    {"name": "Sinhala",           "region": "Indic",       "flores_code": "sin_Sinh", "candidate": ("llama3-sinhala-8B",    "ihalage/llama3-sinhala")},
    {"name": "Maithili",          "region": "Indic",       "flores_code": "mai_Deva", "candidate": None},
    # ── Middle East & West Asia ───────────────────────────────────────────────
    {"name": "Arabic",            "region": "Middle East", "flores_code": "arb_Arab", "candidate": ("Jais-2-8B",            "inceptionai/Jais-2-8B-Chat")},
    {"name": "Persian",           "region": "Middle East", "flores_code": "pes_Arab", "candidate": ("Maral-7B",             "MaralGPT/Maral-7B-alpha-1")},
    {"name": "Turkish",           "region": "Middle East", "flores_code": "tur_Latn", "candidate": ("Trendyol-8B",          "Trendyol/Trendyol-LLM-8B-T1")},
    {"name": "Hebrew",            "region": "Middle East", "flores_code": "heb_Hebr", "candidate": ("DictaLM-2.0-7B",       "dicta-il/dictalm2.0-instruct")},
    {"name": "Kurdish",           "region": "Middle East", "flores_code": "kmr_Latn", "candidate": ("Mistral-Nemo-Kurdish", "nazimali/Mistral-Nemo-Kurdish-Instruct")},
    {"name": "Azerbaijani",       "region": "Middle East", "flores_code": "azj_Latn", "candidate": ("AzQ-1.7B",             "karabakh-nlp/AzQ-1.7B")},
    {"name": "Uzbek",             "region": "Middle East", "flores_code": "uzn_Latn", "candidate": ("Mistral-7B-Uz",        "behbudiy/Mistral-7B-Instruct-Uz")},
    {"name": "Kazakh",            "region": "Middle East", "flores_code": "kaz_Cyrl", "candidate": ("KazLLM-8B",            "issai/LLama-3.1-KazLLM-1.0-8B")},
    # ── East & Southeast Asia ─────────────────────────────────────────────────
    {"name": "Chinese",           "region": "East Asia",   "flores_code": "zho_Hans", "candidate": ("ChatGLM3-6B",          "THUDM/chatglm3-6b")},
    {"name": "Japanese",          "region": "East Asia",   "flores_code": "jpn_Jpan", "candidate": ("LLM-jp-3-13B",         "llm-jp/llm-jp-3-13b-instruct")},
    {"name": "Korean",            "region": "East Asia",   "flores_code": "kor_Hang", "candidate": ("Polyglot-Ko-12B",      "EleutherAI/polyglot-ko-12.8b")},
    {"name": "Vietnamese",        "region": "SEA",         "flores_code": "vie_Latn", "candidate": ("Arcee-VyLinh-3B",      "arcee-ai/Arcee-VyLinh")},
    {"name": "Thai",              "region": "SEA",         "flores_code": "tha_Thai", "candidate": ("Typhoon2-7B",          "scb10x/typhoon2-qwen2.5-7b-instruct")},
    {"name": "Indonesian",        "region": "SEA",         "flores_code": "ind_Latn", "candidate": ("Nusantara-7B",         "kalisai/Nusantara-7b-Indo-Chat")},
    {"name": "Malay",             "region": "SEA",         "flores_code": "zsm_Latn", "candidate": ("MaLLaM-5B",            "mesolitica/mallam-5B-4096")},
    {"name": "Tagalog",           "region": "SEA",         "flores_code": "tgl_Latn", "candidate": None},
    {"name": "Burmese",           "region": "SEA",         "flores_code": "mya_Mymr", "candidate": ("Burmese-GPT-1B",       "WYNN747/Burmese-GPT")},
    {"name": "Khmer",             "region": "SEA",         "flores_code": "khm_Khmr", "candidate": ("PrahokBART-62M",       "nict-astrec-att/prahokbart_base")},
    # ── Africa ───────────────────────────────────────────────────────────────
    {"name": "Swahili",           "region": "Africa",      "flores_code": "swh_Latn", "candidate": ("Swahili-Gemma-7B",     "Mollel/Swahili_Gemma")},
    {"name": "Amharic",           "region": "Africa",      "flores_code": "amh_Ethi", "candidate": ("Walia-LLM-7B",         "israel/LLAMA-Walia-II")},
    {"name": "Hausa",             "region": "Africa",      "flores_code": "hau_Latn", "candidate": ("HausaLlama-8B",        "Jacaranda/HausaLlama")},
    {"name": "Yoruba",            "region": "Africa",      "flores_code": "yor_Latn", "candidate": ("YorubaLlama-8B",       "Jacaranda/YorubaLlama")},
    {"name": "Igbo",              "region": "Africa",      "flores_code": "ibo_Latn", "candidate": ("Kakugo-3B-Igbo",       "ptrdvn/kakugo-3B-ibo")},
    {"name": "Zulu",              "region": "Africa",      "flores_code": "zul_Latn", "candidate": ("Xhosa-ZuluLlama3-8B",  "Jacaranda/Xhosa_ZuluLlama3_v1")},
    {"name": "Xhosa",             "region": "Africa",      "flores_code": "xho_Latn", "candidate": ("Xhosa-ZuluLlama3-8B",  "Jacaranda/Xhosa_ZuluLlama3_v1")},
    {"name": "Somali",            "region": "Africa",      "flores_code": "som_Latn", "candidate": None},
    {"name": "Wolof",             "region": "Africa",      "flores_code": "wol_Latn", "candidate": ("Wolof-Qwen-1.5B",      "ciskoM/wolof-qwen-1.5b")},
    {"name": "Shona",             "region": "Africa",      "flores_code": "sna_Latn", "candidate": None},
    # ── Europe ───────────────────────────────────────────────────────────────
    {"name": "French",            "region": "Europe",      "flores_code": "fra_Latn", "candidate": ("Lucie-7B",             "OpenLLM-France/Lucie-7B")},
    {"name": "German",            "region": "Europe",      "flores_code": "deu_Latn", "candidate": ("LeoLM-7B",             "LeoLM/leo-mistral-hessianai-7b-chat")},
    {"name": "Spanish",           "region": "Europe",      "flores_code": "spa_Latn", "candidate": ("Salamandra-7B",        "BSC-LT/salamandra-7b")},
    {"name": "Portuguese",        "region": "Europe",      "flores_code": "por_Latn", "candidate": ("Gervasio-8B",          "PORTULAN/gervasio-8b-portuguese-ptpt-decoder")},
    {"name": "Italian",           "region": "Europe",      "flores_code": "ita_Latn", "candidate": ("LLaMAntino-3-8B",      "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA")},
    {"name": "Dutch",             "region": "Europe",      "flores_code": "nld_Latn", "candidate": ("Fietje-2",             "BramVanroy/fietje-2")},
    {"name": "Polish",            "region": "Europe",      "flores_code": "pol_Latn", "candidate": ("Bielik-11B",           "speakleash/Bielik-11B-v2.3-Instruct")},
    {"name": "Russian",           "region": "Europe",      "flores_code": "rus_Cyrl", "candidate": ("Vikhr-Nemo-12B",       "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24")},
    {"name": "Ukrainian",         "region": "Europe",      "flores_code": "ukr_Cyrl", "candidate": ("MamayLM-12B",          "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0")},
    {"name": "Romanian",          "region": "Europe",      "flores_code": "ron_Latn", "candidate": ("LLMic-3B",             "faur-ai/LLMic")},
    {"name": "Swedish",           "region": "Europe",      "flores_code": "swe_Latn", "candidate": ("Viking-7B",            "LumiOpen/Viking-7B")},
    {"name": "Czech",             "region": "Europe",      "flores_code": "ces_Latn", "candidate": ("CSMPT-7B",             "BUT-FIT/csmpt7b")},
    {"name": "Greek",             "region": "Europe",      "flores_code": "ell_Grek", "candidate": ("Meltemi-7B",           "ilsp/Meltemi-7B-Instruct-v1.5")},
    # ── Americas ─────────────────────────────────────────────────────────────
    {"name": "Lat.Am. Spanish",   "region": "Americas",    "flores_code": "spa_Latn", "candidate": ("LatamGPT-70B",         "latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0")},
    {"name": "Brazilian Portuguese","region": "Americas",  "flores_code": "por_Latn", "candidate": ("Tucano-2b4",           "TucanoBR/Tucano-2b4-Instruct")},
    {"name": "Quechua",           "region": "Americas",    "flores_code": "quy_Latn", "candidate": None},
    {"name": "Nahuatl",           "region": "Americas",    "flores_code": None,        "candidate": None},  # not in FLORES-200
    {"name": "Haitian Creole",    "region": "Americas",    "flores_code": "hat_Latn", "candidate": None},
    # ── Oceania ───────────────────────────────────────────────────────────────
    {"name": "Māori",             "region": "Oceania",     "flores_code": "mri_Latn", "candidate": ("Goldfish-mri-39M",     "goldfish-models/mri_latn_10mb")},
    {"name": "Samoan",            "region": "Oceania",     "flores_code": "smo_Latn", "candidate": None},
    {"name": "Hawaiian",          "region": "Oceania",     "flores_code": None,        "candidate": None},  # not in FLORES-200
    {"name": "Tok Pisin",         "region": "Oceania",     "flores_code": "tpi_Latn", "candidate": ("Goldfish-tpi-125M",    "goldfish-models/tpi_latn_full")},
]

CSV_FIELDS = [
    "tokenizer_name", "language", "region",
    "fertility", "compression_ratio", "byte_fallback_rate", "unknown_rate",
    "vocab_coverage", "roundtrip_pass_rate", "avg_tokens_per_sent",
    "total_tokens", "total_words", "total_chars", "total_sentences",
]

# ── Byte-fallback detection ───────────────────────────────────────────────────

# GPT-2 byte encoder maps the 256 bytes to specific Unicode codepoints.
# Bytes that aren't printable ASCII (33-126, 161-172, 174-255) get mapped to
# codepoints starting at 256. We flag those as byte fallbacks.
def _build_gpt2_byte_fallback_set():
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    extra, n = [], 0
    for b in range(256):
        if b not in visible:
            extra.append(256 + n)
            n += 1
    return {chr(c) for c in extra}

_GPT2_BYTE_FALLBACK_CHARS = _build_gpt2_byte_fallback_set()
_LLAMA_BYTE_RE = re.compile(r"^<0x[0-9a-fA-F]{2}>$")


def _is_byte_fallback(token_str):
    if _LLAMA_BYTE_RE.match(token_str):
        return True
    # GPT-2/Qwen style: single-char token that's in the "extended" byte range
    if len(token_str) == 1 and token_str in _GPT2_BYTE_FALLBACK_CHARS:
        return True
    return False


# ── Metric helpers ────────────────────────────────────────────────────────────

def _roundtrip_ok(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    norm = lambda s: s.replace("​", "").replace(" ", " ").strip()
    return norm(decoded) == norm(text)


def _vocab_coverage(tokenizer, sentences):
    all_text = "\n".join(sentences)
    unique_chars = set(all_text)
    if not unique_chars:
        return 0.0
    covered = sum(
        1 for c in unique_chars
        if len(tokenizer.encode(c, add_special_tokens=False)) == 1
    )
    return round(covered / len(unique_chars) * 100, 2)


def evaluate(tokenizer, sentences):
    total_tokens = total_words = total_chars = 0
    byte_fallback_tokens = unk_tokens = roundtrip_pass = 0
    total_sentences = len(sentences)
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for sent in sentences:
        words = sent.split()
        ids = tokenizer.encode(sent, add_special_tokens=False)
        toks = tokenizer.convert_ids_to_tokens(ids)

        total_words   += len(words)
        total_chars   += len(sent)
        total_tokens  += len(ids)

        for tok, id_ in zip(toks, ids):
            if tok and _is_byte_fallback(tok):
                byte_fallback_tokens += 1
            if unk_id is not None and id_ == unk_id:
                unk_tokens += 1

        if _roundtrip_ok(tokenizer, sent):
            roundtrip_pass += 1

    vcov = _vocab_coverage(tokenizer, sentences)

    return {
        "fertility":           round(total_tokens / total_words if total_words else 0, 3),
        "compression_ratio":   round(total_chars / total_tokens if total_tokens else 0, 3),
        "byte_fallback_rate":  round(byte_fallback_tokens / total_tokens * 100 if total_tokens else 0, 2),
        "unknown_rate":        round(unk_tokens / total_tokens * 100 if total_tokens else 0, 2),
        "vocab_coverage":      vcov,
        "roundtrip_pass_rate": round(roundtrip_pass / total_sentences * 100 if total_sentences else 0, 2),
        "avg_tokens_per_sent": round(total_tokens / total_sentences if total_sentences else 0, 2),
        "total_tokens":        total_tokens,
        "total_words":         total_words,
        "total_chars":         total_chars,
        "total_sentences":     total_sentences,
    }


# ── FLORES-200 loader ─────────────────────────────────────────────────────────

_flores_cache = {}

def load_flores(flores_code):
    if flores_code in _flores_cache:
        return _flores_cache[flores_code]
    from datasets import load_dataset
    print(f"  Loading FLORES-200 [{flores_code}] ...", flush=True)
    ds = load_dataset("facebook/flores", flores_code, split="devtest", trust_remote_code=True)
    sentences = [row["sentence"] for row in ds]
    _flores_cache[flores_code] = sentences
    return sentences


# ── Tokenizer loader ──────────────────────────────────────────────────────────

def load_tokenizer(hf_id):
    from transformers import AutoTokenizer
    kwargs = {"use_fast": True}
    # Some models require trust_remote_code (e.g. ChatGLM)
    if any(x in hf_id.lower() for x in ("chatglm", "moss")):
        kwargs["trust_remote_code"] = True
    return AutoTokenizer.from_pretrained(hf_id, **kwargs)


# ── Summary computation ───────────────────────────────────────────────────────

def compute_summary(rows):
    from collections import defaultdict
    by_tok = defaultdict(list)
    for r in rows:
        by_tok[r["tokenizer_name"]].append(r)

    numeric = [
        "fertility", "compression_ratio", "byte_fallback_rate", "unknown_rate",
        "vocab_coverage", "roundtrip_pass_rate", "avg_tokens_per_sent",
    ]
    summary = {}
    for tok, tok_rows in by_tok.items():
        total_chars = sum(r["total_chars"] for r in tok_rows)
        unweighted, weighted = {}, {}
        for m in numeric:
            vals = [r[m] for r in tok_rows]
            unweighted[f"avg_{m}"] = round(sum(vals) / len(vals), 3)
            # character-weighted average
            w_sum = sum(r[m] * r["total_chars"] for r in tok_rows)
            weighted[f"weighted_avg_{m}"] = round(w_sum / total_chars if total_chars else 0, 3)
        summary[tok] = {
            "languages_tested":  len(tok_rows),
            "total_chars":       total_chars,
            **unweighted,
            **weighted,
        }
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region",   nargs="+", help="Filter to one or more regions")
    p.add_argument("--language", nargs="+", help="Filter to specific language names")
    p.add_argument("--skip-baselines", action="store_true",
                   help="Only run regional candidate tokenizers (skip Gemma-4/BLOOM/mT5)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run without loading any tokenizers")
    return p.parse_args()


def main():
    args = parse_args()

    langs = LANGUAGES
    if args.region:
        langs = [l for l in langs if l["region"] in args.region]
    if args.language:
        langs = [l for l in langs if l["name"] in args.language]

    # Drop languages not in FLORES-200
    runnable = [l for l in langs if l["flores_code"] is not None]
    skipped  = [l for l in langs if l["flores_code"] is None]

    if skipped:
        print(f"Skipping {len(skipped)} language(s) not in FLORES-200: "
              f"{', '.join(l['name'] for l in skipped)}")

    if args.dry_run:
        print("\nDry run — would evaluate:")
        for lang in runnable:
            cand = lang["candidate"]
            tok_list = ([] if args.skip_baselines else list(BASELINES.keys()))
            if cand:
                tok_list.append(cand[0])
            print(f"  {lang['name']:25s} ({lang['region']}) → {', '.join(tok_list)}")
        return

    rows = []
    total_combos = sum(
        (0 if args.skip_baselines else len(BASELINES)) + (1 if l["candidate"] else 0)
        for l in runnable
    )
    done = 0

    for lang in runnable:
        lang_name  = lang["name"]
        region     = lang["region"]
        flores_code = lang["flores_code"]

        print(f"\n{'─'*60}")
        print(f"Language: {lang_name}  [{region}]  FLORES: {flores_code}")

        try:
            sentences = load_flores(flores_code)
        except Exception as e:
            print(f"  ERROR loading FLORES-200 for {flores_code}: {e}")
            continue

        tokenizers_to_run = {}
        if not args.skip_baselines:
            tokenizers_to_run.update(BASELINES)
        if lang["candidate"]:
            cand_name, cand_id = lang["candidate"]
            tokenizers_to_run[cand_name] = cand_id

        for tok_name, hf_id in tokenizers_to_run.items():
            done += 1
            print(f"  [{done}/{total_combos}] {tok_name} ({hf_id})", end=" ... ", flush=True)
            try:
                tokenizer = load_tokenizer(hf_id)
                metrics   = evaluate(tokenizer, sentences)
                del tokenizer  # free memory immediately
                row = {
                    "tokenizer_name": tok_name,
                    "language":       lang_name,
                    "region":         region,
                    **metrics,
                }
                rows.append(row)
                print(f"fertility={metrics['fertility']}  vcov={metrics['vocab_coverage']}%  "
                      f"roundtrip={metrics['roundtrip_pass_rate']}%")
            except Exception as e:
                print(f"FAILED — {e}")

    if not rows:
        print("\nNo results collected. Exiting.")
        sys.exit(1)

    # Write results.csv
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {RESULTS_CSV}")

    # Write summary.json
    summary = compute_summary(rows)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary → {SUMMARY_JSON}")

    # Quick terminal summary
    print("\n── Unweighted avg fertility per tokenizer ──")
    for tok, stats in sorted(summary.items(), key=lambda x: x[1]["avg_fertility"]):
        print(f"  {tok:30s}  fertility={stats['avg_fertility']}  "
              f"vcov={stats['avg_vocab_coverage']}%")


if __name__ == "__main__":
    main()
