"""
European Language Expansion — tokenizer evaluation.

Tests 8 pan-European challenger models against Gemma-4 baseline across
30 European languages using FLORES-200 devtest sentences.

Gate (must pass ALL three to proceed to qualitative eval):
  fertility         < Gemma-4 fertility for that language
  vocab_coverage    ≥ 80 %
  roundtrip_pass_rate ≥ 95 %

Outputs:
  data/european_results.csv   — one row per model × language
  data/european_summary.json  — gate results + per-model pass counts

Usage:
  python experiments/european_tokenizer_test.py
  python experiments/european_tokenizer_test.py --append
  python experiments/european_tokenizer_test.py --model Teuken-7B EuroLLM-22B
  python experiments/european_tokenizer_test.py --language French German
  python experiments/european_tokenizer_test.py --dry-run
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

RESULTS_CSV  = DATA_DIR / "european_results.csv"
SUMMARY_JSON = DATA_DIR / "european_summary.json"

# Production model is gemma-4-26B-A4B-it; all Gemma-4 variants share the same tokenizer.
# gemma-4-12B-it is publicly accessible and sufficient for tokenizer-only eval.
GEMMA4_HF_ID = "google/gemma-4-12B-it"

# 8 European challenger models
CHALLENGERS = {
    "Llama-3.3-70B":       "meta-llama/Llama-3.3-70B-Instruct",
    "Mistral-Small-3.2":   "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "Teuken-7B":           "openGPT-X/Teuken-7B-instruct-v0.6",
    "EuroLLM-22B":         "utter-project/EuroLLM-22B-Instruct-2512",
    "SauerkrautLM-70B":    "VAGOsolutions/Llama-3.1-SauerkrautLM-70b-Instruct",
    "GEITje-7B":           "BramVanroy/GEITje-7B-ultra",
    "TildeOpen-30B":       "TildeAI/TildeOpen-30b",
    "Aya-Vision-32B":      "CohereLabs/aya-vision-32b",
}

# Models that require trust_remote_code for their tokenizer
TRUST_REMOTE_CODE = {"Teuken-7B", "Aya-Vision-32B"}

# 30 target European languages + FLORES-200 codes
LANGUAGES = [
    {"name": "French",      "flores_code": "fra_Latn"},
    {"name": "German",      "flores_code": "deu_Latn"},
    {"name": "Spanish",     "flores_code": "spa_Latn"},
    {"name": "Italian",     "flores_code": "ita_Latn"},
    {"name": "Portuguese",  "flores_code": "por_Latn"},
    {"name": "Dutch",       "flores_code": "nld_Latn"},
    {"name": "Polish",      "flores_code": "pol_Latn"},
    {"name": "Romanian",    "flores_code": "ron_Latn"},
    {"name": "Ukrainian",   "flores_code": "ukr_Cyrl"},
    {"name": "Swedish",     "flores_code": "swe_Latn"},
    {"name": "Czech",       "flores_code": "ces_Latn"},
    {"name": "Greek",       "flores_code": "ell_Grek"},
    {"name": "Russian",     "flores_code": "rus_Cyrl"},
    {"name": "Danish",      "flores_code": "dan_Latn"},
    {"name": "Finnish",     "flores_code": "fin_Latn"},
    {"name": "Hungarian",   "flores_code": "hun_Latn"},
    {"name": "Turkish",     "flores_code": "tur_Latn"},
    {"name": "Croatian",    "flores_code": "hrv_Latn"},
    {"name": "Slovak",      "flores_code": "slk_Latn"},
    {"name": "Slovenian",   "flores_code": "slv_Latn"},
    {"name": "Bulgarian",   "flores_code": "bul_Cyrl"},
    {"name": "Lithuanian",  "flores_code": "lit_Latn"},
    {"name": "Latvian",     "flores_code": "lvs_Latn"},
    {"name": "Estonian",    "flores_code": "est_Latn"},
    {"name": "Irish",       "flores_code": "gle_Latn"},
    {"name": "Norwegian",   "flores_code": "nob_Latn"},
    {"name": "Maltese",     "flores_code": "mlt_Latn"},
    {"name": "Serbian",     "flores_code": "srp_Cyrl"},
    {"name": "Icelandic",   "flores_code": "isl_Latn"},
    {"name": "Albanian",    "flores_code": "als_Latn"},
]

CSV_FIELDS = [
    "model", "language", "flores_code",
    "fertility", "compression_ratio", "byte_fallback_rate", "unknown_rate",
    "vocab_coverage", "roundtrip_pass_rate", "avg_tokens_per_sent",
    "total_tokens", "total_words", "total_chars", "total_sentences",
]

GATE_FERTILITY_THRESHOLD  = None   # < gemma4 fertility (computed per language)
GATE_VOCAB_COV_MIN        = 80.0   # ≥ 80%
GATE_ROUNDTRIP_MIN        = 95.0   # ≥ 95%


# ── Byte-fallback detection ────────────────────────────────────────────────────

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
    if len(token_str) == 1 and token_str in _GPT2_BYTE_FALLBACK_CHARS:
        return True
    return False


# ── Metric helpers ─────────────────────────────────────────────────────────────

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
        ids   = tokenizer.encode(sent, add_special_tokens=False)
        toks  = tokenizer.convert_ids_to_tokens(ids)

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
        "fertility":            round(total_tokens / total_words if total_words else 0, 3),
        "compression_ratio":    round(total_chars  / total_tokens if total_tokens else 0, 3),
        "byte_fallback_rate":   round(byte_fallback_tokens / total_tokens * 100 if total_tokens else 0, 2),
        "unknown_rate":         round(unk_tokens  / total_tokens * 100 if total_tokens else 0, 2),
        "vocab_coverage":       vcov,
        "roundtrip_pass_rate":  round(roundtrip_pass / total_sentences * 100 if total_sentences else 0, 2),
        "avg_tokens_per_sent":  round(total_tokens / total_sentences if total_sentences else 0, 2),
        "total_tokens":         total_tokens,
        "total_words":          total_words,
        "total_chars":          total_chars,
        "total_sentences":      total_sentences,
    }


# ── FLORES-200 loader ──────────────────────────────────────────────────────────

_flores_cache = {}

def load_flores(flores_code):
    if flores_code in _flores_cache:
        return _flores_cache[flores_code]
    import os
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")
    print(f"  Loading FLORES-200 [{flores_code}] ...", flush=True)
    ds = load_dataset("facebook/flores", flores_code, split="devtest", token=token)
    sentences = [row["sentence"] for row in ds]
    _flores_cache[flores_code] = sentences
    return sentences


# ── Tokenizer loader ───────────────────────────────────────────────────────────

def load_tokenizer(hf_id, trust_remote_code=False):
    import os
    from transformers import AutoTokenizer
    token = os.environ.get("HF_TOKEN")
    return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code, token=token)


# ── Gate evaluation ────────────────────────────────────────────────────────────

def evaluate_gate(rows):
    """
    For each (model, language) row, check if it passes the gate relative to Gemma-4.
    Returns dict: {(model, language): {"passes": bool, "reason": str, "gemma4_fertility": float}}
    """
    gemma4 = {r["language"]: r for r in rows if r["model"] == "Gemma-4"}
    results = {}
    for r in rows:
        if r["model"] == "Gemma-4":
            continue
        lang = r["language"]
        g4 = gemma4.get(lang)
        if g4 is None:
            results[(r["model"], lang)] = {"passes": False, "reason": "no Gemma-4 baseline", "gemma4_fertility": None}
            continue
        g4_fert = g4["fertility"]
        challenger_fert = r["fertility"]
        vcov = r["vocab_coverage"]
        rtrip = r["roundtrip_pass_rate"]

        fails = []
        if challenger_fert >= g4_fert:
            fails.append(f"fertility {challenger_fert} ≥ Gemma-4 {g4_fert}")
        if vcov < GATE_VOCAB_COV_MIN:
            fails.append(f"vocab_coverage {vcov}% < {GATE_VOCAB_COV_MIN}%")
        if rtrip < GATE_ROUNDTRIP_MIN:
            fails.append(f"roundtrip {rtrip}% < {GATE_ROUNDTRIP_MIN}%")

        results[(r["model"], lang)] = {
            "passes": len(fails) == 0,
            "reason": "PASS" if not fails else "; ".join(fails),
            "gemma4_fertility": g4_fert,
        }
    return results


# ── Summary / report ───────────────────────────────────────────────────────────

def print_gate_matrix(gate_results, model_names, lang_names):
    col_w = 18
    header = f"{'Language':<20}" + "".join(f"{m[:col_w]:<{col_w}}" for m in model_names)
    print("\n" + "─" * len(header))
    print("GATE RESULTS (✓ = passes all 3 conditions, ✗ = fails one or more)")
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for lang in lang_names:
        row = f"{lang:<20}"
        for model in model_names:
            r = gate_results.get((model, lang))
            if r is None:
                row += f"{'—':<{col_w}}"
            elif r["passes"]:
                row += f"{'✓ PASS':<{col_w}}"
            else:
                row += f"{'✗ FAIL':<{col_w}}"
        print(row)
    print("─" * len(header))

    # Per-model pass count
    print("\nPass counts per model:")
    for model in model_names:
        passes = sum(1 for lang in lang_names if gate_results.get((model, lang), {}).get("passes"))
        print(f"  {model:<30} {passes}/{len(lang_names)} languages")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    nargs="+", help="Filter to specific model names (display names)")
    p.add_argument("--language", nargs="+", help="Filter to specific language names")
    p.add_argument("--append",   action="store_true",
                   help="Load existing results and skip already-completed model×language pairs")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print what would run without loading any tokenizers")
    p.add_argument("--gate-only", action="store_true",
                   help="Skip running; just re-evaluate gate from existing results CSV")
    return p.parse_args()


def _cast_row(r):
    int_fields   = {"total_tokens", "total_words", "total_chars", "total_sentences"}
    float_fields = {"fertility", "compression_ratio", "byte_fallback_rate", "unknown_rate",
                    "vocab_coverage", "roundtrip_pass_rate", "avg_tokens_per_sent"}
    out = {}
    for k, v in r.items():
        if k in int_fields:
            out[k] = int(v)
        elif k in float_fields:
            out[k] = float(v)
        else:
            out[k] = v
    return out


OLD_RESULTS_CSV = ROOT / "data" / "results.csv"


def _import_gemma4_baselines(rows, existing, lang_names):
    """
    Pull Gemma-4 rows from the Task 1 results.csv for any European language
    not already in the current european_results.csv.
    """
    if not OLD_RESULTS_CSV.exists():
        return
    imported = 0
    with open(OLD_RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["tokenizer_name"] != "Gemma-4":
                continue
            lang = r["language"]
            if lang not in lang_names:
                continue
            if ("Gemma-4", lang) in existing:
                continue
            # Translate old CSV schema → european_results schema
            row = {
                "model":       "Gemma-4",
                "language":    lang,
                "flores_code": next(
                    (l["flores_code"] for l in LANGUAGES if l["name"] == lang), ""
                ),
                "fertility":            float(r["fertility"]),
                "compression_ratio":    float(r["compression_ratio"]),
                "byte_fallback_rate":   float(r["byte_fallback_rate"]),
                "unknown_rate":         float(r["unknown_rate"]),
                "vocab_coverage":       float(r["vocab_coverage"]),
                "roundtrip_pass_rate":  float(r["roundtrip_pass_rate"]),
                "avg_tokens_per_sent":  float(r["avg_tokens_per_sent"]),
                "total_tokens":         int(r["total_tokens"]),
                "total_words":          int(r["total_words"]),
                "total_chars":          int(r["total_chars"]),
                "total_sentences":      int(r["total_sentences"]),
            }
            rows.append(row)
            existing.add(("Gemma-4", lang))
            imported += 1
    if imported:
        print(f"Imported {imported} Gemma-4 baselines from Task 1 results.csv.")


def main():
    args = parse_args()

    # All models to test: Gemma-4 baseline + 8 challengers
    all_models = {"Gemma-4": GEMMA4_HF_ID, **CHALLENGERS}

    models_to_run = all_models
    if args.model:
        # Always include Gemma-4 so the gate can be computed
        models_to_run = {"Gemma-4": GEMMA4_HF_ID}
        for name in args.model:
            if name in all_models:
                models_to_run[name] = all_models[name]
            else:
                print(f"WARNING: unknown model '{name}'. Known: {list(all_models)}")

    langs_to_run = LANGUAGES
    if args.language:
        langs_to_run = [l for l in LANGUAGES if l["name"] in args.language]
        if not langs_to_run:
            print(f"No matching languages. Available: {[l['name'] for l in LANGUAGES]}")
            sys.exit(1)

    # Load existing results
    _INT_FIELDS   = {"total_tokens", "total_words", "total_chars", "total_sentences"}
    _FLOAT_FIELDS = {"fertility", "compression_ratio", "byte_fallback_rate", "unknown_rate",
                     "vocab_coverage", "roundtrip_pass_rate", "avg_tokens_per_sent"}

    existing = set()
    rows = []
    if (args.append or args.gate_only) and RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(_cast_row(r))
                existing.add((r["model"], r["language"]))
        print(f"Loaded {len(rows)} existing rows.")

    # Pull Gemma-4 baselines from old Task 1 results.csv (avoids re-downloading gated model)
    lang_names = {l["name"] for l in LANGUAGES}
    _import_gemma4_baselines(rows, existing, lang_names)

    if args.gate_only:
        gate = evaluate_gate(rows)
        model_names = [m for m in all_models if m != "Gemma-4"]
        lang_names  = [l["name"] for l in LANGUAGES]
        print_gate_matrix(gate, model_names, lang_names)
        return

    if args.dry_run:
        print("\nDry run — would evaluate:")
        total = 0
        for lang in langs_to_run:
            for model_name in models_to_run:
                key = (model_name, lang["name"])
                status = "skip" if key in existing else "run"
                print(f"  {model_name:<30} × {lang['name']:<15} [{status}]")
                if status == "run":
                    total += 1
        print(f"\n{total} evaluations to run.")
        return

    total = sum(
        1 for lang in langs_to_run for m in models_to_run
        if (m, lang["name"]) not in existing
    )
    done = 0

    for lang in langs_to_run:
        lang_name   = lang["name"]
        flores_code = lang["flores_code"]

        print(f"\n{'─'*60}")
        print(f"Language: {lang_name}  [{flores_code}]")

        try:
            sentences = load_flores(flores_code)
        except Exception as e:
            print(f"  ERROR loading FLORES-200 for {flores_code}: {e}")
            continue

        for model_name, hf_id in models_to_run.items():
            key = (model_name, lang_name)
            done += 1
            if key in existing:
                print(f"  [{done}/{total}] {model_name} — already done, skipping")
                continue

            trust = model_name in TRUST_REMOTE_CODE
            print(f"  [{done}/{total}] {model_name} ({hf_id})", end=" ... ", flush=True)
            try:
                tokenizer = load_tokenizer(hf_id, trust_remote_code=trust)
                metrics   = evaluate(tokenizer, sentences)
                del tokenizer
                row = {
                    "model":       model_name,
                    "language":    lang_name,
                    "flores_code": flores_code,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"fertility={metrics['fertility']}  "
                    f"vcov={metrics['vocab_coverage']}%  "
                    f"roundtrip={metrics['roundtrip_pass_rate']}%"
                )
                # Write incrementally so partial runs are saved
                _write_csv(rows)
            except Exception as e:
                print(f"FAILED — {e}")

    if not rows:
        print("\nNo results collected. Exiting.")
        sys.exit(1)

    _write_csv(rows)
    print(f"\nWrote {len(rows)} rows → {RESULTS_CSV}")

    # Gate evaluation
    gate = evaluate_gate(rows)
    model_names = [m for m in all_models if m != "Gemma-4"]
    lang_names  = [l["name"] for l in LANGUAGES]
    print_gate_matrix(gate, model_names, lang_names)

    # Write summary JSON
    summary = {
        "gate_results": {
            f"{model}|{lang}": v
            for (model, lang), v in gate.items()
        },
        "pass_counts": {
            model: sum(1 for lang in lang_names if gate.get((model, lang), {}).get("passes"))
            for model in model_names
        },
        "qualitative_eval_queue": [
            {"model": model, "language": lang, "gemma4_fertility": gate[(model, lang)]["gemma4_fertility"]}
            for (model, lang), v in gate.items()
            if v["passes"]
        ],
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary → {SUMMARY_JSON}")

    winners = summary["qualitative_eval_queue"]
    print(f"\n{len(winners)} (model, language) pairs proceed to qualitative eval.")


def _write_csv(rows):
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
