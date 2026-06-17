"""
Translation Dataset Creator
============================
Builds a translation evaluation dataset for the 17 tokenizer-winner languages
using FLORES-200 sentences (already used in Task 1 tokenizer evaluation).

Each language gets ~2024 samples:
  - ~1012 en → target  (English source, target-language reference)
  - ~1012 target → en  (target-language source, English reference)

Output: data/datasets/translation/<language_slug>/samples.jsonl
        data/datasets/translation/meta.json  (summary across all languages)

Usage:
  python experiments/create_translation_dataset.py
  python experiments/create_translation_dataset.py --language Tamil
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "datasets" / "translation"

HF_TOKEN = os.getenv("HF_TOKEN")

# ── 17 winner languages with their FLORES-200 codes ──────────────────────────
WINNERS = [
    {"name": "Tamil",                "slug": "tamil",       "flores": "tam_Taml", "region": "Indic",      "winner_model": "Tamil-Mistral-7B"},
    {"name": "Marathi",              "slug": "marathi",     "flores": "mar_Deva", "region": "Indic",      "winner_model": "MahaMarathi-7B"},
    {"name": "Kannada",              "slug": "kannada",     "flores": "kan_Knda", "region": "Indic",      "winner_model": "Ambari-7B"},
    {"name": "Gujarati",             "slug": "gujarati",    "flores": "guj_Gujr", "region": "Indic",      "winner_model": "Gujju-Llama-7B"},
    {"name": "Arabic",               "slug": "arabic",      "flores": "arb_Arab", "region": "Middle East","winner_model": "Jais-2-8B"},
    {"name": "Hebrew",               "slug": "hebrew",      "flores": "heb_Hebr", "region": "Middle East","winner_model": "DictaLM-2.0-7B"},
    {"name": "Korean",               "slug": "korean",      "flores": "kor_Hang", "region": "East Asia",  "winner_model": "Polyglot-Ko-12B"},
    {"name": "Malay",                "slug": "malay",       "flores": "zsm_Latn", "region": "SEA",        "winner_model": "MaLLaM-5B"},
    {"name": "Swahili",              "slug": "swahili",     "flores": "swh_Latn", "region": "Africa",     "winner_model": "Swahili-Gemma-7B"},
    {"name": "Amharic",              "slug": "amharic",     "flores": "amh_Ethi", "region": "Africa",     "winner_model": "Walia-LLM-7B"},
    {"name": "French",               "slug": "french",      "flores": "fra_Latn", "region": "Europe",     "winner_model": "Lucie-7B"},
    {"name": "Swedish",              "slug": "swedish",     "flores": "swe_Latn", "region": "Europe",     "winner_model": "Viking-7B"},
    {"name": "Czech",                "slug": "czech",       "flores": "ces_Latn", "region": "Europe",     "winner_model": "CSMPT-7B"},
    {"name": "Greek",                "slug": "greek",       "flores": "ell_Grek", "region": "Europe",     "winner_model": "Meltemi-7B"},
    {"name": "Brazilian Portuguese", "slug": "brazilian_portuguese", "flores": "por_Latn", "region": "Americas", "winner_model": "Tucano-2b4"},
    {"name": "Māori",                "slug": "maori",       "flores": "mri_Latn", "region": "Oceania",    "winner_model": "Goldfish-mri-39M"},
    {"name": "Tok Pisin",            "slug": "tok_pisin",   "flores": "tpi_Latn", "region": "Oceania",    "winner_model": "Goldfish-tpi-125M"},
]

ENGLISH_FLORES = "eng_Latn"


def load_flores(flores_code: str):
    from datasets import load_dataset
    try:
        ds = load_dataset("facebook/flores", flores_code, split="devtest", token=HF_TOKEN)
    except Exception as e:
        raise RuntimeError(
            f"Could not load FLORES-200 config '{flores_code}' from HuggingFace Hub.\n"
            f"  Original error: {e}\n"
            f"  Fix: ensure you have internet access (or set HF_TOKEN env var if the repo is gated).\n"
            f"  The dataset is ~1MB per language config and downloads in seconds."
        ) from e
    return [{"flores_id": row["id"], "text": row["sentence"]} for row in ds]


def build_samples(lang: dict, en_rows: list, tgt_rows: list) -> list:
    samples = []
    lang_name = lang["name"]
    slug = lang["slug"]

    # en → target
    for row in en_rows:
        ref = next((r["text"] for r in tgt_rows if r["flores_id"] == row["flores_id"]), None)
        if ref is None:
            continue
        samples.append({
            "id": f"trans_{slug}_en_tgt_{row['flores_id']:04d}",
            "language": lang_name,
            "region": lang["region"],
            "winner_model": lang["winner_model"],
            "direction": "en→target",
            "source_lang": "English",
            "target_lang": lang_name,
            "source": row["text"],
            "reference": ref,
            "flores_id": row["flores_id"],
        })

    # target → en
    for row in tgt_rows:
        ref = next((r["text"] for r in en_rows if r["flores_id"] == row["flores_id"]), None)
        if ref is None:
            continue
        samples.append({
            "id": f"trans_{slug}_tgt_en_{row['flores_id']:04d}",
            "language": lang_name,
            "region": lang["region"],
            "winner_model": lang["winner_model"],
            "direction": "target→en",
            "source_lang": lang_name,
            "target_lang": "English",
            "source": row["text"],
            "reference": ref,
            "flores_id": row["flores_id"],
        })

    return samples


def process_language(lang: dict, en_rows: list):
    print(f"  Loading FLORES-200 [{lang['flores']}] for {lang['name']}…", end=" ", flush=True)
    try:
        tgt_rows = load_flores(lang["flores"])
    except Exception as e:
        print(f"FAILED — {e}")
        return None

    samples = build_samples(lang, en_rows, tgt_rows)
    out_dir = OUT_DIR / lang["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "samples.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    meta = {
        "language": lang["name"],
        "slug": lang["slug"],
        "region": lang["region"],
        "winner_model": lang["winner_model"],
        "flores_code": lang["flores"],
        "total_samples": len(samples),
        "en_to_target": len([s for s in samples if s["direction"] == "en→target"]),
        "target_to_en": len([s for s in samples if s["direction"] == "target→en"]),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"{len(samples)} samples ({meta['en_to_target']} en→tgt, {meta['target_to_en']} tgt→en)")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", help="Process only this language (e.g. Tamil)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    langs = WINNERS
    if args.language:
        langs = [l for l in WINNERS if l["name"].lower() == args.language.lower()]
        if not langs:
            print(f"Unknown language '{args.language}'. Available: {[l['name'] for l in WINNERS]}")
            sys.exit(1)

    print("Loading English FLORES-200 sentences…")
    try:
        en_rows = load_flores(ENGLISH_FLORES)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("\nThe instruction following dataset (17,000 samples) was already generated separately.")
        print("Run this script again once you have internet access to HuggingFace Hub.")
        sys.exit(1)
    print(f"  {len(en_rows)} English sentences loaded.\n")

    all_meta = []
    for lang in langs:
        meta = process_language(lang, en_rows)
        if meta:
            all_meta.append(meta)

    # Write overall summary
    summary = {
        "total_languages": len(all_meta),
        "total_samples": sum(m["total_samples"] for m in all_meta),
        "languages": all_meta,
    }
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {summary['total_languages']} languages · {summary['total_samples']:,} total samples")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
