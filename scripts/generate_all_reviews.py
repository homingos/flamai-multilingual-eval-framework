"""
Batch-generate all review HTML files from data/modal_cache/ into data/review_static/.
Run this locally before deploying to Vercel.

Usage:
    python scripts/generate_all_reviews.py [--force]
"""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
CACHE_DIR   = REPO_ROOT / "data" / "modal_cache"
STATIC_DIR  = REPO_ROOT / "data" / "review_static"
QUAL_JSON   = REPO_ROOT / "data" / "qualitative_results.json"
GEN_SCRIPT  = REPO_ROOT / "scripts" / "generate_review.py"

SLUG_META = {
    "tamil":                    ("Tamil",                "Tamil-Mistral-7B"),
    "sarvam-m-tamil":           ("Tamil",                "Sarvam-M-24B"),
    "marathi":                  ("Marathi",              "MahaMarathi-7B"),
    "sarvam-m-marathi":         ("Marathi",              "Sarvam-M-24B"),
    "kannada":                  ("Kannada",              "Ambari-7B"),
    "sarvam-m-kannada":         ("Kannada",              "Sarvam-M-24B"),
    "gujarati":                 ("Gujarati",             "Gujju-Llama-7B"),
    "sarvam-m-gujarati":        ("Gujarati",             "Sarvam-M-24B"),
    "arabic":                   ("Arabic",               "Jais-2-8B"),
    "jais-70b":                 ("Arabic",               "Jais-2-70B-Chat"),
    "hebrew":                   ("Hebrew",               "DictaLM-3.0-Nemotron-12B"),
    "korean":                   ("Korean",               "Polyglot-Ko-12B"),
    "exaone-korean":            ("Korean",               "EXAONE-3.5-32B-Instruct"),
    "malay":                    ("Malay",                "MaLLaM-5B"),
    "swahili":                  ("Swahili",              "Swahili-Gemma-7B"),
    "amharic":                  ("Amharic",              "Walia-LLM-7B"),
    "greek":                    ("Greek",                "Meltemi-7B"),
    "krikri-greek":             ("Greek",                "Krikri-8B-Instruct"),
    "french":                   ("French",               "Lucie-7B-Instruct-v1.1"),
    "swedish":                  ("Swedish",              "Viking-7B"),
    "czech":                    ("Czech",                "CSMPT-7B"),
    "maori":                    ("Māori",                "Goldfish-mri-39M"),
    "tok_pisin":                ("Tok Pisin",            "Goldfish-tpi-125M"),
    "brazilian_portuguese":     ("Brazilian Portuguese", "Tucano-2b4"),
}


def main():
    force = "--force" in sys.argv

    data = json.loads(QUAL_JSON.read_text())
    evaluations = data.get("evaluations", [])

    done = skipped = failed = missing = 0

    for ev in evaluations:
        run_id = ev.get("run_id")
        if not run_id:
            continue

        run_dir = CACHE_DIR / run_id
        if not run_dir.exists():
            print(f"  ✗ {run_id}  — cache dir missing")
            missing += 1
            continue

        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            print(f"  ✗ {run_id}  — meta.json missing")
            missing += 1
            continue

        meta = json.loads(meta_path.read_text())
        slug = meta.get("slug")
        task = meta.get("task")
        if not slug or not task:
            print(f"  ✗ {run_id}  — slug/task missing in meta.json")
            missing += 1
            continue

        verdicts_path = run_dir / "verdicts.jsonl"
        gemma4_path   = run_dir / "gemma4.jsonl"
        regional_path = run_dir / "regional.jsonl"

        if not (verdicts_path.exists() and gemma4_path.exists() and regional_path.exists()):
            print(f"  ✗ {run_id}  {slug}/{task}  — JSONL files missing (skipped/failed run)")
            missing += 1
            continue

        out_dir  = STATIC_DIR / f"{slug}_{task}_{run_id}"
        out_file = out_dir / "review.html"

        if out_file.exists() and not force:
            print(f"  — {run_id}  {slug}/{task}  already generated")
            skipped += 1
            continue

        _, model_display = SLUG_META.get(slug, (slug, slug))
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  ⟳ {run_id}  {slug}/{task}  generating…", end="", flush=True)
        result = subprocess.run(
            [
                sys.executable, str(GEN_SCRIPT),
                "--run-id", run_id,
                "--slug", slug,
                "--task", task,
                "--model-display", model_display,
                "--output-dir", str(out_dir),
                "--local-gemma4",   str(gemma4_path),
                "--local-regional", str(regional_path),
                "--local-verdicts", str(verdicts_path),
            ],
            capture_output=True, text=True
        )

        if result.returncode == 0 and out_file.exists():
            size_kb = out_file.stat().st_size // 1024
            print(f"  ✓  ({size_kb}KB)")
            done += 1
        else:
            print(f"  FAILED")
            if result.stderr:
                print("    ", result.stderr[:300])
            failed += 1

    print(f"\nDone. {done} generated, {skipped} already existed, {missing} missing data, {failed} failed.")


if __name__ == "__main__":
    main()
