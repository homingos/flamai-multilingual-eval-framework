"""
Download all completed Modal runs into data/modal_cache/.

Usage:
    python scripts/download_all_runs.py            # download everything new
    python scripts/download_all_runs.py --force    # re-download even if cached
    python scripts/download_all_runs.py --run-id 2026-06-25_094523_dace81  # single run

Files saved per run:
    data/modal_cache/{run_id}/verdicts.jsonl   (~60 KB)
    data/modal_cache/{run_id}/gemma4.jsonl     (~800 KB)
    data/modal_cache/{run_id}/regional.jsonl   (~800 KB)
    data/modal_cache/{run_id}/meta.json        (slug, task, language, region, model, grade)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent.parent
CACHE_DIR    = REPO_ROOT / "data" / "modal_cache"
QUAL_JSON    = REPO_ROOT / "data" / "qualitative_results.json"
MODAL_VOLUME = "phase2a-outputs"
RUN_ID_RE    = re.compile(r'\b(\d{4}-\d{2}-\d{2}_\d{6}_[a-f0-9]+)\b')

# slug → (language, region, model_display)
SLUG_META: dict[str, tuple[str, str, str]] = {
    "tamil":                ("Tamil",                "Indic",        "Tamil-Mistral-7B"),
    "sarvam-m-tamil":       ("Tamil",                "Indic",        "Sarvam-M-24B"),
    "marathi":              ("Marathi",              "Indic",        "MahaMarathi-7B"),
    "sarvam-m-marathi":     ("Marathi",              "Indic",        "Sarvam-M-24B"),
    "kannada":              ("Kannada",              "Indic",        "Ambari-7B"),
    "sarvam-m-kannada":     ("Kannada",              "Indic",        "Sarvam-M-24B"),
    "gujarati":             ("Gujarati",             "Indic",        "Gujju-Llama-7B"),
    "sarvam-m-gujarati":    ("Gujarati",             "Indic",        "Sarvam-M-24B"),
    "arabic":               ("Arabic",               "Middle East",  "Jais-2-8B"),
    "jais-70b":             ("Arabic",               "Middle East",  "Jais-2-70B-Chat"),
    "hebrew":               ("Hebrew",               "Middle East",  "DictaLM-3.0-Nemotron-12B"),
    "korean":               ("Korean",               "East Asia",    "Polyglot-Ko-12B"),
    "exaone-korean":        ("Korean",               "East Asia",    "EXAONE-3.5-32B-Instruct"),
    "malay":                ("Malay",                "SEA",          "MaLLaM-5B"),
    "swahili":              ("Swahili",              "Africa",       "Swahili-Gemma-7B"),
    "amharic":              ("Amharic",              "Africa",       "Walia-LLM-7B"),
    "greek":                ("Greek",                "Europe",       "Meltemi-7B"),
    "krikri-greek":         ("Greek",                "Europe",       "Krikri-8B-Instruct"),
    "french":               ("French",               "Europe",       "Lucie-7B-Instruct-v1.1"),
    "swedish":              ("Swedish",              "Europe",       "Viking-7B"),
    "czech":                ("Czech",                "Europe",       "CSMPT-7B"),
    "german":               ("German",               "Europe",       "EuroLLM-22B"),
    "italian":              ("Italian",              "Europe",       "EuroLLM-22B"),
    "maori":                ("Māori",                "Oceania",      "Goldfish-mri-39M"),
    "tok_pisin":            ("Tok Pisin",            "Oceania",      "Goldfish-tpi-125M"),
    "brazilian_portuguese": ("Brazilian Portuguese", "Americas",     "Tucano-2b4"),
}


def modal_ls(path: str) -> str:
    r = subprocess.run(
        ["modal", "volume", "ls", MODAL_VOLUME, path],
        capture_output=True, text=True,
    )
    return r.stdout + r.stderr


def modal_get(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["modal", "volume", "get", "--force", MODAL_VOLUME, remote, str(local)],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def list_run_ids() -> list[str]:
    output = modal_ls("runs/")
    return list(dict.fromkeys(RUN_ID_RE.findall(output)))


def discover_slug_task(run_id: str) -> tuple[str, str]:
    for line in modal_ls(f"runs/{run_id}/judge/").splitlines():
        m = re.search(r'([^/\s]+)_(instructions|translation)_verdicts\.jsonl', line.strip())
        if m:
            return m.group(1), m.group(2)
    for line in modal_ls(f"runs/{run_id}/regional/").splitlines():
        m = re.search(r'([^/\s]+)_(instructions|translation)_outputs\.jsonl', line.strip())
        if m:
            return m.group(1), m.group(2)
    return "", ""


def compute_grade(win_rate: float) -> str:
    if win_rate >= 60: return "A"
    if win_rate >= 50: return "B"
    if win_rate >= 40: return "C"
    if win_rate >= 20: return "D"
    return "E"


def compute_win_rate(verdicts_path: Path) -> tuple[float, float] | None:
    """Returns (regional_win_rate%, gemma4_win_rate%) or None on failure."""
    spec = importlib.util.spec_from_file_location(
        "generate_review", REPO_ROOT / "scripts" / "generate_review.py"
    )
    gr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gr)

    try:
        records    = gr.load_jsonl(verdicts_path)
        aggregated = gr.aggregate_verdicts(records)
    except Exception as e:
        print(f"    ✗ Could not parse verdicts: {e}")
        return None

    regional = gemma4 = ties = 0
    for dims in aggregated.values():
        w = gr.overall_winner(dims)
        if w == "regional":   regional += 1
        elif w == "gemma4":   gemma4   += 1
        elif w == "tie":      ties     += 1

    total = regional + gemma4 + ties
    if total == 0:
        return None
    return round(regional / total * 100, 1), round(gemma4 / total * 100, 1)


def download_run(run_id: str, force: bool = False) -> bool:
    """
    Download a single run into data/modal_cache/{run_id}/.
    Returns True if successful, False if skipped or failed.
    """
    out_dir  = CACHE_DIR / run_id
    meta_file = out_dir / "meta.json"

    if meta_file.exists() and not force:
        print(f"  ↷ {run_id} — already cached, skipping (use --force to re-download)")
        return True

    print(f"  ⟳ {run_id} — discovering…", flush=True)
    slug, task = discover_slug_task(run_id)
    if not slug or not task:
        print(f"    ✗ No completed judge dir found — run still in progress or failed")
        return False

    meta = SLUG_META.get(slug)
    if meta:
        language, region, model = meta
    else:
        language = slug.replace("-", " ").replace("_", " ").title()
        region   = "Unknown"
        model    = slug

    print(f"    slug={slug}  task={task}  language={language}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    verdicts_path  = out_dir / "verdicts.jsonl"
    gemma4_path    = out_dir / "gemma4.jsonl"
    regional_path  = out_dir / "regional.jsonl"

    print(f"    downloading verdicts…", flush=True)
    if not modal_get(f"runs/{run_id}/judge/{slug}_{task}_verdicts.jsonl", verdicts_path):
        print(f"    ✗ verdicts download failed")
        return False

    print(f"    downloading gemma4 outputs…", flush=True)
    if not modal_get(f"runs/{run_id}/gemma4/{task}_outputs.jsonl", gemma4_path):
        print(f"    ✗ gemma4 outputs download failed")
        return False

    print(f"    downloading regional outputs…", flush=True)
    if not modal_get(f"runs/{run_id}/regional/{slug}_{task}_outputs.jsonl", regional_path):
        print(f"    ✗ regional outputs download failed")
        return False

    wr = compute_win_rate(verdicts_path)
    if wr is None:
        print(f"    ✗ Could not compute win rate from verdicts")
        return False
    win_rate, gemma4_wr = wr
    grade = compute_grade(win_rate)

    meta_dict = {
        "run_id":          run_id,
        "slug":            slug,
        "task":            task,
        "language":        language,
        "region":          region,
        "model":           model,
        "judge_win_rate":  win_rate,
        "gemma4_win_rate": gemma4_wr,
        "grade":           grade,
    }
    meta_file.write_text(json.dumps(meta_dict, indent=2, ensure_ascii=False))
    print(f"    ✓ done  grade={grade}  win_rate={win_rate}%", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Modal runs to data/modal_cache/")
    parser.add_argument("--force",  action="store_true", help="Re-download even if cached")
    parser.add_argument("--run-id", help="Download a specific run ID only")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.run_id:
        run_ids = [args.run_id]
    else:
        print("Listing all runs on Modal…", flush=True)
        run_ids = list_run_ids()
        print(f"Found {len(run_ids)} run(s) on Modal\n", flush=True)

    ok = failed = skipped = 0
    for run_id in run_ids:
        result = download_run(run_id, force=args.force)
        if result:
            ok += 1
        else:
            # Check if it was a skip (meta exists)
            if (CACHE_DIR / run_id / "meta.json").exists() and not args.force:
                skipped += 1
            else:
                failed += 1

    print(f"\nDone. {ok} downloaded, {skipped} already cached, {failed} failed.")
    if failed:
        print("Failed runs may still be in progress on Modal.")


if __name__ == "__main__":
    main()
