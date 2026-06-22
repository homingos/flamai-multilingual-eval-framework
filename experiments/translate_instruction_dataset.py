"""
experiments/translate_instruction_dataset.py
=============================================
Translates user_prompt in the instruction dataset to the target language,
adding a user_prompt_localized field to each sample.

This makes the instruction evaluation realistic: instead of an English user
asking questions to a Greek (or Tamil, etc.) model, the user asks in the
target language — matching how Talking Avatar is actually used.

The system_instruction is left in English (it's set by the developer).
Only the user turn is localized.

Reads/writes: /data/benchmarks/instructions/{slug}/samples.jsonl
              (Modal volume: phase2a-benchmarks)

Idempotent — samples that already have user_prompt_localized are skipped.

Usage:
  modal run experiments/translate_instruction_dataset.py --slug greek
  modal run experiments/translate_instruction_dataset.py --slug all
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

import modal

# ── Modal setup ───────────────────────────────────────────────────────────────

app = modal.App("translate-instruction-dataset")

benchmarks_volume = modal.Volume.from_name("phase2a-benchmarks")
judge_secret      = modal.Secret.from_name("phase2a-judge")
translate_image   = modal.Image.debian_slim()

# ── Language slug → display name for the translation prompt ──────────────────

SLUG_TO_LANGUAGE = {
    "amharic":              "Amharic",
    "arabic":               "Arabic",
    "brazilian_portuguese": "Brazilian Portuguese",
    "czech":                "Czech",
    "french":               "French",
    "greek":                "Greek",
    "gujarati":             "Gujarati",
    "hebrew":               "Hebrew",
    "kannada":              "Kannada",
    "korean":               "Korean",
    "malay":                "Malay",
    "maori":                "Māori",
    "marathi":              "Marathi",
    "swahili":              "Swahili",
    "swedish":              "Swedish",
    "tamil":                "Tamil",
    "tok_pisin":            "Tok Pisin",
}


# ── Gemini translation helper ─────────────────────────────────────────────────

def _translate(text: str, target_language: str, api_key: str, max_retries: int = 3) -> str:
    """
    Calls Gemini to translate text → target_language.
    Returns the translation, or the original text on repeated failure.
    """
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-3.5-flash:generateContent?key={api_key}"
    )
    prompt = (
        f"Translate the following English text to {target_language}. "
        f"Output only the translation, nothing else.\n\n{text}"
    )
    payload = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256},
    }).encode()

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
            return body["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [translate] failed after {max_retries} attempts: {exc}")
                return text  # fallback: keep English

    return text


# ── Modal function ─────────────────────────────────────────────────────────────

@app.function(
    image=translate_image,
    secrets=[judge_secret],
    volumes={"/data/benchmarks": benchmarks_volume},
    timeout=3600,
)
def translate_slug(slug: str) -> int:
    """
    Translates user_prompt for all samples in the given slug.
    Returns the number of newly-translated samples.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — check the phase2a-judge Modal secret")

    language = SLUG_TO_LANGUAGE.get(slug)
    if not language:
        raise ValueError(f"Unknown slug '{slug}'. Add it to SLUG_TO_LANGUAGE.")

    path = f"/data/benchmarks/instructions/{slug}/samples.jsonl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    # ── Read ──────────────────────────────────────────────────────────────────
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"[{slug}] {len(samples)} samples, language={language}")

    # ── Translate missing samples ─────────────────────────────────────────────
    new_count = 0
    for i, sample in enumerate(samples):
        if sample.get("user_prompt_localized"):
            continue  # already translated — skip

        translated = _translate(sample["user_prompt"], language, api_key)
        sample["user_prompt_localized"] = translated
        new_count += 1

        if (i + 1) % 100 == 0:
            print(f"  [{slug}] {i + 1}/{len(samples)} — {new_count} new")

        time.sleep(0.15)  # gentle rate limiting

    print(f"[{slug}] {new_count} new translations (skipped {len(samples) - new_count} existing)")

    # ── Write back ────────────────────────────────────────────────────────────
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    benchmarks_volume.commit()
    print(f"[{slug}] Volume committed")
    return new_count


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(slug: str = "greek"):
    """
    Args:
      --slug greek   translate one language
      --slug all     translate all 17 languages (sequential)
    """
    if slug == "all":
        slugs = sorted(SLUG_TO_LANGUAGE.keys())
    else:
        slugs = [slug]

    total = 0
    for s in slugs:
        print(f"\n── {s} ──────────────────────────────────")
        count = translate_slug.remote(s)
        total += count
        print(f"  {s}: {count} samples translated")

    print(f"\nDone — {total} total translations across {len(slugs)} language(s)")
