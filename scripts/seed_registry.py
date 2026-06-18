"""
Seed the Phase 2A registry with all 17 regional models + Gemma-4 baseline,
all hardware presets, all metrics, and task configurations.

Usage:
    REGISTRY_URL=https://your-registry.modal.run \\
    JWT_TOKEN=$(python scripts/jwt_token_generator.py --secret $JWT_SECRET --scopes registry:write) \\
    python scripts/seed_registry.py

Idempotent: 409 Conflict responses (already-exists) are silently ignored.
"""
from __future__ import annotations

import os
import sys

import requests

REGISTRY_URL = os.environ.get("REGISTRY_URL", "http://localhost:8000")
JWT_TOKEN    = os.environ.get("JWT_TOKEN", "")

HEADERS = {
    "Authorization": f"Bearer {JWT_TOKEN}",
    "Content-Type":  "application/json",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

MODELS = [
    {
        "id": "gemma-4-26b",
        "name": "Gemma-4",
        "hf_model_id": "google/gemma-4-26B-A4B-it",
        "language": "Baseline",
        "slug": "baseline",
        "region": "Global",
        "gpu_preset": "a100_80gb",
        "params_billions": 26.0,
        "notes": "Gemma-4 26B A4B IT — Phase 2A baseline model",
    },
    {
        "id": "tamil-mistral-7b",
        "name": "Tamil-Mistral-7B",
        "hf_model_id": "Hemanth-thunder/Tamil-Mistral-7B-Instruct-v0.1",
        "language": "Tamil",
        "slug": "tamil",
        "region": "Indic",
        "gpu_preset": "l4",
        "params_billions": 7.0,
    },
    {
        "id": "mahamarathi-7b",
        "name": "MahaMarathi-7B",
        "hf_model_id": "marathi-llm/MahaMarathi-7B-v24.01-Base",
        "language": "Marathi",
        "slug": "marathi",
        "region": "Indic",
        "gpu_preset": "l4",
        "params_billions": 7.0,
    },
    {
        "id": "ambari-7b",
        "name": "Ambari-7B",
        "hf_model_id": "Cognitive-Lab/Ambari-7B-base-v0.1",
        "language": "Kannada",
        "slug": "kannada",
        "region": "Indic",
        "gpu_preset": "l4",
        "params_billions": 7.0,
    },
    {
        "id": "gujju-llama-7b",
        "name": "Gujju-Llama-7B",
        "hf_model_id": "sampoorna42/gujju-llama-base-v1.0",
        "language": "Gujarati",
        "slug": "gujarati",
        "region": "Indic",
        "gpu_preset": "l4",
        "params_billions": 7.0,
    },
    {
        "id": "jais-2-8b",
        "name": "Jais-2-8B",
        "hf_model_id": "inceptionai/Jais-2-8B-Chat",
        "language": "Arabic",
        "slug": "arabic",
        "region": "Middle East",
        "gpu_preset": "l4",
        "params_billions": 8.0,
    },
    {
        "id": "dictalm-2-7b",
        "name": "DictaLM-2.0-7B",
        "hf_model_id": "dicta-il/dictalm2.0-instruct",
        "language": "Hebrew",
        "slug": "hebrew",
        "region": "Middle East",
        "gpu_preset": "l4",
        "params_billions": 7.0,
    },
    {
        "id": "polyglot-ko-12b",
        "name": "Polyglot-Ko-12B",
        "hf_model_id": "EleutherAI/polyglot-ko-12.8b",
        "language": "Korean",
        "slug": "korean",
        "region": "East Asia",
        "gpu_preset": "l40s",
        "params_billions": 12.8,
    },
    {
        "id": "mallam-5b",
        "name": "MaLLaM-5B",
        "hf_model_id": "mesolitica/mallam-5B-4096",
        "language": "Malay",
        "slug": "malay",
        "region": "SEA",
        "gpu_preset": "l4",
        "params_billions": 5.0,
    },
    {
        "id": "swahili-gemma-7b",
        "name": "Swahili-Gemma-7B",
        "hf_model_id": "Mollel/Swahili_Gemma",
        "language": "Swahili",
        "slug": "swahili",
        "region": "Africa",
        "gpu_preset": "l4",
        "params_billions": 7.0,
    },
    {
        "id": "walia-llm-7b",
        "name": "Walia-LLM-7B",
        "hf_model_id": "israel/LLAMA-Walia-II",
        "language": "Amharic",
        "slug": "amharic",
        "region": "Africa",
        "gpu_preset": "l4",
        "params_billions": 7.0,
    },
    {
        "id": "lucie-7b",
        "name": "Lucie-7B",
        "hf_model_id": "OpenLLM-France/Lucie-7B",
        "language": "French",
        "slug": "french",
        "region": "Europe",
        "gpu_preset": "l4",
        "params_billions": 7.0,
    },
    {
        "id": "viking-7b",
        "name": "Viking-7B",
        "hf_model_id": "LumiOpen/Viking-7B",
        "language": "Swedish",
        "slug": "swedish",
        "region": "Europe",
        "gpu_preset": "l4",
        "params_billions": 7.0,
    },
    {
        "id": "csmpt-7b",
        "name": "CSMPT-7B",
        "hf_model_id": "BUT-FIT/csmpt7b",
        "language": "Czech",
        "slug": "czech",
        "region": "Europe",
        "gpu_preset": "l4",
        "params_billions": 7.0,
    },
    {
        "id": "meltemi-7b",
        "name": "Meltemi-7B",
        "hf_model_id": "ilsp/Meltemi-7B-Instruct-v1.5",
        "language": "Greek",
        "slug": "greek",
        "region": "Europe",
        "gpu_preset": "l4",
        "params_billions": 7.0,
    },
    {
        "id": "tucano-2b4",
        "name": "Tucano-2b4",
        "hf_model_id": "TucanoBR/Tucano-2b4-Instruct",
        "language": "Brazilian Portuguese",
        "slug": "brazilian_portuguese",
        "region": "Americas",
        "gpu_preset": "l4",
        "params_billions": 2.4,
    },
    {
        "id": "goldfish-mri-39m",
        "name": "Goldfish-mri-39M",
        "hf_model_id": "goldfish-models/mri_latn_10mb",
        "language": "Māori",
        "slug": "maori",
        "region": "Oceania",
        "gpu_preset": "t4",
        "params_billions": 0.039,
    },
    {
        "id": "goldfish-tpi-125m",
        "name": "Goldfish-tpi-125M",
        "hf_model_id": "goldfish-models/tpi_latn_full",
        "language": "Tok Pisin",
        "slug": "tok_pisin",
        "region": "Oceania",
        "gpu_preset": "t4",
        "params_billions": 0.125,
    },
]

HARDWARE = [
    {"id": "t4",           "gpu": "T4",           "vram_gb": 16, "cost_per_hr": 0.59,
     "use_for": "Tiny models <500M (Goldfish series)"},
    {"id": "l4",           "gpu": "L4",           "vram_gb": 24, "cost_per_hr": 0.80,
     "use_for": "7B/8B inference + model metrics (COMET, BERTScore)"},
    {"id": "l40s",         "gpu": "L40S",         "vram_gb": 48, "cost_per_hr": 1.95,
     "use_for": "12B models (Polyglot-Ko-12B)"},
    {"id": "a100_80gb",    "gpu": "A100-80GB",    "vram_gb": 80, "cost_per_hr": 2.50,
     "use_for": "Gemma-4 26B baseline"},
    {"id": "rtx_pro_6000", "gpu": "RTX_PRO_6000", "vram_gb": 96, "cost_per_hr": 3.03,
     "use_for": "Future 30B+ models"},
]

METRICS = [
    {"name": "bleu",              "stage": "post_inference", "compute_tier": "light", "task_types": ["translation"],  "category": None,                   "weight": 1.0},
    {"name": "chrf",              "stage": "post_inference", "compute_tier": "light", "task_types": ["translation"],  "category": None,                   "weight": 1.5},
    {"name": "comet",             "stage": "post_inference", "compute_tier": "model", "task_types": ["translation"],  "category": None,                   "weight": 2.0},
    {"name": "bertscore",         "stage": "post_inference", "compute_tier": "model", "task_types": ["translation"],  "category": None,                   "weight": 1.5},
    {"name": "back_translation",  "stage": "post_inference", "compute_tier": "model", "task_types": ["translation"],  "category": None,                   "weight": 1.0},
    {"name": "lang_adherence",    "stage": "post_inference", "compute_tier": "light", "task_types": ["instructions"], "category": None,                   "weight": 2.0},
    {"name": "length_accuracy",   "stage": "post_inference", "compute_tier": "light", "task_types": ["instructions"], "category": "length_constraint",    "weight": 1.0},
    {"name": "format_compliance", "stage": "post_inference", "compute_tier": "light", "task_types": ["instructions"], "category": "structured_output",    "weight": 1.0},
    {"name": "topic_boundary",    "stage": "post_inference", "compute_tier": "model", "task_types": ["instructions"], "category": "topic_boundary",       "weight": 1.5},
    {"name": "tone_detection",    "stage": "post_judge",     "compute_tier": "model", "task_types": ["instructions"], "category": "tone_style",           "weight": 1.0},
    {"name": "digit_by_digit",    "stage": "post_inference", "compute_tier": "light", "task_types": ["instructions"], "category": "number_verbalization", "weight": 1.0},
    {"name": "word_form",         "stage": "post_inference", "compute_tier": "light", "task_types": ["instructions"], "category": "number_verbalization", "weight": 1.0},
    {"name": "digit_leakage",     "stage": "post_inference", "compute_tier": "light", "task_types": ["instructions"], "category": "number_verbalization", "weight": 1.0},
    {"name": "digit_preservation","stage": "post_inference", "compute_tier": "light", "task_types": ["instructions"], "category": "number_verbalization", "weight": 1.0},
    {"name": "currency_unit",     "stage": "post_inference", "compute_tier": "light", "task_types": ["instructions"], "category": "number_verbalization", "weight": 1.0},
    {"name": "number_type_cls",   "stage": "post_inference", "compute_tier": "light", "task_types": ["instructions"], "category": "number_verbalization", "weight": 1.0},
    {"name": "number_language",   "stage": "post_inference", "compute_tier": "light", "task_types": ["instructions"], "category": "number_verbalization", "weight": 1.0},
    {"name": "mixed_consistency", "stage": "post_inference", "compute_tier": "light", "task_types": ["instructions"], "category": "number_verbalization", "weight": 1.0},
]

TASKS_TO_UPDATE = [
    {"name": "translation",   "active": True,  "description": "English ↔ target language translation (FLORES-200)"},
    {"name": "instructions",  "active": True,  "description": "Talking Avatar instruction following (6 categories)"},
    {"name": "summarization", "active": False, "description": "Summarization (not yet active)"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(path: str, payload: dict) -> None:
    resp = requests.post(f"{REGISTRY_URL}{path}", json=payload, headers=HEADERS)
    if resp.status_code == 409:
        print(f"  SKIP  {path} — already exists")
    elif resp.status_code in (200, 201):
        print(f"  OK    {path}")
    else:
        print(f"  ERROR {path} — {resp.status_code}: {resp.text}", file=sys.stderr)


def _patch(path: str, payload: dict) -> None:
    resp = requests.patch(f"{REGISTRY_URL}{path}", json=payload, headers=HEADERS)
    if resp.status_code == 200:
        print(f"  OK    PATCH {path}")
    else:
        print(f"  ERROR PATCH {path} — {resp.status_code}: {resp.text}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(update: bool = False) -> None:
    if not JWT_TOKEN:
        print("ERROR: JWT_TOKEN env var is required.", file=sys.stderr)
        sys.exit(1)

    if update:
        print(f"\n=== Updating models in Phase 2A registry at {REGISTRY_URL} ===\n")
        print("--- Updating model hf_model_id / name fields ---")
        for m in MODELS:
            model_id = m["id"]
            _patch(f"/models/{model_id}", {
                "hf_model_id": m["hf_model_id"],
                "name": m["name"],
            })
        print("\n=== Update done ===")
        return

    print(f"\n=== Seeding Phase 2A registry at {REGISTRY_URL} ===\n")

    # 1. Models
    print("--- Models ---")
    for m in MODELS:
        _post("/models", m)

    # 2. Hardware
    print("\n--- Hardware ---")
    for h in HARDWARE:
        _post("/hardware", h)

    # 3. Metrics
    print("\n--- Metrics ---")
    for m in METRICS:
        _post("/metrics", m)

    # 4. Tasks (pre-seeded; update descriptions/active flags)
    print("\n--- Tasks ---")
    for t in TASKS_TO_UPDATE:
        name = t.pop("name")
        _patch(f"/tasks/{name}", t)

    print("\n=== Done ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="PATCH existing models instead of POSTing new ones")
    args = parser.parse_args()
    main(update=args.update)
