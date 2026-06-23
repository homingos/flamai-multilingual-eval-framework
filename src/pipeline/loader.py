"""
pipeline/loader.py
==================
Dataset reader and schema validator.
No Modal imports. No network calls.
All reads from /data/benchmarks/.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Iterator, List, Optional

BENCHMARKS_ROOT = os.environ.get("BENCHMARKS_ROOT", "/data/benchmarks")

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

TRANSLATION_REQUIRED = {
    "id", "language", "region", "winner_model",
    "direction", "source_lang", "target_lang",
    "source", "reference", "flores_id",
}

INSTRUCTION_REQUIRED = {
    "id", "language", "region", "winner_model",
    "category", "system_instruction", "user_prompt", "expected_constraints",
}

VALID_DIRECTIONS = {"en→target", "target→en"}

VALID_CATEGORIES = {
    "tone_style", "length_constraint", "language_compliance",
    "topic_boundary", "structured_output", "number_verbalization",
}


# ---------------------------------------------------------------------------
# Meta readers
# ---------------------------------------------------------------------------

def read_task_meta(task: str) -> dict:
    """Reads /data/benchmarks/{task}/meta.json. Raises FileNotFoundError if missing."""
    path = os.path.join(BENCHMARKS_ROOT, task, "meta.json")
    with open(path) as f:
        return json.load(f)


def read_language_meta(task: str, slug: str) -> dict:
    """Reads /data/benchmarks/{task}/{slug}/meta.json."""
    path = os.path.join(BENCHMARKS_ROOT, task, slug, "meta.json")
    with open(path) as f:
        return json.load(f)


def list_available_slugs(task: str) -> List[str]:
    """
    Scans /data/benchmarks/{task}/ for subdirectories containing samples.jsonl.
    Returns sorted list of slug strings.
    """
    task_dir = os.path.join(BENCHMARKS_ROOT, task)
    if not os.path.isdir(task_dir):
        return []
    slugs = []
    for entry in os.scandir(task_dir):
        if entry.is_dir():
            if os.path.exists(os.path.join(entry.path, "samples.jsonl")):
                slugs.append(entry.name)
    return sorted(slugs)


# kept for backwards compatibility with Phase 1 tests
def list_languages(task: str) -> list[dict]:
    meta = read_task_meta(task)
    return meta.get("languages", [])


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def load_samples(task: str, slug: str, limit: Optional[int] = None) -> List[dict]:
    """
    Reads /data/benchmarks/{task}/{slug}/samples.jsonl.
    Validates each sample. Returns list of validated sample dicts.
    Raises FileNotFoundError if samples.jsonl does not exist.
    Raises ValueError (with row + field) if schema validation fails.
    """
    return list(iter_samples(task, slug, limit=limit))


def iter_samples(task: str, slug: str, limit: Optional[int] = None) -> Iterator[dict]:
    """Memory-efficient streaming version. Yields one validated sample at a time."""
    path = os.path.join(BENCHMARKS_ROOT, task, slug, "samples.jsonl")
    count = 0
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if task == "translation":
                validate_translation_sample(sample, lineno)
            elif task == "instructions":
                validate_instruction_sample(sample, lineno)
            yield sample
            count += 1
            if limit is not None and count >= limit:
                break


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_translation_sample(sample: dict, row: int = 0) -> None:
    """
    Raises ValueError("Row {row}: missing field '{field}'") for missing fields.
    Raises ValueError("Row {row}: invalid direction '{val}'") for bad direction.
    """
    for field in TRANSLATION_REQUIRED:
        if field not in sample:
            raise ValueError(f"Row {row}: missing field '{field}'")
    direction = sample.get("direction", "")
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Row {row}: invalid direction '{direction}'")


def validate_instruction_sample(sample: dict, row: int = 0) -> None:
    """
    Raises ValueError("Row {row}: missing field '{field}'") for missing fields.
    Raises ValueError("Row {row}: invalid category '{val}'") for bad category.
    """
    for field in INSTRUCTION_REQUIRED:
        if field not in sample:
            raise ValueError(f"Row {row}: missing field '{field}'")
    category = sample.get("category", "")
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Row {row}: invalid category '{category}'")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_samples(task: str, slug: str) -> int:
    """Fast line count without loading into memory. Returns 0 if file does not exist."""
    path = os.path.join(BENCHMARKS_ROOT, task, slug, "samples.jsonl")
    try:
        count = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                count += chunk.count(b"\n")
        return count
    except FileNotFoundError:
        return 0


def compute_file_hash(task: str, slug: str) -> str:
    """Returns SHA-256 hex digest of samples.jsonl file content."""
    path = os.path.join(BENCHMARKS_ROOT, task, slug, "samples.jsonl")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# kept for backwards compatibility
def compute_benchmark_hash(task: str, slug: str) -> str:
    return f"sha256:{compute_file_hash(task, slug)}"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_translation_prompt(sample: dict) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for a translation sample.

    System prompt instructs the model to translate from source_lang to target_lang.
    User prompt is the source text.
    """
    system_prompt = (
        f"You are a professional translator. Translate the following text from "
        f"{sample['source_lang']} to {sample['target_lang']}. "
        f"Output only the translation, nothing else."
    )
    user_prompt = sample["source"]
    return system_prompt, user_prompt


def build_instruction_prompt(sample: dict) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for an instruction sample.

    System prompt is sample["system_instruction"] (English, set by developer).
    User prompt is sample["user_prompt_localized"] (target language) — required.
    Raises ValueError if missing: all instruction samples must be pre-translated
    before evaluation so regional models are never tested on English input.
    """
    localized = sample.get("user_prompt_localized")
    if not localized:
        raise ValueError(
            f"Sample '{sample.get('id')}' missing 'user_prompt_localized'. "
            "Run the pre-translation job before evaluation."
        )
    return sample["system_instruction"], localized


def build_prompt(task: str, sample: dict) -> tuple[str, str]:
    """Dispatches to build_translation_prompt or build_instruction_prompt."""
    if task == "translation":
        return build_translation_prompt(sample)
    elif task == "instructions":
        return build_instruction_prompt(sample)
    else:
        raise ValueError(f"Unknown task: '{task}'. Expected 'translation' or 'instructions'.")
