"""
pipeline/run.py
===============
Experiment tracking helpers. All path resolution lives here.
No Modal imports. No network calls. No registry calls.
Pure file I/O at /data/outputs/.
"""
from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from typing import Optional

OUTPUTS_ROOT = os.environ.get("OUTPUTS_ROOT", "/data/outputs")


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """
    Returns a unique run ID of the form: '2026-06-17_143022_a3f9b1'
    Format: YYYY-MM-DD_HHMMSS_<6-char hex>
    Two calls in the same second produce different IDs due to the random suffix.
    """
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d_%H%M%S")
    suffix = secrets.token_hex(3)  # 6 hex chars
    return f"{stamp}_{suffix}"


# ---------------------------------------------------------------------------
# Path helpers — all paths are under /data/outputs/
# ---------------------------------------------------------------------------

def run_dir(run_id: str) -> str:
    return f"{OUTPUTS_ROOT}/runs/{run_id}"


def gemma_output_path(run_id: str, task: str) -> str:
    return f"{OUTPUTS_ROOT}/runs/{run_id}/gemma4/{task}_outputs.jsonl"


def regional_output_path(run_id: str, slug: str, task: str) -> str:
    return f"{OUTPUTS_ROOT}/runs/{run_id}/regional/{slug}_{task}_outputs.jsonl"


def judge_path(run_id: str, slug: str, task: str) -> str:
    return f"{OUTPUTS_ROOT}/runs/{run_id}/judge/{slug}_{task}_verdicts.jsonl"


def metric_path(run_id: str, metric_name: str, slug: str) -> str:
    return f"{OUTPUTS_ROOT}/runs/{run_id}/metrics/{metric_name}_{slug}.jsonl"


def manifest_path(run_id: str) -> str:
    return f"{OUTPUTS_ROOT}/runs/{run_id}/run_manifest.json"


def report_path(run_id: str) -> str:
    return f"{OUTPUTS_ROOT}/runs/{run_id}/reports/final_report.json"


def checkpoint_path(run_id: str, model_id: str, task: str) -> str:
    """'/data/outputs/runs/{run_id}/checkpoints/{model_id}_{task}.json'"""
    return f"{OUTPUTS_ROOT}/runs/{run_id}/checkpoints/{model_id}_{task}.json"


def runs_index_path() -> str:
    return f"{OUTPUTS_ROOT}/runs_index.json"


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def write_manifest(run_id: str, manifest: dict) -> None:
    """Atomically writes run_manifest.json. Creates parent directories if missing."""
    path = manifest_path(run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_write(path, manifest)


def read_manifest(run_id: str) -> dict:
    """Reads run_manifest.json. Raises FileNotFoundError if it doesn't exist."""
    with open(manifest_path(run_id)) as f:
        return json.load(f)


def update_manifest_status(run_id: str, status: str) -> None:
    """Reads manifest, sets status field, writes back atomically."""
    manifest = read_manifest(run_id)
    manifest["status"] = status
    _atomic_write(manifest_path(run_id), manifest)


# ---------------------------------------------------------------------------
# Runs index
# ---------------------------------------------------------------------------

def append_run_to_index(run_id: str, status: str = "started") -> None:
    """Appends {run_id, status, created_at} to runs_index.json atomically."""
    index_path = runs_index_path()
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    try:
        with open(index_path) as f:
            entries: list = json.load(f)
    except FileNotFoundError:
        entries = []
    entries.append({
        "run_id": run_id,
        "status": status,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    _atomic_write(index_path, entries)


def update_run_in_index(run_id: str, status: str) -> None:
    """Updates the status field of an existing run entry in runs_index.json."""
    index_path = runs_index_path()
    try:
        with open(index_path) as f:
            entries: list = json.load(f)
    except FileNotFoundError:
        return
    for entry in entries:
        if entry["run_id"] == run_id:
            entry["status"] = status
            break
    _atomic_write(index_path, entries)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def write_checkpoint(
    run_id: str, model_id: str, task: str, last_id: str, count: int
) -> None:
    """
    Writes checkpoint JSON atomically.
    Called after every CHECKPOINT_EVERY prompts.
    """
    path = checkpoint_path(run_id, model_id, task)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "last_completed_id": last_id,
        "completed_count": count,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _atomic_write(path, data)


def read_checkpoint(run_id: str, model_id: str, task: str) -> Optional[dict]:
    """Returns checkpoint dict or None if no checkpoint exists."""
    path = checkpoint_path(run_id, model_id, task)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Output JSONL helpers
# ---------------------------------------------------------------------------

def append_output(path: str, record: dict) -> None:
    """
    Appends a single JSON record as a new line to a JSONL output file.
    Creates parent directories if missing.
    Opens in append mode — inherently safe for incremental writes.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_completed_ids(path: str) -> set:
    """
    Reads a JSONL output file and returns the set of all prompt_id values.
    Returns empty set if file does not exist.
    """
    try:
        ids = set()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    ids.add(record["prompt_id"])
        return ids
    except FileNotFoundError:
        return set()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atomic_write(path: str, data: dict | list) -> None:
    """Write to a temp file then os.replace — prevents corrupt reads mid-write."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)
