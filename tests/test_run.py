"""Tests for pipeline/run.py — path helpers, manifest I/O, checkpoint, index."""
import json
import os
import re

import pytest


@pytest.fixture(autouse=True)
def tmp_outputs(monkeypatch, tmp_path):
    import src.pipeline.run as run_mod
    monkeypatch.setattr(run_mod, "OUTPUTS_ROOT", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def test_generate_run_id_format():
    from src.pipeline.run import generate_run_id
    run_id = generate_run_id()
    assert re.match(r"\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-f]{6}$", run_id), run_id


def test_generate_run_id_unique():
    from src.pipeline.run import generate_run_id
    assert generate_run_id() != generate_run_id()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def test_gemma_output_path():
    from src.pipeline.run import gemma_output_path
    p = gemma_output_path("run1", "translation")
    assert p.endswith("gemma4/translation_outputs.jsonl")


def test_regional_output_path():
    from src.pipeline.run import regional_output_path
    p = regional_output_path("run1", "tamil", "translation")
    assert "regional/tamil_translation_outputs.jsonl" in p


def test_checkpoint_path():
    from src.pipeline.run import checkpoint_path
    p = checkpoint_path("run1", "tamil-mistral-7b", "translation")
    assert "checkpoints" in p
    assert "tamil-mistral-7b" in p
    assert "translation" in p


def test_judge_path():
    from src.pipeline.run import judge_path
    p = judge_path("run-id", "tamil", "instructions")
    assert "judge/tamil_instructions_verdicts.jsonl" in p


def test_metric_path():
    from src.pipeline.run import metric_path
    p = metric_path("run-id", "bleu", "tamil")
    assert "metrics/bleu_tamil.jsonl" in p


def test_manifest_path():
    from src.pipeline.run import manifest_path
    p = manifest_path("run-id")
    assert p.endswith("run_manifest.json")
    assert "run-id" in p


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def test_write_and_read_manifest(tmp_outputs):
    from src.pipeline.run import write_manifest, read_manifest, generate_run_id
    run_id = generate_run_id()
    manifest = {"run_id": run_id, "status": "started", "task_scope": ["translation"]}
    write_manifest(run_id, manifest)
    result = read_manifest(run_id)
    assert result["run_id"] == run_id
    assert result["status"] == "started"


def test_update_manifest_status(tmp_outputs):
    from src.pipeline.run import write_manifest, update_manifest_status, read_manifest, generate_run_id
    run_id = generate_run_id()
    write_manifest(run_id, {"run_id": run_id, "status": "started"})
    update_manifest_status(run_id, "completed")
    assert read_manifest(run_id)["status"] == "completed"


# ---------------------------------------------------------------------------
# Runs index
# ---------------------------------------------------------------------------

def test_append_run_to_index_creates_file(tmp_outputs):
    from src.pipeline.run import append_run_to_index, runs_index_path
    append_run_to_index("run1", "started")
    with open(runs_index_path()) as f:
        records = json.load(f)
    assert records[0]["run_id"] == "run1"


def test_append_run_to_index_appends(tmp_outputs):
    from src.pipeline.run import append_run_to_index, runs_index_path
    append_run_to_index("run1", "started")
    append_run_to_index("run2", "started")
    with open(runs_index_path()) as f:
        records = json.load(f)
    assert len(records) == 2
    assert records[1]["run_id"] == "run2"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def test_write_and_read_checkpoint(tmp_outputs):
    from src.pipeline.run import write_checkpoint, read_checkpoint
    write_checkpoint("run1", "tamil-mistral-7b", "translation", "prompt_042", 42)
    cp = read_checkpoint("run1", "tamil-mistral-7b", "translation")
    assert cp["last_completed_id"] == "prompt_042"
    assert cp["completed_count"] == 42


def test_read_checkpoint_none_if_missing(tmp_outputs):
    from src.pipeline.run import read_checkpoint
    assert read_checkpoint("run1", "no-model", "translation") is None


# ---------------------------------------------------------------------------
# Output JSONL helpers
# ---------------------------------------------------------------------------

def test_append_output_creates_file(tmp_outputs):
    from src.pipeline.run import append_output, gemma_output_path
    path = gemma_output_path("run1", "translation")
    append_output(path, {"prompt_id": "p1", "output": "hello"})
    with open(path) as f:
        records = [json.loads(line) for line in f]
    assert records[0]["prompt_id"] == "p1"


def test_append_output_appends_multiple(tmp_outputs):
    from src.pipeline.run import append_output, gemma_output_path
    path = gemma_output_path("run1", "translation")
    append_output(path, {"prompt_id": "p1", "output": "a"})
    append_output(path, {"prompt_id": "p2", "output": "b"})
    with open(path) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 2


def test_load_completed_ids_empty_if_missing(tmp_outputs):
    from src.pipeline.run import load_completed_ids, gemma_output_path
    ids = load_completed_ids(gemma_output_path("run1", "translation"))
    assert ids == set()


def test_load_completed_ids_after_write(tmp_outputs):
    from src.pipeline.run import append_output, load_completed_ids, gemma_output_path
    path = gemma_output_path("run1", "translation")
    append_output(path, {"prompt_id": "p1", "output": "x"})
    append_output(path, {"prompt_id": "p2", "output": "y"})
    ids = load_completed_ids(path)
    assert ids == {"p1", "p2"}
