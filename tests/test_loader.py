"""Tests for pipeline/loader.py — schema validation, counting, hashing, prompt building."""
import json
import os

import pytest

SAMPLE_TRANSLATION = {
    "id": "trans_tamil_en_tgt_0000",
    "language": "Tamil",
    "region": "Indic",
    "winner_model": "Tamil-Mistral-7B",
    "direction": "en→target",
    "source_lang": "English",
    "target_lang": "Tamil",
    "source": "Hello world.",
    "reference": "வணக்கம் உலகம்.",
    "flores_id": 0,
}

SAMPLE_INSTRUCTION = {
    "id": "inst_tamil_tone_style_001",
    "language": "Tamil",
    "region": "Indic",
    "winner_model": "Tamil-Mistral-7B",
    "category": "tone_style",
    "system_instruction": "You are a helpful assistant.",
    "user_prompt": "How do I reset my password?",
    "expected_constraints": {"tone": "friendly"},
}


@pytest.fixture
def mock_benchmarks(tmp_path, monkeypatch):
    """Minimal benchmarks directory structure in tmp_path."""
    import src.pipeline.loader as loader_mod
    monkeypatch.setattr(loader_mod, "BENCHMARKS_ROOT", str(tmp_path))

    # Translation
    trans_dir = tmp_path / "translation" / "tamil"
    trans_dir.mkdir(parents=True)
    with open(trans_dir / "samples.jsonl", "w") as f:
        for i in range(5):
            sample = {**SAMPLE_TRANSLATION, "id": f"trans_tamil_en_tgt_{i:04d}", "flores_id": i}
            f.write(json.dumps(sample) + "\n")
    with open(trans_dir / "meta.json", "w") as f:
        json.dump({
            "language": "Tamil", "slug": "tamil", "region": "Indic",
            "winner_model": "Tamil-Mistral-7B", "flores_code": "tam_Taml",
            "total_samples": 5, "en_to_target": 5, "target_to_en": 0,
        }, f)
    with open(tmp_path / "translation" / "meta.json", "w") as f:
        json.dump({"languages": [{"language": "Tamil", "slug": "tamil"}]}, f)

    # Instructions
    inst_dir = tmp_path / "instructions" / "tamil"
    inst_dir.mkdir(parents=True)
    with open(inst_dir / "samples.jsonl", "w") as f:
        for i in range(3):
            sample = {**SAMPLE_INSTRUCTION, "id": f"inst_tamil_tone_style_{i:03d}"}
            f.write(json.dumps(sample) + "\n")

    return tmp_path


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def test_load_samples_translation(mock_benchmarks):
    from src.pipeline.loader import load_samples
    samples = load_samples("translation", "tamil")
    assert len(samples) == 5


def test_load_samples_limit(mock_benchmarks):
    from src.pipeline.loader import load_samples
    samples = load_samples("translation", "tamil", limit=2)
    assert len(samples) == 2


def test_load_samples_file_not_found(mock_benchmarks):
    from src.pipeline.loader import load_samples
    with pytest.raises(FileNotFoundError):
        load_samples("translation", "nonexistent")


def test_load_samples_instructions(mock_benchmarks):
    from src.pipeline.loader import load_samples
    samples = load_samples("instructions", "tamil")
    assert len(samples) == 3


# ---------------------------------------------------------------------------
# Validation — missing fields
# ---------------------------------------------------------------------------

def test_validate_translation_missing_field():
    import src.pipeline.loader as loader_mod
    bad = {**SAMPLE_TRANSLATION}
    del bad["reference"]
    with pytest.raises(ValueError, match="reference"):
        loader_mod.validate_translation_sample(bad, row=0)


def test_validate_translation_bad_direction():
    import src.pipeline.loader as loader_mod
    bad = {**SAMPLE_TRANSLATION, "direction": "sideways"}
    with pytest.raises(ValueError, match="direction"):
        loader_mod.validate_translation_sample(bad, row=0)


def test_validate_instruction_missing_field():
    import src.pipeline.loader as loader_mod
    bad = {**SAMPLE_INSTRUCTION}
    del bad["category"]
    with pytest.raises(ValueError, match="category"):
        loader_mod.validate_instruction_sample(bad, row=0)


def test_validate_instruction_bad_category():
    import src.pipeline.loader as loader_mod
    bad = {**SAMPLE_INSTRUCTION, "category": "made_up"}
    with pytest.raises(ValueError, match="category"):
        loader_mod.validate_instruction_sample(bad, row=0)


def test_load_samples_raises_on_invalid_schema(mock_benchmarks, tmp_path):
    import src.pipeline.loader as loader_mod
    bad_dir = tmp_path / "translation" / "badslug"
    bad_dir.mkdir(parents=True)
    with open(bad_dir / "samples.jsonl", "w") as f:
        f.write(json.dumps({"id": "x"}) + "\n")
    with pytest.raises(ValueError):
        loader_mod.load_samples("translation", "badslug")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def test_count_samples(mock_benchmarks):
    from src.pipeline.loader import count_samples
    assert count_samples("translation", "tamil") == 5


def test_count_samples_zero_if_missing(mock_benchmarks):
    from src.pipeline.loader import count_samples
    assert count_samples("translation", "ghost") == 0


def test_compute_file_hash(mock_benchmarks):
    from src.pipeline.loader import compute_file_hash
    h = compute_file_hash("translation", "tamil")
    assert len(h) == 64  # SHA-256 hex digest


def test_compute_file_hash_deterministic(mock_benchmarks):
    from src.pipeline.loader import compute_file_hash
    assert compute_file_hash("translation", "tamil") == compute_file_hash("translation", "tamil")


# ---------------------------------------------------------------------------
# Meta readers
# ---------------------------------------------------------------------------

def test_read_language_meta(mock_benchmarks):
    from src.pipeline.loader import read_language_meta
    meta = read_language_meta("translation", "tamil")
    assert meta["flores_code"] == "tam_Taml"
    assert meta["total_samples"] == 5


def test_list_available_slugs(mock_benchmarks):
    from src.pipeline.loader import list_available_slugs
    slugs = list_available_slugs("translation")
    assert "tamil" in slugs


def test_read_task_meta(mock_benchmarks):
    from src.pipeline.loader import read_task_meta
    meta = read_task_meta("translation")
    assert len(meta["languages"]) >= 1


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def test_build_translation_prompt():
    from src.pipeline.loader import build_translation_prompt
    sys_p, user_p = build_translation_prompt(SAMPLE_TRANSLATION)
    assert "English" in sys_p and "Tamil" in sys_p
    assert user_p == SAMPLE_TRANSLATION["source"]


def test_build_instruction_prompt():
    from src.pipeline.loader import build_instruction_prompt
    sys_p, user_p = build_instruction_prompt(SAMPLE_INSTRUCTION)
    assert sys_p == SAMPLE_INSTRUCTION["system_instruction"]
    assert user_p == SAMPLE_INSTRUCTION["user_prompt"]


def test_build_prompt_dispatcher_translation():
    from src.pipeline.loader import build_prompt
    sys_p, user_p = build_prompt("translation", SAMPLE_TRANSLATION)
    assert "Tamil" in sys_p


def test_build_prompt_dispatcher_instructions():
    from src.pipeline.loader import build_prompt
    sys_p, user_p = build_prompt("instructions", SAMPLE_INSTRUCTION)
    assert sys_p == SAMPLE_INSTRUCTION["system_instruction"]


def test_build_prompt_dispatcher_unknown_task():
    from src.pipeline.loader import build_prompt
    with pytest.raises(ValueError):
        build_prompt("unknown_task", SAMPLE_TRANSLATION)
