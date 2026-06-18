"""Tests for BaseMetric and all 13 light metric implementations."""
import pytest
from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRANSLATION_OUTPUTS = [
    {"prompt_id": "t1", "direction": "en→target",
     "source": "Hello.", "reference": "வணக்கம்.", "output": "வணக்கம்."},
    {"prompt_id": "t2", "direction": "en→target",
     "source": "Good morning.", "reference": "காலை வணக்கம்.", "output": "காலை வணக்கம்."},
    {"prompt_id": "t3", "direction": "target→en",
     "source": "வணக்கம்.", "reference": "Hello.", "output": "Hello."},
]

INSTRUCTION_LANG_OUTPUTS = [
    {
        "prompt_id": "i1", "category": "tone_style",
        "output": "Hello! I can help you reset your password. Please follow these steps carefully.",
        "expected_constraints": {"tone": "friendly"},
    },
]

NUMVERB_OUTPUTS = [
    {
        "prompt_id": "n1", "category": "number_verbalization",
        "output": "postal code ஐந்து ஆறு பூஜ்ஜியம் பூஜ்ஜியம் ஆறு ஆறு",
        "expected_constraints": {
            "number_type": "postal_code", "rule": "digit_by_digit",
            "numbers_in_prompt": ["560066"],
            "digit_by_digit_types": ["postal_code", "phone_number", "otp_pin"],
            "word_form_types": ["currency", "decimal_measurement"],
        },
    },
    {
        "prompt_id": "n2", "category": "number_verbalization",
        "output": "postal code 560066",
        "expected_constraints": {
            "number_type": "postal_code", "rule": "digit_by_digit",
            "numbers_in_prompt": ["560066"],
            "digit_by_digit_types": ["postal_code"],
            "word_form_types": [],
        },
    },
]

# ---------------------------------------------------------------------------
# BaseMetric
# ---------------------------------------------------------------------------

def test_base_metric_is_abstract():
    with pytest.raises(TypeError):
        BaseMetric()


def test_applies_to_translation_only():
    from src.metrics.translation.bleu import BLEUMetric
    m = BLEUMetric()
    assert m.applies_to("translation") is True
    assert m.applies_to("instructions") is False


def test_applies_to_all_instruction_categories():
    from src.metrics.instruction.lang_adherence import LangAdherenceMetric
    m = LangAdherenceMetric()
    assert m.applies_to("instructions") is True
    assert m.applies_to("instructions", category="tone_style") is True
    assert m.applies_to("translation") is False


def test_applies_to_specific_category():
    from src.metrics.instruction.length_accuracy import LengthAccuracyMetric
    m = LengthAccuracyMetric()
    assert m.applies_to("instructions", category="length_constraint") is True
    assert m.applies_to("instructions", category="tone_style") is False


# ---------------------------------------------------------------------------
# Translation — BLEU
# ---------------------------------------------------------------------------

def test_bleu_returns_both_directions():
    from src.metrics.translation.bleu import BLEUMetric
    r = BLEUMetric().compute(TRANSLATION_OUTPUTS, "Tamil", "translation")
    assert isinstance(r, MetricResult)
    assert "bleu_en_to_target" in r.scores
    assert "bleu_target_to_en" in r.scores


def test_bleu_perfect_outputs_score_high():
    from src.metrics.translation.bleu import BLEUMetric
    # Use multiple identical longer outputs — enough tokens to avoid brevity penalty
    outputs = [
        {"direction": "en→target",
         "output":    "the quick brown fox jumps over the lazy dog",
         "reference": "the quick brown fox jumps over the lazy dog"}
        for _ in range(10)
    ]
    r = BLEUMetric().compute(outputs, "Tamil", "translation")
    assert r.scores.get("bleu_en_to_target", 0) > 50


def test_bleu_no_raise_on_empty():
    from src.metrics.translation.bleu import BLEUMetric
    r = BLEUMetric().compute([], "Tamil", "translation")
    assert isinstance(r, MetricResult)
    assert r.scores == {}


# ---------------------------------------------------------------------------
# Translation — chrF
# ---------------------------------------------------------------------------

def test_chrf_returns_scores():
    from src.metrics.translation.chrf import ChrFMetric
    r = ChrFMetric().compute(TRANSLATION_OUTPUTS, "Tamil", "translation")
    assert "chrf_en_to_target" in r.scores
    assert r.scores["chrf_en_to_target"] > 0


def test_chrf_has_notes():
    from src.metrics.translation.chrf import ChrFMetric
    r = ChrFMetric().compute(TRANSLATION_OUTPUTS, "Tamil", "translation")
    assert r.notes is not None and "Indic" in r.notes


# ---------------------------------------------------------------------------
# Instruction — lang adherence
# ---------------------------------------------------------------------------

def test_lang_adherence_flags_english():
    from src.metrics.instruction.lang_adherence import LangAdherenceMetric
    r = LangAdherenceMetric().compute(INSTRUCTION_LANG_OUTPUTS, "Tamil", "instructions")
    assert r.scores["lang_adherence_rate"] < 1.0


def test_lang_adherence_unknown_language():
    from src.metrics.instruction.lang_adherence import LangAdherenceMetric
    r = LangAdherenceMetric().compute(INSTRUCTION_LANG_OUTPUTS, "Klingon", "instructions")
    assert r.errors


# ---------------------------------------------------------------------------
# Instruction — length accuracy
# ---------------------------------------------------------------------------

def test_length_accuracy_one_sentence_pass_fail():
    from src.metrics.instruction.length_accuracy import LengthAccuracyMetric
    outputs = [
        {"category": "length_constraint", "output": "One sentence.",
         "expected_constraints": {"max_sentences": 1}},
        {"category": "length_constraint", "output": "One. Two. Three.",
         "expected_constraints": {"max_sentences": 1}},
    ]
    r = LengthAccuracyMetric().compute(outputs, "Tamil", "instructions")
    assert r.scores["length_accuracy"] == 0.5


def test_length_accuracy_no_raise_on_empty():
    from src.metrics.instruction.length_accuracy import LengthAccuracyMetric
    r = LengthAccuracyMetric().compute([], "Tamil", "instructions")
    assert isinstance(r, MetricResult)


# ---------------------------------------------------------------------------
# Instruction — format compliance
# ---------------------------------------------------------------------------

def test_format_compliance_numbered_list():
    from src.metrics.instruction.format_compliance import FormatComplianceMetric
    outputs = [
        {"category": "structured_output", "output": "1. Step one\n2. Step two",
         "expected_constraints": {"format": "numbered_list"}},
        {"category": "structured_output", "output": "Just prose here.",
         "expected_constraints": {"format": "numbered_list"}},
    ]
    r = FormatComplianceMetric().compute(outputs, "Tamil", "instructions")
    assert r.scores["format_compliance"] == 0.5


def test_format_compliance_unknown_format_skipped():
    from src.metrics.instruction.format_compliance import FormatComplianceMetric
    outputs = [
        {"category": "structured_output", "output": "anything",
         "expected_constraints": {"format": "unknown_format_xyz"}},
    ]
    r = FormatComplianceMetric().compute(outputs, "Tamil", "instructions")
    assert r.scores["total"] == 0


# ---------------------------------------------------------------------------
# Number verbalization — N1 digit_by_digit
# ---------------------------------------------------------------------------

def test_digit_by_digit_pass_fail():
    from src.metrics.number_verbalization.digit_by_digit import DigitByDigitMetric
    r = DigitByDigitMetric().compute(NUMVERB_OUTPUTS, "Tamil", "instructions")
    assert r.scores["digit_by_digit_compliance"] == 0.5   # n1 passes, n2 fails


# ---------------------------------------------------------------------------
# Number verbalization — N3 digit_leakage
# ---------------------------------------------------------------------------

def test_digit_leakage_skips_digit_by_digit():
    from src.metrics.number_verbalization.digit_leakage import DigitLeakageMetric
    r = DigitLeakageMetric().compute(NUMVERB_OUTPUTS, "Tamil", "instructions")
    assert r.sample_count == 0   # all samples are digit_by_digit — skipped


# ---------------------------------------------------------------------------
# Number verbalization — N4 digit_preservation
# ---------------------------------------------------------------------------

def test_digit_preservation_all_digits_present():
    from src.metrics.number_verbalization.digit_preservation import DigitPreservationMetric
    outputs = [{"expected_constraints": {"numbers_in_prompt": ["123"]},
                "output": "one 1 two 2 three 3"}]
    r = DigitPreservationMetric().compute(outputs, "Tamil", "instructions")
    assert r.scores["digit_preservation_accuracy"] == 1.0


def test_digit_preservation_digits_missing():
    from src.metrics.number_verbalization.digit_preservation import DigitPreservationMetric
    outputs = [{"expected_constraints": {"numbers_in_prompt": ["123"]},
                "output": "one two three"}]
    r = DigitPreservationMetric().compute(outputs, "Tamil", "instructions")
    assert r.scores["digit_preservation_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Number verbalization — N5 currency_unit
# ---------------------------------------------------------------------------

def test_currency_unit_no_currency_samples():
    from src.metrics.number_verbalization.currency_unit import CurrencyUnitMetric
    outputs = [{"expected_constraints": {"word_form_types": ["percentage"]},
                "output": "fifty percent"}]
    r = CurrencyUnitMetric().compute(outputs, "Tamil", "instructions")
    assert r.sample_count == 0
    assert r.scores["currency_unit_accuracy"] is None


def test_currency_unit_tamil_rupee():
    from src.metrics.number_verbalization.currency_unit import CurrencyUnitMetric
    outputs = [{"expected_constraints": {"word_form_types": ["currency"]},
                "output": "இது ரூபாய் ஐந்து ஆகும்"}]
    r = CurrencyUnitMetric().compute(outputs, "Tamil", "instructions")
    assert r.scores["currency_unit_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Number verbalization — N7 number_language
# ---------------------------------------------------------------------------

def test_number_language_no_english_words():
    from src.metrics.number_verbalization.number_language import NumberLanguageMetric
    outputs = [{"output": "ஐந்து ஆறு பூஜ்ஜியம்"}]
    r = NumberLanguageMetric().compute(outputs, "Tamil", "instructions")
    assert r.scores["language_of_number_words"] == 1.0


def test_number_language_english_words_flagged():
    from src.metrics.number_verbalization.number_language import NumberLanguageMetric
    outputs = [{"output": "five six zero"}]
    r = NumberLanguageMetric().compute(outputs, "Tamil", "instructions")
    assert r.scores["language_of_number_words"] == 0.0


# ---------------------------------------------------------------------------
# Number verbalization — N8 mixed_sentence
# ---------------------------------------------------------------------------

def test_mixed_sentence_no_mixed_samples():
    from src.metrics.number_verbalization.mixed_sentence import MixedSentenceMetric
    r = MixedSentenceMetric().compute(NUMVERB_OUTPUTS, "Tamil", "instructions")
    assert r.sample_count == 0   # NUMVERB_OUTPUTS are all digit_by_digit


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discover_finds_all_13_light_metrics():
    from src.workers.light_metrics import _discover_light_metrics
    names = {m.name for m in _discover_light_metrics()}
    expected = {
        "bleu", "chrf",
        "lang_adherence", "length_accuracy", "format_compliance",
        "digit_by_digit", "word_form", "digit_leakage", "digit_preservation",
        "currency_unit", "number_type_cls", "number_language", "mixed_sentence",
    }
    missing = expected - names
    assert not missing, f"Missing metrics: {missing}"


def test_all_discovered_are_light_tier():
    from src.workers.light_metrics import _discover_light_metrics
    for m in _discover_light_metrics():
        assert m.compute_tier == ComputeTier.LIGHT, f"{m.name} is not LIGHT tier"


def test_no_metric_imports_modal():
    import ast
    import os
    metrics_dir = os.path.join(os.path.dirname(__file__), "../src/metrics")
    for root, _, files in os.walk(metrics_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            with open(path) as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = (
                        [a.name for a in node.names]
                        if isinstance(node, ast.Import)
                        else ([node.module] if node.module else [])
                    )
                    for name in names:
                        assert not (name or "").startswith("modal"), \
                            f"{path} imports modal — metrics must be pure Python"
