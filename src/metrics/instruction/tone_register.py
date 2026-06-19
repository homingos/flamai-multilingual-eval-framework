"""
src/metrics/instruction/tone_register.py
===========================================
I5 — Register / Tone Detection.

Applies to: tone_style category.
Checks whether the model's output matches the expected register
(formal/informal/friendly/empathetic/direct) specified in
expected_constraints.tone.

Method: rule-based proxy using per-language honorific and register
markers, per docs/evaluation-metrics.md §2.5 ("rule-based proxies e.g.
presence of formal honorifics per language"). A multilingual tone
classifier was considered but rejected — no single off-the-shelf model
covers register distinctions reliably across all 17 languages, and a
wrong classifier verdict is worse than an honest "low confidence" rule
match. This is why the spec table marks I5 as "Partial."

Coverage caveat: marker lists are necessarily incomplete and asymmetric
across languages — some have well-documented formal/informal lexical
splits (Korean, Japanese-adjacent honorific systems; Tamil/Marathi
second-person forms), others have weaker or no marker(s) (e.g. French
has tu/vous but few would-be informal lexical tells beyond pronoun
choice, which doesn't reliably surface in short responses). Treat this
metric's scores as directional, not authoritative — the LLM judge's
holistic dimensions are a stronger signal where they're available.
"""
from __future__ import annotations

from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage

# Per-language (formal_markers, informal_markers) — presence of either
# set of markers in the response is used to classify the realized register.
# Markers chosen for precision (distinctive forms), not recall — many
# correct responses will show zero markers and are scored as "indeterminate"
# rather than guessed.
REGISTER_MARKERS = {
    "Tamil":     {"formal": ["நீங்கள்", "தாங்கள்", "வணக்கம்"], "informal": ["நீ ", "உனக்கு"]},
    "Gujarati":  {"formal": ["તમે", "આપ "],                     "informal": ["તું "]},
    "Kannada":   {"formal": ["ನೀವು", "ತಾವು"],                   "informal": ["ನೀನು"]},
    "Marathi":   {"formal": ["आपण", "तुम्ही"],                   "informal": ["तू "]},
    "Arabic":    {"formal": ["حضرتك", "سيادتكم"],                "informal": ["انت ", "إنت "]},
    "Hebrew":    {"formal": ["אדוני", "גברתי", "בבקשה"],          "informal": ["אתה ", "את "]},
    "Korean":    {"formal": ["습니다", "십시오", "께서"],          "informal": ["야", "어", "지마"]},
    "Malay":     {"formal": ["anda", "tuan", "puan"],            "informal": ["kau ", "lu "]},
    "Swahili":   {"formal": ["tafadhali", "bwana", "bibi"],      "informal": ["wewe "]},
    "Amharic":   {"formal": ["እርስዎ", "አቶ", "ወይዘሮ"],            "informal": ["አንቴ"]},
    "French":    {"formal": ["vous ", "veuillez", "monsieur", "madame"], "informal": ["tu ", "salut"]},
    "Swedish":   {"formal": ["ni ", "vänligen"],                 "informal": ["du ", "hej "]},
    "Czech":     {"formal": ["vy ", "prosím", "vážený"],         "informal": ["ty "]},
    "Greek":     {"formal": ["εσείς", "παρακαλώ", "κύριε", "κυρία"], "informal": ["εσύ "]},
    "Brazilian Portuguese": {"formal": ["senhor", "senhora", "por favor"], "informal": ["você ", "tu "]},
    "Māori":     {"formal": ["koutou", "tēnā koe"],              "informal": ["koe "]},
    "Tok Pisin": {"formal": ["plis", "tenkyu tru"],              "informal": ["yu ", "olsem wanem"]},
}

# Maps the dataset's expected_constraints.tone values onto a target register.
# "friendly"/"empathetic" lean informal-warm; "formal"/"professional" lean formal;
# "direct" is register-neutral (no check performed — always counted as a pass
# since directness is about content brevity, not formality markers).
TONE_TO_REGISTER = {
    "formal":       "formal",
    "professional": "formal",
    "friendly":     "informal",
    "empathetic":   "informal",
    "direct":       None,   # no register signal expected — not scored
}


def _classify_register(text: str, markers: dict) -> str:
    """Returns 'formal', 'informal', or 'indeterminate' based on marker presence."""
    lowered = text  # most marker languages are non-Latin scripts; case-fold is a no-op there
    formal_hit   = any(m in lowered for m in markers.get("formal", []))
    informal_hit = any(m in lowered for m in markers.get("informal", []))
    if formal_hit and not informal_hit:
        return "formal"
    if informal_hit and not formal_hit:
        return "informal"
    return "indeterminate"


class ToneRegisterMetric(BaseMetric):
    name         = "tone_register"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = "tone_style"

    def compute(self, outputs, language, task):
        markers = REGISTER_MARKERS.get(language)
        if markers is None:
            return MetricResult(
                metric_name="tone_register", language=language, task=task,
                scores={}, sample_count=0,
                errors=[f"No register markers defined for language '{language}'"],
            )

        correct = scored = indeterminate = skipped = 0

        for o in outputs:
            tone = o.get("expected_constraints", {}).get("tone", "")
            target_register = TONE_TO_REGISTER.get(tone)

            if target_register is None:
                skipped += 1
                continue

            text = o.get("output", "")
            detected = _classify_register(text, markers)

            if detected == "indeterminate":
                indeterminate += 1
                continue

            scored += 1
            if detected == target_register:
                correct += 1

        rate = round(correct / scored, 4) if scored > 0 else None
        return MetricResult(
            metric_name="tone_register", language=language, task=task,
            scores={
                "tone_register_accuracy": rate,
                "correct":      correct,
                "scored":       scored,
                "indeterminate": indeterminate,
                "skipped_neutral_tone": skipped,
            },
            sample_count=len(outputs),
            notes=(
                "Rule-based marker proxy, not a trained classifier — treat as "
                "directional. High 'indeterminate' count means most responses "
                "showed no detectable formality marker either way; this is "
                "expected for short or neutral responses and is not a failure."
            ),
        )