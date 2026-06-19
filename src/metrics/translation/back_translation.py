"""
src/metrics/translation/back_translation.py
===============================================
T5 — Back-translation Consistency.

Applies to: translation, en→target direction only (per spec — verifies
the round trip from English to target and back preserves meaning).

Method: translate the model's target-language output back to English
using a fixed MarianMT model, then score sacrebleu against the
original English source. This is NOT comparing the regional model to
Gemma-4 — it's an internal consistency check on each model's own
output independent of the other model.

MODEL tier: loads a MarianMT checkpoint (~300MB) per language. Runs in
ModelMetricWorker alongside COMET/BERTScore.

Model coverage — IMPORTANT:
Helsinki-NLP/opus-mt does not have a dedicated bilingual {lang}-en
checkpoint for every language in this evaluation. Where no dedicated
model exists, this metric uses the appropriate Helsinki-NLP multilingual
checkpoint (opus-mt-mul-en or opus-mt-inc-en) as a documented fallback.
Where no checkpoint of any kind is confirmed to cover the language, the
metric explicitly skips it and reports why — it does NOT guess a model
ID, because a wrong/low-quality back-translation model would produce a
score that looks like a translation-quality signal but is actually a
back-translation-model-quality signal. A skipped result is more honest
than a wrong number.

This mapping should be re-verified against current Hugging Face Hub
listings before relying on it for Korean, Malay, Swahili, Brazilian
Portuguese, and Tok Pisin specifically — these were not confirmed
during implementation and are marked UNVERIFIED below.
"""
from __future__ import annotations

from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage

# {language: (model_id, coverage_note)}
# coverage_note is surfaced in MetricResult.notes so a low/high score is
# never read without knowing which checkpoint produced it.
BACK_TRANSLATION_MODELS: dict[str, tuple[str, str]] = {
    "Tamil":                ("Helsinki-NLP/opus-mt-mul-en", "multilingual fallback (no dedicated ta-en checkpoint)"),
    "Gujarati":              ("Helsinki-NLP/opus-mt-inc-en", "Indic-family dedicated checkpoint"),
    "Kannada":               ("Helsinki-NLP/opus-mt-mul-en", "multilingual fallback (no dedicated kn-en checkpoint)"),
    "Marathi":               ("Helsinki-NLP/opus-mt-inc-en", "Indic-family dedicated checkpoint"),
    "Arabic":                ("Helsinki-NLP/opus-mt-ar-en",  "dedicated bilingual checkpoint"),
    "Hebrew":                ("Helsinki-NLP/opus-mt-mul-en", "multilingual fallback (no dedicated he-en checkpoint)"),
    "Korean":                ("Helsinki-NLP/opus-mt-ko-en",  "dedicated bilingual checkpoint — UNVERIFIED at implementation time, confirm before trusting"),
    "Malay":                 (None, "no confirmed ms-en checkpoint — skipped, do not guess"),
    "Swahili":               (None, "no confirmed sw-en checkpoint — skipped, do not guess"),
    "Amharic":               ("Helsinki-NLP/opus-mt-mul-en", "multilingual fallback (am covered under opus-mt-en-sem family upstream)"),
    "French":                ("Helsinki-NLP/opus-mt-fr-en",  "dedicated bilingual checkpoint"),
    "Swedish":               ("Helsinki-NLP/opus-mt-mul-en", "multilingual fallback (no dedicated sv-en checkpoint confirmed)"),
    "Czech":                 ("Helsinki-NLP/opus-mt-mul-en", "multilingual fallback (no dedicated cs-en checkpoint confirmed)"),
    "Greek":                 ("Helsinki-NLP/opus-mt-mul-en", "multilingual fallback (no dedicated el-en checkpoint confirmed)"),
    "Brazilian Portuguese":  (None, "opus-mt-mul-en covers generic 'pt', not BR-specific — skipped to avoid PT/BR conflation"),
    "Māori":                 ("Helsinki-NLP/opus-mt-mul-en", "multilingual fallback (mi included in mul-en language list)"),
    "Tok Pisin":             (None, "no confirmed tpi-en checkpoint of any kind — skipped, do not guess"),
}

# Module-level cache — {model_id: (model, tokenizer)}. Avoids reloading
# the same checkpoint across languages that share opus-mt-mul-en / inc-en.
_model_cache: dict = {}


def _get_marian(model_id: str):
    if model_id not in _model_cache:
        from transformers import MarianMTModel, MarianTokenizer
        tokenizer = MarianTokenizer.from_pretrained(model_id)
        model     = MarianMTModel.from_pretrained(model_id)
        _model_cache[model_id] = (model, tokenizer)
    return _model_cache[model_id]


def _back_translate(texts: list[str], model_id: str, batch_size: int = 8) -> list[str]:
    model, tokenizer = _get_marian(model_id)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # MarianMT multilingual checkpoints (mul-en, inc-en) require a
        # target-language prefix token on the source side for some
        # variants; opus-mt-mul-en/inc-en are many-to-one (→en) so no
        # prefix is needed here — they infer source language automatically.
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**encoded, max_length=512)
        results.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
    return results


class BackTranslationMetric(BaseMetric):
    name         = "back_translation"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.MODEL
    task_types   = ["translation"]
    category     = None

    def compute(self, outputs, language, task):
        mapping = BACK_TRANSLATION_MODELS.get(language)
        if mapping is None:
            return MetricResult(
                metric_name="back_translation", language=language, task=task,
                scores={}, sample_count=0,
                errors=[f"No back-translation model mapping defined for '{language}'"],
            )

        model_id, coverage_note = mapping
        if model_id is None:
            return MetricResult(
                metric_name="back_translation", language=language, task=task,
                scores={"back_translation_bleu": None}, sample_count=0,
                notes=f"Skipped — {coverage_note}",
            )

        # Spec: only applies to en→target direction samples
        en_to_tgt = [o for o in outputs if o.get("direction") == "en→target"]
        if not en_to_tgt:
            return MetricResult(
                metric_name="back_translation", language=language, task=task,
                scores={"back_translation_bleu": None}, sample_count=0,
                notes="No en→target samples in this batch",
            )

        try:
            import sacrebleu
        except ImportError:
            return MetricResult(
                metric_name="back_translation", language=language, task=task,
                scores={}, sample_count=0, errors=["sacrebleu not installed"],
            )

        hypotheses = [o.get("output", "") for o in en_to_tgt]
        sources    = [o.get("source", "")  for o in en_to_tgt]

        # Skip empty hypotheses — back-translating an empty string is
        # meaningless and would drag the corpus BLEU down for reasons
        # unrelated to back-translation consistency itself.
        pairs = [(h, s) for h, s in zip(hypotheses, sources) if h.strip()]
        if not pairs:
            return MetricResult(
                metric_name="back_translation", language=language, task=task,
                scores={"back_translation_bleu": None}, sample_count=0,
                notes="All en→target outputs were empty",
            )
        hyps, srcs = zip(*pairs)

        try:
            back_translated = _back_translate(list(hyps), model_id)
        except Exception as exc:
            return MetricResult(
                metric_name="back_translation", language=language, task=task,
                scores={}, sample_count=len(pairs),
                errors=[f"Back-translation model '{model_id}' failed: {exc}"],
            )

        try:
            result = sacrebleu.corpus_bleu(back_translated, [list(srcs)])
            score = round(result.score, 2)
        except Exception as exc:
            return MetricResult(
                metric_name="back_translation", language=language, task=task,
                scores={}, sample_count=len(pairs),
                errors=[f"sacrebleu scoring failed: {exc}"],
            )

        return MetricResult(
            metric_name="back_translation", language=language, task=task,
            scores={"back_translation_bleu": score},
            sample_count=len(pairs),
            notes=f"Model: {model_id} ({coverage_note})",
        )