from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage


class LangAdherenceMetric(BaseMetric):
    name         = "lang_adherence"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = None   # applies to ALL instruction categories

    LANG_CODES = {
        "Tamil":                "ta",
        "Gujarati":             "gu",
        "Kannada":              "kn",
        "Marathi":              "mr",
        "Arabic":               "ar",
        "Hebrew":               "he",
        "Korean":               "ko",
        "Malay":                "ms",
        "Swahili":              "sw",
        "Amharic":              "am",
        "French":               "fr",
        "Swedish":              "sv",
        "Czech":                "cs",
        "Greek":                "el",
        "Brazilian Portuguese": "pt",
        "Māori":                "mi",
        "Tok Pisin":            "tpi",
    }

    def compute(self, outputs, language, task):
        from langdetect import detect, LangDetectException

        expected_code = self.LANG_CODES.get(language)
        if expected_code is None:
            return MetricResult(
                metric_name="lang_adherence", language=language, task=task,
                scores={}, sample_count=len(outputs),
                errors=[f"No ISO code mapped for language '{language}'"],
            )

        correct = total = 0
        errors = []
        for o in outputs:
            text = o.get("output", "")
            if not text.strip():
                continue
            total += 1
            try:
                detected = detect(text)
                if detected == expected_code:
                    correct += 1
            except LangDetectException as exc:
                errors.append(str(exc))

        rate = round(correct / total, 4) if total > 0 else 0.0
        return MetricResult(
            metric_name="lang_adherence", language=language, task=task,
            scores={"lang_adherence_rate": rate, "correct": correct, "total": total},
            sample_count=len(outputs), errors=errors[:5],
        )
