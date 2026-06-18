from src.metrics.base import BaseMetric, ComputeTier, MetricResult, MetricStage

CURRENCY_WORDS = {
    "Tamil":                ["ரூபாய்", "ரூ", "rupee", "rupees"],
    "Gujarati":             ["રૂપિયા", "rupee", "rupees"],
    "Kannada":              ["ರೂಪಾಯಿ", "rupee", "rupees"],
    "Marathi":              ["रुपये", "रुपया", "rupee", "rupees"],
    "Arabic":               ["دينار", "درهم", "dollar", "دولار"],
    "Hebrew":               ["שקל", "shekel", "shekels"],
    "Korean":               ["원", "달러", "won", "dollar"],
    "Malay":                ["ringgit", "RM", "dollar"],
    "Swahili":              ["shilingi", "dola", "dollar"],
    "Amharic":              ["ብር", "birr", "dollar"],
    "French":               ["euro", "euros", "dollar", "dollars"],
    "Swedish":              ["krona", "kronor", "dollar"],
    "Czech":                ["koruna", "korun", "dollar"],
    "Greek":                ["ευρώ", "euro", "δολάριο"],
    "Brazilian Portuguese": ["real", "reais", "dólar", "dollar"],
    "Māori":                ["dollar", "dollars", "tāra"],
    "Tok Pisin":            ["kina", "toea", "dollar"],
}


class CurrencyUnitMetric(BaseMetric):
    name         = "currency_unit"
    stage        = MetricStage.POST_INFERENCE
    compute_tier = ComputeTier.LIGHT
    task_types   = ["instructions"]
    category     = "number_verbalization"

    def compute(self, outputs, language, task):
        expected_words = CURRENCY_WORDS.get(language, ["dollar"])
        correct = total = 0
        for o in outputs:
            if "currency" not in o.get("expected_constraints", {}).get("word_form_types", []):
                continue
            text = o.get("output", "").lower()
            total += 1
            if any(word.lower() in text for word in expected_words):
                correct += 1

        if total == 0:
            return MetricResult(
                metric_name="currency_unit", language=language, task=task,
                scores={"currency_unit_accuracy": None}, sample_count=0,
                notes="No currency samples in this batch",
            )
        return MetricResult(
            metric_name="currency_unit", language=language, task=task,
            scores={"currency_unit_accuracy": round(correct / total, 4)},
            sample_count=total,
        )
