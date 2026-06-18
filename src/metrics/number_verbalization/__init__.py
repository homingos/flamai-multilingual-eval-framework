"""Shared helpers for number verbalization metrics."""


def get_numbers(sample: dict) -> list:
    return sample.get("expected_constraints", {}).get("numbers_in_prompt", [])


def get_rule(sample: dict) -> str:
    return sample.get("expected_constraints", {}).get("rule", "")


def should_be_digit_by_digit(sample: dict) -> bool:
    return get_rule(sample) in ("digit_by_digit", "mixed")


def should_be_word_form(sample: dict) -> bool:
    return get_rule(sample) in ("word_form", "mixed")
