"""
Stratified random sampler for evaluation datasets.

Picks n samples reproducibly, stratified by:
  - translation: direction (en→target, target→en) — n/2 each
  - instructions: category — n/6 each (±1 for rounding)
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Optional

from src.pipeline.loader import load_samples


def sample_stratified(
    task: str,
    slug: str,
    n: int = 200,
    seed: int = 42,
) -> list[dict]:
    """
    Return n samples stratified by direction (translation) or category (instructions).
    Reproducible: same task/slug/n/seed always yields the same samples.
    """
    all_samples = load_samples(task, slug)
    rng = random.Random(seed)

    if task == "translation":
        return _stratify_by_field(all_samples, "direction", n, rng)
    elif task == "instructions":
        return _stratify_by_field(all_samples, "category", n, rng)
    else:
        raise ValueError(f"Unknown task: {task!r}")


def _stratify_by_field(
    samples: list[dict],
    field: str,
    n: int,
    rng: random.Random,
) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        buckets[s[field]].append(s)

    keys = sorted(buckets)
    k = len(keys)
    if k == 0:
        return []

    # Distribute n across k buckets: floor for all, then +1 for the remainder
    base, remainder = divmod(n, k)
    quota: dict[str, int] = {key: base for key in keys}
    for key in keys[:remainder]:
        quota[key] += 1

    result: list[dict] = []
    for key in keys:
        pool = buckets[key]
        take = min(quota[key], len(pool))
        result.extend(rng.sample(pool, take))

    rng.shuffle(result)
    return result
