"""
src/workers/reporter.py
========================
ReportGenerator — aggregates pointwise judge verdicts for a run into a
final structured report. v2: no Gemma-4 comparison, score-based grading.

Output: /data/outputs/runs/{run_id}/reports/final_report.json

Report structure:
{
  "run_id":       str,
  "generated_at": str,
  "languages": {
    "<slug>": {
      "language":               str,
      "regional_model":         str,
      "classification":         "A"|"B"|"C"|"D",
      "classification_rationale": str,
      "avg_score":              float,  # 0–1
      "sample_count":           int,
      # per-dimension avgs (translation)
      "fluency_avg":            float,
      "adequacy_avg":           float,
      "overall_avg":            float,
      # per-dimension avgs (instructions)
      "language_compliance_avg":   float,
      "instruction_following_avg": float,
      "helpfulness_avg":           float,
    }
  }
}

Grade mapping:
  A: avg_score >= 0.75
  B: avg_score >= 0.50
  C: avg_score >= 0.25
  D: avg_score < 0.25

No Modal imports. Pure Python. CPU only.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional


class ReportGenerator:

    def generate(
        self,
        run_id: str,
        slug: str,
        task: str,
        language: str,
        regional_model_id: str,
    ) -> dict:
        from src.pipeline.run import judge_path, report_path, run_dir

        vpath    = judge_path(run_id, slug, task)
        verdicts = _load_jsonl(vpath)
        scores   = _aggregate_scores(verdicts)
        cls, rationale = _classify(scores)

        lang_entry: dict = {
            "language":               language,
            "regional_model":         regional_model_id,
            "classification":         cls,
            "classification_rationale": rationale,
            "avg_score":              scores.get("avg_score"),
            "sample_count":           scores.get("sample_count", 0),
        }
        for k, v in scores.items():
            if k.endswith("_avg") or k == "avg_score":
                lang_entry[k] = v

        report: dict = {
            "run_id":       run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "languages":    {slug: lang_entry},
        }

        rpath = report_path(run_id)
        os.makedirs(os.path.dirname(rpath), exist_ok=True)

        # Merge with existing report (multi-language fan-out)
        existing = {}
        try:
            with open(rpath) as f:
                existing = json.load(f)
        except FileNotFoundError:
            pass

        if existing:
            existing.setdefault("languages", {}).update(report["languages"])
            existing["generated_at"] = report["generated_at"]
            report = existing

        tmp = rpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(report, f, indent=2, default=str)
        os.replace(tmp, rpath)

        print(f"[ReportGenerator] Written to {rpath}")
        print(f"[ReportGenerator] {slug}/{task}: {cls} | avg_score={scores.get('avg_score', '?')}")
        return report


# ── Read-side API ─────────────────────────────────────────────────────────────

def load_report(run_id: str) -> dict:
    from src.pipeline.run import report_path
    try:
        with open(report_path(run_id)) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_language_summary(report: dict, slug: str, task: str = "translation") -> dict:
    """Extracts the fields for one language — consumed by entrypoints.py."""
    lang = report.get("languages", {}).get(slug, {})
    return {
        "language":               lang.get("language", slug),
        "regional_model":         lang.get("regional_model", ""),
        "classification":         lang.get("classification", "?"),
        "avg_score":              lang.get("avg_score"),
        "sample_count":           lang.get("sample_count", 0),
        "fluency_avg":            lang.get("fluency_avg"),
        "adequacy_avg":           lang.get("adequacy_avg"),
        "overall_avg":            lang.get("overall_avg"),
        "language_compliance_avg":    lang.get("language_compliance_avg"),
        "instruction_following_avg":  lang.get("instruction_following_avg"),
        "helpfulness_avg":            lang.get("helpfulness_avg"),
    }


def summarize_run(run_id: str, slugs: list, task: str = "translation") -> dict:
    report  = load_report(run_id)
    results = {slug: get_language_summary(report, slug, task) for slug in slugs}
    counts  = {
        grade: sum(1 for r in results.values() if r["classification"] == grade)
        for grade in ["A", "B", "C", "D", "?"]
    }
    return {"results": results, "classification_counts": counts}


# ── Score aggregation ─────────────────────────────────────────────────────────

def _aggregate_scores(verdicts: List[dict]) -> dict:
    """
    Given a list of pointwise verdict dicts (each with 'score' and 'dimension'),
    returns avg score per dimension + overall avg + sample count.
    """
    if not verdicts:
        return {"avg_score": None, "sample_count": 0}

    dim_scores: Dict[str, List[float]] = {}
    prompt_ids = set()

    for v in verdicts:
        if v.get("error"):
            continue
        score = v.get("score")
        dim   = v.get("dimension")
        pid   = v.get("prompt_id")
        if score is None or not dim:
            continue
        dim_scores.setdefault(dim, []).append(float(score))
        if pid:
            prompt_ids.add(pid)

    if not dim_scores:
        return {"avg_score": None, "sample_count": 0}

    dim_avgs    = {dim: round(sum(vals) / len(vals), 4) for dim, vals in dim_scores.items()}
    overall_avg = round(sum(dim_avgs.values()) / len(dim_avgs), 4)

    result: dict = {"avg_score": overall_avg, "sample_count": len(prompt_ids)}
    for dim, avg in dim_avgs.items():
        result[f"{dim}_avg"] = avg

    return result


# ── Classification ────────────────────────────────────────────────────────────

def _classify(scores: dict) -> tuple[str, str]:
    avg = scores.get("avg_score")
    if avg is None:
        return "?", "No judge verdicts available"
    if avg >= 0.75:
        return "A", f"Good quality — avg score {avg:.3f}"
    if avg >= 0.50:
        return "B", f"Acceptable quality — avg score {avg:.3f}"
    if avg >= 0.25:
        return "C", f"Marginal quality — avg score {avg:.3f}"
    return "D", f"Poor quality — avg score {avg:.3f}"


def _load_jsonl(path: str) -> List[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []
