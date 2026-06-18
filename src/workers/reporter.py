"""
src/workers/reporter.py
========================
ReportGenerator — aggregates all metric results and judge verdicts
for a run into a final structured report.

Output: /data/outputs/runs/{run_id}/reports/final_report.json

Report structure:
{
  "run_id":    str,
  "generated_at": str,
  "languages": {
    "<slug>": {
      "language":       str,
      "regional_model": str,
      "tasks": {
        "translation": {
          "light_metrics":  {metric_name: scores_dict},
          "model_metrics":  {metric_name: scores_dict},
          "judge": {
            "total_judgments":    int,
            "regional_win_rate":  float,   # 0.0–1.0
            "gemma4_win_rate":    float,
            "tie_rate":           float,
            "by_dimension": {
              dim: {"regional": int, "gemma4": int, "tie": int}
            },
            "consistency":        float,   # agreement across swap_runs (0–1)
          }
        }
      },
      "classification": "A"|"B"|"C"|"D"|"E",
      "classification_rationale": str,
    }
  }
}

Classifications:
  A — Regional Superior:  regional wins >60% overall, COMET delta > +0.02
  B — Regional Preferred: regional wins 50–60% or COMET delta 0.00–0.02
  C — Comparable:         win rate 40–60%, COMET within ±0.01
  D — Gemma-4 Preferred:  gemma4 wins 50–60% or COMET delta < -0.02
  E — Gemma-4 Superior:   gemma4 wins >60%, COMET delta < -0.05

No Modal imports. Pure Python. CPU only.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional


class ReportGenerator:
    """
    Aggregates metrics and judge verdicts into a final report.
    Called from the Phase 4 orchestrator after all scoring is done.
    """

    def generate(
        self,
        run_id: str,
        slug: str,
        task: str,
        language: str,
        regional_model_id: str,
    ) -> dict:
        """
        Builds the report section for one (slug, task) pair.
        Reads from the outputs volume, writes final_report.json.
        Returns the report dict.
        """
        from src.pipeline.run import (
            judge_path,
            metric_path,
            report_path,
            run_dir,
            update_manifest_status,
        )

        # ── Load light metrics ───────────────────────────────────────────────
        light_metrics = {}
        model_metrics = {}

        metrics_dir = os.path.join(run_dir(run_id), "metrics")
        if os.path.isdir(metrics_dir):
            for fname in os.listdir(metrics_dir):
                if not fname.endswith(".jsonl"):
                    continue
                fpath = os.path.join(metrics_dir, fname)
                records = _load_jsonl(fpath)
                for rec in records:
                    mname = rec.get("metric", fname.replace(".jsonl", ""))
                    # Determine tier by metric name prefix
                    if mname in ("comet", "bertscore"):
                        model_metrics[mname] = rec.get("scores", {})
                    else:
                        light_metrics[mname] = rec.get("scores", {})

        # ── Load judge verdicts ──────────────────────────────────────────────
        vpath    = judge_path(run_id, slug, task)
        verdicts = _load_jsonl(vpath)
        judge_summary = _summarize_verdicts(verdicts)

        # ── Classify ─────────────────────────────────────────────────────────
        classification, rationale = _classify(
            light_metrics=light_metrics,
            model_metrics=model_metrics,
            judge_summary=judge_summary,
        )

        # ── Assemble report ──────────────────────────────────────────────────
        report = {
            "run_id":        run_id,
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "languages": {
                slug: {
                    "language":       language,
                    "regional_model": regional_model_id,
                    "tasks": {
                        task: {
                            "light_metrics": light_metrics,
                            "model_metrics": model_metrics,
                            "judge":         judge_summary,
                        }
                    },
                    "classification":           classification,
                    "classification_rationale": rationale,
                }
            },
        }

        # ── Write report ──────────────────────────────────────────────────────
        rpath = report_path(run_id)
        os.makedirs(os.path.dirname(rpath), exist_ok=True)

        # Merge with existing report if present (multi-language runs)
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

        print(f"[ReportGenerator] Report written to {rpath}")
        print(f"[ReportGenerator] Classification for {slug}/{task}: {classification} — {rationale}")

        return report


# ── Verdict aggregation ──────────────────────────────────────────────────────

def _summarize_verdicts(verdicts: List[dict]) -> dict:
    if not verdicts:
        return {
            "total_judgments":   0,
            "regional_win_rate": None,
            "gemma4_win_rate":   None,
            "tie_rate":          None,
            "by_dimension":      {},
            "consistency":       None,
        }

    total     = len(verdicts)
    regional  = sum(1 for v in verdicts if v.get("winner_model") == "regional")
    gemma4    = sum(1 for v in verdicts if v.get("winner_model") == "gemma4")
    tie_count = sum(1 for v in verdicts if v.get("winner_model") == "tie")

    # By-dimension breakdown
    by_dim: Dict[str, dict] = {}
    for v in verdicts:
        dim = v.get("dimension", "unknown")
        by_dim.setdefault(dim, {"regional": 0, "gemma4": 0, "tie": 0})
        model = v.get("winner_model", "tie")
        if model in by_dim[dim]:
            by_dim[dim][model] += 1

    # Consistency: for each (prompt_id, dimension) pair, do swap_run 0 and 1 agree on winner_model?
    from collections import defaultdict
    pair_votes: dict = defaultdict(list)
    for v in verdicts:
        key = (v.get("prompt_id"), v.get("dimension"))
        pair_votes[key].append(v.get("winner_model"))

    consistent = sum(
        1 for votes in pair_votes.values()
        if len(votes) >= 2 and votes[0] == votes[1]
    )
    total_pairs = len(pair_votes)
    consistency = round(consistent / total_pairs, 4) if total_pairs else None

    return {
        "total_judgments":   total,
        "regional_win_rate": round(regional / total, 4),
        "gemma4_win_rate":   round(gemma4  / total, 4),
        "tie_rate":          round(tie_count / total, 4),
        "by_dimension":      by_dim,
        "consistency":       consistency,
    }


# ── Classification ────────────────────────────────────────────────────────────

def _classify(
    light_metrics: dict,
    model_metrics: dict,
    judge_summary: dict,
) -> tuple[str, str]:
    """
    Returns (class_letter, rationale_string).

    Primary signal: judge win rate.
    Secondary signal: COMET delta (regional - gemma4), if available.
    Fallback to BLEU/chrF if neither judge nor COMET available.
    """
    win_rate = judge_summary.get("regional_win_rate")
    comet_r  = _get_comet(model_metrics, "regional")
    comet_g  = _get_comet(model_metrics, "gemma4")
    comet_delta = (comet_r - comet_g) if (comet_r is not None and comet_g is not None) else None

    # ── Judge-based classification ─────────────────────────────────────────
    if win_rate is not None:
        if win_rate > 0.60:
            cls = "A"
            rationale = f"Regional wins {win_rate:.0%} of judge comparisons"
        elif win_rate >= 0.50:
            cls = "B"
            rationale = f"Regional wins {win_rate:.0%} of judge comparisons (modest preference)"
        elif win_rate >= 0.40:
            cls = "C"
            rationale = f"Win rate {win_rate:.0%} — models comparable"
        elif win_rate >= 0.30:
            cls = "D"
            rationale = f"Gemma-4 preferred — regional wins only {win_rate:.0%}"
        else:
            cls = "E"
            rationale = f"Gemma-4 strongly preferred — regional wins only {win_rate:.0%}"

        # Refine with COMET if available
        if comet_delta is not None:
            comet_str = f"; COMET Δ={comet_delta:+.4f}"
            if comet_delta > 0.02 and cls in ("B", "C"):
                cls = "A"
                rationale = f"COMET confirms regional superior{comet_str}"
            elif comet_delta < -0.05 and cls in ("C", "D"):
                cls = "E"
                rationale = f"COMET confirms Gemma-4 strongly preferred{comet_str}"
            else:
                rationale += comet_str
        return cls, rationale

    # ── COMET-only fallback ────────────────────────────────────────────────
    if comet_delta is not None:
        if comet_delta > 0.05:
            return "A", f"COMET Δ={comet_delta:+.4f} — regional superior"
        elif comet_delta > 0.00:
            return "B", f"COMET Δ={comet_delta:+.4f} — regional modestly preferred"
        elif comet_delta >= -0.01:
            return "C", f"COMET Δ={comet_delta:+.4f} — comparable"
        elif comet_delta >= -0.05:
            return "D", f"COMET Δ={comet_delta:+.4f} — Gemma-4 preferred"
        else:
            return "E", f"COMET Δ={comet_delta:+.4f} — Gemma-4 strongly preferred"

    # ── BLEU-only last resort ──────────────────────────────────────────────
    bleu_scores = light_metrics.get("bleu", {})
    if bleu_scores:
        avg = sum(bleu_scores.values()) / len(bleu_scores)
        if avg > 20:
            return "B", f"BLEU {avg:.1f} — plausible translations, no judge or COMET"
        elif avg > 5:
            return "C", f"BLEU {avg:.1f} — modest translation quality, no judge"
        else:
            return "D", f"BLEU {avg:.1f} — low translation quality, no judge"

    return "C", "Insufficient signal to classify — no judge verdicts, COMET, or BLEU available"


def _get_comet(model_metrics: dict, model_role: str) -> Optional[float]:
    """
    Extracts an average COMET score from the model_metrics dict.
    model_role: 'regional' or 'gemma4' — not actually used since we load
    separate metric files per model; just returns the mean COMET score.
    """
    comet_scores = model_metrics.get("comet", {})
    if not comet_scores:
        return None
    vals = [v for v in comet_scores.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def _load_jsonl(path: str) -> List[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []