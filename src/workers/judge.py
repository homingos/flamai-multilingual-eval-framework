"""
src/workers/judge.py
=====================
JudgeWorker — LLM-as-judge for pairwise output comparison.

Supports two judge providers, selected automatically by model string prefix:
  - Anthropic  →  judge_model starts with "claude-"   (e.g. "claude-haiku-4-5")
  - Gemini     →  judge_model starts with "gemini-"   (e.g. "gemini-2.0-flash")

The worker reads whichever API key is present in the environment:
  ANTHROPIC_API_KEY  — for Anthropic models
  GEMINI_API_KEY     — for Gemini models

Both env vars can coexist in the same Modal secret so you can switch
judge models without redeploying.

Each pair is judged twice with A/B order swapped (swap_runs=2) to
control for position bias.

Output written to:
  /data/outputs/runs/{run_id}/judge/{slug}_{task}_verdicts.jsonl

Each verdict record:
  {
    "prompt_id":      str,
    "task":           str,
    "language":       str,
    "regional_model": str,
    "dimension":      str,           # e.g. "fluency"
    "winner":         "A"|"B"|"tie",
    "winner_model":   "regional"|"gemma4"|"tie",
    "confidence":     "high"|"medium"|"low",
    "reasoning":      str,
    "swap_run":       0|1,           # 0 = regional=A, 1 = regional=B
    "judge_model":    str,           # which model was used
    "judged_at":      str,
  }

No Modal imports. No GPU. CPU-only (remote API calls).
Registered in modal_app.py as JudgeWorkerModal.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import List, Optional


# ── Judge prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a rigorous multilingual output evaluator. You will be shown two AI-generated
responses (Response A and Response B) to the same prompt. Your task is to judge which
response is better according to the specified dimension.

Rules:
- Be BLIND to which model generated each response — judge content only.
- If both responses are equally good (or bad), declare a tie.
- Be concise: 1-2 sentences of reasoning maximum.
- Output JSON only — no preamble, no markdown fences.

Output format (JSON):
{
  "winner": "A" | "B" | "tie",
  "confidence": "high" | "medium" | "low",
  "reasoning": "<1-2 sentence explanation>"
}
"""

DIMENSIONS = {
    "translation": [
        ("fluency",    "Which response reads more naturally and fluently in the target language?"),
        ("adequacy",   "Which response preserves the meaning of the source text more accurately?"),
        ("overall",    "Overall, which response is the better translation?"),
    ],
    "instructions": [
        ("instruction_following", "Which response follows the given instructions more precisely?"),
        ("language_quality",      "Which response has better language quality and naturalness?"),
        ("overall",               "Overall, which response better addresses the prompt?"),
    ],
}

# ── Shared user message builder ───────────────────────────────────────────────

def _build_user_message(
    dimension_label: str,
    dimension_question: str,
    source_or_instruction: str,
    output_a: str,
    output_b: str,
) -> str:
    return f"""\
Dimension: {dimension_label}
Question: {dimension_question}

Prompt / Source text:
{source_or_instruction}

Response A:
{output_a}

Response B:
{output_b}

Judge which response better satisfies the dimension. Output JSON only."""


# ── Shared JSON parser ────────────────────────────────────────────────────────

def _parse_verdict(raw_text: str) -> dict:
    """Strips markdown fences and parses JSON. Returns raw dict."""
    text = raw_text.strip()
    if text.startswith("```"):
        # ```json\n...\n``` or ```\n...\n```
        parts = text.split("```")
        # parts[1] is the content block
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ── Anthropic provider ────────────────────────────────────────────────────────

def _call_anthropic(
    user_message: str,
    judge_model: str,
    api_key: str,
    max_retries: int = 3,
) -> dict:
    """
    Calls the Anthropic Messages API.
    Returns {"winner", "confidence", "reasoning", "error"}.
    """
    import urllib.request

    payload = json.dumps({
        "model":      judge_model,
        "max_tokens": 256,
        "system":     SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": user_message}],
    }).encode()

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type":      "application/json",
                    "x-api-key":         api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())

            raw_text = body["content"][0]["text"]
            verdict  = _parse_verdict(raw_text)
            return {
                "winner":     verdict.get("winner", "tie"),
                "confidence": verdict.get("confidence", "low"),
                "reasoning":  verdict.get("reasoning", ""),
                "error":      None,
            }
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"winner": "tie", "confidence": "low", "reasoning": "", "error": str(exc)}

    return {"winner": "tie", "confidence": "low", "reasoning": "", "error": "max_retries exceeded"}


# ── Gemini provider ───────────────────────────────────────────────────────────

def _call_gemini(
    user_message: str,
    judge_model: str,
    api_key: str,
    max_retries: int = 3,
) -> dict:
    """
    Calls the Google Gemini generateContent REST API.
    Endpoint: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    Returns {"winner", "confidence", "reasoning", "error"}.

    The system prompt is passed as a system_instruction field (Gemini ≥1.5).
    """
    import urllib.request

    # Gemini REST endpoint — no SDK needed
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{judge_model}:generateContent?key={api_key}"
    )

    payload = json.dumps({
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role":  "user",
                "parts": [{"text": user_message}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 256,
            "temperature":     0.0,
        },
    }).encode()

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())

            # Gemini response shape:
            # body["candidates"][0]["content"]["parts"][0]["text"]
            raw_text = body["candidates"][0]["content"]["parts"][0]["text"]
            verdict  = _parse_verdict(raw_text)
            return {
                "winner":     verdict.get("winner", "tie"),
                "confidence": verdict.get("confidence", "low"),
                "reasoning":  verdict.get("reasoning", ""),
                "error":      None,
            }
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"winner": "tie", "confidence": "low", "reasoning": "", "error": str(exc)}

    return {"winner": "tie", "confidence": "low", "reasoning": "", "error": "max_retries exceeded"}


# ── Provider router ───────────────────────────────────────────────────────────

def _call_judge(
    user_message: str,
    judge_model: str,
    anthropic_api_key: str,
    gemini_api_key: str,
    max_retries: int = 3,
) -> dict:
    """
    Routes to the correct provider based on model string prefix:
      "claude-*"  → Anthropic
      "gemini-*"  → Gemini
    Raises RuntimeError if the required API key is missing.
    """
    if judge_model.startswith("gemini-"):
        if not gemini_api_key:
            raise RuntimeError(
                f"Judge model '{judge_model}' requires GEMINI_API_KEY but it is not set. "
                "Add it to the phase2a-judge Modal secret."
            )
        return _call_gemini(user_message, judge_model, gemini_api_key, max_retries)

    elif judge_model.startswith("claude-"):
        if not anthropic_api_key:
            raise RuntimeError(
                f"Judge model '{judge_model}' requires ANTHROPIC_API_KEY but it is not set. "
                "Add it to the phase2a-judge Modal secret."
            )
        return _call_anthropic(user_message, judge_model, anthropic_api_key, max_retries)

    else:
        raise ValueError(
            f"Unknown judge model '{judge_model}'. "
            "Model string must start with 'claude-' (Anthropic) or 'gemini-' (Google)."
        )


# ── JudgeWorker ───────────────────────────────────────────────────────────────

class JudgeWorker:
    """
    LLM-as-judge worker. Compares regional model outputs vs Gemma-4 outputs.

    Supports Anthropic (claude-*) and Gemini (gemini-*) judge models.
    The active provider is selected by the judge_model argument — no code
    change needed when switching between them.

    Usage (from orchestrator):
        judge = JudgeWorkerModal()
        verdicts = judge.judge.remote(
            run_id="...", slug="tamil", task="translation",
            language="Tamil", regional_model_id="tamil-mistral-7b",
            judge_model="gemini-2.0-flash", swap_runs=2,
        )
    """

    def judge(
        self,
        run_id: str,
        slug: str,
        task: str,
        language: str,
        regional_model_id: str,
        judge_model: str = "gemini-2.0-flash",
        swap_runs: int = 2,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """
        Runs pairwise judgment for all prompts in (run_id, slug, task).

        For each prompt:
        - Loads regional output and Gemma-4 output
        - Judges each configured dimension
        - Repeats swap_runs times with A/B order swapped
        - Writes each verdict to JSONL and returns all verdicts

        Returns list of verdict dicts.
        """
        from src.pipeline.run import (
            gemma_output_path,
            judge_path,
            regional_output_path,
            append_output,
        )

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        gemini_key    = os.environ.get("GEMINI_API_KEY", "")

        # Fail fast — check key availability before starting any work
        if judge_model.startswith("gemini-") and not gemini_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to the phase2a-judge Modal secret:\n"
                "  modal secret create phase2a-judge GEMINI_API_KEY=<your-key>"
            )
        if judge_model.startswith("claude-") and not anthropic_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to the phase2a-judge Modal secret:\n"
                "  modal secret create phase2a-judge ANTHROPIC_API_KEY=<your-key>"
            )

        print(f"[JudgeWorker] Provider: {'Gemini' if judge_model.startswith('gemini-') else 'Anthropic'}")
        print(f"[JudgeWorker] Model:    {judge_model}")

        # Load outputs
        reg_path   = regional_output_path(run_id, slug, task)
        gemma_path = gemma_output_path(run_id, task)

        reg_outputs   = _load_jsonl(reg_path)
        gemma_outputs = _load_jsonl(gemma_path)

        if not reg_outputs or not gemma_outputs:
            print(f"[JudgeWorker] Missing outputs — regional: {len(reg_outputs)}, gemma: {len(gemma_outputs)}")
            return []

        # Index by prompt_id for fast lookup
        reg_by_id   = {o["prompt_id"]: o for o in reg_outputs}
        gemma_by_id = {o["prompt_id"]: o for o in gemma_outputs}

        common_ids = sorted(set(reg_by_id.keys()) & set(gemma_by_id.keys()))
        if limit:
            common_ids = common_ids[:limit]

        out_path       = judge_path(run_id, slug, task)
        already_judged = _load_judged_keys(out_path)

        dimensions = DIMENSIONS.get(task, DIMENSIONS["instructions"])
        verdicts   = []
        total      = len(common_ids) * swap_runs * len(dimensions)
        done       = 0

        print(f"[JudgeWorker] {len(common_ids)} prompts × {swap_runs} swaps × "
              f"{len(dimensions)} dimensions = {total} judgments")

        for prompt_id in common_ids:
            reg_out   = reg_by_id[prompt_id]
            gemma_out = gemma_by_id[prompt_id]

            if task == "translation":
                source_text = reg_out.get("source", "")
            else:
                source_text = (
                    reg_out.get("system_instruction", "")
                    + "\n\n"
                    + reg_out.get("user_prompt", "")
                )

            for swap_run in range(swap_runs):
                # swap_run 0: A=regional, B=gemma  |  swap_run 1: A=gemma, B=regional
                if swap_run == 0:
                    output_a, output_b = reg_out.get("output", ""), gemma_out.get("output", "")
                else:
                    output_a, output_b = gemma_out.get("output", ""), reg_out.get("output", "")

                for dim_label, dim_question in dimensions:
                    key = f"{prompt_id}_{swap_run}_{dim_label}"
                    if key in already_judged:
                        done += 1
                        continue

                    user_msg = _build_user_message(
                        dimension_label=dim_label,
                        dimension_question=dim_question,
                        source_or_instruction=source_text,
                        output_a=output_a,
                        output_b=output_b,
                    )

                    raw = _call_judge(
                        user_message=user_msg,
                        judge_model=judge_model,
                        anthropic_api_key=anthropic_key,
                        gemini_api_key=gemini_key,
                    )

                    # Translate A/B winner back to model identity
                    winner_letter = raw["winner"]
                    if winner_letter == "A":
                        winner_model = "regional" if swap_run == 0 else "gemma4"
                    elif winner_letter == "B":
                        winner_model = "gemma4" if swap_run == 0 else "regional"
                    else:
                        winner_model = "tie"

                    verdict = {
                        "prompt_id":      prompt_id,
                        "task":           task,
                        "language":       language,
                        "regional_model": regional_model_id,
                        "dimension":      dim_label,
                        "winner":         winner_letter,
                        "winner_model":   winner_model,
                        "confidence":     raw["confidence"],
                        "reasoning":      raw["reasoning"],
                        "swap_run":       swap_run,
                        "judge_model":    judge_model,
                        "judged_at":      datetime.now(timezone.utc).isoformat(),
                        "error":          raw.get("error"),
                    }
                    verdicts.append(verdict)
                    append_output(out_path, verdict)
                    done += 1

                    if done % 50 == 0:
                        print(f"[JudgeWorker] {done}/{total} judgments complete")

        print(f"[JudgeWorker] Done — {len(verdicts)} new verdicts written to {out_path}")
        return verdicts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> List[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def _load_judged_keys(path: str) -> set:
    """Returns set of '{prompt_id}_{swap_run}_{dimension}' already in the verdicts file."""
    keys = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                v = json.loads(line)
                k = f"{v['prompt_id']}_{v['swap_run']}_{v['dimension']}"
                keys.add(k)
    except FileNotFoundError:
        pass
    return keys