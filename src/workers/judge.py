"""
src/workers/judge.py
=====================
JudgeWorker — pointwise LLM-as-judge for standalone output evaluation.

Supports two judge providers, selected by model string prefix:
  - Anthropic  →  judge_model starts with "claude-"   (e.g. "claude-haiku-4-5")
  - Gemini     →  judge_model starts with "gemini-"   (e.g. "gemini-3.5-flash")

The worker scores each regional model output on a 0–1 scale per rubric dimension.
No Gemma-4 comparison — the regional output is evaluated standalone.
Rubrics are loaded from config/rubrics/{task}.yaml.

Output written to:
  /data/outputs/runs/{run_id}/judge/{slug}_{task}_verdicts.jsonl

Each verdict record:
  {
    "prompt_id":      str,
    "task":           str,
    "language":       str,
    "regional_model": str,
    "dimension":      str,
    "score":          0.0 | 0.25 | 0.5 | 0.75 | 1.0,
    "confidence":     "high" | "medium" | "low",
    "reasoning":      str,
    "judge_model":    str,
    "judged_at":      str,
    "error":          str | null,
  }

No Modal imports. No GPU. CPU-only (remote API calls).
Registered in modal_app.py as JudgeWorkerModal.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


# ── Rubric loader ─────────────────────────────────────────────────────────────

def _load_rubric(task: str) -> dict:
    """
    Loads config/rubrics/{task}.yaml from the repo root.
    Falls back to /data/registry/rubrics/{task}.yaml for Modal containers
    where the repo root may be at a different path.
    """
    import yaml  # pyyaml

    candidates = [
        Path(__file__).resolve().parent.parent.parent / "config" / "rubrics" / f"{task}.yaml",
        Path("/data/registry/rubrics") / f"{task}.yaml",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Rubric for task '{task}' not found. "
        f"Searched: {[str(c) for c in candidates]}"
    )


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a rigorous multilingual output evaluator. You will be shown an AI-generated
response to a prompt. Your task is to score it on a specific evaluation dimension
using the provided rubric.

Rules:
- Score the response on its own merits. Do NOT compare to any baseline or other model.
- Choose the score value that best matches the rubric criteria. Use only these values: 0.0, 0.25, 0.5, 0.75, 1.0
- Be concise: 1 sentence of reasoning maximum.
- Output JSON only — no preamble, no markdown fences, no text outside the JSON object.

Output format (JSON):
{
  "score": <0.0 | 0.25 | 0.5 | 0.75 | 1.0>,
  "confidence": "high" | "medium" | "low",
  "reasoning": "<1 sentence citing specific evidence>"
}
"""


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_pointwise_prompt(
    dimension_name: str,
    dimension_question: str,
    criteria: dict,
    context: str,
    candidate_text: str,
) -> str:
    criteria_lines = "\n".join(
        f"  {score}: {desc}" for score, desc in sorted(criteria.items(), reverse=True)
    )
    return f"""\
Dimension: {dimension_name}
Question: {dimension_question}

Scoring rubric:
{criteria_lines}

Context / Source:
{context}

Response to evaluate:
{candidate_text}

Score this response on the dimension above. Output JSON only."""


# ── Robust JSON parser ────────────────────────────────────────────────────────

def _parse_verdict(raw_text: str) -> dict:
    """
    Extracts the first valid JSON object from raw_text, tolerating common
    LLM output decorations: markdown fences, preamble, trailing text, backticks.
    Parses 'score' as float. Raises ValueError if no valid JSON found.
    """
    text = raw_text.strip()

    def _extract(s: str) -> dict:
        d = json.loads(s)
        if "score" in d:
            d["score"] = float(d["score"])
        return d

    # 1. Direct parse
    try:
        return _extract(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Fenced: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return _extract(fence_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Single backtick inline: `{...}`
    inline_match = re.search(r"`(\{.*?\})`", text, re.DOTALL)
    if inline_match:
        try:
            return _extract(inline_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. Brace-level scan for first complete object
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return _extract(text[start: i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break

    raise ValueError(f"No valid JSON object found in model output: {repr(text[:300])}")


# ── Anthropic provider ────────────────────────────────────────────────────────

def _call_anthropic(
    user_message: str,
    judge_model: str,
    api_key: str,
    max_retries: int = 3,
) -> dict:
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
                "score":      verdict.get("score", 0.0),
                "confidence": verdict.get("confidence", "low"),
                "reasoning":  verdict.get("reasoning", ""),
                "error":      None,
            }
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"score": 0.0, "confidence": "low", "reasoning": "", "error": str(exc)}

    return {"score": 0.0, "confidence": "low", "reasoning": "", "error": "max_retries exceeded"}


# ── Gemini provider ───────────────────────────────────────────────────────────

def _call_gemini(
    user_message: str,
    judge_model: str,
    api_key: str,
    max_retries: int = 3,
) -> dict:
    """
    Calls Google Gemini generateContent REST API.

    temperature=0.1 (not 0.0): avoids deterministic collapse on near-equal responses.
    No responseMimeType: without it Gemini reasons first then formats, producing
    genuine scores rather than the minimal-valid-JSON shortcut.
    """
    import urllib.request

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{judge_model}:generateContent?key={api_key}"
    )

    payload = json.dumps({
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {
            "maxOutputTokens": 8192,
            "temperature":     0.1,
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

            # Gemini thinking models return multiple parts; find last text without thoughtSignature
            parts    = body["candidates"][0]["content"]["parts"]
            raw_text = None
            for part in reversed(parts):
                if "text" in part and "thoughtSignature" not in part:
                    raw_text = part["text"]
                    break
            if raw_text is None:
                for part in reversed(parts):
                    if "text" in part:
                        raw_text = part["text"]
                        break
            if raw_text is None:
                raise ValueError(f"No text part found. Parts: {parts}")

            verdict = _parse_verdict(raw_text)
            return {
                "score":      verdict.get("score", 0.0),
                "confidence": verdict.get("confidence", "low"),
                "reasoning":  verdict.get("reasoning", ""),
                "error":      None,
            }

        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = 30 * (2 ** attempt)
                print(f"[JudgeWorker] 429 rate limit — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    return {"score": 0.0, "confidence": "low", "reasoning": "",
                            "error": f"HTTP 429: rate limited after {max_retries} retries"}
            else:
                err_body = exc.read().decode()[:200]
                return {"score": 0.0, "confidence": "low", "reasoning": "",
                        "error": f"HTTP {exc.code}: {err_body}"}
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"score": 0.0, "confidence": "low", "reasoning": "", "error": str(exc)}

    return {"score": 0.0, "confidence": "low", "reasoning": "", "error": "max_retries exceeded"}


# ── Provider router ───────────────────────────────────────────────────────────

def _call_judge(
    user_message: str,
    judge_model: str,
    anthropic_api_key: str,
    gemini_api_key: str,
    max_retries: int = 3,
) -> dict:
    if judge_model.startswith("gemini-"):
        if not gemini_api_key:
            raise RuntimeError(
                f"Judge model '{judge_model}' requires GEMINI_API_KEY but it is not set."
            )
        return _call_gemini(user_message, judge_model, gemini_api_key, max_retries)

    elif judge_model.startswith("claude-"):
        if not anthropic_api_key:
            raise RuntimeError(
                f"Judge model '{judge_model}' requires ANTHROPIC_API_KEY but it is not set."
            )
        return _call_anthropic(user_message, judge_model, anthropic_api_key, max_retries)

    else:
        raise ValueError(
            f"Unknown judge model '{judge_model}'. "
            "Must start with 'claude-' or 'gemini-'."
        )


# ── JudgeWorker ───────────────────────────────────────────────────────────────

class JudgeWorker:
    """
    Pointwise LLM-as-judge: scores each regional model output 0–1 per dimension.
    No Gemma-4 — outputs are evaluated standalone against the rubric.

    Usage (from orchestrator):
        judge = JudgeWorkerModal()
        verdicts = judge.judge.remote(
            run_id="...", slug="german", task="translation",
            language="German", regional_model_id="eurollm-22b",
            judge_model="gemini-3.5-flash",
        )
    """

    def judge(
        self,
        run_id: str,
        slug: str,
        task: str,
        language: str,
        regional_model_id: str,
        judge_model: str = "gemini-3.5-flash",
        limit: Optional[int] = None,
    ) -> List[dict]:
        """
        Scores all outputs in (run_id, slug, task) on every rubric dimension.
        Idempotent: skips already-judged (prompt_id, dimension) pairs.
        Returns list of verdict dicts.
        """
        from src.pipeline.run import regional_output_path, judge_path, append_output

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        gemini_key    = os.environ.get("GEMINI_API_KEY", "")

        if judge_model.startswith("gemini-") and not gemini_key:
            raise RuntimeError("GEMINI_API_KEY not set. Add it to the phase2a-judge Modal secret.")
        if judge_model.startswith("claude-") and not anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to the phase2a-judge Modal secret.")

        rubric     = _load_rubric(task)
        dimensions = rubric["dimensions"]

        print(f"[JudgeWorker] Provider: {'Gemini' if judge_model.startswith('gemini-') else 'Anthropic'}")
        print(f"[JudgeWorker] Model:    {judge_model}")
        print(f"[JudgeWorker] Task:     {task}  |  Dimensions: {[d['name'] for d in dimensions]}")

        # Free tier Gemini Flash ≈ 15 RPM → 4s keeps us safely under.
        call_delay_s = 4.0 if judge_model.startswith("gemini-") else 1.0

        reg_path = regional_output_path(run_id, slug, task)
        outputs  = _load_jsonl(reg_path)

        if not outputs:
            print(f"[JudgeWorker] No outputs found at {reg_path}")
            return []

        if limit:
            outputs = outputs[:limit]

        out_path       = judge_path(run_id, slug, task)
        already_judged = _load_judged_keys(out_path)

        verdicts = []
        total    = len(outputs) * len(dimensions)
        done     = 0

        print(f"[JudgeWorker] {len(outputs)} outputs × {len(dimensions)} dimensions = {total} judgments")

        for output in outputs:
            prompt_id = output["prompt_id"]

            # Build context string for this task type
            if task == "translation":
                context = (
                    f"Source ({output.get('direction', 'en→target')}):\n"
                    f"{output.get('source', '')}\n\n"
                    f"Reference translation:\n"
                    f"{output.get('reference', '')}"
                )
                candidate_text = output.get("output", "")
            else:
                context = (
                    f"System instruction:\n{output.get('system_instruction', '')}\n\n"
                    f"User prompt:\n{output.get('user_prompt', '')}"
                )
                candidate_text = output.get("output", "")

            for dim in dimensions:
                dim_name     = dim["name"]
                dim_question = dim["question"]
                criteria     = dim["criteria"]

                key = f"{prompt_id}_{dim_name}"
                if key in already_judged:
                    done += 1
                    continue

                user_msg = _build_pointwise_prompt(
                    dimension_name=dim_name,
                    dimension_question=dim_question,
                    criteria=criteria,
                    context=context,
                    candidate_text=candidate_text,
                )

                raw = _call_judge(
                    user_message=user_msg,
                    judge_model=judge_model,
                    anthropic_api_key=anthropic_key,
                    gemini_api_key=gemini_key,
                )

                verdict = {
                    "prompt_id":      prompt_id,
                    "task":           task,
                    "language":       language,
                    "regional_model": regional_model_id,
                    "dimension":      dim_name,
                    "score":          raw["score"],
                    "confidence":     raw["confidence"],
                    "reasoning":      raw["reasoning"],
                    "judge_model":    judge_model,
                    "judged_at":      datetime.now(timezone.utc).isoformat(),
                    "error":          raw.get("error"),
                }
                verdicts.append(verdict)
                append_output(out_path, verdict)
                done += 1

                if done % 20 == 0:
                    print(f"[JudgeWorker] {done}/{total} judgments complete")

                time.sleep(call_delay_s)

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
    """Returns set of '{prompt_id}_{dimension}' already in the verdicts file."""
    keys = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                v = json.loads(line)
                keys.add(f"{v['prompt_id']}_{v['dimension']}")
    except FileNotFoundError:
        pass
    return keys
