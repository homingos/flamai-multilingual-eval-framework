"""
test_judge_local.py
====================
Standalone local test for the Gemini judge pipeline.
Runs 2 real Greek translation pairs through _call_gemini and _parse_verdict,
printing every intermediate value so failures are immediately visible.

Usage:
    GEMINI_API_KEY=your-key python3 test_judge_local.py
    GEMINI_API_KEY=your-key python3 test_judge_local.py --model gemini-2.5-flash
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# ── Copy of SYSTEM_PROMPT from judge.py ──────────────────────────────────────

SYSTEM_PROMPT = """\
You are a rigorous multilingual output evaluator. You will be shown two AI-generated
responses (Response A and Response B) to the same prompt. Your task is to judge which
response is better according to the specified dimension.

Rules:
- Be BLIND to which model generated each response — judge content only.
- You MUST pick a winner (A or B) whenever there is ANY detectable difference,
  however small — word choice, naturalness, accuracy, register, fluency.
- Only declare "tie" when the two responses are genuinely IDENTICAL or have
  zero detectable difference after careful reading. Ties should be rare.
- Be concise: 1-2 sentences of reasoning maximum.
- Output JSON only — no preamble, no markdown fences.

Output format (JSON):
{
  "winner": "A" | "B" | "tie",
  "confidence": "high" | "medium" | "low",
  "reasoning": "<1-2 sentence explanation citing a specific difference>"
}
"""

# ── Two real Greek translation pairs ─────────────────────────────────────────
# Source: FLORES-200 English sentences
# Response A: Meltemi-7B output (regional)
# Response B: Gemma-4 output (baseline)
# These are realistic close comparisons — exactly the hard cases for the judge.

TEST_PAIRS = [
    {
        "prompt_id": "test_0001",
        "source":    "The researchers discovered a new method for treating cancer using targeted immunotherapy.",
        "output_a":  "Οι ερευνητές ανακάλυψαν μια νέα μέθοδο για τη θεραπεία του καρκίνου χρησιμοποιώντας στοχευμένη ανοσοθεραπεία.",
        "output_b":  "Οι επιστήμονες ανακάλυψαν μία νέα μέθοδο θεραπείας του καρκίνου με τη χρήση στοχευμένης ανοσοθεραπείας.",
    },
    {
        "prompt_id": "test_0002",
        "source":    "The government announced new environmental regulations to reduce carbon emissions by 2030.",
        "output_a":  "Η κυβέρνηση ανακοίνωσε νέους περιβαλλοντικούς κανονισμούς για τη μείωση των εκπομπών άνθρακα έως το 2030.",
        "output_b":  "Η κυβέρνηση ανακοίνωσε νέες περιβαλλοντικές ρυθμίσεις για τη μείωση των εκπομπών διοξειδίου του άνθρακα μέχρι το 2030.",
    },
]

DIMENSIONS = [
    ("fluency",   "Which response reads more naturally and fluently in the target language?"),
    ("adequacy",  "Which response preserves the meaning of the source text more accurately?"),
    ("overall",   "Overall, which response is the better translation?"),
]


# ── Copy of _parse_verdict from judge.py ─────────────────────────────────────

def _parse_verdict(raw_text: str) -> dict:
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    inline_match = re.search(r"`(\{.*?\})`", text, re.DOTALL)
    if inline_match:
        try:
            return json.loads(inline_match.group(1))
        except json.JSONDecodeError:
            pass
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
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No valid JSON found in: {repr(text[:300])}")


# ── Single API call ───────────────────────────────────────────────────────────

def call_gemini_once(user_message: str, model: str, api_key: str) -> dict:
    """Makes one call, returns full debug info."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = json.dumps({
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {
            "maxOutputTokens": 8192,
            "temperature":     0.1,
        },
    }).encode()

    try:
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())

        # ── Debug: print full parts structure ─────────────────────────────
        parts    = body["candidates"][0]["content"]["parts"]
        finish   = body["candidates"][0].get("finishReason")
        usage    = body.get("usageMetadata", {})
        service_tier = usage.get("serviceTier", "unknown")

        print(f"\n         [DEBUG] finish={finish} tier={service_tier} parts={len(parts)}")
        for i, p in enumerate(parts):
            text_preview = repr(p.get("text", "")[:120])
            has_thought  = "thoughtSignature" in p
            print(f"         [DEBUG] parts[{i}]: thought={has_thought} text={text_preview}")

        # ── Extract the actual answer text ─────────────────────────────────
        # Gemini thinking models (2.5+, 3.x) return parts like:
        #   parts[0]: {"text": "<thinking>...", "thoughtSignature": "..."}  ← internal reasoning
        #   parts[1]: {"text": "{\"winner\": ...}"}                         ← actual answer
        # Non-thinking models return a single parts[0] with the answer.
        # Strategy: find the last part that has text but no thoughtSignature.
        raw_text = None
        for part in reversed(parts):
            if "text" in part and "thoughtSignature" not in part:
                raw_text = part["text"]
                break
        # Fallback: if all parts have thoughtSignature, take the last text
        if raw_text is None:
            for part in reversed(parts):
                if "text" in part:
                    raw_text = part["text"]
                    break

        if raw_text is None:
            return {
                "ok": False, "http_error": None,
                "error_body": f"No text part found in response. Parts: {parts}",
                "raw_text": None, "verdict": {},
            }

        try:
            verdict   = _parse_verdict(raw_text)
            parse_ok  = True
            parse_err = None
        except ValueError as e:
            verdict   = {}
            parse_ok  = False
            parse_err = str(e)

        return {
            "ok":            True,
            "raw_text":      raw_text,
            "finish_reason": finish,
            "service_tier":  service_tier,
            "total_tokens":  usage.get("totalTokenCount", 0),
            "parse_ok":      parse_ok,
            "parse_err":     parse_err,
            "verdict":       verdict,
            "http_error":    None,
        }

    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode()
        return {
            "ok": False, "http_error": exc.code,
            "error_body": err_body, "raw_text": None, "verdict": {},
        }
    except Exception as exc:
        return {
            "ok": False, "http_error": None,
            "error_body": str(exc), "raw_text": None, "verdict": {},
        }


# ── Main test runner ──────────────────────────────────────────────────────────

def run_tests(model: str, api_key: str, call_delay: float) -> None:
    print(f"\n{'='*60}")
    print(f"  Judge local test")
    print(f"  Model:       {model}")
    print(f"  Call delay:  {call_delay}s")
    print(f"  Pairs:       {len(TEST_PAIRS)}")
    print(f"  Dimensions:  {len(DIMENSIONS)}")
    print(f"  Total calls: {len(TEST_PAIRS) * len(DIMENSIONS)}")
    print(f"{'='*60}\n")

    results = []
    call_n  = 0
    total   = len(TEST_PAIRS) * len(DIMENSIONS)

    for pair in TEST_PAIRS:
        print(f"── Pair {pair['prompt_id']} ──")
        print(f"   Source: {pair['source'][:70]}...")
        print(f"   A: {pair['output_a'][:60]}...")
        print(f"   B: {pair['output_b'][:60]}...")
        print()

        for dim_label, dim_question in DIMENSIONS:
            call_n += 1
            user_msg = f"""\
Dimension: {dim_label}
Question: {dim_question}

Prompt / Source text:
{pair['source']}

Response A:
{pair['output_a']}

Response B:
{pair['output_b']}

Judge which response better satisfies the dimension. Output JSON only."""

            print(f"  [{call_n}/{total}] {dim_label}...", end=" ", flush=True)
            result = call_gemini_once(user_msg, model, api_key)

            if not result["ok"]:
                print(f"FAILED  HTTP {result['http_error']}")
                print(f"         {result['error_body'][:200]}")
                results.append({"pair": pair["prompt_id"], "dim": dim_label, "ok": False})
            else:
                verdict = result["verdict"]
                winner  = verdict.get("winner", "MISSING")
                conf    = verdict.get("confidence", "?")
                reason  = verdict.get("reasoning", "")[:80]

                status = "OK" if result["parse_ok"] else "PARSE_ERR"
                print(f"{status}  winner={winner}  confidence={conf}  tier={result['service_tier']}")
                if not result["parse_ok"]:
                    print(f"         parse error: {result['parse_err']}")
                    print(f"         raw: {repr(result['raw_text'][:200])}")
                else:
                    print(f"         reasoning: {reason}")

                results.append({
                    "pair":    pair["prompt_id"],
                    "dim":     dim_label,
                    "ok":      result["ok"] and result["parse_ok"],
                    "winner":  winner,
                    "conf":    conf,
                    "raw":     result["raw_text"],
                })

            if call_n < total:
                time.sleep(call_delay)

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    ok_count   = sum(1 for r in results if r["ok"])
    tie_count  = sum(1 for r in results if r.get("winner") == "tie")
    fail_count = sum(1 for r in results if not r["ok"])
    a_count    = sum(1 for r in results if r.get("winner") == "A")
    b_count    = sum(1 for r in results if r.get("winner") == "B")

    print(f"  Total calls:    {total}")
    print(f"  Succeeded:      {ok_count}")
    print(f"  Failed:         {fail_count}")
    print(f"  Winners — A:    {a_count}  B: {b_count}  tie: {tie_count}")
    print()

    if fail_count > 0:
        print("  ❌ Some calls failed — check HTTP errors above")
        sys.exit(1)
    elif tie_count == total:
        print("  ❌ All ties — judge is stuck (prompt or API config issue)")
        sys.exit(1)
    elif tie_count > total * 0.5:
        print("  ⚠️  High tie rate — judge may still be constrained")
    else:
        print("  ✅ Judge is working — producing genuine verdicts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gemini judge locally")
    parser.add_argument("--model",       default="gemini-3.5-flash",
                        help="Gemini model to test (default: gemini-3.5-flash)")
    parser.add_argument("--delay",       type=float, default=4.0,
                        help="Seconds between API calls (default: 4.0)")
    parser.add_argument("--api-key",     default=os.environ.get("GEMINI_API_KEY", ""),
                        help="Gemini API key (default: $GEMINI_API_KEY)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        print("  export GEMINI_API_KEY=your-key")
        print("  or pass --api-key your-key")
        sys.exit(1)

    run_tests(model=args.api_key and args.model,
              api_key=args.api_key,
              call_delay=args.delay)