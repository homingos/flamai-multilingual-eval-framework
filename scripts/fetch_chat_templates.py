"""
Fetch and store chat templates for all active models in the registry.

For each model, calls POST /models/{id}/fetch-chat-template which:
  1. Downloads the HuggingFace tokenizer
  2. Extracts the Jinja2 chat_template string
  3. Stores it in the registry (or stores None if the model has no template)

Usage:
    REGISTRY_URL=https://ai-team-core--phase2a-registry-dev.modal.run \\
    JWT_TOKEN=$(python scripts/jwt_token_generator.py --secret $JWT_SECRET --scopes registry:write) \\
    python scripts/fetch_chat_templates.py

Options:
    --model-id <id>   Fetch for a single model only
    --force           Re-fetch even if chat_template is already set

The registry container must have HF_TOKEN set as a Modal secret if any models
are in gated HuggingFace repos (add it to phase2a-auth-secrets).
"""
from __future__ import annotations

import argparse
import os
import sys

import requests

REGISTRY_URL = os.environ.get("REGISTRY_URL", "http://localhost:8000")
JWT_TOKEN    = os.environ.get("JWT_TOKEN", "")

HEADERS = {
    "Authorization": f"Bearer {JWT_TOKEN}",
    "Content-Type":  "application/json",
}


def fetch_for_model(model_id: str) -> dict:
    resp = requests.post(
        f"{REGISTRY_URL}/models/{model_id}/fetch-chat-template",
        headers=HEADERS,
    )
    if resp.status_code == 200:
        return resp.json()
    else:
        return {"error": f"{resp.status_code}: {resp.text}"}


def get_active_models() -> list[dict]:
    resp = requests.get(f"{REGISTRY_URL}/models", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()["models"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", help="Fetch for a single model only")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if chat_template already set")
    args = parser.parse_args()

    if not JWT_TOKEN:
        print("ERROR: JWT_TOKEN env var required.", file=sys.stderr)
        sys.exit(1)

    if args.model_id:
        models = [{"id": args.model_id, "chat_template": None}]
    else:
        models = get_active_models()

    print(f"\nFetching chat templates for {len(models)} models from {REGISTRY_URL}\n")
    print(f"  {'Model ID':<28} {'Template':>10}  {'Length':>8}  Status")
    print(f"  {'-'*28} {'-'*10}  {'-'*8}  ------")

    found = skipped = failed = 0
    for m in models:
        mid = m["id"]

        # Skip if already set (unless --force)
        if not args.force and m.get("chat_template"):
            print(f"  {mid:<28} {'already set':>10}  {'':>8}  SKIP")
            skipped += 1
            continue

        result = fetch_for_model(mid)

        if "error" in result:
            print(f"  {mid:<28} {'':>10}  {'':>8}  ERROR: {result['error']}")
            failed += 1
        elif result.get("template_found"):
            length = result["template_length"]
            print(f"  {mid:<28} {'found':>10}  {length:>8}  OK")
            found += 1
        else:
            print(f"  {mid:<28} {'none':>10}  {0:>8}  OK (no template — plain-text fallback)")
            found += 1  # still a success, just no template

    print(f"\n  Done: {found} processed, {skipped} skipped, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()