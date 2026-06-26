"""
Vercel entry point — imports the FastAPI app from scripts/dashboard.py.
Vercel's Python runtime serves the ASGI app directly (no uvicorn needed).
"""
import sys
from pathlib import Path

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dashboard import app  # noqa: F401 — Vercel looks for `app`
