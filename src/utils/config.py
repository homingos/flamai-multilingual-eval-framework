"""
Environment variable config — keep verbatim, do not modify.

Imported by auth.py and deps.py. No Modal, no FastAPI imports.
"""
import os

JWT_SECRET: str = os.environ.get("JWT_SECRET", "dev-secret-change-in-prod")
JWT_ALGORITHM: str = "HS256"
JWT_EXPIRY_MINUTES: int = int(os.environ.get("JWT_EXPIRY_MINUTES", "60"))
PERSISTENCE_BACKEND: str = os.environ.get("PERSISTENCE_BACKEND", "volume")
DEPLOY_ENV: str = os.environ.get("DEPLOY_ENV", "dev")
