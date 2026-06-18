"""
Generates a signed JWT for local testing and the seed script.

Usage:
    python scripts/jwt_token_generator.py --secret <JWT_SECRET> --scopes registry:read registry:write
    python scripts/jwt_token_generator.py  # uses JWT_SECRET env var, default scopes
"""
import argparse
import datetime
import os
import sys

import jwt


def make_token(
    secret: str,
    scopes: list[str],
    subject: str = "dev-user-2",
    expiry_minutes: int = 86400,
) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": subject,
        "scopes": scopes,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=expiry_minutes),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Phase 2A JWT token")
    parser.add_argument("--secret", default=os.environ.get("JWT_SECRET", "dev-secret-change-in-prod"))
    parser.add_argument("--scopes", nargs="*", default=["registry:read", "registry:write"])
    parser.add_argument("--subject", default="dev-user")
    parser.add_argument("--expiry-minutes", type=int, default=60)
    args = parser.parse_args()

    token = make_token(
        secret=args.secret,
        scopes=args.scopes,
        subject=args.subject,
        expiry_minutes=args.expiry_minutes,
    )
    print(token)


if __name__ == "__main__":
    main()
