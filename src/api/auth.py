"""
JWT auth helpers — keep verbatim.

verify_token  — decode and validate JWT, return TokenPayload
get_current_user  — FastAPI dependency for any authenticated route
require_scope(scope)  — returns a FastAPI dependency that enforces a specific scope
"""
from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.schemas import TokenPayload
from src.utils.config import JWT_ALGORITHM, JWT_SECRET

_bearer = HTTPBearer(auto_error=False)


def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> TokenPayload:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )
        return TokenPayload(
            sub=payload.get("sub", "anonymous"),
            scopes=payload.get("scopes", []),
            exp=payload.get("exp"),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")


def get_current_user(token: TokenPayload = Depends(verify_token)) -> TokenPayload:
    return token


def require_scope(scope: str):
    """Return a FastAPI dependency that enforces the given JWT scope."""
    def _check(token: TokenPayload = Depends(verify_token)) -> TokenPayload:
        if scope not in (token.scopes or []):
            raise HTTPException(
                status_code=403,
                detail=f"Missing required scope: '{scope}'",
            )
        return token
    return _check
