"""Security, authentication, and cryptographic utilities for Lumora backend."""

import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from app.core.config import settings

# Algorithm used for JWT encoding/decoding
ALGORITHM = "HS256"
PBKDF2_ITERATIONS = 100_000


def get_password_hash(password: str) -> str:
    """Hash a password using secure PBKDF2-HMAC-SHA256 with a random 16-byte salt."""
    salt = os.urandom(16)
    salt_hex = salt.hex()
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    hash_hex = pwd_hash.hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_hex}${hash_hex}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a stored hash string.

    Supports pbkdf2_sha256 format and bcrypt (if passlib is installed).
    """
    if not hashed_password or not plain_password:
        return False

    if hashed_password.startswith("pbkdf2_sha256$"):
        parts = hashed_password.split("$")
        if len(parts) != 4:
            return False
        try:
            iterations = int(parts[1])
            salt = bytes.fromhex(parts[2])
            expected_hash = bytes.fromhex(parts[3])
            computed_hash = hashlib.pbkdf2_hmac(
                "sha256",
                plain_password.encode("utf-8"),
                salt,
                iterations,
            )
            return hmac.compare_digest(computed_hash, expected_hash)
        except Exception:
            return False

    # Attempt passlib/bcrypt fallback if hash comes from bcrypt
    try:
        from passlib.context import CryptContext

        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        # Fallback SHA256 comparison for legacy simple hashes
        simple_hash = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(simple_hash, hashed_password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except (jwt.PyJWTError, Exception):
        return None
