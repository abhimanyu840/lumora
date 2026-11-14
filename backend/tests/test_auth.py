"""Tests for User authentication and profile endpoints."""

import uuid
from app.core.security import create_access_token, decode_access_token, get_password_hash, verify_password


def test_password_hashing_and_verification():
    raw_password = "SecretPassword123!"
    hashed = get_password_hash(raw_password)
    assert hashed != raw_password
    assert hashed.startswith("pbkdf2_sha256$")
    assert verify_password(raw_password, hashed) is True
    assert verify_password("WrongPassword", hashed) is False
    assert verify_password("", hashed) is False
    assert verify_password(raw_password, "") is False


def test_jwt_token_roundtrip():
    payload = {"sub": "testuser_jwt"}
    token = create_access_token(payload)
    assert isinstance(token, str)
    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "testuser_jwt"
    assert "exp" in decoded

    # Invalid token decode returns None
    assert decode_access_token("invalid.token.here") is None


def test_user_registration_and_login_flow(client):
    uid = uuid.uuid4().hex[:8]
    username = f"photographer_{uid}"
    email = f"{username}@lumora.art"
    password = "JanePassword456!"

    user_payload = {
        "username": username,
        "email": email,
        "password": password,
    }

    # 1. Register new user
    res = client.post("/api/v1/auth/register", json=user_payload)
    assert res.status_code == 201
    user_data = res.json()
    assert user_data["username"] == username
    assert user_data["email"] == email
    assert "hashed_password" not in user_data
    assert "id" in user_data

    # 2. Duplicate registration returns 400
    dup_res = client.post("/api/v1/auth/register", json=user_payload)
    assert dup_res.status_code == 400

    # 3. Login with correct credentials
    login_res = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    token = token_data["access_token"]

    # 4. Login with incorrect password returns 401
    bad_login = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": "WrongPassword"},
    )
    assert bad_login.status_code == 401

    # 5. Get current profile via Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    me_res = client.get("/api/v1/auth/me", headers=headers)
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["username"] == username
    assert me_data["email"] == email

    # 6. Unauthenticated request to /me returns 401
    unauth_res = client.get("/api/v1/auth/me")
    assert unauth_res.status_code == 401

    # 7. Malformed token returns 401
    bad_token_res = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer bad_garbage_token"},
    )
    assert bad_token_res.status_code == 401

    # 8. Login using email address instead of username
    email_login = client.post(
        "/api/v1/auth/login",
        json={"username": email, "password": password},
    )
    assert email_login.status_code == 200
    assert "access_token" in email_login.json()


def test_model_imports_and_aliases():
    from app.models import Preset, PresetModel, User, UserModel
    assert User is UserModel
    assert Preset is PresetModel
