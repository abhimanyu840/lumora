"""Authentication API endpoints for Lumora."""

from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.database import SQLALCHEMY_AVAILABLE, get_db
from app.core.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)
from app.models.user import UserModel
from app.schemas.user import Token, UserCreate, UserLogin, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def get_user_by_username(db, username: str) -> Optional[UserModel]:
    """Helper to query a user by username across SQLAlchemy and SQLite."""
    if SQLALCHEMY_AVAILABLE and hasattr(db, "query"):
        return db.query(UserModel).filter(UserModel.username == username).first()
    elif hasattr(db, "conn"):
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT id, username, email, hashed_password, is_active, created_at FROM users WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
        if row:
            return UserModel(
                id=row["id"],
                username=row["username"],
                email=row["email"],
                hashed_password=row["hashed_password"],
                is_active=bool(row["is_active"]),
                created_at=row["created_at"],
            )
    return None


def get_user_by_email(db, email: str) -> Optional[UserModel]:
    """Helper to query a user by email across SQLAlchemy and SQLite."""
    if SQLALCHEMY_AVAILABLE and hasattr(db, "query"):
        return db.query(UserModel).filter(UserModel.email == email).first()
    elif hasattr(db, "conn"):
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT id, username, email, hashed_password, is_active, created_at FROM users WHERE email = ?",
            (email,),
        )
        row = cursor.fetchone()
        if row:
            return UserModel(
                id=row["id"],
                username=row["username"],
                email=row["email"],
                hashed_password=row["hashed_password"],
                is_active=bool(row["is_active"]),
                created_at=row["created_at"],
            )
    return None


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db=Depends(get_db),
) -> UserModel:
    """Dependency that decodes the Bearer token and returns the current user."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials or token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user_by_username(db, username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account",
        )
    return user


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(user_in: UserCreate, db=Depends(get_db)):
    """Register a new user account."""
    if get_user_by_username(db, user_in.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
    if get_user_by_email(db, user_in.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    hashed_pw = get_password_hash(user_in.password)
    now_iso = datetime.now(timezone.utc).isoformat()

    if SQLALCHEMY_AVAILABLE and hasattr(db, "add"):
        db_user = UserModel(
            username=user_in.username,
            email=user_in.email,
            hashed_password=hashed_pw,
            is_active=True,
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    elif hasattr(db, "conn"):
        cursor = db.conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, hashed_password, is_active, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_in.username, user_in.email, hashed_pw, 1, now_iso),
        )
        db.conn.commit()
        new_id = cursor.lastrowid
        return UserResponse(
            id=new_id,
            username=user_in.username,
            email=user_in.email,
            is_active=True,
            created_at=now_iso,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database session unavailable",
        )


@router.post("/login", response_model=Token)
def login(login_data: UserLogin, db=Depends(get_db)):
    """Authenticate via JSON credentials and obtain a JWT access token."""
    user = get_user_by_username(db, login_data.username) or get_user_by_email(db, login_data.username)
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account",
        )
    token = create_access_token(data={"sub": user.username})
    return Token(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(current_user: UserModel = Depends(get_current_user)):
    """Retrieve the authenticated user's profile."""
    return current_user
