"""Preset and marketplace API endpoints for Lumora."""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.database import SQLALCHEMY_AVAILABLE, get_db
from app.models.preset import PresetModel
from app.schemas.preset import PresetCreate, PresetResponse

router = APIRouter(prefix="/presets", tags=["Presets"])


def _row_to_response(r) -> PresetResponse:
    downloads = r["downloads"] if ("downloads" in r.keys() and r["downloads"] is not None) else 0
    return PresetResponse(
        id=r["id"],
        name=r["name"],
        author=r["author"],
        author_id=r["author_id"] if ("author_id" in r.keys() and r["author_id"] is not None) else None,
        category=r["category"] if ("category" in r.keys() and r["category"]) else "portrait",
        description=r["description"] if ("description" in r.keys() and r["description"] is not None) else None,
        data=r["data"],
        tags=r["tags"] if ("tags" in r.keys() and r["tags"] is not None) else "",
        downloads=downloads,
        downloads_count=downloads,
        upvotes=r["upvotes"] if ("upvotes" in r.keys() and r["upvotes"] is not None) else 0,
        rating=float(r["rating"]) if ("rating" in r.keys() and r["rating"] is not None) else 5.0,
        created_at=r["created_at"] if "created_at" in r.keys() else None,
    )


@router.get("", response_model=List[PresetResponse])
def list_presets(
    category: Optional[str] = Query(None, description="Filter by category (portrait, landscape, etc.)"),
    tag: Optional[str] = Query(None, description="Filter by tag (e.g. moody, film)"),
    search: Optional[str] = Query(None, description="Search by name or description"),
    sort_by: Optional[str] = Query("popular", description="Sort by 'popular' or 'newest'"),
    db=Depends(get_db),
):
    """List and discover presets with marketplace filtering and sorting."""
    has_cat = bool(category and category.strip() and category.strip().lower() != "all")
    clean_cat = category.strip() if has_cat else None

    if SQLALCHEMY_AVAILABLE and hasattr(db, "query"):
        query = db.query(PresetModel)
        if clean_cat:
            query = query.filter(PresetModel.category.ilike(clean_cat))
        if tag:
            query = query.filter(PresetModel.tags.ilike(f"%{tag.strip()}%"))
        if search:
            s = search.strip()
            query = query.filter(
                (PresetModel.name.ilike(f"%{s}%"))
                | (PresetModel.description.ilike(f"%{s}%"))
            )

        if sort_by == "popular":
            query = query.order_by(PresetModel.upvotes.desc(), PresetModel.downloads.desc())
        elif sort_by == "newest":
            query = query.order_by(PresetModel.id.desc())
        elif sort_by == "rating":
            query = query.order_by(PresetModel.rating.desc(), PresetModel.upvotes.desc())
        else:
            query = query.order_by(PresetModel.id.asc())

        items = query.all()
        # populate downloads_count explicitly
        results = []
        for item in items:
            resp = PresetResponse.model_validate(item)
            resp.downloads_count = item.downloads or 0
            results.append(resp)
        return results

    elif hasattr(db, "conn"):
        cursor = db.conn.cursor()
        sql = "SELECT id, name, author, author_id, category, description, data, tags, downloads, upvotes, rating, created_at FROM presets WHERE 1=1"
        params = []

        if clean_cat:
            sql += " AND LOWER(category) = LOWER(?)"
            params.append(clean_cat)
        if tag:
            sql += " AND LOWER(tags) LIKE ?"
            params.append(f"%{tag.strip().lower()}%")
        if search:
            sql += " AND (LOWER(name) LIKE ? OR COALESCE(LOWER(description), '') LIKE ?)"
            s_param = f"%{search.strip().lower()}%"
            params.extend([s_param, s_param])

        if sort_by == "popular":
            sql += " ORDER BY upvotes DESC, downloads DESC"
        elif sort_by == "newest":
            sql += " ORDER BY id DESC"
        elif sort_by == "rating":
            sql += " ORDER BY rating DESC, upvotes DESC"
        else:
            sql += " ORDER BY id ASC"

        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        return [_row_to_response(r) for r in rows]

    return []


@router.get("/{preset_id}", response_model=PresetResponse)
def get_preset(preset_id: int, db=Depends(get_db)):
    """Retrieve a specific preset by ID."""
    if SQLALCHEMY_AVAILABLE and hasattr(db, "query"):
        preset = db.query(PresetModel).filter(PresetModel.id == preset_id).first()
        if preset:
            resp = PresetResponse.model_validate(preset)
            resp.downloads_count = preset.downloads or 0
            return resp
    elif hasattr(db, "conn"):
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT id, name, author, author_id, category, description, data, tags, downloads, upvotes, rating, created_at FROM presets WHERE id = ?",
            (preset_id,),
        )
        r = cursor.fetchone()
        if r:
            return _row_to_response(r)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Preset {preset_id} not found",
    )


@router.post("", response_model=PresetResponse, status_code=status.HTTP_201_CREATED)
def create_preset(preset_in: PresetCreate, db=Depends(get_db)):
    """Upload or publish a new preset to the marketplace."""
    now_iso = datetime.now(timezone.utc).isoformat()
    if SQLALCHEMY_AVAILABLE and hasattr(db, "add"):
        db_obj = PresetModel(
            name=preset_in.name,
            author=preset_in.author,
            author_id=preset_in.author_id,
            category=preset_in.category,
            description=preset_in.description,
            data=preset_in.data,
            tags=preset_in.tags or "",
            downloads=0,
            upvotes=0,
            rating=5.0,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        resp = PresetResponse.model_validate(db_obj)
        resp.downloads_count = db_obj.downloads or 0
        return resp
    elif hasattr(db, "conn"):
        cursor = db.conn.cursor()
        cursor.execute(
            """INSERT INTO presets (name, author, author_id, category, description, data, tags, downloads, upvotes, rating, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 5.0, ?)""",
            (
                preset_in.name,
                preset_in.author,
                preset_in.author_id,
                preset_in.category,
                preset_in.description,
                preset_in.data,
                preset_in.tags or "",
                now_iso,
            ),
        )
        db.conn.commit()
        new_id = cursor.lastrowid
        return PresetResponse(
            id=new_id,
            name=preset_in.name,
            author=preset_in.author,
            author_id=preset_in.author_id,
            category=preset_in.category,
            description=preset_in.description,
            data=preset_in.data,
            tags=preset_in.tags or "",
            downloads=0,
            downloads_count=0,
            upvotes=0,
            rating=5.0,
            created_at=now_iso,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database session unavailable",
        )


@router.post("/{preset_id}/upvote", response_model=PresetResponse)
def upvote_preset(preset_id: int, db=Depends(get_db)):
    """Increment the upvote count for a preset."""
    if SQLALCHEMY_AVAILABLE and hasattr(db, "query"):
        preset = db.query(PresetModel).filter(PresetModel.id == preset_id).first()
        if not preset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset {preset_id} not found",
            )
        preset.upvotes = (preset.upvotes or 0) + 1
        db.commit()
        db.refresh(preset)
        resp = PresetResponse.model_validate(preset)
        resp.downloads_count = preset.downloads or 0
        return resp
    elif hasattr(db, "conn"):
        cursor = db.conn.cursor()
        cursor.execute("SELECT id FROM presets WHERE id = ?", (preset_id,))
        if not cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset {preset_id} not found",
            )
        cursor.execute("UPDATE presets SET upvotes = upvotes + 1 WHERE id = ?", (preset_id,))
        db.conn.commit()
        cursor.execute(
            "SELECT id, name, author, author_id, category, description, data, tags, downloads, upvotes, rating, created_at FROM presets WHERE id = ?",
            (preset_id,),
        )
        return _row_to_response(cursor.fetchone())

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Database session unavailable",
    )


@router.post("/{preset_id}/download", response_model=PresetResponse)
def download_preset(preset_id: int, db=Depends(get_db)):
    """Increment the download count when a user downloads or imports a preset."""
    if SQLALCHEMY_AVAILABLE and hasattr(db, "query"):
        preset = db.query(PresetModel).filter(PresetModel.id == preset_id).first()
        if not preset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset {preset_id} not found",
            )
        preset.downloads = (preset.downloads or 0) + 1
        db.commit()
        db.refresh(preset)
        resp = PresetResponse.model_validate(preset)
        resp.downloads_count = preset.downloads or 0
        return resp
    elif hasattr(db, "conn"):
        cursor = db.conn.cursor()
        cursor.execute("SELECT id FROM presets WHERE id = ?", (preset_id,))
        if not cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset {preset_id} not found",
            )
        cursor.execute("UPDATE presets SET downloads = downloads + 1 WHERE id = ?", (preset_id,))
        db.conn.commit()
        cursor.execute(
            "SELECT id, name, author, author_id, category, description, data, tags, downloads, upvotes, rating, created_at FROM presets WHERE id = ?",
            (preset_id,),
        )
        return _row_to_response(cursor.fetchone())

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Database session unavailable",
    )


@router.delete("/{preset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_preset(preset_id: int, db=Depends(get_db)):
    """Delete a preset by ID."""
    if SQLALCHEMY_AVAILABLE and hasattr(db, "query"):
        preset = db.query(PresetModel).filter(PresetModel.id == preset_id).first()
        if preset:
            db.delete(preset)
            db.commit()
            return None
    elif hasattr(db, "conn"):
        cursor = db.conn.cursor()
        cursor.execute("SELECT id FROM presets WHERE id = ?", (preset_id,))
        if cursor.fetchone():
            cursor.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
            db.conn.commit()
            return None

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Preset {preset_id} not found",
    )
