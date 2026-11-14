from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from app.core.database import get_db, SQLALCHEMY_AVAILABLE
from app.schemas.preset import PresetCreate, PresetResponse
from app.models.preset import PresetModel

router = APIRouter(prefix="/presets", tags=["Presets"])


@router.get("", response_model=List[PresetResponse])
def list_presets(db=Depends(get_db)):
    """List available presets in the marketplace."""
    if SQLALCHEMY_AVAILABLE and hasattr(db, "query"):
        presets = db.query(PresetModel).all()
        return presets
    elif hasattr(db, "conn"):
        cursor = db.conn.cursor()
        cursor.execute("SELECT id, name, author, category, description, data, downloads, rating FROM presets")
        rows = cursor.fetchall()
        return [
            PresetResponse(
                id=r["id"],
                name=r["name"],
                author=r["author"],
                category=r["category"],
                description=r["description"],
                data=r["data"],
                downloads=r["downloads"],
                rating=r["rating"],
            )
            for r in rows
        ]
    return []


@router.get("/{preset_id}", response_model=PresetResponse)
def get_preset(preset_id: int, db=Depends(get_db)):
    """Retrieve a specific preset by ID."""
    if SQLALCHEMY_AVAILABLE and hasattr(db, "query"):
        preset = db.query(PresetModel).filter(PresetModel.id == preset_id).first()
        if preset:
            return preset
    elif hasattr(db, "conn"):
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT id, name, author, category, description, data, downloads, rating FROM presets WHERE id = ?",
            (preset_id,),
        )
        r = cursor.fetchone()
        if r:
            return PresetResponse(
                id=r["id"],
                name=r["name"],
                author=r["author"],
                category=r["category"],
                description=r["description"],
                data=r["data"],
                downloads=r["downloads"],
                rating=r["rating"],
            )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Preset {preset_id} not found",
    )


@router.post("", response_model=PresetResponse, status_code=status.HTTP_201_CREATED)
def create_preset(preset_in: PresetCreate, db=Depends(get_db)):
    """Upload or publish a new preset."""
    if SQLALCHEMY_AVAILABLE and hasattr(db, "add"):
        db_obj = PresetModel(
            name=preset_in.name,
            author=preset_in.author,
            category=preset_in.category,
            description=preset_in.description,
            data=preset_in.data,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    elif hasattr(db, "conn"):
        cursor = db.conn.cursor()
        cursor.execute(
            "INSERT INTO presets (name, author, category, description, data, downloads, rating) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                preset_in.name,
                preset_in.author,
                preset_in.category,
                preset_in.description,
                preset_in.data,
                0,
                5.0,
            ),
        )
        db.conn.commit()
        new_id = cursor.lastrowid
        return PresetResponse(
            id=new_id,
            name=preset_in.name,
            author=preset_in.author,
            category=preset_in.category,
            description=preset_in.description,
            data=preset_in.data,
            downloads=0,
            rating=5.0,
        )
    else:
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
