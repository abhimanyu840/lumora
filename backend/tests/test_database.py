from app.core.database import get_db, init_db


def test_init_db():
    # Should run without error and register models
    init_db()


def test_get_db_session():
    generator = get_db()
    session = next(generator)
    assert session is not None
    # Close session via generator teardown
    try:
        next(generator)
    except StopIteration:
        pass


def test_preset_crud_lifecycle(client):
    payload = {
        "name": "Cinematic Moody",
        "author": "LumoraStudio",
        "category": "Moody",
        "description": "Deep greens and warm skin tones",
        "data": '{"exposure": -0.2, "curves": "sample"}',
    }
    # 1. Create preset
    response = client.post("/api/v1/presets", json=payload)
    assert response.status_code == 201
    created = response.json()
    assert created["name"] == "Cinematic Moody"
    assert created["author"] == "LumoraStudio"
    assert "id" in created
    preset_id = created["id"]

    # 2. List presets
    list_res = client.get("/api/v1/presets")
    assert list_res.status_code == 200
    presets = list_res.json()
    assert any(p["id"] == preset_id for p in presets)

    # 3. Get preset by ID
    get_res = client.get(f"/api/v1/presets/{preset_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Cinematic Moody"

    # 4. Get non-existent preset -> 404
    missing_res = client.get("/api/v1/presets/999999")
    assert missing_res.status_code == 404

    # 5. Delete preset
    del_res = client.delete(f"/api/v1/presets/{preset_id}")
    assert del_res.status_code == 204

    # 6. Verify deleted -> 404
    assert client.get(f"/api/v1/presets/{preset_id}").status_code == 404
    assert client.delete(f"/api/v1/presets/{preset_id}").status_code == 404


def test_preset_validation_errors(client):
    # Missing required field "data"
    bad_payload = {
        "name": "Invalid Preset",
        "author": "Tester",
    }
    response = client.post("/api/v1/presets", json=bad_payload)
    assert response.status_code == 422


def test_multiple_presets_roundtrip(client):
    presets_data = [
        {"name": "Preset A", "author": "Alice", "category": "Portrait", "data": "{}"},
        {"name": "Preset B", "author": "Bob", "category": "Landscape", "data": "{}"},
    ]
    created_ids = []
    for p in presets_data:
        res = client.post("/api/v1/presets", json=p)
        assert res.status_code == 201
        created_ids.append(res.json()["id"])

    list_res = client.get("/api/v1/presets")
    assert list_res.status_code == 200
    ids_in_list = [item["id"] for item in list_res.json()]
    for cid in created_ids:
        assert cid in ids_in_list
        # Cleanup
        client.delete(f"/api/v1/presets/{cid}")
