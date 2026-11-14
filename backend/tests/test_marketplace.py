"""Tests for Preset Marketplace discovery, filtering, upvoting, and download metrics."""


def test_preset_marketplace_categories_and_tags(client):
    presets = [
        {
            "name": "Cyberpunk Neon 2077",
            "author": "NeoEditor",
            "category": "cyberpunk",
            "description": "Cyan shadows and magenta highlights with crushed blacks",
            "tags": "cyberpunk,neon,night,futuristic",
            "data": '{"exposure": -0.1, "hsl": {"magenta": 0.4, "cyan": 0.5}}',
        },
        {
            "name": "Kodak Portra 400",
            "author": "AnalogMaster",
            "category": "vintage",
            "description": "Warm golden skin tones with organic film grain",
            "tags": "vintage,film,kodak,warm",
            "data": '{"exposure": 0.1, "curves": "portra_s_curve"}',
        },
        {
            "name": "Nordic Landscape",
            "author": "FjordVisuals",
            "category": "landscape",
            "description": "Muted greens and atmospheric mist for nature photography",
            "tags": "landscape,nature,nordic,cold",
            "data": '{"exposure": -0.05, "clarity": 0.2}',
        },
        {
            "name": "Dramatic B&W Noir",
            "author": "ClassicDarkroom",
            "category": "black_and_white",
            "description": "High-contrast monochrome with deep shadows",
            "tags": "black_and_white,monochrome,noir,contrast",
            "data": '{"saturation": -1.0, "contrast": 0.35}',
        },
    ]

    created_ids = []
    for p in presets:
        res = client.post("/api/v1/presets", json=p)
        assert res.status_code == 201
        data = res.json()
        assert data["category"] == p["category"]
        assert data["tags"] == p["tags"]
        assert data["upvotes"] == 0
        assert data["downloads"] == 0
        assert data["downloads_count"] == 0
        created_ids.append(data["id"])

    # 1. Filter by category
    res_cat = client.get("/api/v1/presets?category=cyberpunk")
    assert res_cat.status_code == 200
    cyber_items = res_cat.json()
    assert len(cyber_items) >= 1
    assert any(i["name"] == "Cyberpunk Neon 2077" for i in cyber_items)
    assert all(i["category"] == "cyberpunk" for i in cyber_items)

    # 1b. Filter by 'all' should return all items without error
    res_all = client.get("/api/v1/presets?category=all")
    assert res_all.status_code == 200
    assert len(res_all.json()) >= 4

    # 2. Filter by tag
    res_tag = client.get("/api/v1/presets?tag=film")
    assert res_tag.status_code == 200
    film_items = res_tag.json()
    assert len(film_items) >= 1
    assert any(i["name"] == "Kodak Portra 400" for i in film_items)

    # 3. Search query across name and description
    res_search = client.get("/api/v1/presets?search=Nordic")
    assert res_search.status_code == 200
    search_items = res_search.json()
    assert any(i["name"] == "Nordic Landscape" for i in search_items)

    # 4. Upvote endpoint
    target_id = created_ids[0]
    upvote_res1 = client.post(f"/api/v1/presets/{target_id}/upvote")
    assert upvote_res1.status_code == 200
    assert upvote_res1.json()["upvotes"] == 1

    upvote_res2 = client.post(f"/api/v1/presets/{target_id}/upvote")
    assert upvote_res2.status_code == 200
    assert upvote_res2.json()["upvotes"] == 2

    # 5. Download counter increment
    down_res1 = client.post(f"/api/v1/presets/{target_id}/download")
    assert down_res1.status_code == 200
    assert down_res1.json()["downloads"] == 1
    assert down_res1.json()["downloads_count"] == 1

    down_res2 = client.post(f"/api/v1/presets/{target_id}/download")
    assert down_res2.status_code == 200
    assert down_res2.json()["downloads"] == 2
    assert down_res2.json()["downloads_count"] == 2

    # 6. Popular sorting (most upvoted first)
    pop_res = client.get("/api/v1/presets?sort_by=popular")
    assert pop_res.status_code == 200
    pop_items = pop_res.json()
    assert pop_items[0]["id"] == target_id
    assert pop_items[0]["upvotes"] == 2

    # 6b. Rating sorting
    rating_res = client.get("/api/v1/presets?sort_by=rating")
    assert rating_res.status_code == 200
    assert len(rating_res.json()) >= 4

    # 7. Edge cases: Upvote or download non-existent preset -> 404
    assert client.post("/api/v1/presets/999999/upvote").status_code == 404
    assert client.post("/api/v1/presets/999999/download").status_code == 404

    # Cleanup created presets
    for pid in created_ids:
        client.delete(f"/api/v1/presets/{pid}")
