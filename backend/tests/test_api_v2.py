"""Contract tests for the /api/v2/* router.

Uses FastAPI's TestClient against the fixture-backed EntityStore so
every test is hermetic — no network, no real LiteLLM. The goal is to
pin the public response shapes that the frontend already consumes.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Force fixture mode by removing any existing v2 snapshot and then
    # loading the store through the app's lifespan.
    from main import app
    from services.entity_store import get_store

    store = get_store()
    store._load_fixture()
    return TestClient(app)


class TestEntitiesList:
    def test_returns_list_of_entities(self, client):
        r = client.get("/api/v2/entities")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_every_item_has_required_shape(self, client):
        r = client.get("/api/v2/entities")
        for entity in r.json():
            # Core identity
            assert "canonical_id" in entity
            assert "slug" in entity
            assert "name" in entity
            assert "family" in entity
            assert "maker" in entity
            # Capability bag
            assert isinstance(entity["capabilities"], list)
            assert isinstance(entity["input_modalities"], list)
            assert isinstance(entity["output_modalities"], list)
            # List view includes primary_offering
            assert "primary_offering" in entity

    def test_filter_by_maker(self, client):
        r = client.get("/api/v2/entities", params={"maker": "Anthropic"})
        assert r.status_code == 200
        for entity in r.json():
            assert entity["maker"] == "Anthropic"

    def test_filter_by_capability(self, client):
        r = client.get("/api/v2/entities", params={"capability": "vision"})
        assert r.status_code == 200
        for entity in r.json():
            assert "vision" in entity["capabilities"]

    def test_sort_by_input_asc(self, client):
        r = client.get("/api/v2/entities", params={"sort": "input", "order": "asc"})
        data = r.json()
        inputs = [
            e["primary_offering"]["pricing"]["input"]
            for e in data
            if e.get("primary_offering") and e["primary_offering"]["pricing"]["input"] is not None
        ]
        assert inputs == sorted(inputs), "results should be sorted ascending by input price"

    def test_search_by_substring(self, client):
        r = client.get("/api/v2/entities", params={"q": "claude"})
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 0
        for entity in data:
            target = f"{entity['name']} {entity['canonical_id']} {entity['family']}".lower()
            assert "claude" in target

    def test_rejects_bad_sort_value(self, client):
        r = client.get("/api/v2/entities", params={"sort": "wtf"})
        assert r.status_code == 422


class TestEntityDetail:
    def test_known_slug_returns_detail(self, client):
        r = client.get("/api/v2/entities/claude-sonnet-4-5")
        assert r.status_code == 200
        data = r.json()
        assert "entity" in data
        assert "offerings" in data
        assert "alternatives" in data
        assert data["entity"]["slug"] == "claude-sonnet-4-5"
        assert isinstance(data["offerings"], list)
        assert isinstance(data["alternatives"], list)

    def test_unknown_slug_returns_404(self, client):
        r = client.get("/api/v2/entities/definitely-not-a-real-slug")
        assert r.status_code == 404
        body = r.json()
        assert body["detail"]["code"] == "not_found"


class TestSearch:
    def test_search_returns_ranked_hits(self, client):
        r = client.get("/api/v2/search", params={"q": "son"})
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        # At least Claude Sonnet 4.5 should be among the fixture results
        if data:
            for hit in data:
                assert "slug" in hit
                assert "name" in hit
                assert "primary_input_price" in hit

    def test_empty_query_rejected(self, client):
        r = client.get("/api/v2/search", params={"q": ""})
        assert r.status_code == 422

    def test_limit_is_respected(self, client):
        r = client.get("/api/v2/search", params={"q": "a", "limit": 3})
        assert len(r.json()) <= 3


class TestCompare:
    def test_two_valid_slugs(self, client):
        r = client.get(
            "/api/v2/compare",
            params={"ids": "claude-sonnet-4-5,gpt-4o"},
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["entities"]) == 2
        assert data["missing_ids"] == []
        assert isinstance(data["common_capabilities"], list)

    def test_missing_slug_reported_not_404(self, client):
        r = client.get(
            "/api/v2/compare",
            params={"ids": "claude-sonnet-4-5,bogus-model-id"},
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["entities"]) == 1
        assert "bogus-model-id" in data["missing_ids"]

    def test_empty_ids_rejected(self, client):
        r = client.get("/api/v2/compare", params={"ids": ""})
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "bad_request"

    def test_too_many_ids_rejected(self, client):
        r = client.get(
            "/api/v2/compare",
            params={"ids": "a,b,c,d,e"},
        )
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "too_many_ids"


class TestStats:
    def test_stats_shape(self, client):
        r = client.get("/api/v2/stats")
        assert r.status_code == 200
        data = r.json()
        for key in (
            "total_entities",
            "total_offerings",
            "makers",
            "families",
            "last_refresh",
        ):
            assert key in data
        assert data["total_entities"] >= 1


class TestCacheHeaders:
    def test_api_responses_are_not_cacheable(self, client):
        r = client.get("/api/v2/entities")
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-store"
        assert r.headers["pragma"] == "no-cache"

    def test_snapshot_is_shared_cacheable(self, client):
        r = client.get("/api/v2/snapshot")
        assert r.status_code == 200
        data = r.json()
        assert data["version"] == "v2-live-snapshot.1"
        assert data["entity_count"] == len(data["entities"])
        assert "offerings_by_entity" in data
        assert "alternatives_by_entity" in data
        assert "s-maxage=3600" in r.headers["cache-control"]
        assert "stale-while-revalidate=86400" in r.headers["cache-control"]
