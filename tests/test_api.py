"""API endpoint integration tests.

These tests verify the API endpoints work correctly.
Tests that require loaded models are skipped if models are not available.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app

# Initialize client
client = TestClient(app)


# ==================== Fixtures ====================

@pytest.fixture(scope="module")
def health_response():
    """Get health response once per module."""
    return client.get("/health").json()


def has_models(health_response):
    """Check if models are loaded."""
    models_loaded = health_response.get("models_loaded", {})
    return models_loaded.get("lightgbm", False)


# ==================== Health Endpoint Tests ====================

class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_status_healthy(self):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"
    
    def test_health_shows_models_loaded(self):
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], dict)
    
    def test_health_has_version(self):
        response = client.get("/health")
        assert "version" in response.json()


# ==================== Models Endpoint Tests ====================

class TestModelsEndpoint:
    """Tests for /v1/models endpoint."""
    
    def test_models_returns_200(self):
        response = client.get("/v1/models")
        assert response.status_code == 200
    
    def test_models_lists_loaded_models(self):
        response = client.get("/v1/models")
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)


class TestMetricsEndpoint:
    """Tests for /v1/metrics endpoint."""
    
    def test_metrics_returns_200(self):
        response = client.get("/v1/metrics")
        assert response.status_code == 200


# ==================== Score Endpoint Tests (Require Models) ====================

class TestScoreEndpointWithModels:
    """Tests for /v1/score endpoint when models are loaded.
    
    These tests only run if models are available.
    """
    
    @pytest.fixture(autouse=True)
    def check_models(self, health_response):
        """Skip tests if models are not loaded."""
        if not has_models(health_response):
            pytest.skip("Models not loaded - run train_all.py first")
    
    def test_score_returns_200(self, health_response):
        response = client.post("/v1/score", json={
            "dst_port": 443,
            "bytes_in": 1500,
            "bytes_out": 500,
        })
        assert response.status_code == 200
    
    def test_score_normal_https_traffic(self, health_response):
        """Normal HTTPS browsing should be classified as NORMAL."""
        response = client.post("/v1/score", json={
            "dst_port": 443,
            "bytes_in": 1500,
            "bytes_out": 500,
            "duration": 0.5,
            "protocol": "https",
        })
        data = response.json()
        assert data["classification"] == "NORMAL"
        assert data["anomaly_score"] < 0.5
    
    def test_score_brute_force_attack(self, health_response):
        """Failed SSH login should be classified as ATTACK."""
        response = client.post("/v1/score", json={
            "dst_port": 22,
            "bytes_in": 100,
            "bytes_out": 50,
            "duration": 0.01,
            "protocol": "ssh",
            "status": "failed",
        })
        data = response.json()
        assert data["classification"] == "ATTACK"
        assert data["anomaly_score"] >= 0.5
    
    def test_score_data_exfiltration(self, health_response):
        """Large outbound data transfer should be flagged."""
        response = client.post("/v1/score", json={
            "dst_port": 443,
            "bytes_in": 100,
            "bytes_out": 5000000,
            "duration": 0.001,
            "protocol": "https",
        })
        data = response.json()
        assert data["anomaly_score"] >= 0.5
        assert data["classification"] == "ATTACK"
    
    def test_score_response_structure(self, health_response):
        """Response should have all required fields."""
        response = client.post("/v1/score", json={"dst_port": 80})
        data = response.json()
        
        assert "anomaly_score" in data
        assert "classification" in data
        assert "severity" in data
        assert "model_scores" in data
        assert "features_used" in data
        
        assert isinstance(data["anomaly_score"], float)
        assert 0 <= data["anomaly_score"] <= 1
        assert data["classification"] in ["NORMAL", "ATTACK"]


class TestBatchScoreEndpointWithModels:
    """Tests for /v1/score/batch endpoint when models are loaded."""
    
    @pytest.fixture(autouse=True)
    def check_models(self, health_response):
        """Skip tests if models are not loaded."""
        if not has_models(health_response):
            pytest.skip("Models not loaded - run train_all.py first")
    
    def test_batch_returns_200(self, health_response):
        response = client.post("/v1/score/batch", json={
            "events": [
                {"dst_port": 443, "protocol": "https"},
                {"dst_port": 22, "protocol": "ssh", "status": "failed"},
            ]
        })
        assert response.status_code == 200
    
    def test_batch_returns_correct_count(self, health_response):
        events = [
            {"dst_port": 80},
            {"dst_port": 443},
            {"dst_port": 22},
        ]
        response = client.post("/v1/score/batch", json={"events": events})
        data = response.json()
        assert len(data) == 3
    
    def test_batch_each_has_score(self, health_response):
        response = client.post("/v1/score/batch", json={
            "events": [
                {"dst_port": 443},
                {"dst_port": 22, "status": "failed"},
            ]
        })
        for item in response.json():
            assert "anomaly_score" in item
            assert "classification" in item


# Run with: pytest tests/test_api.py -v
