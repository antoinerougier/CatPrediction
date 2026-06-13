import io
import pytest
import torch
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def make_mock_model():
    mock = MagicMock()
    mock.return_value = torch.tensor([[2.0, 0.5]])
    return mock


@pytest.fixture(autouse=True)
def mock_model_loading():
    with patch("api.main.get_model") as mock_get:
        mock_get.return_value = (make_mock_model(), ["cat", "dog"])
        yield


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "classes" in response.json()


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_returns_valid_structure(sample_cat_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("cat.jpg", sample_cat_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in ["cat", "dog"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_feedback_invalid_id():
    response = client.post("/feedback", json={"prediction_id": 999999, "correct": True})
    assert response.status_code == 404
