from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert set(data["classes"]) == {"cat", "dog"}


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello world", "text/plain")},
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
    assert set(data["probabilities"].keys()) == {"cat", "dog"}


def test_feedback_invalid_id():
    response = client.post("/feedback", json={"prediction_id": 999999, "correct": True})
    assert response.status_code == 404
