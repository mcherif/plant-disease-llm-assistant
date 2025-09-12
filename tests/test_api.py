import os
import io
import pytest
from fastapi.testclient import TestClient
from src.interface.api import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "index_dir" in data


def test_rag():
    payload = {
        "query": "How do I treat powdery mildew on tomato?",
        "plant": "tomato",
        "disease": "powdery mildew",
        "top_k": 2
    }
    resp = client.post("/rag", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data


def test_classify(monkeypatch):
    from PIL import Image
    import numpy as np
    import torch

    def dummy_from_pretrained(*args, **kwargs):
        print("Dummy model loaded")

        class DummyModel:
            def eval(self): print("DummyModel.eval()"); return self

            def to(self, device): print(
                f"DummyModel.to({device})"); return self

            def __call__(self, **inputs):
                print(f"DummyModel.__call__ inputs: {inputs}")

                class DummyLogits:
                    logits = torch.tensor([[1.0, 0.0, 0.0]])
                return DummyLogits()
        return DummyModel()

    def dummy_processor(*args, **kwargs):
        print("Dummy processor loaded")

        class DummyProcessor:
            def __call__(self, images, return_tensors):
                print(
                    f"DummyProcessor.__call__ images: {images}, return_tensors: {return_tensors}")
                return {"pixel_values": torch.zeros(1, 3, 32, 32)}
        return DummyProcessor()

    monkeypatch.setattr(
        "transformers.AutoModelForImageClassification.from_pretrained", dummy_from_pretrained)
    monkeypatch.setattr(
        "transformers.AutoImageProcessor.from_pretrained", dummy_processor)
    monkeypatch.setattr("builtins.open", lambda *a, **k: io.StringIO(
        '{"0": "Powdery Mildew", "1": "Leaf Spot", "2": "Healthy"}'))

    img = Image.new("RGB", (32, 32), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    print("Sending request to /api/classify")
    resp = client.post(
        "/api/classify", files={"file": ("test.jpg", buf, "image/jpeg")})
    print(f"Response status: {resp.status_code}")
    print(f"Response body: {resp.text}")
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data


def test_feedback():
    payload = {
        "query": "How do I treat powdery mildew on tomato?",
        "answer": "Use fungicides and remove infected leaves...",
        "feedback": "up",
        "timestamp": "2025-09-12T12:34:56Z"
    }
    resp = client.post("/api/feedback", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
