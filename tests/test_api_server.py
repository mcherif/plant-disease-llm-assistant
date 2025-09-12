import requests

# Health check
resp = requests.get("http://localhost:8000/health")
print(resp.json())

# RAG endpoint
payload = {
    "query": "How do I treat Leaf mold on tomato?",
    "plant": "tomato",
    "disease": "leaf mold",
    "top_k": 2
}
resp = requests.post("http://localhost:8000/rag", json=payload)
print(resp.json())

# Classify endpoint
with open("tests/test_image.JPG", "rb") as f:
    files = {"file": ("test_image.jpg", f, "image/jpeg")}
    resp = requests.post("http://localhost:8000/api/classify", files=files)
    print(resp.json())

# Feedback endpoint
feedback = {
    "query": "How do I treat powdery mildew on tomato?",
    "answer": "Use fungicides and remove infected leaves...",
    "feedback": "up",
    "timestamp": "2025-09-12T12:34:56Z"
}
resp = requests.post("http://localhost:8000/api/feedback", json=feedback)
print(resp.json())
