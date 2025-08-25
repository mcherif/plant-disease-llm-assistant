from fastapi import FastAPI, UploadFile, File
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import json
import os

app = FastAPI(title="Plant Disease Classifier API")

MODEL_DIR = "models/vit-finetuned"

# Load model and processor at startup
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load class mapping
with open(os.path.join(MODEL_DIR, "class_mapping.json")) as f:
    class_mapping = json.load(f)
    idx_to_class = {v: k for k, v in class_mapping.items()}


@app.post("/predict/")
def predict(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=1).item()
        pred_class = idx_to_class[pred_idx]
        confidence = torch.softmax(outputs.logits, dim=1)[0, pred_idx].item()
    return {"class": pred_class, "confidence": confidence}


@app.get("/")
def root():
    return {"message": "Plant Disease Classifier API. Use /predict/ to POST an image."}
