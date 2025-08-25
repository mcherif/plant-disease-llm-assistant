from fastapi import FastAPI, UploadFile, File
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import json
import os
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plant-disease-api")

app = FastAPI(title="Plant Disease Classifier API")

# Serve static files (like index.html) from a 'static' directory
app.mount("/static", StaticFiles(directory=os.path.join(
    os.path.dirname(__file__), "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    logger.info("Serving index.html")
    index_path = os.path.join(os.path.dirname(
        __file__), "static", "index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error loading index.html: {e}")
        return HTMLResponse(content="Error loading page.", status_code=500)

# Remove any duplicate root() definition elsewhere in this file to fix F811.


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
    logger.info(f"Received file: {file.filename}")
    try:
        image_bytes = file.file.read()
        logger.info(f"Image bytes length: {len(image_bytes)}")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        logger.info(f"Inputs keys: {inputs.keys()}")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            pred_idx = torch.argmax(outputs.logits, dim=1).item()
            pred_class = idx_to_class[pred_idx]
            confidence = torch.softmax(outputs.logits, dim=1)[
                0, pred_idx].item()
        logger.info(f"Prediction: {pred_class}, Confidence: {confidence}")
        return {"class": pred_class, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        return {"error": str(e)}
