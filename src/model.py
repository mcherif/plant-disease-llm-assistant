from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import json
import os

MODEL_DIR = "models/vit-finetuned"


def load_model():
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Load class mapping
    with open(os.path.join(MODEL_DIR, "class_mapping.json")) as f:
        class_mapping = json.load(f)
        idx_to_class = {v: k for k, v in class_mapping.items()}
    return model, processor, idx_to_class, device


def preprocess_image(image, processor, device):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def predict_image(model, img_tensor, idx_to_class):
    with torch.no_grad():
        outputs = model(**img_tensor)
        pred_idx = torch.argmax(outputs.logits, dim=1).item()
        pred_class = idx_to_class[pred_idx]
        confidence = torch.softmax(outputs.logits, dim=1)[0, pred_idx].item()
    return {"class": pred_class, "confidence": confidence}
