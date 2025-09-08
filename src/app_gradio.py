"""
OBSOLETE: Use streamlit_app.py instead.

This Gradio app is deprecated and replaced by the Streamlit UI in src/interface/streamlit_app.py.
All new UI features and maintenance should be done in the Streamlit app.
"""
"""
Plant Disease LLM Assistant - Gradio app.

Summary:
- Provides a web UI (Gradio Blocks) to classify plant diseases from an uploaded image.
- Loads a fine‑tuned Hugging Face image classification model from models/vit-finetuned.
- Runs inference on CPU or GPU (CUDA if available), shows a result card and top‑k probabilities.
- Embeds a local README and logo (if present) into the UI.

How to run:
- Local development:  python src/app_gradio.py --local    -> http://127.0.0.1:7860
- HF Spaces/containers: python src/app_gradio.py          -> binds 0.0.0.0 using PORT env var

Requirements:
- gradio, torch, transformers; the model dir must include class_mapping.json.
- Expected files:
  models/vit-finetuned/
    - model weights/config (e.g., pytorch_model.bin, config.json, preprocessor_config.json)
    - class_mapping.json mapping class_name -> index
  images/plant-disease-llm-assistant-logo.png (optional)
  README.md (optional)

Notes:
- Toggle DEBUG=True to print additional diagnostic messages.
"""

import gradio as gr
import torch
import os
import json
import base64
from transformers import AutoImageProcessor, AutoModelForImageClassification
import sys

# Debug flag: set to True to enable, False to disable
DEBUG = False


def dbg(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")


MODEL_DIR = "models/vit-finetuned"

# Build a safe path to the logo and expose it via Gradio's file= URL
LOGO_REL = "images/plant-disease-llm-assistant-logo.png"
LOGO_ABS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", LOGO_REL))

# Path to README for embedding
README_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "README.md"))
SPACE_URL = "https://huggingface.co/spaces/mcherif/Plant-Disease-LLM-Assistant"


def logo_block():
    if os.path.exists(LOGO_ABS):
        try:
            with open(LOGO_ABS, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            html = f'<img src="data:image/png;base64,{b64}" width="200" />'
            dbg_msg = f"Logo OK: {LOGO_ABS}"
        except Exception as e:
            html = '<div style="color:red;">Logo image failed to load!</div>'
            dbg_msg = f"Logo read error at {LOGO_ABS}: {e}"
    else:
        html = '<div style="color:red;">Logo image not found!</div>'
        dbg_msg = f"Logo missing at: {LOGO_ABS}"
    dbg(dbg_msg)
    return html, dbg_msg


def load_readme():
    try:
        with open(README_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        dbg(f"README load failed: {e}")
        return "# About\nREADME.md not found."


# Preload README markdown
readme_md = load_readme()

logo_html, logo_dbg = logo_block()


def predict(image):
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dbg(f"Using device: {device}")

    with open(os.path.join(MODEL_DIR, "class_mapping.json")) as f:
        class_mapping = json.load(f)
        idx_to_class = {v: k for k, v in class_mapping.items()}
    dbg(f"Loaded {len(class_mapping)} classes from {os.path.join(MODEL_DIR, 'class_mapping.json')}")

    image = image.convert("RGB")
    dbg(f"Image mode={image.mode}, size={image.size}")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_class = idx_to_class[pred_idx]
        confidence = float(probs[pred_idx].item())

        # Build Top-K (up to 3) dict for gr.Label
        k = min(3, probs.shape[0])
        top_vals, top_idxs = torch.topk(probs, k)
        topk = {idx_to_class[int(i)]: float(v) for v, i in zip(
            top_vals.tolist(), top_idxs.tolist())}

    dbg(f"Prediction: {pred_class} (idx={pred_idx}, conf={confidence:.4f})")

    # Pretty result card (HTML)
    result_card = f"""
    <div style="padding:12px;border-radius:10px;background:#f0fff4;border:1px solid #c6f6d5;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:1.1em;">🌿 <b>{pred_class}</b></div>
        <div style="font-size:0.95em;">Confidence: <b>{confidence*100:.1f}%</b></div>
      </div>
    </div>
    """
    return result_card, topk


with gr.Blocks() as demo:
    # Header: logo left, title + subtitle + form right
    with gr.Row():
        with gr.Column(scale=0, min_width=220):
            gr.HTML(logo_html)
        with gr.Column(scale=1, min_width=500):
            gr.Markdown("# Plant Disease Classifier")
            gr.Markdown("Upload a plant image to classify its disease.")
            # Upload + preview directly under the subtitle
            img_in = gr.Image(type="pil", label="Input image")
            submit = gr.Button("Classify", variant="primary")
            # Replace plain text with a card + Top-K label
            result_card = gr.HTML(label="Result")
            result_probs = gr.Label(label="Top predictions", num_top_classes=3)

            submit.click(fn=predict, inputs=img_in, outputs=[
                         result_card, result_probs])

    # Collapsible README content inside the app (optional)
    with gr.Accordion("About this app", open=False):
        gr.Markdown(readme_md)

    if DEBUG:
        gr.Markdown(f"<sub>{logo_dbg}</sub>")

if __name__ == "__main__":
    LOCAL = "--local" in sys.argv

    if LOCAL:
        # Local dev: bind to loopback so the browser check works on Windows
        demo.launch(server_name="127.0.0.1", server_port=7860)
    else:
        # HF Spaces / containers: bind to 0.0.0.0 and use provided PORT
        port = int(os.getenv("PORT", "7860"))
        demo.launch(server_name="0.0.0.0", server_port=port)
