import os
import json
import torch
import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEBUG_MINIMAL = False  # set True to sanity-check Space/Container boot

MODEL_DIR = "models/vit-finetuned"

st.set_page_config(page_title="Plant Disease LLM Assistant", page_icon="🌿")

if DEBUG_MINIMAL:
    st.title("Plant Disease Classifier")
    st.write("Minimal debug mode: Streamlit is working!")
else:
    st.title("Plant Disease LLM Assistant")
    st.caption("Upload an image to classify with the finetuned ViT model.")

    @st.cache_resource
    def load_model_and_processor():
        model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
        processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)
        return model, processor, device

    @st.cache_data
    def id2label():
        try:
            with open(os.path.join(MODEL_DIR, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return {int(k): v for k, v in cfg.get("id2label", {}).items()}
        except Exception:
            return {}

    uploaded = st.file_uploader(
        "Upload a plant image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        with st.spinner("Loading model..."):
            model, processor, device = load_model_and_processor()
        labels = id2label()

        with st.spinner("Running inference..."):
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                topk = min(5, probs.shape[-1])
                scores, idxs = torch.topk(probs, topk)

        st.subheader("Top predictions")
        for score, idx in zip(scores.tolist(), idxs.tolist()):
            label = labels.get(idx, f"class_{idx}")
            st.write(f"- {label}: {score:.3f}")
