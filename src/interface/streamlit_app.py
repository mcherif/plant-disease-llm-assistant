"""
Plant Disease RAG Assistant — Streamlit UI

This is the main interactive frontend for the plant disease RAG assistant.
Features:
- Upload plant images for disease classification (ViT finetuned model)
- Ask questions about plant diseases and management using RAG (retrieval-augmented generation)
- Uses OpenAI LLM backend and PlantVillage + Wikipedia knowledge base
- Sidebar settings: index directory, top-k context, device, detected labels
- Displays sources and context for answers
- Replaces the previous Gradio app (see app_gradio.py, now obsolete)

Usage:
- Run via Streamlit: `streamlit run src/interface/streamlit_app.py`
- Configure index and API key via sidebar or environment variables

Author: Mohamed Cherif / innerloopinc@gmail.com
"""

import os
import json
import torch
import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from src.llm.rag_pipeline import RAGPipeline, RetrievalConfig
import re
import pandas as pd

DEBUG_MINIMAL = False  # set True to sanity-check Space/Container boot

LOGO_PATH = "images/plant-disease-rag-assistant-logo.png"

st.set_page_config(page_title="Plant Disease RAG Assistant", layout="wide")
st.sidebar.image(LOGO_PATH, use_container_width=True)
st.title("Plant Disease RAG Assistant")

# Sidebar config (move this block up, before any function that uses MODEL_DIR)
st.sidebar.header("Settings")
MODEL_DIR = st.sidebar.text_input("Model directory", "models/vit-finetuned")
index_dir = st.sidebar.text_input("Index dir", "models/index/kb-faiss-bge")
top_k = st.sidebar.slider("Top-k context", 1, 6, 3)
retrieval_device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
model_env = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
st.sidebar.caption(f"Judge/LLM model (env): {model_env}")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("OPENAI_API_KEY not set. Answers won’t work.", icon="⚠️")

if st.sidebar.button("Show Dashboard"):
    st.header("Feedback Dashboard")
    try:
        df = pd.read_json("data/feedback/feedback.jsonl", lines=True)
        st.bar_chart(df["feedback"].value_counts())
        st.line_chart(df.groupby("timestamp").size())
        st.write("Recent feedback:", df.tail(10))
        # Add more charts as needed (latency, retrieval count, etc.)
    except Exception:
        st.warning("No feedback data yet or error loading dashboard.")

# ---- helpers (must be defined before use) ----


def _canon_plant(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    aliases = {
        "peach tree": "Peach",
        "maize": "Corn (maize)",
        "corn": "Corn (maize)",
    }
    key = s.lower()
    return aliases.get(key, s.title())


def _canon_disease(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    aliases = {
        # common classifier labels -> KB names
        "peach bacterial": "Bacterial spot",
        "bacterial spot of peach": "Bacterial spot",
        "powdery mildew": "Powdery mildew",
        "northern leaf blight": "Northern Leaf Blight",
        "gray leaf spot": "Cercospora leaf spot Gray leaf spot",
    }
    key = s.lower()
    return aliases.get(key, re.sub(r"\s+", " ", s).strip())


def _infer_labels_from_classifier(raw: str):
    """
    Heuristic: infer (plant, disease) from a raw classifier label like 'peach bacterial'.
    Uses canonicalizers and simple keyword detection.
    """
    if not raw:
        return None, None
    s = re.sub(r"[_\-]+", " ", str(raw)).strip()
    low = s.lower()
    plant_keys = ["peach", "tomato", "potato", "apple", "grape", "corn", "maize",
                  "pepper", "orange", "banana", "cucumber", "zucchini", "strawberry", "cherry"]
    plant = None
    matched_key = None
    for k in plant_keys:
        if k in low:
            plant = _canon_plant(k)
            matched_key = k
            break
    disease_raw = s
    if matched_key:
        disease_raw = re.sub(
            rf"\b{re.escape(matched_key)}\b", " ", disease_raw, flags=re.I)
    disease = _canon_disease(disease_raw.strip())
    if not disease or disease.lower() == (plant or "").lower():
        disease = _canon_disease(s)
    return plant, disease or None
# ---- helpers end ----


if DEBUG_MINIMAL:
    st.header("Plant Disease Classifier")
    st.write("Minimal debug mode: Streamlit is working!")
else:
    st.caption("Upload an image to classify with the finetuned ViT model.")

    @st.cache_resource
    def load_model_and_processor():
        model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
        processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
        model_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(model_device)
        return model, processor, model_device

    @st.cache_data
    def id2label():
        # 1) Prefer class_mapping.json (name -> index) and invert to index -> name
        try:
            with open(os.path.join(MODEL_DIR, "class_mapping.json"), "r", encoding="utf-8") as f:
                name2idx = json.load(f)
            if isinstance(name2idx, dict) and name2idx:
                return {int(v): k for k, v in name2idx.items()}
        except Exception:
            pass
        # 2) Fallback to config.json id2label (may be LABEL_x)
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
        st.image(image, caption="Uploaded image",
                 use_container_width=False, width=300)

        with st.spinner("Loading model..."):
            model, processor, model_device = load_model_and_processor()
        labels = id2label()

        with st.spinner("Running inference..."):
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                topk = min(5, probs.shape[-1])
                scores, idxs = torch.topk(probs, topk)

        st.subheader("Top predictions")
        for score, idx in zip(scores.tolist(), idxs.tolist()):
            label = labels.get(idx, f"class_{idx}")
            st.write(f"- {label}: {score:.3f}")

        # Take top-1 as detected labels and store to session
        top1_label = labels.get(int(idxs[0]), f"class_{int(idxs[0])}")
        plant_guess, disease_guess = _infer_labels_from_classifier(top1_label)
        st.session_state["detected_plant"] = plant_guess or st.session_state.get(
            "detected_plant")
        st.session_state["detected_disease"] = disease_guess or st.session_state.get(
            "detected_disease")

# RAG pipeline (re-init when settings change)
cfg = RetrievalConfig(index_dir=index_dir, top_k=top_k,
                      device=retrieval_device)
rag = RAGPipeline(cfg)

# Detected labels from classifier
detected_plant = st.session_state.get("detected_plant")
detected_disease = st.session_state.get("detected_disease")

# Inputs
query = st.text_area(
    "Question", placeholder="e.g., What can I do to treat this?")
col1, col2 = st.columns(2)

# Initialize inputs and optionally auto-fill BEFORE creating widgets
use_labels = st.checkbox(
    "Use detected plant/disease (if available)", value=True)
# Set defaults if keys missing
if "plant_input" not in st.session_state:
    st.session_state["plant_input"] = (detected_plant or "")
if "disease_input" not in st.session_state:
    st.session_state["disease_input"] = (detected_disease or "")
# If user opted-in and fields are empty, copy detected labels now
if use_labels and (detected_plant or detected_disease):
    if detected_plant and not (st.session_state.get("plant_input") or "").strip():
        st.session_state["plant_input"] = detected_plant
    if detected_disease and not (st.session_state.get("disease_input") or "").strip():
        st.session_state["disease_input"] = detected_disease
    st.caption(
        f"Detected: plant={detected_plant or '-'} | disease={detected_disease or '-'}"
    )

# Now create the text inputs (they will read session_state defaults)
with col1:
    plant = st.text_input("Plant (optional)",
                          key="plant_input", placeholder="Peach")
with col2:
    disease = st.text_input("Disease (optional)",
                            key="disease_input", placeholder="Bacterial spot")

run = st.button("Get answer", type="primary")

# Run
DEFAULT_QUESTION = "What can I do to treat this?"
if run:
    q = (query or "").strip() or DEFAULT_QUESTION
    plant_norm = _canon_plant(st.session_state.get("plant_input") or "")
    disease_norm = _canon_disease(st.session_state.get("disease_input") or "")
    plant_norm = plant_norm or None
    disease_norm = disease_norm or None
    if use_labels and (plant_norm or disease_norm):
        q = " ".join([x for x in [plant_norm, disease_norm, q] if x])

    with st.spinner("Retrieving and generating..."):
        try:
            res = rag.answer(q, plant=plant_norm, disease=disease_norm)
        except Exception as e:
            st.error(f"RAG error: {e}")
        else:
            st.subheader("Answer")
            st.write(res.get("answer", ""))
            retrieved = res.get("retrieved", []) or []
            if retrieved:
                st.subheader("Sources")
                for i, doc in enumerate(retrieved, start=1):
                    meta = doc.get("meta", {})
                    title = meta.get("title") or meta.get(
                        "doc_id") or f"Doc {i}"
                    url = meta.get("url")
                    with st.expander(f"[{i}] {title}"):
                        if url:
                            st.markdown(f"[{url}]({url})")
                        st.write(meta.get("text", "")[:1200])
            else:
                st.info("No sources retrieved.")
