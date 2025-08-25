<<<<<<< HEAD
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

indices = np.linspace(0, 1, num_points)
theta = 2 * np.pi * num_turns * indices
radius = indices

x = radius * np.cos(theta)
y = radius * np.sin(theta)

df = pd.DataFrame({
    "x": x,
    "y": y,
    "idx": indices,
    "rand": np.random.randn(num_points),
})

st.altair_chart(alt.Chart(df, height=700, width=700)
    .mark_point(filled=True)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        color=alt.Color("idx", legend=None, scale=alt.Scale()),
        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
    ))
=======
import streamlit as st

DEBUG_MINIMAL = False  # Set to False to run the full app


if DEBUG_MINIMAL:
    st.write("Streamlit app file loaded")
    st.title("Plant Disease Classifier")
    st.write("Minimal debug mode: Streamlit is working!")
else:
    from PIL import Image
    import torch
    import os
    import json
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    # Model directory
    MODEL_DIR = "models/vit-finetuned"

    st.title("Plant Disease Classifier")
    st.write("App is starting...")
    st.write("b4 uploaded_file")
    uploaded_file = st.file_uploader(
        "Upload a plant image", type=["jpg", "jpeg", "png"])
    st.write("after uploaded_file")
    if uploaded_file is not None:
        # Load model and processor only when needed
        @st.cache_resource
        def load_model_and_processor():
            model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
            processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
            model.eval()
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            return model, processor, device
        st.write("Loading model and processor...")
        model, processor, device = load_model_and_processor()

        # Load class mapping
        with open(os.path.join(MODEL_DIR, "class_mapping.json")) as f:
            class_mapping = json.load(f)
            idx_to_class = {v: k for k, v in class_mapping.items()}
        st.write("Model and processor loaded.")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Image uploaded and displayed.")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        st.write("Inputs processed, running inference...")
        with torch.no_grad():
            outputs = model(**inputs)
            pred_idx = torch.argmax(outputs.logits, dim=1).item()
            pred_class = idx_to_class[pred_idx]
            confidence = torch.softmax(outputs.logits, dim=1)[
                0, pred_idx].item()

        st.markdown(f"**Prediction:** {pred_class}")
        st.markdown(f"**Confidence:** {confidence:.2f}")
>>>>>>> e886e3e (Initial scaffold for Plant Disease LLM Assistant, the purpose is to extend my previous Plant Disease Classifier Repo with retrieval-augmented generation and LLMs to create a plant health assistant)
