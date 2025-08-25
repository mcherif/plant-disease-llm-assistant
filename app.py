from src.model import load_model, preprocess_image, predict_image
from PIL import Image
import streamlit as st
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

logging.debug("App started. Waiting for user to upload an image.")

# Display logo and title side by side
col1, col2 = st.columns([1, 3])
with col1:
    st.image(
        "images/plant-disease-logo.png",
        width=120,
    )
with col2:
    st.title("Plant Disease Classifier")
    st.write("Upload a plant leaf image to get disease prediction.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"])
logging.debug("File uploader rendered. Awaiting user input.")

if uploaded_file is not None:
    logging.debug(f"User uploaded file: {uploaded_file.name}")
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        logging.debug("Image opened successfully.")

        # Load model (cache to avoid reloading on every run)
        @st.cache_resource
        def get_model():
            logging.debug("Loading model and processor...")
            return load_model()

        model, processor, idx_to_class, device = get_model()
        logging.debug("Model, processor, and class mapping loaded.")

        # Preprocess the image for the model
        img_tensor = preprocess_image(image, processor, device)
        logging.debug(f"Image preprocessed. Tensor shape: {img_tensor.shape}")

        # Make prediction
        prediction = predict_image(model, img_tensor, idx_to_class)
        logging.debug(f"Prediction result: {prediction}")

        st.write(f"**Prediction:** {prediction['class']}")
        st.write(f"**Confidence:** {prediction['confidence']:.2f}")
    except Exception as e:
        logging.exception("Error during prediction")
        st.error(f"Error: {e}")

# Streamlit app ends here. Remove Gradio runner lines below to fix undefined names.
# (Use src/app_gradio.py for Gradio.)

# Removed:
# sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
# if __name__ == "__main__":
#     port = int(os.getenv("PORT", "7860"))
#     demo.launch(server_name="0.0.0.0", server_port=port)
