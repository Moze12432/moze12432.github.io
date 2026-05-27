import streamlit as st
from PIL import Image
import os

from forensic.ela import perform_ela
from detector.model import load_detector
from detector.predict import predict_image

# Create uploads directory
os.makedirs("uploads", exist_ok=True)

# Load TFLite model
model = load_detector()

# Streamlit page config
st.set_page_config(
    page_title="Deepfake Forensics AI",
    layout="wide"
)

# Title
st.title("AI-Based Fake Image Detection System")

st.write(
    "Upload an image for AI-powered digital forensic analysis."
)

# Upload section
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# If uploaded
if uploaded_file is not None:

    # Save uploaded image
    upload_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Open image
    image = Image.open(upload_path)

    # Create columns
    col1, col2 = st.columns(2)

    # Original image
    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

    # ELA analysis
    ela_image = perform_ela(upload_path)

    with col2:

        st.subheader("ELA Analysis")

        st.image(
            ela_image,
            use_container_width=True
        )

    # Divider
    st.divider()

    # AI prediction
    label, confidence = predict_image(
        model,
        upload_path
    )

    # Display result
    st.subheader("AI Detection Result")

    if label == "FAKE":

        st.error(
            f"Prediction: {label} "
            f"({confidence:.2f}% confidence)"
        )

    else:

        st.success(
            f"Prediction: {label} "
            f"({confidence:.2f}% confidence)"
        )

    # Confidence bar
    st.progress(
        min(int(confidence), 100)
    )

    # Final forensic summary
    st.subheader("Forensic Summary")

    st.write(
        f"""
        The uploaded image was analyzed using:
        - Error Level Analysis (ELA)
        - AI-based Deepfake Detection
        - TensorFlow Lite Inference

        Final Prediction:
        **{label}**
        with confidence score of
        **{confidence:.2f}%**
        """
    )
