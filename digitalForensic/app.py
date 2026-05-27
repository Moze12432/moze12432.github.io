import streamlit as st
from PIL import Image
import os

from forensic.ela import perform_ela

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Streamlit page settings
st.set_page_config(
    page_title="Deepfake Forensics AI",
    layout="wide"
)

# App title
st.title("AI-Based Fake Image Detection System")

st.write(
    "Upload an image for digital forensic analysis using "
    "Error Level Analysis (ELA)."
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# If image uploaded
if uploaded_file is not None:

    # Save uploaded image
    upload_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Open original image
    image = Image.open(upload_path)

    # Create 2 columns
    col1, col2 = st.columns(2)

    # Show original image
    with col1:
        st.subheader("Original Image")
        st.image(
            image,
            use_container_width=True
        )

    # Perform ELA
    ela_image = perform_ela(upload_path)

    # Show ELA image
    with col2:
        st.subheader("ELA Analysis")
        st.image(
            ela_image,
            use_container_width=True
        )

    # Success message
    st.success("ELA forensic analysis completed successfully.")
