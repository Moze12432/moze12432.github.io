import streamlit as st
from PIL import Image
import os

from forensic.ela import perform_ela

st.set_page_config(
    page_title="Deepfake Forensics AI",
    layout="wide"
)

st.title("AI-Based Fake Image Detection System")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    upload_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(upload_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    ela_image = perform_ela(upload_path)

    with col2:
        st.subheader("ELA Analysis")
        st.image(ela_image, use_container_width=True)

    st.success("ELA forensic analysis completed.")
