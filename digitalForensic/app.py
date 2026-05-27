import streamlit as st
from PIL import Image

st.title("Deepfake Forensics AI")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    st.success("Image ready for analysis")
