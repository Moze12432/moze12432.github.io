import streamlit as st
import os

from detector.predict import predict_image

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="AI Fake Image Detector",
    layout="centered"
)

st.title("AI-Based Fake Image Detection")

st.write("""
This system analyzes uploaded images to determine
whether they are:
- Real
- AI-generated
- Manipulated
""")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(
        file_path,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("")

    with st.spinner("Analyzing image..."):

        prediction = predict_image(file_path)

    fake_score = prediction * 100
    real_score = 100 - fake_score

    st.subheader("Detection Results")

    if prediction > 0.5:

        st.error(
            f"Fake Image Detected ({fake_score:.2f}%)"
        )

    else:

        st.success(
            f"Real Image ({real_score:.2f}%)"
        )

    st.progress(float(prediction))

    st.write(f"Fake Probability: {fake_score:.2f}%")
    st.write(f"Real Probability: {real_score:.2f}%")

    st.subheader("Forensic Interpretation")

    if prediction > 0.5:

        st.warning("""
Possible indicators of image manipulation detected.

The uploaded image may contain:
- AI-generated artifacts
- Deepfake inconsistencies
- Synthetic textures
- Unrealistic visual patterns
""")

    else:

        st.info("""
The uploaded image appears authentic based on
the trained forensic AI model.
""")
