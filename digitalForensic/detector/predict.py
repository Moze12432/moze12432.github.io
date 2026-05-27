import numpy as np
from PIL import Image
from detector.model import load_model

session = load_model()

def preprocess_image(image_path):
    # Get expected input shape from model
    input_info = session.get_inputs()[0]
    _, height, width, channels = input_info.shape

    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height))

    img = np.array(image).astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def predict_image(image_path):
    img = preprocess_image(image_path)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})
    prediction = outputs[0][0][0]

    return float(prediction)
