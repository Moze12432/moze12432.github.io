import numpy as np

from PIL import Image

def predict_image(interpreter, image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image).astype(np.float32)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Input/output details
    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(
        input_details[0]['index'],
        image_array
    )

    # Run inference
    interpreter.invoke()

    # Get prediction
    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )[0][0]

    confidence = float(prediction) * 100

    if prediction >= 0.5:
        label = "FAKE"
    else:
        label = "REAL"
        confidence = 100 - confidence

    return label, confidence
