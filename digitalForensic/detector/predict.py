import numpy as np
from PIL import Image

def predict_image(interpreter, image_path):

    # Load image
    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image).astype(np.float32)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Get input/output details
    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    # Handle quantized input
    input_scale, input_zero_point = input_details[0]['quantization']

    if input_scale > 0:

        image_array = image_array / input_scale + input_zero_point

        image_array = image_array.astype(
            input_details[0]['dtype']
        )

    # Set tensor
    interpreter.set_tensor(
        input_details[0]['index'],
        image_array
    )

    # Run inference
    interpreter.invoke()

    # Get output
    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )

    # Handle quantized output
    output_scale, output_zero_point = output_details[0]['quantization']

    if output_scale > 0:

        prediction = (
            prediction.astype(np.float32)
            - output_zero_point
        ) * output_scale

    prediction = prediction[0][0]

    confidence = float(prediction) * 100

    # Binary classification
    if prediction >= 0.5:

        label = "FAKE"

    else:

        label = "REAL"

        confidence = 100 - confidence

    return label, confidence
