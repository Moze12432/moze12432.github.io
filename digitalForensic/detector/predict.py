from PIL import Image
import torch

def predict_image(processor, model, image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    predicted_class = logits.argmax(-1).item()

    confidence = torch.softmax(
        logits,
        dim=1
    )[0][predicted_class].item()

    labels = ["REAL", "FAKE"]

    label = labels[predicted_class]

    return label, confidence * 100
