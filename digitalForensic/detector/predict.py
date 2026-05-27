from PIL import Image
import torch
from torchvision import transforms

# ImageNet labels
LABELS = [
    "REAL",
    "FAKE"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(model, image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)

    probabilities = torch.softmax(outputs[0], dim=0)

    confidence, predicted = torch.max(probabilities, 0)

    # Temporary fake mapping
    label = "FAKE" if predicted.item() % 2 == 0 else "REAL"

    return label, confidence.item() * 100
