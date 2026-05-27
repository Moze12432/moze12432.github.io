from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification

def load_model():

    processor = AutoImageProcessor.from_pretrained(
        "prithivMLmods/Deep-Fake-Detector-Model"
    )

    model = AutoModelForImageClassification.from_pretrained(
        "prithivMLmods/Deep-Fake-Detector-Model"
    )

    return processor, model
