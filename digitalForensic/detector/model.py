import timm
import torch

def load_model():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True
    )

    model.eval()

    return model
