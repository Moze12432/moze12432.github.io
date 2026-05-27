import os
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_detector.onnx")

session = ort.InferenceSession(MODEL_PATH)

def load_model():
    return session
