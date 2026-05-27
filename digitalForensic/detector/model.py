import os
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_detector.onnx")

session = ort.InferenceSession(MODEL_PATH)

# Debug: print model input details
for inp in session.get_inputs():
    print(f"Input name: {inp.name}, shape: {inp.shape}, type: {inp.type}")

def load_model():
    return session
