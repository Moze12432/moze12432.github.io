import onnxruntime as ort

MODEL_PATH = "models/deepfake_detector.onnx"

session = ort.InferenceSession(MODEL_PATH)

def load_model():
    return session
