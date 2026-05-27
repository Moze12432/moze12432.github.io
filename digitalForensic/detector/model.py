import tflite_runtime.interpreter as tflite

def load_detector():

    interpreter = tflite.Interpreter(
        model_path="models/deepfake_detector.tflite"
    )

    interpreter.allocate_tensors()

    return interpreter
