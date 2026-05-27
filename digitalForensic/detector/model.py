import tensorflow as tf

def load_detector():

    interpreter = tf.lite.Interpreter(
        model_path="models/deepfake_detector.tflite"
    )

    interpreter.allocate_tensors()

    return interpreter
