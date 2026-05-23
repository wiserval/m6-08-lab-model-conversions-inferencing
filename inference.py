import numpy as np
import onnxruntime as ort
from PIL import Image

class FlowerClassifier:
    def __init__(self, onnx_path):
        options = ort.SessionOptions()
        options.log_severity_level = 3

        self.session = ort.InferenceSession(onnx_path, sess_options=options)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB").resize((232, 232))

        left = (232 - 224) // 2
        img = img.crop((left, left, left + 224, left + 224))

        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0

        return ((x - self.mean) / self.std).astype(np.float32)

    def predict(self, image_path, k=5):
        x = self.preprocess(image_path)
        logits = self.session.run(None, {"input": x})[0][0]

        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        top = np.argsort(probs)[::-1][:k]
        return [(int(i), float(probs[i])) for i in top]
