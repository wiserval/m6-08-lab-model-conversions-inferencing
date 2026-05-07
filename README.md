![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Model Conversions & Inferencing

## Overview

You spent yesterday training a fine-tuned ResNet on Flowers-102. Today you'll turn that trained model into a **deployable inference artefact**: export it to ONNX, build a clean inference-only pipeline, run it through ONNX Runtime, apply INT8 quantisation, and benchmark every step.

This is the exact workflow you'll apply on Friday's assessment to your cat-detection model — so think of today as a dress rehearsal for tomorrow's deliverable.

## Learning Goals

By the end of this lab you should be able to:

- Export a trained PyTorch model to ONNX with proper `dynamic_axes`.
- Validate an ONNX model with `onnx.checker.check_model`.
- Run inference with ONNX Runtime and confirm numerical equivalence to the PyTorch model.
- Apply post-training dynamic INT8 quantisation and measure the speed/size impact.
- Build an inference pipeline as a self-contained Python module — no training code, no PyTorch dependency at runtime if you want to drop it.

## Setup and Context

You'll work in a single Jupyter Notebook. You'll need a trained model from yesterday's transfer learning lab — if you don't have one, the repo provides a small fallback checkpoint trained on Flowers-102.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install numpy pandas matplotlib torch torchvision onnx onnxruntime
```

> If `onnxruntime` install fails on your platform, try `onnxruntime-cpu`. For GPU acceleration, install `onnxruntime-gpu` (requires CUDA).

## Getting Started

1. Create a notebook called **`m6-08-model-conversions-inferencing.ipynb`**.
2. Standard imports:

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import onnx
import onnxruntime as ort
import time

device = "cpu"  # we'll deliberately use CPU to demonstrate quantisation gains
torch.manual_seed(42)
```

3. Load your trained model from yesterday (or the provided checkpoint):

```python
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 102)
model.load_state_dict(torch.load("flowers102_resnet18.pth", map_location=device))
model.eval()
```

## Tasks

### Task 1 — Export to ONNX

1. Define an example input matching the validation pipeline:

```python
example = torch.randn(1, 3, 224, 224)
```

2. Export to ONNX with **explicit dynamic batch axis**:

```python
torch.onnx.export(
    model, example, "flowers_resnet18.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
```

3. Validate the export:

```python
onnx_model = onnx.load("flowers_resnet18.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")
```

4. Print the file size of the exported model in MB.

### Task 2 — Numerical Equivalence Check

Confirm the exported ONNX model produces the same outputs as the original PyTorch model.

1. Load the ONNX model into an ONNX Runtime session:

```python
session = ort.InferenceSession("flowers_resnet18.onnx")
```

2. Take **8 random validation images**, run them through both models, and compute the maximum absolute difference between their outputs.
3. Assert the difference is below `1e-4`. If not, investigate (different normalisation, dropout still on, etc.).

### Task 3 — Build an Inference Pipeline

Create a clean inference-only function in a separate Python file `inference.py` that:

- Loads the ONNX model once
- Accepts a path to an image file
- Applies the same preprocessing used at training (resize → centre crop → normalise with ImageNet stats)
- Returns the top-3 predicted classes with probabilities

Suggested skeleton:

```python
import numpy as np, onnxruntime as ort
from PIL import Image

class FlowerClassifier:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB").resize((232, 232))
        # centre crop to 224x224
        left = (232 - 224) // 2
        img = img.crop((left, left, left + 224, left + 224))
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return ((x - self.mean) / self.std).astype(np.float32)

    def predict(self, image_path, k=3):
        x = self.preprocess(image_path)
        logits = self.session.run(None, {"input": x})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        top = np.argsort(probs)[::-1][:k]
        return [(int(i), float(probs[i])) for i in top]
```

In your notebook:
1. Import this class and run inference on 5 test images. Print the top-3 predictions for each.
2. Verify the predictions match what your PyTorch model produces for the same images.

### Task 4 — Quantise to INT8

Apply post-training dynamic quantisation:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="flowers_resnet18.onnx",
    model_output="flowers_resnet18.int8.onnx",
    weight_type=QuantType.QInt8,
)
```

Then:

1. Print the file size of the quantised model and the size ratio vs FP32.
2. Load the quantised model into a new ONNX Runtime session.
3. Run it on the same validation set used in Task 2 and compare the outputs with the FP32 ONNX model — report the maximum and mean absolute difference.
4. Run it on the held-out test set (or whatever test loader you have from yesterday) and report the test accuracy. Compare to the FP32 ONNX model's test accuracy.

In a markdown cell, comment on the trade-off: how much accuracy did you lose, and how much smaller is the model?

### Task 5 — Latency Benchmark

Benchmark all three variants on the same hardware.

1. Pick a single image. For each model — PyTorch eager mode, FP32 ONNX, INT8 ONNX — measure the average latency over 100 runs (use `time.perf_counter()`).
2. Report the results in a table:

| Model | File size (MB) | Avg latency (ms) | Speedup vs PyTorch |
|---|---|---|---|
| PyTorch (FP32) | … | … | 1.00× |
| ONNX (FP32) | … | … | … |
| ONNX (INT8) | … | … | … |

3. In a markdown cell, comment on what you see. Was the speedup what you expected? Where did most of the gain come from?

### Task 6 — Edge Case (Stretch)

This is optional but recommended.

1. Try inferring on an image with an unusual aspect ratio (very wide or very tall). Confirm your preprocessing handles it gracefully.
2. Try a batch of 16 images at once and compare the per-image latency to single-image inference. Quantify the throughput gain.
3. In a markdown cell, comment on when you'd choose to batch and when you'd serve single inputs.

## Submission

### What to submit

- `m6-08-model-conversions-inferencing.ipynb` — completed notebook.
- `inference.py` — your inference module.
- `flowers_resnet18.onnx` and `flowers_resnet18.int8.onnx` — exported model files (these can be a few MB each, that's fine for a lab).

### Definition of done (checklist)

- [ ] PyTorch model exported to ONNX with dynamic batch axis and validated.
- [ ] Numerical equivalence between PyTorch and FP32 ONNX confirmed.
- [ ] Standalone `inference.py` runs and matches PyTorch predictions.
- [ ] INT8 quantised model produced; size and accuracy compared to FP32.
- [ ] Latency benchmark table with all three variants.
- [ ] `Kernel → Restart & Run All` produces no errors.

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete model conversions and inferencing"
git push origin main
```

Then open a **Pull Request** on the original repository describing your benchmark results. Tomorrow you'll do this for the cat-detection model.
