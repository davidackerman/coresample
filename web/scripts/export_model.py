"""Export DINOv2-small to ONNX for the browser: input (batch, 3, 112, 112) ImageNet-normalized,
output (batch, 768) = [CLS, mean of patch tokens], each L2-normalized (the same embedding as
embed.py). Writes web/public/models/dinov2-small-112.onnx and checks it against PyTorch."""

import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))
OUT = ROOT / "web/public/models/dinov2-small-112.onnx"


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        h = self.m(pixel_values=x).last_hidden_state
        cls = torch.nn.functional.normalize(h[:, 0], dim=1)
        avg = torch.nn.functional.normalize(h[:, 1:].mean(1), dim=1)
        return torch.cat([cls, avg], 1)


def main():
    w = Wrap(AutoModel.from_pretrained("facebook/dinov2-small").eval()).eval()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(w, (torch.randn(4, 3, 112, 112),), str(OUT), input_names=["pixel_values"],
                      output_names=["embedding"], opset_version=17, dynamo=False,
                      dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}})
    import onnxruntime as ort

    x = torch.randn(7, 3, 112, 112)
    diff = np.abs(w(x).detach().numpy() - ort.InferenceSession(str(OUT)).run(None, {"pixel_values": x.numpy()})[0]).max()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB), max |torch - onnx| = {diff:.2e}")


if __name__ == "__main__":
    main()
