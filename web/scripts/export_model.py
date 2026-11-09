"""Export DINOv2-small to ONNX for the browser: 112 px (pooled), 224 px and 112 px (tokens).

Input (batch, 3, S, S) ImageNet-normalized.

  112 px: output (batch, 1152) = [CLS | mean of patch tokens | centre-weighted mean of patch
          tokens (Gaussian, sigma = a quarter of the token grid)], each 384-d part L2-normalized,
          so the page can match on any combination (experiments/test5_matching.py compares them).
  224 px: output (batch, 257, 384) = the last hidden state (CLS, then the 16 x 16 patch tokens),
          raw. The page pools the same 1152-d embedding from it for "detailed" matching, and
          matches windows of tokens for the tile-token (Mars CTX explorer) mode (experiments/test7_template.py).
  112 px, tokens: output (batch, 65, 384), the last hidden state, raw: for a brush (which parts of
          the box matter), whose weighted mean of tokens the page pools; only fetched then, so the
          usual search keeps the small pooled output (experiments/test12_brush.py).

Writes web/public/models/dinov2-small-{112,224,112-tokens}.onnx and checks each against PyTorch."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))  # before transformers reads it at import

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel  # noqa: E402

OUT = ROOT / "web/public/models"
EXPORTS = {"112": (112, False), "224": (224, True), "112-tokens": (112, True)}  # file name: input size, outputs its tokens


class Pooled(torch.nn.Module):
    def __init__(self, m, size):
        super().__init__()
        self.m = m
        g = size // 14
        yy, xx = np.mgrid[0:g, 0:g] - (g - 1) / 2
        w = np.exp(-(yy ** 2 + xx ** 2) / (2 * (g / 4) ** 2)).ravel()
        self.register_buffer("w", torch.from_numpy((w / w.sum()).astype(np.float32)))

    def forward(self, x):
        h = self.m(pixel_values=x).last_hidden_state
        tok = h[:, 1:]
        n = torch.nn.functional.normalize
        return torch.cat([n(h[:, 0], dim=1), n(tok.mean(1), dim=1), n((tok * self.w[None, :, None]).sum(1), dim=1)], 1)


class Tokens(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).last_hidden_state


def main(names=tuple(EXPORTS)):
    import onnxruntime as ort

    m = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    OUT.mkdir(parents=True, exist_ok=True)
    for key in names:
        size, tokens = EXPORTS[key]
        w = (Tokens(m) if tokens else Pooled(m, size)).eval()
        out = OUT / f"dinov2-small-{key}.onnx"
        name = "tokens" if tokens else "embedding"
        torch.onnx.export(w, (torch.randn(2, 3, size, size),), str(out), input_names=["pixel_values"],
                          output_names=[name], opset_version=17, dynamo=False,
                          dynamic_axes={"pixel_values": {0: "batch"}, name: {0: "batch"}})
        x = torch.randn(5, 3, size, size)
        diff = np.abs(w(x).detach().numpy() - ort.InferenceSession(str(out)).run(None, {"pixel_values": x.numpy()})[0]).max()
        print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB), output {name}, max |torch - onnx| = {diff:.2e}")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or tuple(EXPORTS))
