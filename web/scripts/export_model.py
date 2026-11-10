"""Export the page's models to ONNX for the browser: DINOv2-small (112 px pooled, 224 px and
112 px tokens), MorphEm and PE-Core-T (112 px tokens).

Input (batch, C, S, S), normalized as each model was trained (the page does it: embedder.ts
`prepare`): DINOv2 ImageNet mean and std; MorphEm one grey channel, each patch to zero mean and
unit std; PE-Core-T 0.5 mean and std.

  dinov2-small-112: output (batch, 1152) = [CLS | mean of patch tokens | centre-weighted mean
          of patch tokens (Gaussian, sigma = a quarter of the token grid)], each 384-d part
          L2-normalized, so the page can match on any combination (experiments/test5_matching.py).
  dinov2-small-224: output (batch, 257, 384) = the last hidden state (CLS, then the 16 x 16 patch
          tokens), raw. The page pools the same 1152-d embedding from it for "detailed" matching,
          and matches windows of tokens for the tile-token (Mars CTX explorer) mode (test 7).
  dinov2-small-112-tokens: output (batch, 65, 384), the last hidden state, raw: for a brush
          (which parts of the box matter), whose weighted mean of tokens the page pools; only
          fetched then, so the usual search keeps the small pooled output (test 12).
  morphem-112-tokens: MorphEm (CaicedoLab/MorphEm, MIT; a ViT-S/16 trained with DINO on single
          fluorescence channels), output (batch, 50, 384): CLS, then the 7 x 7 patch tokens, raw.
          The page pools the same parts from it as from DINOv2's (test 6: +0.06 p@10 on HeLa EM).
  pecore-t16-112-tokens: PE-Core-T16 (Meta's Perception Encoder, Apache-2.0, timm), output
          (batch, 50, 384): its 192-d tokens with 192 zeros after them, so every part is 384-d
          as the page expects; zeros change no cosine (test 6: 4.6x faster on the CPU, -0.06 p@10).

Writes web/public/models/<name>.onnx and checks each against PyTorch.
    python web/scripts/export_model.py [name ...]"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))  # before transformers reads it at import

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = ROOT / "web/public/models"
WIDTH = 384  # every part of the page's embeddings


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
    """The last hidden state (prefix token, then the patch tokens), padded to WIDTH."""

    def __init__(self, m, hf=True):
        super().__init__()
        self.m, self.hf = m, hf

    def forward(self, x):
        h = self.m(pixel_values=x).last_hidden_state if self.hf else self.m.forward_features(x)
        return torch.nn.functional.pad(h, (0, WIDTH - h.shape[-1])) if h.shape[-1] < WIDTH else h


def dinov2():
    from transformers import AutoModel
    return AutoModel.from_pretrained("facebook/dinov2-small").eval()


def morphem(size=112):
    """MorphEm's weights in timm's ViT-S/16 (one input channel; its DINO head left out), the
    position embeddings resampled to the size: as experiments/test6_models.py loads it."""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from timm.layers import resample_abs_pos_embed

    m = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0, in_chans=1, img_size=size)
    sd = {k: v for k, v in load_file(hf_hub_download("CaicedoLab/MorphEm", "model.safetensors")).items() if not k.startswith("head.")}
    sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], (size // 16, size // 16), num_prefix_tokens=1)
    m.load_state_dict(sd, strict=True)
    return m.eval()


def pecore_t(size=112):
    import timm
    return timm.create_model("vit_pe_core_tiny_patch16_384.fb", pretrained=True, num_classes=0, img_size=size).eval()


# file name: (loader, input size, input channels, wrapper)
EXPORTS = {
    "dinov2-small-112": (dinov2, 112, 3, lambda m: Pooled(m, 112)),
    "dinov2-small-224": (dinov2, 224, 3, Tokens),
    "dinov2-small-112-tokens": (dinov2, 112, 3, Tokens),
    "morphem-112-tokens": (morphem, 112, 1, lambda m: Tokens(m, hf=False)),
    "pecore-t16-112-tokens": (pecore_t, 112, 3, lambda m: Tokens(m, hf=False)),
}


def main(names=tuple(EXPORTS)):
    import onnxruntime as ort

    OUT.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for key in names:
        load, size, ch, wrap = EXPORTS[key]
        if load not in loaded:
            loaded[load] = load()
        m = loaded[load]
        m.requires_grad_(False)
        w = wrap(m).eval()
        out = OUT / f"{key}.onnx"
        name = "embedding" if isinstance(w, Pooled) else "tokens"
        torch.onnx.export(w, (torch.randn(2, ch, size, size),), str(out), input_names=["pixel_values"],
                          output_names=[name], opset_version=17, dynamo=False,
                          dynamic_axes={"pixel_values": {0: "batch"}, name: {0: "batch"}})
        x = torch.randn(5, ch, size, size)
        with torch.no_grad():
            ref = w(x).numpy()
        diff = np.abs(ref - ort.InferenceSession(str(out)).run(None, {"pixel_values": x.numpy()})[0]).max()
        print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB), output {name} {ref.shape[1:]}, max |torch - onnx| = {diff:.2e}")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or tuple(EXPORTS))
