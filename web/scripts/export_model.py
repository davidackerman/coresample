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

Each float32 weight of more than a thousand values is stored as float16, with a Cast back to
float32 after it: the files are half the size (the page downloads them), while the graph still
computes in float32, as ONNX Runtime folds the Casts into float32 weights when it creates the
session (no slower to run). Each export is checked against PyTorch: the cosine of each 384-d part
(or token) and the largest difference. The weights come from pinned Hugging Face revisions, the
packages from web/scripts/export-requirements.txt (the Pages workflow caches the models on both).
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
FP16_MIN = 1024  # float32 weights with more values than this are stored as float16
# the Hugging Face revisions the weights come from (a change here re-exports in the Pages workflow)
DINOV2 = "ed25f3a31f01632728cabb09d1542f84ab7b0056"  # facebook/dinov2-small
MORPHEM = "0e8d58787421f83f975634d72420d85c5dfc9c2c"  # CaicedoLab/MorphEm
PECORE_T = "29e4ebb6cc0502b84ba432add45499f4ffa44238"  # timm/vit_pe_core_tiny_patch16_384.fb


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
    return AutoModel.from_pretrained("facebook/dinov2-small", revision=DINOV2).eval()


def morphem(size=112):
    """MorphEm's weights in timm's ViT-S/16 (one input channel; its DINO head left out), the
    position embeddings resampled to the size: as experiments/test6_models.py loads it."""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from timm.layers import resample_abs_pos_embed

    m = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0, in_chans=1, img_size=size)
    sd = {k: v for k, v in load_file(hf_hub_download("CaicedoLab/MorphEm", "model.safetensors", revision=MORPHEM)).items() if not k.startswith("head.")}
    sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], (size // 16, size // 16), num_prefix_tokens=1)
    m.load_state_dict(sd, strict=True)
    return m.eval()


def pecore_t(size=112):
    import timm
    return timm.create_model("vit_pe_core_tiny_patch16_384.fb", pretrained=True, num_classes=0, img_size=size,
                             pretrained_cfg_overlay={"hf_hub_id": f"timm/vit_pe_core_tiny_patch16_384.fb@{PECORE_T}"}).eval()


def fp16_weights(path, least=FP16_MIN):
    """Rewrite an ONNX file so each float32 initializer of more than `least` values is stored as
    float16, a Cast to float32 giving it its old name: half the bytes, the same float32 graph.
    The small ones (LayerNorm, biases, scalars) stay float32; they are little of the file."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(path))
    g = m.graph
    inits, casts = [], []
    for t in g.initializer:
        a = numpy_helper.to_array(t) if t.data_type == TensorProto.FLOAT and np.prod(t.dims) > least else None
        if a is None or np.abs(a).max() > np.finfo(np.float16).max:
            inits.append(t)
            continue
        inits.append(numpy_helper.from_array(a.astype(np.float16), f"{t.name}.fp16"))
        casts.append(helper.make_node("Cast", [f"{t.name}.fp16"], [t.name], name=f"{t.name}.cast", to=TensorProto.FLOAT))
    nodes = casts + list(g.node)  # the Casts first, so the nodes stay in topological order
    g.ClearField("initializer")
    g.initializer.extend(inits)
    g.ClearField("node")
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return len(casts)


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
        n16 = fp16_weights(out)
        x = torch.randn(5, ch, size, size)
        with torch.no_grad():
            ref = w(x).numpy()
        got = ort.InferenceSession(str(out)).run(None, {"pixel_values": x.numpy()})[0]
        a, b = ref.reshape(-1, WIDTH), got.reshape(-1, WIDTH)  # each 384-d part, or each token
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB, {n16} weights as float16), output {name} {ref.shape[1:]}, "
              f"torch vs onnx: min cosine {cos.min():.6f}, max |diff| {np.abs(ref - got).max():.2e}")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or tuple(EXPORTS))
