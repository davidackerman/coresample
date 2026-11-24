"""Export the page's models to ONNX for the browser: DINOv2-small (112 px pooled, 224 px and
112 px tokens), MorphEm and PE-Core-T (112 px tokens), PE-Core-S (image and text embeddings).

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
  pecore-s16-112-clip: PE-Core-S16 (open_clip's PE-Core-S-16-384, Apache-2.0), output (batch, 512):
          the image embedding its text tower was trained against (attention-pooled, projected),
          unit length. One part, matched whole: what a typed query is compared with.
  pecore-s16-text: its text tower. Input (batch, 32): the CLIP tokens of a text (the page's
          cliptok.ts; start, text, end, zeros) as float32, rounded to integers in the graph (the
          page's ONNX wrapper feeds float32); output (batch, 512), unit length. 127 MB, so written
          as pecore-s16-text.part0.onnx and .part1.onnx (PART_BYTES each at most), which the page
          joins (embedder.ts: the model's textParts).

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
PART_BYTES = 64_000_000  # a file larger than this is written in parts this long: the site is a git branch, and GitHub takes no file over 100 MB
# the Hugging Face revisions the weights come from (a change here re-exports in the Pages workflow)
DINOV2 = "ed25f3a31f01632728cabb09d1542f84ab7b0056"  # facebook/dinov2-small
MORPHEM = "0e8d58787421f83f975634d72420d85c5dfc9c2c"  # CaicedoLab/MorphEm
PECORE_T = "29e4ebb6cc0502b84ba432add45499f4ffa44238"  # timm/vit_pe_core_tiny_patch16_384.fb
PECORE_S_CLIP = "3249a38eb1c432ec19231c6fe1774acb6a4e4efe"  # timm/PE-Core-S-16-384 (open_clip: image and text towers)
CONTEXT = 32  # PE-Core's text length, in tokens


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


def pecore_s_clip(size=112):
    """PE-Core-S16 with its text tower, through open_clip (timm has the image tower alone), the
    image tower's position embeddings resized to `size`."""
    import open_clip
    from huggingface_hub import hf_hub_download
    weights = hf_hub_download("timm/PE-Core-S-16-384", "open_clip_model.safetensors", revision=PECORE_S_CLIP)
    return open_clip.create_model("PE-Core-S-16-384", pretrained=weights, force_image_size=size).eval()


class ClipImage(torch.nn.Module):
    """The image embedding the text tower is matched against, unit length."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.nn.functional.normalize(self.m.encode_image(x), dim=-1)


class ClipText(torch.nn.Module):
    """The text embedding of CLIP tokens given as float32 (rounded here), unit length."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, tokens):
        return torch.nn.functional.normalize(self.m.encode_text(tokens.round().long()), dim=-1)


def text_tokens(n):
    """`n` rows of plausible tokens: start, a few words' ids, end, zeros."""
    rng = np.random.default_rng(n)
    t = np.zeros((n, CONTEXT), np.float32)
    for row in t:
        k = int(rng.integers(1, CONTEXT - 2))
        row[0], row[1:1 + k], row[1 + k] = 49406, rng.integers(0, 49406, k), 49407
    return torch.from_numpy(t)


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


def split(path, most=PART_BYTES):
    """A file larger than `most` rewritten as <stem>.part0.onnx, .part1.onnx... (the whole file
    removed, so it is not published); the number of files it is now."""
    data = path.read_bytes()
    for old in path.parent.glob(f"{path.stem}.part*.onnx"):
        old.unlink()
    if len(data) <= most:
        return 1
    n = -(-len(data) // most)
    for i in range(n):
        (path.parent / f"{path.stem}.part{i}.onnx").write_bytes(data[i * most:(i + 1) * most])
    path.unlink()
    return n


# file name: (loader, input size, input channels, wrapper)
EXPORTS = {
    "dinov2-small-112": (dinov2, 112, 3, lambda m: Pooled(m, 112)),
    "dinov2-small-224": (dinov2, 224, 3, Tokens),
    "dinov2-small-112-tokens": (dinov2, 112, 3, Tokens),
    "morphem-112-tokens": (morphem, 112, 1, lambda m: Tokens(m, hf=False)),
    "pecore-t16-112-tokens": (pecore_t, 112, 3, lambda m: Tokens(m, hf=False)),
    "pecore-s16-112-clip": (pecore_s_clip, 112, 3, ClipImage),
    "pecore-s16-text": (pecore_s_clip, 0, 0, ClipText),  # no image: its input is tokens
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
        text = isinstance(w, ClipText)
        name = "embedding" if isinstance(w, (Pooled, ClipImage, ClipText)) else "tokens"
        inp = "tokens" if text else "pixel_values"
        sample = (lambda n: text_tokens(n)) if text else (lambda n: torch.randn(n, ch, size, size))
        torch.onnx.export(w, (sample(2),), str(out), input_names=[inp],
                          output_names=[name], opset_version=17, dynamo=False,
                          dynamic_axes={inp: {0: "batch"}, name: {0: "batch"}})
        n16 = fp16_weights(out)
        x = sample(5)
        with torch.no_grad():
            ref = w(x).numpy()
        got = ort.InferenceSession(str(out)).run(None, {inp: x.numpy()})[0]
        width = ref.shape[-1] if isinstance(w, (ClipImage, ClipText)) else WIDTH
        a, b = ref.reshape(-1, width), got.reshape(-1, width)  # each 384-d part or token, or the one embedding
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        size_mb, parts = out.stat().st_size / 1e6, split(out)
        print(f"wrote {out} ({size_mb:.0f} MB{f' in {parts} parts' if parts > 1 else ''}, {n16} weights as float16), output {name} {ref.shape[1:]}, "
              f"torch vs onnx: min cosine {cos.min():.6f}, max |diff| {np.abs(ref - got).max():.2e}")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or tuple(EXPORTS))
