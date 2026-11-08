"""Test 6: other matching models and descriptors, against DINOv2-small, on labelled EM.

Same candidates (the brute-force grid of the region) and the same queries (seed 2, the 30 of
test 5's large run) for every matcher, so only the matcher varies. Top-10/25 precision and
recall@100 with standard errors over queries.

Matchers:
  dinov2-small   the current model (112 px, CLS + mean token)
  dinov2-base    4x the parameters (112 px, CLS + mean token)
  dinov3-small   DINOv3 ViT-S/16 (gated on Hugging Face: needs an accepted license and a token);
                 128 px so it sees the same 8 x 8 token grid, CLS + mean of patch tokens (its 4
                 register tokens left out)
  siglip-base    image-text model (224 px, pooled image embedding)
  clip-b16       image-text model (224 px, pooled image embedding)
  cem500k        ResNet-50 pretrained (MoCo v2) on 500k EM images (224 px grey, one input channel,
                 the checkpoint's own normalization, global average pool);
                 its checkpoint is loaded with torch's weights_only loader (tensors only, no
                 pickled code) and skipped if that loader refuses it
  texture        classical: rotation-invariant LBP histograms + Gabor energies (no learning)

    uv run python experiments/test6_models.py [--targets vesicle mito_full] [--models ...] [--queries 30] [--tag _x]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

import test2_coarse2fine as t2
from common import CACHE, RESULTS, Mirror

ROOT = Path(__file__).resolve().parents[1]  # the repository (its .hf holds the model downloads)
os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))
import torch  # noqa: E402

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
MODELS = ["dinov2-small", "dinov2-base", "siglip-base", "clip-b16", "cem500k", "texture"]  # + dinov3-small (gated)
CEM500K_URL = "https://zenodo.org/records/6453140/files/cem500k_mocov2_resnet50_200ep.pth.tar?download=1"


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def to_tensor(patches, size, mean, std, channels=3):
    x = torch.from_numpy(patches.astype(np.float32) / 255.0)[:, None]
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, channels, 1, 1)
    return (x - torch.from_numpy(mean)) / torch.from_numpy(std)


class Matcher:
    def __init__(self, name):
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", os.cpu_count())))
        self.name, self.channels = name, 3
        if name.startswith("dinov2"):
            from transformers import AutoModel

            self.m = AutoModel.from_pretrained(f"facebook/{name}").eval()
            self.size, self.mean, self.std = 112, IMNET_MEAN, IMNET_STD
        elif name == "dinov3-small":
            from transformers import AutoModel

            self.m = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").eval()
            self.size, self.mean, self.std = 128, IMNET_MEAN, IMNET_STD
            self.skip = 1 + int(self.m.config.num_register_tokens)  # CLS, then the register tokens
        elif name == "siglip-base":
            from transformers import SiglipVisionModel

            self.m = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").eval()
            self.size = 224
            self.mean = self.std = np.array([0.5, 0.5, 0.5], np.float32)[:, None, None]
        elif name == "clip-b16":
            from transformers import CLIPVisionModelWithProjection

            self.m = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").eval()
            self.size = 224
            self.mean = np.array([0.4815, 0.4578, 0.4082], np.float32)[:, None, None]
            self.std = np.array([0.2686, 0.2613, 0.2758], np.float32)[:, None, None]
        elif name == "cem500k":
            import torchvision

            f = CACHE / "cem500k_mocov2_resnet50_200ep.pth.tar"
            if not f.exists():
                urllib.request.urlretrieve(CEM500K_URL, f)
            # tensors and plain containers only: a downloaded checkpoint must not run code
            ck = torch.load(f, map_location="cpu", weights_only=True)
            sd = ck.get("state_dict", ck)
            # the released checkpoint holds the query encoder's weights unprefixed; a raw MoCo
            # training checkpoint would prefix them with module.encoder_q.
            pre = "module.encoder_q."
            if any(k.startswith(pre) for k in sd):
                sd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
            net = torchvision.models.resnet50()
            # trained on grey images: its first layer takes one channel
            net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            missing, _ = net.load_state_dict(sd, strict=False)
            if any(not k.startswith("fc.") for k in missing):
                raise RuntimeError(f"cem500k: unexpected missing weights {missing[:5]}")
            net.fc = torch.nn.Identity()
            self.m = net.eval()
            self.size, self.channels = 224, 1
            # the grey mean and standard deviation it was trained with (stored in the checkpoint)
            mu, sigma = ck.get("norms", [0.57287007, 0.12740536])
            self.mean = np.array([mu], np.float32)[:, None, None]
            self.std = np.array([sigma], np.float32)[:, None, None]
        elif name == "texture":
            self.m = None
        else:
            raise ValueError(name)

    def __call__(self, patches: np.ndarray, batch=64) -> np.ndarray:
        if self.name == "texture":
            return texture_descriptor(patches)
        out = []
        with torch.inference_mode():
            for i in range(0, len(patches), batch):
                x = to_tensor(patches[i:i + batch], self.size, self.mean, self.std, self.channels)
                if self.name.startswith("dinov"):
                    h = self.m(pixel_values=x).last_hidden_state
                    skip = getattr(self, "skip", 1)
                    v = torch.cat([torch.nn.functional.normalize(h[:, 0], dim=1),
                                   torch.nn.functional.normalize(h[:, skip:].mean(1), dim=1)], 1)
                elif self.name == "siglip-base":
                    v = self.m(pixel_values=x).pooler_output
                elif self.name == "clip-b16":
                    v = self.m(pixel_values=x).image_embeds
                else:
                    v = self.m(x)
                out.append(v.numpy().astype(np.float32))
        return unit(np.concatenate(out))


def texture_descriptor(patches: np.ndarray) -> np.ndarray:
    """Rotation-invariant LBP histograms (radius 1 and 3) and Gabor energies averaged over
    orientations (3 frequencies): a classical, learning-free texture signature per patch."""
    from scipy import ndimage as ndi
    from skimage.feature import local_binary_pattern
    from skimage.filters import gabor_kernel

    kernels = [[np.real(gabor_kernel(freq, theta=th)) for th in np.arange(4) * np.pi / 4] for freq in (0.08, 0.16, 0.32)]
    feats = []
    for p in patches:
        f = []
        for P, R in ((8, 1), (16, 3)):
            lbp = local_binary_pattern(p, P, R, method="uniform")
            h, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2), density=True)
            f.extend(h)
        x = (p.astype(np.float32) - p.mean()) / (p.std() + 1e-6)
        for ks in kernels:
            e = [float(np.mean(ndi.convolve(x, k, mode="reflect") ** 2)) for k in ks]
            f.extend([np.mean(e), np.std(e)])
        f.extend([p.mean() / 255.0, p.std() / 255.0])
        feats.append(f)
    return np.asarray(feats, np.float32)


def queries_for(data, cfg, gt, rlo, rhi, n, seed):
    """The same query selection as test 5 (instances big enough, an object voxel near the
    centroid, well inside the region), so its seed-2 queries are reused exactly."""
    W, P = cfg["working"], cfg["patch"]
    cents = gt["centroids"]
    rng = np.random.default_rng(seed)
    ok = gt["big"] & np.all((cents >= rlo + P) & (cents < rhi - P), axis=1)
    out = []
    for j in rng.permutation(np.nonzero(ok)[0]):
        pt = t2.inside_point(data.mirror, cfg["label"], W, int(gt["ids"][j]), cents[j], P // 2)
        if pt is not None and np.all((pt >= rlo + P) & (pt < rhi - P)):
            out.append(np.asarray(pt, int))
        if len(out) >= n:
            break
    return out


def run(target, args, data, log):
    name, full = ("mito", True) if target == "mito_full" else (target, False)
    cfg = t2.TARGETS[name]
    W, P = cfg["working"], cfg["patch"]
    data.working = W
    t2.REGION_S0 = (np.zeros(3, int), np.array([6368, 1600, 12000])) if full else (np.array([640, 0, 512]), np.array([2688, 1600, 4608]))
    rlo, rhi = t2.REGION_S0[0] // 2 ** W, t2.REGION_S0[1] // 2 ** W
    gt = t2.instances(data.mirror, cfg["label"], W, rlo, rhi, cfg["min_voxels"])
    cents = gt["centroids"]
    queries = queries_for(data, cfg, gt, rlo, rhi, args.queries, args.seed)
    gc = t2.grid_centers(rlo, rhi, P)
    log(f"[{target}] {len(cents)} instances, {len(gc)} candidates, {len(queries)} queries")
    h = P // 2
    qpatches = np.stack([data.read(W, q - [0, h, h], q + [1, h, h])[0] for q in queries])
    res = {"target": target, "n_instances": int(len(cents)), "n_candidates": int(len(gc)),
           "queries": [q.tolist() for q in queries], "rows": []}
    for mname in args.models:
        t = time.time()
        try:
            m = Matcher(mname)
        except Exception as e:  # a model that cannot be loaded safely is reported, not forced
            log(f"[{target}] {mname}: skipped ({e})")
            continue
        f = CACHE / f"test6_{target}_{mname}.npy"
        if f.exists():
            E = np.load(f)
        else:
            vol = data.read(W, rlo, rhi)
            E = np.concatenate([m(t2.patches_from(vol, rlo, gc[i:i + 2048], P)) for i in range(0, len(gc), 2048)])
            np.save(f, E.astype(np.float16))
            del vol
        E = E.astype(np.float32)
        Q = m(qpatches)
        if mname == "texture":  # standardize the hand-made features over the candidates, then compare
            mu, sd = E.mean(0), E.std(0) + 1e-6
            E, Q = unit((E - mu) / sd), unit((Q - mu) / sd)
        gl = []
        for qc, q in zip(queries, Q):
            notself = ~np.all(np.abs(gc - qc) < P / 2, 1)
            top = t2.rank_nms(gc[notself], (E @ q)[notself], P)
            e = t2.evaluate(top, cents, P)
            gl.append({k: e[k] for k in ("p@10", "p@25", "r@100")})
        row = {"model": mname, "seconds": time.time() - t}
        for k in ("p@10", "p@25", "r@100"):
            v = np.array([g[k] for g in gl])
            row[k], row[k + "_se"], row[k + "_per_query"] = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), v.tolist()
        res["rows"].append(row)
        log(f"[{target}] {mname:<13} p@10 {row['p@10']:.2f}+/-{row['p@10_se']:.2f}  p@25 {row['p@25']:.2f}+/-{row['p@25_se']:.2f}"
            f"  r@100 {row['r@100']:.3f}  ({row['seconds']:.0f}s)")
        (RESULTS / f"test6_{target}{args.tag}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["vesicle", "mito_full"])
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tag", default="", help="suffix for the result and log files (a separate run)")
    args = ap.parse_args()
    logf = open(RESULTS / f"test6{args.tag}.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    data = t2.Data(Mirror())
    for t in args.targets:
        run(t, args, data, log)


if __name__ == "__main__":
    main()
