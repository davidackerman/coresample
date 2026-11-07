"""DINOv2-small patch embeddings with a disk cache (the self-building index).

Every embedding is stored in cache/embeddings.sqlite keyed by
(dataset/array, level, patch centre, patch size, view, model, input size), so any
later search that asks for the same patch gets it without reading or embedding.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from common import CACHE

MODEL = "facebook/dinov2-small"
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


class Embedder:
    def __init__(self, input_size: int = 112, threads: int | None = None, cache_path: Path | None = None,
                 batch: int = 64):
        import torch
        from transformers import AutoModel

        self.torch = torch
        torch.set_num_threads(threads or os.cpu_count())
        self.model = AutoModel.from_pretrained(MODEL).eval()
        self.input_size = input_size
        self.batch = batch
        self.tag = f"{MODEL}@{input_size}"
        self.db = sqlite3.connect(str(cache_path or CACHE / "embeddings.sqlite"))
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        self.stats = {"computed": 0, "cached": 0, "seconds": 0.0}

    def _prep(self, patches: np.ndarray):
        torch = self.torch
        x = torch.from_numpy(patches.astype(np.float32) / 255.0)[:, None]  # (n,1,P,P)
        x = torch.nn.functional.interpolate(x, size=(self.input_size,) * 2, mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        return (x - torch.from_numpy(_MEAN)) / torch.from_numpy(_STD)

    def embed_patches(self, patches: np.ndarray) -> np.ndarray:
        """(n, P, P) uint8 -> (n, 768): [CLS, mean of patch tokens], each L2-normalized."""
        torch = self.torch
        out = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for i in range(0, len(patches), self.batch):
                h = self.model(pixel_values=self._prep(patches[i : i + self.batch])).last_hidden_state
                cls = torch.nn.functional.normalize(h[:, 0], dim=1)
                avg = torch.nn.functional.normalize(h[:, 1:].mean(1), dim=1)
                out.append(torch.cat([cls, avg], 1).numpy())
        self.stats["seconds"] += time.perf_counter() - t0
        self.stats["computed"] += len(patches)
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def key(self, array: str, level: int, center, patch: int, view: str = "xy") -> str:
        z, y, x = (int(c) for c in center)
        return f"{array}|s{level}|{z},{y},{x}|P{patch}|{view}|{self.tag}"

    def embed_at(self, array: str, level: int, centers: np.ndarray, patch: int, get_patches) -> np.ndarray:
        """Embeddings for xy patches centred at `centers` (working-level voxels).
        get_patches(centers_subset) -> (m, P, P) is only called for cache misses."""
        keys = [self.key(array, level, c, patch) for c in centers]
        found = {}
        for i in range(0, len(keys), 900):
            q = keys[i : i + 900]
            for k, v in self.db.execute(f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(q))})", q):
                found[k] = np.frombuffer(v, np.float32)
        miss = [i for i, k in enumerate(keys) if k not in found]
        self.stats["cached"] += len(keys) - len(miss)
        if miss:
            embs = self.embed_patches(get_patches(centers[miss]))
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                                [(keys[i], e.astype(np.float32).tobytes()) for i, e in zip(miss, embs)])
            self.db.commit()
            for i, e in zip(miss, embs):
                found[keys[i]] = e.astype(np.float32)
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, 768), np.float32)
