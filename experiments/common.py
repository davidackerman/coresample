"""Shared data access for the coresample feasibility tests.

Everything here is about *getting bytes*: S3 listings (chunk sizes without reading
pixels), a local chunk mirror (so repeated experiments do not re-download), and
chunk-touch accounting (so "fraction of data read" is computed from the chunks a
method needed, independent of caching).

Scoring and descent logic lives in search.py and does not import this module.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorstore as ts

BUCKET = "janelia-cosem-datasets"
BASE_URL = f"https://{BUCKET}.s3.amazonaws.com"
HERE = Path(__file__).resolve().parent  # experiments/
CACHE = HERE / "cache"
RESULTS = HERE / "results"

DATASET = "jrc_hela-2"
EM_ZARR = f"{DATASET}/{DATASET}.zarr/recon-1/em/fibsem-uint8"
EM_N5 = f"{DATASET}/{DATASET}.n5/em/fibsem-uint16"
LABELS_N5 = f"{DATASET}/{DATASET}.n5/labels"

_ns = re.compile(r"<CommonPrefixes><Prefix>(.*?)</Prefix>")
_keys = re.compile(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", re.S)
_token = re.compile(r"<NextContinuationToken>(.*?)</NextContinuationToken>")


def _get(url: str, timeout: float = 60.0, tries: int = 5) -> bytes:
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except Exception as e:  # network hiccup: retry
            last = e
    raise RuntimeError(f"GET {url} failed: {last}")


def list_prefix(prefix: str, delimiter: str | None = "/"):
    """S3 ListObjectsV2 (paginated). Returns (child prefixes, {key: size})."""
    prefixes, keys, token = [], {}, None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        t = _get(f"{BASE_URL}/?{urllib.parse.urlencode(q)}").decode()
        prefixes += _ns.findall(t)
        for k, s in _keys.findall(t):
            keys[k] = int(s)
        m = _token.search(t)
        if not m:
            return prefixes, keys
        token = m.group(1).replace("&amp;", "&")


def list_sizes(prefix: str, workers: int = 16) -> dict[str, int]:
    """All object sizes under prefix. Splits by first-level children to parallelize."""
    children, top = list_prefix(prefix)
    out = dict(top)
    with ThreadPoolExecutor(workers) as ex:
        for _, keys in ex.map(lambda p: list_prefix(p, delimiter=None), children):
            out.update(keys)
    return out


def get_json(key: str):
    return json.loads(_get(f"{BASE_URL}/{key}"))


def get_bytes(key: str) -> bytes:
    return _get(f"{BASE_URL}/{key}")


# --------------------------------------------------------------------------------------
# Array geometry
# --------------------------------------------------------------------------------------


@dataclass
class ArrayInfo:
    """One stored array (one pyramid level), always described in z, y, x order."""

    path: str
    fmt: str  # "zarr2" or "n5"
    shape: tuple[int, int, int]
    chunks: tuple[int, int, int]
    dtype: str
    compressor: str

    @property
    def grid(self) -> tuple[int, int, int]:
        return tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))

    def chunk_key(self, idx) -> str:
        z, y, x = (int(i) for i in idx)
        rel = f"{z}/{y}/{x}" if self.fmt == "zarr2" else f"{x}/{y}/{z}"
        return f"{self.path}/{rel}"

    def key_to_index(self, key: str):
        a, b, c = (int(p) for p in key[len(self.path) + 1 :].split("/"))
        return (a, b, c) if self.fmt == "zarr2" else (c, b, a)


def array_info(path: str) -> ArrayInfo:
    """Read .zarray (zarr v2) or attributes.json (N5) for one level; z, y, x order."""
    try:
        m = get_json(f"{path}/.zarray")
        comp = m.get("compressor") or {}
        return ArrayInfo(
            path,
            "zarr2",
            tuple(m["shape"]),
            tuple(m["chunks"]),
            m["dtype"],
            f"{comp.get('id')}-{comp.get('level', comp.get('clevel', ''))}",
        )
    except urllib.error.HTTPError:
        m = get_json(f"{path}/attributes.json")
        comp = m.get("compression") or {}
        return ArrayInfo(
            path,
            "n5",
            tuple(m["dimensions"][::-1]),
            tuple(m["blockSize"][::-1]),
            m["dataType"],
            f"{comp.get('type')}-{comp.get('level', '')}",
        )


def chunk_size_grid(info: ArrayInfo, sizes: dict[str, int] | None = None) -> np.ndarray:
    """Compressed byte size of every chunk on the grid (0 = not stored = fill value)."""
    if sizes is None:
        sizes = list_sizes(info.path + "/")
    g = np.zeros(info.grid, dtype=np.int64)
    for k, s in sizes.items():
        tail = k[len(info.path) + 1 :]
        if tail.count("/") != 2 or not tail.replace("/", "").isdigit():
            continue
        g[info.key_to_index(k)] = s
    return g


# --------------------------------------------------------------------------------------
# Local chunk mirror + touch accounting
# --------------------------------------------------------------------------------------


class Mirror:
    """Copies remote chunk files into cache/mirror/<same path> on demand.

    Reads go through tensorstore's file driver on the mirror, so they decode exactly
    like a remote read would. Every chunk a read needs is recorded in `touched`
    (per array path), which is what the experiments use for data-read accounting.
    """

    def __init__(self, root: Path | None = None, workers: int = 32):
        # Many small files: keep the mirror on local disk if possible (CORESAMPLE_MIRROR).
        self.root = Path(root or os.environ.get("CORESAMPLE_MIRROR") or CACHE / "mirror")
        self.workers = workers
        self.touched: dict[str, set] = {}
        self.boxes: list[tuple[str, np.ndarray, np.ndarray]] = []  # every read box, for cost models
        self._infos: dict[str, ArrayInfo] = {}
        self._stores: dict[str, ts.TensorStore] = {}
        self._lock = threading.Lock()
        self._ctx = ts.Context({"cache_pool": {"total_bytes_limit": 4 << 30}})

    def info(self, path: str) -> ArrayInfo:
        if path not in self._infos:
            meta_local = self.root / path
            meta_local.mkdir(parents=True, exist_ok=True)
            info = array_info(path)
            name = ".zarray" if info.fmt == "zarr2" else "attributes.json"
            (meta_local / name).write_bytes(get_bytes(f"{path}/{name}"))
            self._infos[path] = info
        return self._infos[path]

    def store(self, path: str) -> ts.TensorStore:
        if path not in self._stores:
            info = self.info(path)
            spec = {
                "driver": "zarr" if info.fmt == "zarr2" else "n5",
                "kvstore": {"driver": "file", "path": str(self.root / path) + "/"},
                "recheck_cached_data": False,
            }
            self._stores[path] = ts.open(spec, read=True, context=self._ctx).result()
        return self._stores[path]

    def _fetch(self, info: ArrayInfo, idx) -> None:
        key = info.chunk_key(idx)
        local = self.root / key
        miss = local.with_suffix(".missing")
        if local.exists() or miss.exists():
            return
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = get_bytes(key)
        except urllib.error.HTTPError:
            miss.touch()  # not stored remotely = fill value
            return
        tmp = local.with_name(local.name + f".tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_bytes(data)
        tmp.replace(local)

    def chunks_for(self, path: str, lo, hi) -> list[tuple[int, int, int]]:
        info = self.info(path)
        lo = np.maximum(np.asarray(lo, int), 0)
        hi = np.minimum(np.asarray(hi, int), info.shape)
        if np.any(hi <= lo):
            return []
        c = np.asarray(info.chunks)
        a, b = lo // c, (hi - 1) // c
        return [
            (z, y, x)
            for z in range(a[0], b[0] + 1)
            for y in range(a[1], b[1] + 1)
            for x in range(a[2], b[2] + 1)
        ]

    def ensure(self, path: str, chunk_list) -> None:
        info = self.info(path)
        with self._lock:
            self.touched.setdefault(path, set()).update(chunk_list)
        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(lambda i: self._fetch(info, i), chunk_list))

    def read(self, path: str, lo, hi, pad_value=0) -> np.ndarray:
        """Read voxel box [lo, hi) (z, y, x); parts outside the array are pad_value."""
        info = self.info(path)
        lo = np.asarray(lo, int)
        hi = np.asarray(hi, int)
        with self._lock:
            self.boxes.append((path, lo.copy(), hi.copy()))
        self.ensure(path, self.chunks_for(path, lo, hi))
        out = np.full(tuple(hi - lo), pad_value, dtype=np.dtype(self.store(path).dtype.numpy_dtype))
        clo = np.maximum(lo, 0)
        chi = np.minimum(hi, info.shape)
        if np.all(chi > clo):
            arr = self.store(path)
            if info.fmt == "n5":  # tensorstore n5 dims are x, y, z
                sub = arr[clo[2] : chi[2], clo[1] : chi[1], clo[0] : chi[0]].read().result().transpose(2, 1, 0)
            else:
                sub = arr[clo[0] : chi[0], clo[1] : chi[1], clo[2] : chi[2]].read().result()
            out[tuple(slice(a - l, b - l) for a, b, l in zip(clo, chi, lo))] = sub
        return out

    def reset_touched(self) -> None:
        with self._lock:
            self.touched = {}
            self.boxes = []


def level_path(base: str, level: int) -> str:
    return f"{base}/s{level}"
