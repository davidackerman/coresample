"""Discover a dataset's arrays, pyramid levels, chunking and compressors from S3.

Walks the bucket prefix with delimiter listings (never lists chunk directories),
recognises zarr v2 (.zarray), zarr v3 (zarr.json), N5 (attributes.json with
"dimensions") and Neuroglancer precomputed (info), and writes
results/layout.json + results/layout.md.

    uv run python explore_layout.py [--prefix jrc_hela-2/] [--skip groundtruth]
"""

from __future__ import annotations

import argparse
import json
import urllib.error
from concurrent.futures import ThreadPoolExecutor

from common import RESULTS, get_json, list_prefix, list_sizes


def _try_json(key):
    try:
        return get_json(key)
    except (urllib.error.HTTPError, ValueError):
        return None


def describe(prefix: str):
    """Probe the metadata documents first (no listing). Returns (kind, metadata);
    kind None means a group whose children should be listed."""
    names = [".zarray", "zarr.json", "attributes.json", "info", ".zattrs"]
    docs = {n: _try_json(prefix + n) for n in names}
    if docs[".zarray"]:
        m = docs[".zarray"]
        return "zarr2-array", {"shape": m["shape"], "chunks": m["chunks"], "dtype": m["dtype"],
                               "compressor": m.get("compressor")}
    if docs["zarr.json"]:
        m = docs["zarr.json"]
        if m.get("node_type") == "array":
            return "zarr3-array", {"shape": m["shape"], "chunks": m["chunk_grid"]["configuration"]["chunk_shape"],
                                   "dtype": m["data_type"], "codecs": m.get("codecs")}
        return None, m.get("attributes", {})
    if docs["attributes.json"]:
        m = docs["attributes.json"]
        if "dimensions" in m:
            return "n5-array", {"shape_xyz": m["dimensions"], "chunks_xyz": m["blockSize"],
                                "dtype": m["dataType"], "compression": m.get("compression")}
        return None, m
    if docs["info"]:
        return "precomputed", {k: v for k, v in docs["info"].items() if k in ("@type", "type", "data_type", "scales")}
    return None, docs[".zattrs"] or {}


def walk(prefix: str, skip: set[str], max_depth: int = 8, pool=None):
    """Breadth-first: describe every node of one depth in parallel, then list groups."""
    root = {"path": prefix.rstrip("/"), "children": []}
    frontier = [(prefix, root)]
    for depth in range(max_depth + 1):
        if not frontier:
            break
        for (p, node), (kind, meta) in zip(frontier, pool.map(lambda f: describe(f[0]), frontier)):
            node["kind"], node["meta"] = kind or "group", meta
        groups = [(p, n) for p, n in frontier if n["kind"] == "group" and depth < max_depth]
        nxt = []
        for (p, node), (children, _) in zip(groups, pool.map(lambda g: list_prefix(g[0]), groups)):
            for c in children:
                if c.rstrip("/").split("/")[-1] in skip:
                    continue
                child = {"path": c.rstrip("/"), "children": []}
                node["children"].append(child)
                nxt.append((c, child))
        frontier = nxt
    return root


def flatten(node, out):
    if node["kind"] != "group":
        out.append(node)
    for c in node["children"]:
        flatten(c, out)
    return out


def multiscale_levels(node):
    """Levels named in OME-Zarr 'multiscales' or COSEM N5 'multiscales' attrs."""
    ms = node["meta"].get("multiscales") if isinstance(node["meta"], dict) else None
    if not ms:
        return None
    out = []
    for d in ms[0].get("datasets", []):
        if "coordinateTransformations" in d:
            t = {x["type"]: x.get(x["type"]) for x in d["coordinateTransformations"]}
            out.append((d["path"], t.get("scale"), t.get("translation")))
        elif "transform" in d:
            out.append((d["path"], d["transform"].get("scale"), d["transform"].get("translate")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="jrc_hela-2/")
    ap.add_argument("--skip", nargs="*", default=["groundtruth"], help="child names not to descend into")
    ap.add_argument("--count-chunks", nargs="*", default=["recon-1/em/fibsem-uint8"],
                    help="array groups whose levels get a full chunk listing (count + bytes)")
    args = ap.parse_args()

    with ThreadPoolExecutor(256) as pool:
        tree = walk(args.prefix, set(args.skip), pool=pool)
        # groups that hold multiscales
        groups = []

        def find_ms(n):
            lv = multiscale_levels(n)
            if lv:
                groups.append((n["path"], lv))
            for c in n["children"]:
                find_ms(c)

        find_ms(tree)
    arrays = flatten(tree, [])

    RESULTS.mkdir(exist_ok=True)
    lines = [f"# Layout of s3://janelia-cosem-datasets/{args.prefix}", ""]
    lines += ["## Multiscale groups", "", "| group | levels | s0 scale (z,y,x) | coarsest scale |", "|---|---|---|---|"]
    for path, lv in groups:
        lines.append(f"| `{path[len(args.prefix):]}` | {len(lv)} | {lv[0][1]} | {lv[-1][1]} |")
    lines += ["", "## Arrays", "", "| array | kind | shape (z,y,x) | chunks (z,y,x) | dtype | compressor |", "|---|---|---|---|---|---|"]
    for a in arrays:
        m = a["meta"]
        if a["kind"] == "n5-array":
            shape, chunks, comp = m["shape_xyz"][::-1], m["chunks_xyz"][::-1], m["compression"]
            comp = f"{comp.get('type')} {comp.get('level', '')}"
        elif a["kind"] in ("zarr2-array", "zarr3-array"):
            shape, chunks = m["shape"], m["chunks"]
            comp = m.get("compressor") or m.get("codecs")
            comp = f"{comp.get('id')} {comp.get('level', '')}" if isinstance(comp, dict) else str(comp)
        else:
            shape, chunks, comp = "-", "-", a["kind"]
        lines.append(f"| `{a['path'][len(args.prefix):]}` | {a['kind']} | {shape} | {chunks} | {m.get('dtype', '-')} | {comp} |")

    counts = {}
    for a in arrays:
        parent = a["path"].rsplit("/", 1)[0]
        if a["kind"] == "precomputed" or not any(parent.endswith(g) for g in args.count_chunks):
            continue
        sizes = list_sizes(a["path"] + "/")
        chunk_sizes = sorted(s for k, s in sizes.items() if k.rsplit("/", 1)[-1].isdigit())
        m = a["meta"]
        shape = m.get("shape") or m["shape_xyz"][::-1]
        chunks = m.get("chunks") or m["chunks_xyz"][::-1]
        grid = 1
        for s, c in zip(shape, chunks):
            grid *= -(-s // c)
        counts[a["path"]] = {
            "grid_chunks": grid,
            "stored_chunks": len(chunk_sizes),
            "stored_bytes": sum(chunk_sizes),
            "median_chunk_bytes": chunk_sizes[len(chunk_sizes) // 2] if chunk_sizes else 0,
        }
    if counts:
        lines += ["", "## Stored chunks (from listing only, no pixel reads)", "",
                  "| array | grid chunks | stored chunks | stored GB | median chunk KB |", "|---|---|---|---|---|"]
        for p, c in counts.items():
            lines.append(f"| `{p[len(args.prefix):]}` | {c['grid_chunks']} | {c['stored_chunks']} | "
                         f"{c['stored_bytes']/1e9:.2f} | {c['median_chunk_bytes']/1e3:.0f} |")

    (RESULTS / "layout.json").write_text(json.dumps({"tree": tree, "multiscales": groups, "chunk_counts": counts}, indent=1))
    (RESULTS / "layout.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
