#!/usr/bin/env bash
# Changes to the Neuroglancer source the site builds (run inside its checkout, before the build).
# The cache key of the build includes this file, so editing it rebuilds the viewer.
set -euo pipefail

# One-byte zarr types with a byte order. Neuroglancer accepts "|u1" and "|i1" only, but
# writers such as omero-zarr (IDR's OME-Zarr) store "<u1" or ">u1": byte order means nothing
# for one byte, so read them as "|u1" and "|i1".
f=src/util/numpy_dtype.ts
grep -q 'supportedDataTypes.set("|u1"' "$f"
cat >> "$f" <<'TS'

// coresample: one-byte types written with a byte order ("<u1", ">u1", as omero-zarr does)
for (const order of ["<", ">"]) {
  supportedDataTypes.set(`${order}u1`, supportedDataTypes.get("|u1")!);
  supportedDataTypes.set(`${order}i1`, supportedDataTypes.get("|i1")!);
}
TS
