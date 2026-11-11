# Makes the small TIFFs test/remotetiff.test.ts reads (values by formulas the test recomputes).
# Run from the repository root: uv run python web/test/fixtures/make_fixtures.py web/test/fixtures
import numpy as np, tifffile, sys
out = sys.argv[1]
def grid(h, w): return np.mgrid[0:h, 0:w]
y, x = grid(500, 700)
tifffile.imwrite(f"{out}/strips_u8.tif", ((x * x + 3 * y) % 251).astype(np.uint8), rowsperstrip=1, compression=None)
with tifffile.TiffWriter(f"{out}/cog_u16.tif") as t:
    for k, (h, w) in enumerate([(480, 640), (240, 320), (120, 160)]):
        yy, xx = grid(h, w)
        t.write(((xx * 37 + yy * 11 + 1000 * k) % 4000).astype(np.uint16), tile=(128, 128), compression="deflate", predictor=True,
                subfiletype=1 if k else 0)
y, x = grid(200, 300)
rgb = np.stack([(x + y) % 256, (2 * x) % 256, (3 * y + x // 7) % 256], -1).astype(np.uint8)
tifffile.imwrite(f"{out}/deflate_rgb.tif", rgb, photometric="rgb", rowsperstrip=8, compression="deflate")
y, x = grid(300, 260)
pl = np.stack([(x * 5 - y * 3), (x * y) % 3001 - 1500]).astype(np.int16)
tifffile.imwrite(f"{out}/planar_be.tif", pl, byteorder=">", planarconfig="separate", photometric="minisblack", rowsperstrip=16, compression=None)
