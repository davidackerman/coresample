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
# JPEG (YCbCr, chroma halved both ways: GDAL's default for RGB JPEG), and big-endian with a
# predictor (horizontal on integers, floating-point on floats); these need imagecodecs:
#   uv run --with imagecodecs python web/test/fixtures/make_fixtures.py web/test/fixtures
y, x = grid(96, 160)
r, g, b = x + 40.0, 2 * y + 20.0, (x + y) // 2 + 60.0
# tifffile stores the samples given as they are: Y, Cb, Cr made here (JFIF), as libtiff does from RGB
ycc = np.stack([0.299 * r + 0.587 * g + 0.114 * b, 128 - 0.168736 * r - 0.331264 * g + 0.5 * b,
                128 + 0.5 * r - 0.418688 * g - 0.081312 * b], -1).round().clip(0, 255).astype(np.uint8)
tifffile.imwrite(f"{out}/ycbcr_jpeg.tif", ycc, photometric="ycbcr", subsampling=(2, 2), compression="jpeg",
                 compressionargs={"level": 95}, rowsperstrip=16)
y, x = grid(120, 90)
tifffile.imwrite(f"{out}/be_predictor_u16.tif", ((x * 701 + y * 13) % 60000).astype(np.uint16), byteorder=">",
                 compression="deflate", predictor=True, rowsperstrip=16)
tifffile.imwrite(f"{out}/be_predictor_f32.tif", (x * 0.5 - y * 0.25).astype(np.float32), byteorder=">",
                 compression="deflate", predictor=3, rowsperstrip=16)
