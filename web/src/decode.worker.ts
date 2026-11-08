// Image files to pixels off the page's main thread (see decode.ts).
self.onmessage = async (e: MessageEvent<{ id: number; bytes: ArrayBuffer | Blob; type: string; planes?: number; raw: boolean }>) => {
  const { id, bytes, type, planes, raw } = e.data;
  try {
    const blob = bytes instanceof Blob ? bytes : new Blob([bytes], { type });
    const bmp = await createImageBitmap(blob, raw ? { colorSpaceConversion: "none", premultiplyAlpha: "none" } : {});
    const { width, height } = bmp;
    const g = new OffscreenCanvas(width, height).getContext("2d", { willReadFrequently: true })!;
    g.drawImage(bmp, 0, 0);
    bmp.close();
    const rgba = g.getImageData(0, 0, width, height).data;
    let data: Uint8ClampedArray | Uint8Array = rgba;
    if (planes) {  // channel-major planes of the first `planes` channels
      const n = width * height, out = new Uint8Array(n * planes);
      for (let c = 0; c < planes; c++) for (let i = 0; i < n; i++) out[c * n + i] = rgba[4 * i + c];
      data = out;
    }
    (self as unknown as Worker).postMessage({ id, data, width, height }, [data.buffer]);
  } catch (err) {
    (self as unknown as Worker).postMessage({ id, error: String(err) });
  }
};
