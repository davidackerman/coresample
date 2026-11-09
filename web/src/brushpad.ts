// The brush over the query box: which of its 8 x 8 parts matter (the 112 px model's token grid,
// so a part is exactly what the model sees as one piece). Drag to mark parts; a drag that starts
// on a marked part unmarks instead. The box is shown big, the parts not marked dimmed.
import { BRUSH } from "./embedder.ts";

export class BrushPad {
  w = new Float32Array(BRUSH * BRUSH);
  private img = document.createElement("canvas");
  private paint: number | null = null;  // the value a drag sets (1 mark, 0 unmark), while dragging
  constructor(private cv: HTMLCanvasElement, private changed: () => void) {
    cv.width = cv.height = 256;
    const at = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect();
      const x = Math.floor(((e.clientX - r.left) / r.width) * BRUSH), y = Math.floor(((e.clientY - r.top) / r.height) * BRUSH);
      return x >= 0 && y >= 0 && x < BRUSH && y < BRUSH ? y * BRUSH + x : -1;
    };
    cv.addEventListener("pointerdown", (e) => {
      const i = at(e);
      if (i < 0) return;
      cv.setPointerCapture(e.pointerId);
      this.paint = this.w[i] > 0 ? 0 : 1;
      this.set(i);
    });
    cv.addEventListener("pointermove", (e) => { if (this.paint !== null) this.set(at(e)); });
    const end = () => { this.paint = null; };
    cv.addEventListener("pointerup", end);
    cv.addEventListener("pointercancel", end);
  }
  private set(i: number) {
    if (i < 0 || this.paint === null || this.w[i] === this.paint) return;
    this.w[i] = this.paint;
    this.draw();
    this.changed();
  }
  /** The query box to mark (P x P, C channels, 0-255) and the marks it starts with. */
  load(pix: Uint8Array, P: number, C: number, w: Float32Array | null) {
    const im = new ImageData(P, P);
    for (let i = 0; i < P * P; i++) {
      im.data[4 * i] = pix[i * C]; im.data[4 * i + 1] = pix[i * C + (C === 3 ? 1 : 0)]; im.data[4 * i + 2] = pix[i * C + (C === 3 ? 2 : 0)];
      im.data[4 * i + 3] = 255;
    }
    this.img.width = this.img.height = P;
    this.img.getContext("2d")!.putImageData(im, 0, 0);
    this.w.set(w ?? new Float32Array(BRUSH * BRUSH));
    this.draw();
  }
  fill(v: number) { this.w.fill(v); this.draw(); this.changed(); }
  get marked() { return this.w.reduce((n, v) => n + (v > 0 ? 1 : 0), 0); }
  private draw() {
    const g = this.cv.getContext("2d")!, S = this.cv.width, c = S / BRUSH;
    g.imageSmoothingEnabled = false;
    g.drawImage(this.img, 0, 0, S, S);
    const any = this.marked > 0;
    g.fillStyle = "#000000a0";
    for (let i = 0; i < this.w.length; i++) if (any && !this.w[i]) g.fillRect((i % BRUSH) * c, Math.floor(i / BRUSH) * c, c, c);
    g.strokeStyle = "#ffffff22"; g.lineWidth = 1;
    for (let k = 1; k < BRUSH; k++) {
      g.beginPath(); g.moveTo(k * c + 0.5, 0); g.lineTo(k * c + 0.5, S); g.stroke();
      g.beginPath(); g.moveTo(0, k * c + 0.5); g.lineTo(S, k * c + 0.5); g.stroke();
    }
    g.strokeStyle = "#5ab0ff"; g.lineWidth = 2;  // the marked parts' outline
    for (let i = 0; i < this.w.length; i++) {
      if (!this.w[i]) continue;
      const x = i % BRUSH, y = Math.floor(i / BRUSH), on = (dx: number, dy: number) => {
        const X = x + dx, Y = y + dy;
        return X >= 0 && Y >= 0 && X < BRUSH && Y < BRUSH && this.w[Y * BRUSH + X] > 0;
      };
      g.beginPath();
      if (!on(0, -1)) { g.moveTo(x * c, y * c + 1); g.lineTo((x + 1) * c, y * c + 1); }
      if (!on(0, 1)) { g.moveTo(x * c, (y + 1) * c - 1); g.lineTo((x + 1) * c, (y + 1) * c - 1); }
      if (!on(-1, 0)) { g.moveTo(x * c + 1, y * c); g.lineTo(x * c + 1, (y + 1) * c); }
      if (!on(1, 0)) { g.moveTo((x + 1) * c - 1, y * c); g.lineTo((x + 1) * c - 1, (y + 1) * c); }
      g.stroke();
    }
  }
}
