// Neuroglancer as the viewer for volumes (OME-Zarr): it streams the raw data natively, from the
// data's own host, at every scale. It runs in a same-origin iframe (./ng/: the site's own build,
// or appspot relayed in development) so the page can drive it as chunkmirage's pages do, through
// the frame's `viewer`: clicks and zoom are read from it, hits are a local annotation layer, and
// the hover box is drawn on a transparent canvas over the frame.
import type { Layer } from "./datasets.ts";
import type { Vec3 } from "./sources.ts";

/** What the viewer needs of a volume Neuroglancer can read itself (OME-Zarr, precomputed). */
export interface NgSource {
  name: string; url: string; dims: 2 | 3;
  ngSource: string;
  ngDimensions(): Record<string, [number, string]> | null;
  ngOffset?: Vec3;  // z, y, x: where voxel 0 is in Neuroglancer's coordinates
}

export interface NgBox { fz: number; fy: number; fx: number; side: number; label?: string }
type Json = Record<string, any>;

const LAYER = "coresample";

export class NgViewer {
  iframe: HTMLIFrameElement;
  private viewer: any = null;
  private names: string[] = [];  // the viewer's global dimension names, in order
  ready: Promise<void>;
  private hoverCtx: CanvasRenderingContext2D;
  private mouse: { x: number; y: number } | null = null;
  private down: { x: number; y: number } | null = null;
  private initScale = 1;
  private explicit = false;  // the first state already had the place and coordinate space

  private off: Vec3;
  constructor(private container: HTMLElement, private src: NgSource, init: { cx: number; cy: number; z: number; scale: number },
              private hover: HTMLCanvasElement,
              private on: { click: (fz: number, fy: number, fx: number) => void; hoverBox: () => { side: number; text: string } | null; moved: () => void },
              layers: Layer[] = []) {
    this.hoverCtx = hover.getContext("2d")!;
    this.off = src.ngOffset ?? [0, 0, 0];
    this.iframe = document.createElement("iframe");
    this.iframe.style.cssText = "position:absolute;inset:0;width:100%;height:100%;border:0;background:#000";
    const dpr = window.devicePixelRatio || 1;
    const state: Json = {
      layers: [{ type: "image", name: src.name, source: src.ngSource },
               // segmentations: every object of a layer in its colour, hidden layers fetch nothing
               ...layers.map((l) => (l.kind === "segmentation"
                 ? { type: "segmentation", name: l.name, source: l.url, visible: !!l.visible, segmentDefaultColor: l.color, selectedAlpha: 0.4, hoverHighlight: false }
                 : { type: "image", name: l.name, source: l.url, visible: !!l.visible })),
               { type: "annotation", name: LAYER, source: "local://annotations", annotationColor: "#ffd24a", annotations: [] }],
      // a volume: the slice and a 3-D view, where every hit box shows wherever it is in z
      layout: src.dims === 3 ? "xy-3d" : "xy", showSlices: true, selectedLayer: { visible: false }, layerListPanel: { visible: false },
      crossSectionScale: init.scale * dpr,
    };
    // the place, zoom and coordinate space in the first state when they can be known from the
    // metadata: the viewer then loads the data there once, instead of first loading a fit of
    // the whole volume and having those requests cancelled when the view moves
    const dims = src.ngDimensions();
    if (dims) {
      this.explicit = true;
      state.dimensions = dims;
      state.position = Object.keys(dims).map((n) => (n === "x" ? init.cx + this.off[2] : n === "y" ? init.cy + this.off[1] : init.z + this.off[0]) + 0.5);
      state.projectionScale = init.scale * dpr * Math.max(300, container.clientHeight) * 3;
      state.layers[state.layers.length - 1].source = { url: "local://annotations", transform: { outputDimensions: dims } };
    }
    this.iframe.src = `ng/index.html#!${encodeURIComponent(JSON.stringify(state))}`;
    container.appendChild(this.iframe);
    this.initScale = init.scale;
    this.ready = this.attach(init);
  }

  private async attach(init: { cx: number; cy: number; z: number }) {
    const t0 = performance.now();
    while (performance.now() - t0 < 60_000) {  // the viewer, once its page has made it and the data opened
      const v = (this.iframe.contentWindow as any)?.viewer;
      const names: string[] | undefined = v?.coordinateSpace?.value?.names;
      if (v?.state && names?.length) { this.viewer = v; this.names = names; break; }
      await new Promise((r) => setTimeout(r, 200));
    }
    if (!this.viewer) throw new Error("Neuroglancer did not start");
    try { this.viewer.uiConfiguration.showUIControls.value = false; } catch { /* older viewers */ }
    if (this.explicit) {
      // opened at the place already; should the viewer still fit the view to the data, go back
      setTimeout(() => {
        const c = this.center();
        if (Math.abs(c.cx - init.cx) > 2 || Math.abs(c.cy - init.cy) > 2) this.goTo(init.cx, init.cy, init.z, this.initScale);
      }, 1500);
    } else {
      // the viewer fits position and zoom to the data once it has opened, over whatever was set
      // before: wait for that, then go to the preset's place
      const t1 = performance.now();
      while (performance.now() - t1 < 20_000) {
        const z = this.viewer.navigationState?.zoomFactor?.value;
        if (Number.isFinite(z) && z < 1e6) break;
        await new Promise((r) => setTimeout(r, 200));
      }
      // the annotation layer in the viewer's own coordinate space (made before the data opened,
      // its local source had none): one state restore, before the camera is set
      try {
        const cs = this.viewer.coordinateSpace.value, dims: Record<string, [number, string]> = {};
        cs.names.forEach((n: string, i: number) => (dims[n] = [cs.scales[i], cs.units[i]]));
        const st = this.json(), layer = (st.layers as Json[]).find((l) => l.name === LAYER);
        if (layer) { layer.source = { url: "local://annotations", transform: { outputDimensions: dims } }; this.viewer.state.restoreState(st); }
      } catch { /* keep the layer as it is */ }
      await new Promise((r) => setTimeout(r, 300));
      this.goTo(init.cx, init.cy, init.z, this.initScale);
      setTimeout(() => this.goTo(init.cx, init.cy, init.z, this.initScale), 1500);  // in case the viewer fits again
    }
    const doc = this.iframe.contentDocument!;
    doc.addEventListener("mousemove", (e) => { this.mouse = { x: e.clientX, y: e.clientY }; this.drawHover(); }, true);
    doc.addEventListener("mouseleave", () => { this.mouse = null; this.drawHover(); }, true);
    doc.addEventListener("wheel", () => setTimeout(() => { this.drawHover(); this.on.moved(); }, 0), true);
    doc.addEventListener("mousedown", (e) => { if (e.button === 0) this.down = { x: e.clientX, y: e.clientY }; }, true);
    doc.addEventListener("mouseup", (e) => {
      const d = this.down;
      this.down = null;
      if (!d || e.button !== 0 || e.shiftKey || e.ctrlKey || e.altKey || e.metaKey) return;
      if (Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 4) return;  // a drag, not a click
      const p = this.mousePosition(e.clientX, e.clientY);
      if (p) { this.mouse = null; this.drawHover(); this.on.click(p[0], p[1], p[2]); }
    }, true);
  }

  /** Show or hide one of the extra layers. */
  setLayerVisible(name: string, visible: boolean) {
    const m = this.viewer?.layerManager?.getLayerByName?.(name);
    if (!m) return;
    if (typeof m.setVisible === "function") m.setVisible(visible);
    else m.visible = visible;
  }

  private idx(n: string) { return this.names.indexOf(n); }
  /** Voxel offset of an axis (Neuroglancer coordinate = array index + offset). */
  private o(n: string) { return n === "z" ? this.off[0] : n === "y" ? this.off[1] : n === "x" ? this.off[2] : 0; }
  /** The voxel under the mouse, (z, y, x) at the finest level: the viewer's own mouse position
   * when it is current, else from the click's offset from the view centre at this zoom (the
   * single xy panel fills the frame). */
  private mousePosition(cx: number, cy: number): [number, number, number] {
    const ms = this.viewer.mouseState;
    if (ms?.active && ms.position?.length) {
      const p = ms.position as Float32Array, at = (n: string) => (this.idx(n) >= 0 ? Math.floor(p[this.idx(n)] - this.o(n)) : 0);
      return [at("z"), at("y"), at("x")];
    }
    // the panel under the click (the slice panel need not fill the frame: xy-3d layout)
    const el = this.iframe.contentDocument?.elementFromPoint(cx, cy)?.closest(".neuroglancer-rendered-data-panel") as HTMLElement | null;
    const r = el?.getBoundingClientRect() ?? { left: 0, top: 0, width: this.iframe.clientWidth, height: this.iframe.clientHeight };
    const c = this.center(), k = this.scale() * (window.devicePixelRatio || 1);
    const dx = (cx - (r.left + r.width / 2)) * k, dy = (cy - (r.top + r.height / 2)) * k;
    return [Math.floor(c.z), Math.floor(c.cy + dy), Math.floor(c.cx + dx)];
  }
  private json(): Json { return this.viewer.state.toJSON(); }

  /** Finest-level voxels per device pixel (the canvas viewer's measure of zoom). */
  scale() {
    const z = this.viewer?.navigationState?.zoomFactor?.value ?? (this.viewer ? this.json().crossSectionScale : 1);
    return z / (window.devicePixelRatio || 1);
  }
  center() {
    const p = this.viewer?.position?.value as Float32Array | undefined;
    if (!p) return { cx: 0, cy: 0, z: 0 };
    const at = (n: string) => (this.idx(n) >= 0 ? p[this.idx(n)] - this.o(n) : 0);
    return { cx: at("x"), cy: at("y"), z: at("z") };
  }
  extentPx() {
    const { cx, cy } = this.center(), s = this.scale() * (window.devicePixelRatio || 1);
    const w = this.iframe.clientWidth * s, h = this.iframe.clientHeight * s;
    return { x0: cx - w / 2, x1: cx + w / 2, y0: cy - h / 2, y1: cy + h / 2 };
  }

  goTo(fx: number, fy: number, fz: number, scale: number | null) {
    if (!this.viewer) return;
    const cur = this.viewer.position?.value as Float32Array | undefined;
    const pos = cur?.length ? Float32Array.from(cur) : new Float32Array(this.names.length);
    const set = (n: string, v: number) => { if (this.idx(n) >= 0) pos[this.idx(n)] = v + this.o(n) + 0.5; };
    set("x", fx); set("y", fy); if (this.src.dims === 3) set("z", fz);
    try {  // the viewer's own controls; a state restore if they are not there
      this.viewer.position.value = pos;
      if (scale !== null) {
        this.viewer.navigationState.zoomFactor.value = scale * (window.devicePixelRatio || 1);
        // the 3-D panel's own zoom (projection scale = the view's height in voxels): a few
        // slice-views around the place, so the hit boxes nearby show in 3-D
        const pz = this.viewer.perspectiveNavigationState?.zoomFactor;
        if (pz) pz.value = scale * (window.devicePixelRatio || 1) * this.iframe.clientHeight * 3;
      }
    } catch {
      const st = this.json();
      st.position = Array.from(pos);
      if (scale !== null) st.crossSectionScale = scale * (window.devicePixelRatio || 1);
      this.viewer.state.restoreState(st);
    }
  }

  /** Query and hits as boxes of the annotation layer (the rest of the state untouched). */
  setBoxes(query: NgBox | null, hits: NgBox[]) {
    if (!this.viewer) return;
    const st = this.json();
    const layer = (st.layers as Json[]).find((l) => l.name === LAYER);
    if (!layer) return;
    const box = (b: NgBox, id: string) => {
      const a = this.names.map(() => 0), c = this.names.map(() => 0);
      const h = b.side / 2, put = (n: string, lo: number, hi: number) => { const i = this.idx(n); if (i >= 0) { a[i] = lo + this.o(n); c[i] = hi + this.o(n); } };
      put("x", b.fx - h, b.fx + h); put("y", b.fy - h, b.fy + h);
      put("z", this.src.dims === 3 ? b.fz - h : 0, this.src.dims === 3 ? b.fz + h : 1);
      return { type: "axis_aligned_bounding_box", id, pointA: a, pointB: c, description: b.label ?? "" };
    };
    const list = [...hits.map((b, k) => box(b, `hit${k}`)), ...(query ? [box(query, "query")] : [])];
    // the annotation layer's own list, replaced in place: a whole-state restore resets the
    // camera and drops local annotations on this viewer
    const local = this.viewer.layerManager?.getLayerByName?.(LAYER)?.layer?.localAnnotations;
    if (local?.restoreState) { local.restoreState(list); return; }
    layer.annotations = list;
    this.viewer.state.restoreState(st);
  }

  drawHover() {
    const g = this.hoverCtx, dpr = window.devicePixelRatio || 1;
    g.clearRect(0, 0, this.hover.width, this.hover.height);
    const hb = this.on.hoverBox();
    if (!hb || !this.mouse || !this.viewer) return;
    const side = (hb.side / this.scale()) , x = this.mouse.x * dpr, y = this.mouse.y * dpr;
    g.setLineDash([6 * dpr, 4 * dpr]); g.lineWidth = 2 * dpr; g.strokeStyle = "#5ab0ff";
    g.strokeRect(x - side / 2, y - side / 2, side, side);
    g.setLineDash([]);
    g.font = `${12 * dpr}px system-ui`;
    const tw = g.measureText(hb.text).width;
    g.fillStyle = "#000a"; g.fillRect(x - side / 2 - 3 * dpr, y - side / 2 - 19 * dpr, tw + 6 * dpr, 17 * dpr);
    g.fillStyle = "#cfe6ff"; g.fillText(hb.text, x - side / 2, y - side / 2 - 6 * dpr);
  }

  destroy() { this.iframe.remove(); this.hoverCtx.clearRect(0, 0, this.hover.width, this.hover.height); }
}
