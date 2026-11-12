// Neuroglancer as the viewer for volumes (OME-Zarr): it streams the raw data natively, from the
// data's own host, at every scale. It runs in a same-origin iframe (./ng/: the site's own build,
// or appspot relayed in development) so the page can drive it as chunkmirage's pages do, through
// the frame's `viewer`: clicks and zoom are read from it, hits are a local annotation layer, and
// the hover box is drawn on a transparent canvas over the frame.
import type { Layer } from "./datasets.ts";
import type { Channel, Vec3 } from "./sources.ts";
import type { Heat } from "./search.ts";
import { heatCanvas, heatGrid } from "./heat.ts";

/** The similarity heat over the slice: the places compared (working-level centres, `foot`
 * working voxels apart), the scores shown (lo to hi), and how strongly. */
export interface NgHeat { h: Heat; lo: number; hi: number; f: number; foot: number; opacity: number }

/** What the viewer needs of a volume Neuroglancer can read itself (OME-Zarr, precomputed). */
export interface NgSource {
  name: string; url: string; dims: 2 | 3;
  ngSource: string;
  ngDimensions(): Record<string, [number, string]> | null;
  ngOffset?: Vec3;  // z, y, x: where voxel 0 is in Neuroglancer's coordinates
  ngDisplay?(): string[];  // the axes to show, when the viewer's default (its first three) would be wrong
  channels?: 1 | 3;       // 3: a colour image, shown as red, green and blue layers added together
  chans?: Channel[];      // the channels of a multi-channel image: one layer each, its colour and range in its controls
  ngRange?: [number, number] | null;  // the values shown black to white (data that is not 8-bit), as the search sees them
}

/** Each channel's layer name: its label (a bare number as "channel 3"), made unique. */
export function channelNames(chans: Channel[]): string[] {
  const seen = new Set<string>();
  return chans.map((c) => {
    let n = /^\d+$/.test(c.label) ? `channel ${c.label}` : c.label;
    if (seen.has(n)) n = `${n} (${c.index})`;
    seen.add(n);
    return n;
  });
}
const hexOf = (c: [number, number, number]) => "#" + c.map((v) => Math.round(v * 255).toString(16).padStart(2, "0")).join("");
function rgbOf(h: unknown): [number, number, number] | null {
  const m = /^#([0-9a-f]{6})$/i.exec(String(h ?? ""));
  if (!m) return null;
  const v = parseInt(m[1], 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

/** The data's image layers. A multi-channel image: one layer per channel, added together, each
 * with a colour and a range in its controls (the search reads them back: channelMix). Else one
 * layer; a colour image with no channel list (precomputed) as red, green and blue layers. */
function imageLayers(src: NgSource): Json[] {
  if (src.chans?.length) {
    const names = channelNames(src.chans);
    // the controls' defaults are the channel's own, so a control left alone reads back as it opened
    return src.chans.map((c, k) => ({
      type: "image", name: names[k], source: src.ngSource, localPosition: [c.index], blend: "additive", opacity: 1, visible: c.visible,
      shader: `#uicontrol vec3 color color(default="${hexOf(c.color)}")\n#uicontrol invlerp normalized(range=[${c.range[0]}, ${c.range[1]}])\n`
            + "void main() { emitRGB(color * normalized()); }",
    }));
  }
  // the viewer's default stretch is the data type's whole range: 16-bit EM with values in a
  // narrow band (IDR's organoid: 1,500-6,000 of 65,535) would look black
  const range = src.ngRange ? { shaderControls: { normalized: { range: src.ngRange } } } : {};
  if (src.channels !== 3) return [{ type: "image", name: src.name, source: src.ngSource, ...range }];
  return ["red", "green", "blue"].map((c, k) => ({
    type: "image", name: k ? `${src.name} (${c})` : src.name, source: src.ngSource, localPosition: [k], blend: "additive", opacity: 1,
    shader: `void main() { float v = toNormalized(getDataValue()); emitRGB(vec3(${k === 0 ? "v" : "0."}, ${k === 1 ? "v" : "0."}, ${k === 2 ? "v" : "0."})); }`,
  }));
}

export interface NgBox { fz: number; fy: number; fx: number; side: number; label?: string }
type Json = Record<string, any>;

const LAYER = "coresample";
const HIT_COLOUR = "#ffd24a", QUERY_COLOUR = "#5ab0ff";  // the page's yellow (hits) and blue (what you clicked)

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
              private on: { click: (fz: number, fy: number, fx: number, add: boolean) => void; hoverBox: () => { side: number; text: string } | null; moved: () => void;
                            pointer?: (fx: number, fy: number, fz: number) => void },
              layers: Layer[] = []) {
    this.hoverCtx = hover.getContext("2d")!;
    this.off = src.ngOffset ?? [0, 0, 0];
    this.iframe = document.createElement("iframe");
    this.iframe.style.cssText = "position:absolute;inset:0;width:100%;height:100%;border:0;background:#000";
    const dpr = window.devicePixelRatio || 1;
    const state: Json = {
      layers: [...imageLayers(src),
               // segmentations: every object of a layer in its colour, hidden layers fetch nothing
               ...layers.map((l) => (l.kind === "segmentation"
                 ? { type: "segmentation", name: l.name, source: l.url, visible: !!l.visible, segmentDefaultColor: l.color, selectedAlpha: 0.4, hoverHighlight: false }
                 : { type: "image", name: l.name, source: l.url, visible: !!l.visible })),
               // each box carries its colour: the query and its examples in the page's blue, hits in yellow
               { type: "annotation", name: LAYER, source: "local://annotations", annotationColor: HIT_COLOUR, annotations: [],
                 annotationProperties: [{ id: "color", type: "rgb", default: HIT_COLOUR }],
                 shader: "void main() { setColor(prop_color()); }" }],
      // a volume: the slice and a 3-D view, where every hit box shows wherever it is in z
      layout: src.dims === 3 ? "xy-3d" : "xy", showSlices: true,
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
    // the heat over the slice panel (under the hover box, which goes above everything)
    this.heatCv.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none";
    container.appendChild(this.heatCv);
    window.addEventListener("resize", this.queueHeat);
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
      await this.showAxes();
      this.goTo(init.cx, init.cy, init.z, this.initScale);
      setTimeout(() => this.goTo(init.cx, init.cy, init.z, this.initScale), 1500);  // in case the viewer fits again
    }
    const doc = this.iframe.contentDocument!;
    // only the data panels search and show the box: the viewer's own controls (layer bar,
    // position, layer panel) work as usual
    const onData = (e: MouseEvent) => !!(e.target as HTMLElement | null)?.closest?.(".neuroglancer-rendered-data-panel");
    doc.addEventListener("mousemove", (e) => {
      if (!onData(e)) { if (this.mouse) { this.mouse = null; this.drawHover(); } return; }
      this.mouse = { x: e.clientX, y: e.clientY };
      this.drawHover();
      if (this.on.pointer) { const p = this.mousePosition(e.clientX, e.clientY); this.on.pointer(p[2], p[1], p[0]); }
    }, true);
    try { this.viewer.position.changed.add(() => this.on.moved()); } catch { /* older viewers */ }  // panned or moved in z
    try { this.viewer.navigationState.changed.add(this.queueHeat); } catch { /* older viewers */ }  // the heat follows the view
    doc.addEventListener("mouseleave", () => { this.mouse = null; this.drawHover(); }, true);
    doc.addEventListener("wheel", () => setTimeout(() => { this.drawHover(); this.on.moved(); }, 0), true);
    doc.addEventListener("mousedown", (e) => { this.down = e.button === 0 && onData(e) ? { x: e.clientX, y: e.clientY } : null; }, true);
    doc.addEventListener("mouseup", (e) => {
      const d = this.down;
      this.down = null;
      if (!d || e.button !== 0 || e.ctrlKey || e.altKey || e.metaKey) return;
      if (Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 4) return;  // a drag, not a click
      const p = this.mousePosition(e.clientX, e.clientY);
      if (p) { this.mouse = null; this.drawHover(); this.on.click(p[0], p[1], p[2], e.shiftKey); }  // shift: another example
    }, true);
  }

  /** The channels as the viewer shows them now: each on or off (a layer removed: off), its
   * colour and range from its controls, and the channel it shows; null before the viewer is up
   * or for an image without channels. A shader rewritten by hand keeps only what it still has. */
  channelMix(): Channel[] | null {
    const chans = this.src.chans;
    if (!chans?.length || !this.viewer) return null;
    const layers = (this.json().layers ?? []) as Json[], names = channelNames(chans);
    return chans.map((c, k) => {
      const l = layers.find((x) => x.name === names[k]);
      if (!l) return { ...c, visible: false };
      const sc = l.shaderControls ?? {}, r = sc.normalized?.range, lp = l.localPosition?.[0];
      return { ...c, visible: l.visible !== false && !l.archived, color: rgbOf(sc.color) ?? c.color,
               range: Array.isArray(r) && r.length === 2 && r.every(Number.isFinite) && r[1] !== r[0] ? [r[0], r[1]] : c.range,
               index: Number.isFinite(lp) ? Math.floor(lp) : c.index };
    });
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

  /** The axes to show (x and y, and z for a volume): an image with no units in its metadata
   * (IDR's) otherwise opens on its first three axes, t-z-y. Set once the coordinate space is
   * back after the state restore above (set without one, the viewer drops it). */
  private async showAxes() {
    const show = this.src.ngDisplay?.();
    if (!show?.length) return;
    const t0 = performance.now();
    while (performance.now() - t0 < 10_000) {
      const names: string[] = this.viewer.coordinateSpace?.value?.names ?? [];
      if (show.every((n) => names.includes(n))) break;
      await new Promise((r) => setTimeout(r, 100));
    }
    try { this.viewer.navigationState.pose.displayDimensions.restoreState(show); } catch { /* keep the viewer's */ }
  }

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

  /** Query (its examples) and hits as boxes of the annotation layer (the rest of the state untouched). */
  setBoxes(queries: NgBox[], hits: NgBox[]) {
    if (!this.viewer) return;
    const st = this.json();
    const layer = (st.layers as Json[]).find((l) => l.name === LAYER);
    if (!layer) return;
    const box = (b: NgBox, id: string, color: string) => {
      const a = this.names.map(() => 0), c = this.names.map(() => 0);
      const h = b.side / 2, put = (n: string, lo: number, hi: number) => { const i = this.idx(n); if (i >= 0) { a[i] = lo + this.o(n); c[i] = hi + this.o(n); } };
      put("x", b.fx - h, b.fx + h); put("y", b.fy - h, b.fy + h);
      put("z", this.src.dims === 3 ? b.fz - h : 0, this.src.dims === 3 ? b.fz + h : 1);
      return { type: "axis_aligned_bounding_box", id, pointA: a, pointB: c, description: b.label ?? "", props: [color] };
    };
    const list = [...hits.map((b, k) => box(b, `hit${k}`, HIT_COLOUR)), ...queries.map((b, k) => box(b, k ? `query${k}` : "query", QUERY_COLOUR))];
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

  // ---- the similarity heat over the slice ----------------------------------------------
  private heatCv = document.createElement("canvas");
  private heat: NgHeat | null = null;
  private slab: { key: string; h: Heat } | null = null;
  private heatQueued = false;

  /** Show the heat over the slice panel (null hides it). */
  setHeat(x: NgHeat | null) { this.heat = x; this.queueHeat(); }
  private heatImg: { key: string; img: HTMLCanvasElement; cell: number; gx0: number; gy0: number; gw: number; gh: number } | null = null;
  private queueHeat = () => {
    if (this.heatQueued) return;
    this.heatQueued = true;
    requestAnimationFrame(() => { this.heatQueued = false; this.drawHeat(); });
  };

  /** The places on the slice shown: in a volume, those whose centre is within half a place
   * spacing of the slice's z (the plane of places through it), so the heat is what was found
   * at this depth, not a projection; in an image, every place. */
  private slabOf(x: NgHeat, zw: number): Heat {
    if (this.src.dims !== 3) return x.h;
    const key = `${x.h.version}|${x.h.n}|${Math.floor(zw)}`;
    if (this.slab?.key === key) return this.slab.h;
    const { c, s, n } = x.h, half = x.foot / 2, keep: number[] = [];
    for (let i = 0; i < n; i++) if (Math.abs(c[3 * i] - zw) <= half && s[i] > -Infinity) keep.push(i);
    const cc = new Int32Array(3 * keep.length), ss = new Float32Array(keep.length);
    keep.forEach((i, k) => { cc.set(c.subarray(3 * i, 3 * i + 3), 3 * k); ss[k] = s[i]; });
    this.slab = { key, h: { c: cc, s: ss, n: keep.length, version: x.h.version } };
    return this.slab.h;
  }

  private drawHeat() {
    const cv = this.heatCv, dpr = window.devicePixelRatio || 1;
    const W = Math.round(this.container.clientWidth * dpr), H = Math.round(this.container.clientHeight * dpr);
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const g = cv.getContext("2d")!;
    g.clearRect(0, 0, W, H);
    const x = this.heat;
    if (!x || !this.viewer || !x.h.n) return;
    // the slice panel (the first; in the slice + 3-D layout the 3-D view is the second)
    const panel = this.iframe.contentDocument?.querySelector(".neuroglancer-rendered-data-panel") as HTMLElement | null;
    if (!panel) return;
    const r = panel.getBoundingClientRect(), c = this.center(), k = this.scale() * dpr;  // finest voxels per CSS pixel
    const f = x.f, x0 = (c.cx - (r.width / 2) * k) / f, x1 = (c.cx + (r.width / 2) * k) / f;
    const y0 = (c.cy - (r.height / 2) * k) / f, y1 = (c.cy + (r.height / 2) * k) / f;
    const sub = this.slabOf(x, c.z / f);
    if (!sub.n) return;
    // one cell per place, coarser when the view holds very many; made for the view and half a view
    // around it, and kept while the view stays inside that at about the same zoom, so panning
    // only moves the picture (made again on every frame, it went through every place each time)
    const vw = x1 - x0, vh = y1 - y0, cell0 = x.foot * Math.max(1, Math.ceil(Math.sqrt((vw / x.foot) * (vh / x.foot) / 1e6)));
    const key = `${sub.version}|${sub.n}|${Math.floor(c.z / f)}|${x.lo}|${x.hi}|${x.foot}`, k0 = this.heatImg;
    const fits = k0 && k0.key === key && k0.cell >= cell0 / 2 && k0.cell <= cell0 * 2
      && k0.gx0 <= x0 && k0.gy0 <= y0 && k0.gx0 + k0.gw * k0.cell >= x1 && k0.gy0 + k0.gh * k0.cell >= y1;
    if (!fits) {
      const cell = x.foot * Math.max(1, Math.ceil(Math.sqrt(((2 * vw) / x.foot) * ((2 * vh) / x.foot) / 2e6)));
      const gx0 = Math.floor((x0 - vw / 2) / cell) * cell, gy0 = Math.floor((y0 - vh / 2) / cell) * cell;
      const gw = Math.max(1, Math.ceil((x1 + vw / 2 - gx0) / cell)), gh = Math.max(1, Math.ceil((y1 + vh / 2 - gy0) / cell));
      const img = heatCanvas(heatGrid(sub, gw, gh, 2, 1, [0, gy0, gx0], 1 / cell, 1 / cell, x.foot), gw, gh, x.lo, x.hi, k0?.img);
      this.heatImg = { key, img, cell, gx0, gy0, gw, gh };
    }
    const { img, cell, gx0, gy0, gw, gh } = this.heatImg!;
    const sx = r.left + r.width / 2 + (gx0 * f - c.cx) / k, sy = r.top + r.height / 2 + (gy0 * f - c.cy) / k;
    g.save();
    g.beginPath(); g.rect(r.left * dpr, r.top * dpr, r.width * dpr, r.height * dpr); g.clip();
    g.globalAlpha = x.opacity;
    g.imageSmoothingEnabled = false;
    g.drawImage(img, sx * dpr, sy * dpr, ((gw * cell * f) / k) * dpr, ((gh * cell * f) / k) * dpr);
    g.restore();
  }

  destroy() {
    window.removeEventListener("resize", this.queueHeat);
    this.iframe.remove(); this.heatCv.remove();
    this.hoverCtx.clearRect(0, 0, this.hover.width, this.hover.height);
  }
}
