// A globe for planetary mosaics (Mars CTX, Moon WAC), drawn by CesiumJS on the body's own
// ellipsoid from the same equirectangular tiles the 2-D map uses (2 x 1 tiles at the top zoom,
// Cesium's geographic tiling). Cesium loads from a pinned CDN copy only when the globe is opened.
// The search still works on the flat mosaic: a click's latitude and longitude become a pixel of it.
import type { TileSource } from "./sources.ts";

const CESIUM = "https://cdn.jsdelivr.net/npm/cesium@1.146.0/Build/Cesium/";

export interface GlobeSpec {
  radii: [number, number, number];  // metres
  template: string;                 // tile URL with {z}, {y}, {x}
  credit?: string;
}

let cesium: Promise<any> | null = null;
function loadCesium(): Promise<any> {
  return (cesium ??= new Promise((resolve, reject) => {
    (window as any).CESIUM_BASE_URL = CESIUM;
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = `${CESIUM}Widgets/widgets.css`;
    document.head.appendChild(css);
    const s = document.createElement("script");
    s.src = `${CESIUM}Cesium.js`;
    s.onload = () => resolve((window as any).Cesium);
    s.onerror = () => reject(new Error("could not load CesiumJS"));
    document.head.appendChild(s);
  }));
}

export interface GBox { fx: number; fy: number; side: number; label?: string }

export class GlobeViewer {
  private C: any;
  private viewer: any;
  private ellipsoid: any;
  private W: number;
  private H: number;
  private hoverE: any = null;
  private boxes: any[] = [];
  private lastMouse: any = null;
  ready: Promise<void>;

  constructor(private div: HTMLElement, private src: TileSource, private spec: GlobeSpec, init: { cx: number; cy: number; scale: number },
              private on: { click: (fx: number, fy: number, add: boolean) => void; hover: () => { side: number; text: string } | null; moved: () => void;
                            pointer?: (fx: number, fy: number) => void }) {
    this.W = src.levels[0].shape[2];
    this.H = src.levels[0].shape[1];
    this.ready = this.start(init);
  }

  private lonlat(fx: number, fy: number) { return [(fx / this.W) * 360 - 180, 90 - (fy / this.H) * 180]; }
  private pixel(lon: number, lat: number) { return [((lon + 180) / 360) * this.W, ((90 - lat) / 180) * this.H]; }
  private get px() { return this.src.levels[0].scale[2]; }  // metres per full-resolution pixel

  private async start(init: { cx: number; cy: number; scale: number }) {
    const C = (this.C = await loadCesium());
    if (this.gone) throw new Error("the globe was closed");  // before Cesium had loaded: no viewer to make
    const [a, b, c] = this.spec.radii;
    this.ellipsoid = new C.Ellipsoid(a, b, c);
    C.Ellipsoid.default = this.ellipsoid;
    const s = this.src.spec;
    this.viewer = new C.Viewer(this.div, {
      baseLayer: false, globe: new C.Globe(this.ellipsoid), skyAtmosphere: false, skyBox: false,
      baseLayerPicker: false, geocoder: false, timeline: false, animation: false, homeButton: false,
      sceneModePicker: false, navigationHelpButton: false, fullscreenButton: false, infoBox: false,
      selectionIndicator: false, requestRenderMode: false,
    });
    this.viewer.scene.globe.baseColor = C.Color.BLACK;
    this.viewer.scene.globe.showGroundAtmosphere = false;
    const tiles = new C.UrlTemplateImageryProvider({
      url: this.spec.template, tileWidth: s.tileSize, tileHeight: s.tileSize, maximumLevel: s.maxZoom,
      tilingScheme: new C.GeographicTilingScheme({ ellipsoid: this.ellipsoid, numberOfLevelZeroTilesX: 2, numberOfLevelZeroTilesY: 1 }),
      credit: this.spec.credit ?? this.src.attribution,
    });
    // every tile through the map's own loader (Cesium's level is the tile zoom): the cache the map
    // and the search share, and tiles a server sends broken (cut off, or an error page, as Esri's
    // Mars cache does at several zooms: black streaks and holes on the globe) rebuilt as the map
    // rebuilds them, from their children or their parent
    tiles.requestImage = async (x: number, y: number, level: number) => {
      const b = await this.src.blob(level, x, y);
      if (!b) throw new Error(`no tile ${level}/${y}/${x}`);
      // flipped vertically, as Cesium makes its own imagery bitmaps (WebGL cannot flip a bitmap
      // on upload): unflipped, every tile was drawn upside down
      return createImageBitmap(b, { imageOrientation: "flipY" });
    };
    this.viewer.imageryLayers.addImageryProvider(tiles);
    this.goTo(init.cx, init.cy, init.scale, false);
    const h = new C.ScreenSpaceEventHandler(this.viewer.scene.canvas);
    const click = (add: boolean) => (e: any) => {
      const ll = this.pick(e.position);
      if (!ll) return;
      this.lastMouse = null;
      this.refreshHover();
      const [fx, fy] = this.pixel(ll[0], ll[1]);
      this.on.click(fx, fy, add);
    };
    h.setInputAction(click(false), C.ScreenSpaceEventType.LEFT_CLICK);
    h.setInputAction(click(true), C.ScreenSpaceEventType.LEFT_CLICK, C.KeyboardEventModifier.SHIFT);  // another example
    h.setInputAction((e: any) => {
      this.lastMouse = e.endPosition;
      this.refreshHover();
      const ll = this.pick(e.endPosition);
      if (ll) { const [fx, fy] = this.pixel(ll[0], ll[1]); this.on.pointer?.(fx, fy); }
    }, C.ScreenSpaceEventType.MOUSE_MOVE);
    this.viewer.camera.changed.addEventListener(() => { this.refreshHover(); this.on.moved(); });
    this.viewer.camera.percentageChanged = 0.05;
  }

  private pick(pos: any): [number, number] | null {
    const C = this.C, cart = this.viewer.camera.pickEllipsoid(pos, this.ellipsoid);
    if (!cart) return null;
    const g = C.Cartographic.fromCartesian(cart, this.ellipsoid);
    return [C.Math.toDegrees(g.longitude), C.Math.toDegrees(g.latitude)];
  }

  /** Full-resolution pixels per device pixel at the view centre (the map's measure of zoom). */
  scale() {
    if (!this.viewer) return 1;
    const cam = this.viewer.camera, h = cam.positionCartographic.height, canvas = this.viewer.scene.canvas;
    const fovy = cam.frustum.fovy ?? cam.frustum.fov ?? Math.PI / 3;
    const mPerCss = (2 * h * Math.tan(fovy / 2)) / Math.max(1, canvas.clientHeight);
    return mPerCss / (window.devicePixelRatio || 1) / this.px;
  }

  extentPx() {
    const C = this.C, r = this.viewer?.camera.computeViewRectangle(this.ellipsoid);
    if (!r) return { x0: 0, y0: 0, x1: this.W, y1: this.H };
    const [x0, y0] = this.pixel(C.Math.toDegrees(r.west), C.Math.toDegrees(r.north));
    const [x1, y1] = this.pixel(C.Math.toDegrees(r.east), C.Math.toDegrees(r.south));
    return { x0, y0, x1, y1 };
  }

  goTo(fx: number, fy: number, scale: number, fly = true) {
    if (!this.viewer) return;
    const C = this.C, [lon, lat] = this.lonlat(fx, fy), canvas = this.viewer.scene.canvas;
    const fovy = this.viewer.camera.frustum.fovy ?? Math.PI / 3;
    const height = (scale * (window.devicePixelRatio || 1) * this.px * Math.max(1, canvas.clientHeight)) / (2 * Math.tan(fovy / 2));
    const dest = C.Cartesian3.fromDegrees(lon, lat, height, this.ellipsoid);
    if (fly) this.viewer.camera.flyTo({ destination: dest, duration: 0.8 });
    else this.viewer.camera.setView({ destination: dest });
  }

  private rect(b: GBox) {
    const C = this.C, h = b.side / 2;
    const [w, n] = this.lonlat(b.fx - h, b.fy - h), [e, s] = this.lonlat(b.fx + h, b.fy + h);
    return C.Rectangle.fromDegrees(w, s, e, n);
  }

  private entity(b: GBox, color: string, dashed = false) {
    const C = this.C, col = C.Color.fromCssColorString(color);
    const r = this.rect(b);
    const corners = [C.Rectangle.northwest(r), C.Rectangle.northeast(r), C.Rectangle.southeast(r), C.Rectangle.southwest(r), C.Rectangle.northwest(r)]
      .map((g: any) => C.Cartesian3.fromRadians(g.longitude, g.latitude, 0, this.ellipsoid));
    return this.viewer.entities.add({
      polyline: { positions: corners, width: 2, clampToGround: false, arcType: C.ArcType.RHUMB,
                  material: dashed ? new C.PolylineDashMaterialProperty({ color: col }) : col },
      position: C.Cartesian3.fromRadians(r.west, r.north, 0, this.ellipsoid),
      label: b.label ? { text: b.label, font: "12px system-ui", fillColor: col, showBackground: true,
                         backgroundColor: C.Color.BLACK.withAlpha(0.6), pixelOffset: new C.Cartesian2(4, -10),
                         horizontalOrigin: C.HorizontalOrigin.LEFT } : undefined,
    });
  }

  refreshHover() {
    if (!this.viewer) return;
    if (this.hoverE) { this.viewer.entities.remove(this.hoverE); this.hoverE = null; }
    const hb = this.lastMouse ? this.on.hover() : null;
    if (!hb) return;
    const ll = this.pick(this.lastMouse);
    if (!ll) return;
    const [fx, fy] = this.pixel(ll[0], ll[1]);
    this.hoverE = this.entity({ fx, fy, side: hb.side, label: hb.text }, "#5ab0ff", true);
  }

  setBoxes(queries: GBox[], hits: GBox[]) {
    if (!this.viewer) return;
    for (const e of this.boxes) this.viewer.entities.remove(e);
    this.boxes = hits.map((b) => this.entity(b, "#ffd24a"));
    for (const q of queries) this.boxes.push(this.entity(q, "#5ab0ff"));
  }

  private gone = false;
  destroy() { this.gone = true; this.viewer?.destroy(); this.viewer = null; this.div.innerHTML = ""; }
}
