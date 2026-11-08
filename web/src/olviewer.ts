// OpenLayers as the viewer for 2-D tile pyramids (planetary maps, Zoomify images, sky cutouts,
// web-map tiles): it draws and caches the source's own tiles, so panning and zooming stay smooth.
// Map coordinates are full-resolution pixels with y negated: pixel (col c, row r) is (c, -r).
import Feature from "ol/Feature";
import ImageLayer from "ol/layer/Image";
import ImageCanvasSource from "ol/source/ImageCanvas";
import type ImageTile from "ol/ImageTile";
import TileState from "ol/TileState";
import Map from "ol/Map";
import View from "ol/View";
import Polygon from "ol/geom/Polygon";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import Projection from "ol/proj/Projection";
import TileImage from "ol/source/TileImage";
import VectorSource from "ol/source/Vector";
import { Fill, Stroke, Style, Text } from "ol/style";
import TileGrid from "ol/tilegrid/TileGrid";
import "ol/ol.css";
import type { TileSource } from "./sources.ts";

export interface Box { fx: number; fy: number; side: number; label?: string }
/** A picture laid over the map: its canvas covers full-resolution pixels [x0, x1) x [y0, y1). */
export interface Overlay { canvas: HTMLCanvasElement; x0: number; y0: number; x1: number; y1: number }

const square = (b: Box) => {
  const h = b.side / 2;
  return new Polygon([[[b.fx - h, -b.fy + h], [b.fx + h, -b.fy + h], [b.fx + h, -b.fy - h], [b.fx - h, -b.fy - h], [b.fx - h, -b.fy + h]]]);
};
const label = (text: string, color: string) => new Text({
  text, font: "12px system-ui", fill: new Fill({ color }), backgroundFill: new Fill({ color: "#000a" }),
  padding: [1, 3, 1, 3], textAlign: "left", textBaseline: "bottom", offsetX: 0, offsetY: -2, overflow: true,
  placement: "point",
});

export class OlViewer {
  map: Map;
  view: View;
  private boxes = new VectorSource();
  private hoverF = new Feature();
  private dpr = window.devicePixelRatio || 1;
  private heat: Overlay | null = null;
  private heatLayer: ImageLayer<ImageCanvasSource>;

  constructor(target: HTMLElement, src: TileSource, init: { cx: number; cy: number; scale: number },
              on: { click: (fx: number, fy: number, add: boolean) => void; hover: () => { side: number; text: string } | null; moved: () => void;
                    pointer?: (fx: number, fy: number) => void }) {
    const s = src.spec, ts = s.tileSize;
    const W = src.levels[0].shape[2], H = src.levels[0].shape[1];
    const extent = [0, -H, W, 0];
    const projection = new Projection({ code: `coresample:${src.name}`, units: "pixels", extent });
    // the server's broken zooms left out: OpenLayers draws the neighbouring zooms there
    const zooms = Array.from({ length: s.maxZoom + 1 }, (_, z) => z).filter((z) => !s.skipZooms?.includes(z));
    const resolutions = zooms.map((z) => 2 ** (s.maxZoom - z));
    const tileGrid = new TileGrid({ extent, origin: [0, 0], resolutions, tileSize: ts });
    const tiles = new TileLayer({
      source: new TileImage({
        projection, tileGrid, transition: 150, wrapX: false,
        tileUrlFunction: (c) => (c ? s.url(zooms[c[0]], c[1], c[2]) : undefined),
        // tiles come through the source's own cached fetch, so viewing and searching share
        // every download (and its retries and rate limit)
        tileLoadFunction: (tile, _url) => {
          const [zi, x, y] = tile.getTileCoord(), z = zooms[zi];
          const img = (tile as ImageTile).getImage() as HTMLImageElement;
          src.blob(z, x, y).then((b) => {
            if (!b) { tile.setState(TileState.ERROR); return; }
            // keep the object URL alive well past "load": Chrome decodes images lazily when they
            // are drawn, and a URL revoked at load left tiles drawn only partly (cut off, black)
            const u = URL.createObjectURL(b);
            img.onerror = () => tile.setState(TileState.ERROR);
            img.src = u;
            setTimeout(() => URL.revokeObjectURL(u), 120_000);
          });
        },
        attributions: src.attribution,
      }),
    });
    this.view = new View({
      projection, center: [init.cx, -init.cy], resolution: init.scale * this.dpr,
      maxResolution: 2 ** s.maxZoom * 2, minResolution: 0.25, extent, constrainOnlyCenter: true,
    });
    const boxStyle = (f: Feature) => {
      const kind = f.get("kind"), color = kind === "query" ? "#5ab0ff" : kind === "hover" ? "#5ab0ff" : "#ffd24a";
      return new Style({
        stroke: new Stroke({ color, width: kind === "query" ? 3 : 2, lineDash: kind === "hover" ? [6, 4] : undefined }),
        text: f.get("label") ? label(f.get("label"), kind === "hit" ? "#ffd24a" : "#cfe6ff") : undefined,
      });
    };
    const overlay = new VectorLayer({ source: this.boxes, style: (f) => boxStyle(f as Feature), updateWhileAnimating: true, updateWhileInteracting: true });
    this.hoverF.set("kind", "hover");
    this.boxes.addFeature(this.hoverF);
    // the similarity heat: one canvas over the search region, drawn into each view unsmoothed
    this.heatLayer = new ImageLayer({
      visible: false,
      source: new ImageCanvasSource({
        projection, ratio: 1,
        canvasFunction: (ext, _res, _pr, size) => {
          const cv = document.createElement("canvas");
          cv.width = size[0]; cv.height = size[1];
          const h = this.heat;
          if (!h) return cv;
          // the cells in view only (deep zooms would otherwise scale the whole canvas enormously)
          const cw = (h.x1 - h.x0) / h.canvas.width, chh = (h.y1 - h.y0) / h.canvas.height;
          const vx0 = ext[0], vx1 = ext[2], vy0 = -ext[3], vy1 = -ext[1];
          const i0 = Math.max(0, Math.floor((vx0 - h.x0) / cw)), i1 = Math.min(h.canvas.width, Math.ceil((vx1 - h.x0) / cw));
          const j0 = Math.max(0, Math.floor((vy0 - h.y0) / chh)), j1 = Math.min(h.canvas.height, Math.ceil((vy1 - h.y0) / chh));
          if (i1 <= i0 || j1 <= j0) return cv;
          const sx = size[0] / (vx1 - vx0), sy = size[1] / (vy1 - vy0), g = cv.getContext("2d")!;
          g.imageSmoothingEnabled = false;
          g.drawImage(h.canvas, i0, j0, i1 - i0, j1 - j0, (h.x0 + i0 * cw - vx0) * sx, (h.y0 + j0 * chh - vy0) * sy, (i1 - i0) * cw * sx, (j1 - j0) * chh * sy);
          return cv;
        },
      }),
    });
    this.map = new Map({ target, layers: [tiles, this.heatLayer, overlay], view: this.view, controls: [] });
    this.map.on("singleclick", (e) => {  // shift-click: another example of the same thing
      this.lastPixel = null; this.refreshHover();
      on.click(e.coordinate[0], -e.coordinate[1], !!(e.originalEvent as MouseEvent).shiftKey);
    });
    this.onHover = on.hover;
    this.map.on("pointermove", (e) => {
      if (e.dragging) return;
      this.lastPixel = e.pixel;
      this.refreshHover();
      on.pointer?.(e.coordinate[0], -e.coordinate[1]);
    });
    this.map.on("moveend", () => on.moved());  // panned as well as zoomed
    target.addEventListener("pointerleave", () => { this.lastPixel = null; this.refreshHover(); });
    // zooming changes the search level (and so the box) without the pointer moving
    this.view.on("change:resolution", () => { this.refreshHover(); on.moved(); });
    this.view.on("change:center", () => this.refreshHover());
  }

  private lastPixel: number[] | null = null;
  private onHover: () => { side: number; text: string } | null = () => null;
  /** The dashed box under the last pointer position, at the current search level's size. */
  refreshHover() {
    const h = this.lastPixel ? this.onHover() : null;
    if (!h || !this.lastPixel) { this.hoverF.setGeometry(undefined); return; }
    const c = this.map.getCoordinateFromPixel(this.lastPixel);
    if (!c) return;
    this.hoverF.setGeometry(square({ fx: c[0], fy: -c[1], side: h.side }));
    this.hoverF.set("label", h.text);
  }

  /** Full-resolution pixels per device pixel (the canvas viewer's measure of zoom). */
  scale() { return (this.view.getResolution() ?? 1) / this.dpr; }
  center() { const c = this.view.getCenter() ?? [0, 0]; return { cx: c[0], cy: -c[1] }; }
  /** Visible region in full-resolution pixels. */
  extentPx() {
    const e = this.view.calculateExtent(this.map.getSize());
    return { x0: e[0], y0: -e[3], x1: e[2], y1: -e[1] };
  }
  goTo(fx: number, fy: number, scale: number) {
    this.view.animate({ center: [fx, -fy], resolution: scale * this.dpr, duration: 250 });
  }
  /** The similarity heat over the map (null: none), at this opacity. */
  setHeat(h: Overlay | null, opacity = 0.6) {
    this.heat = h;
    this.heatLayer.setVisible(!!h);
    this.heatLayer.setOpacity(opacity);
    this.heatLayer.getSource()?.changed();
  }
  setBoxes(queries: Box[], hits: Box[]) {
    for (const f of this.boxes.getFeatures()) if (f !== this.hoverF) this.boxes.removeFeature(f);
    const add = (b: Box, kind: string) => {
      const f = new Feature(square(b));
      f.set("kind", kind);
      if (b.label) f.set("label", b.label);
      this.boxes.addFeature(f);
    };
    hits.forEach((b) => add(b, "hit"));
    queries.forEach((b) => add(b, "query"));
  }
  destroy() { this.map.setTarget(undefined); }
}
