// OpenLayers as the viewer for 2-D tile pyramids (planetary maps, Zoomify images, sky cutouts,
// web-map tiles): it draws and caches the source's own tiles, so panning and zooming stay smooth.
// Map coordinates are full-resolution pixels with y negated: pixel (col c, row r) is (c, -r).
import Feature from "ol/Feature";
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

  constructor(target: HTMLElement, src: TileSource, init: { cx: number; cy: number; scale: number },
              on: { click: (fx: number, fy: number) => void; hover: () => { side: number; text: string } | null; moved: () => void }) {
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
    this.map = new Map({ target, layers: [tiles, overlay], view: this.view, controls: [] });
    this.map.on("singleclick", (e) => { this.lastPixel = null; this.refreshHover(); on.click(e.coordinate[0], -e.coordinate[1]); });
    this.onHover = on.hover;
    this.map.on("pointermove", (e) => {
      if (e.dragging) return;
      this.lastPixel = e.pixel;
      this.refreshHover();
    });
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
  setBoxes(query: Box | null, hits: Box[]) {
    for (const f of this.boxes.getFeatures()) if (f !== this.hoverF) this.boxes.removeFeature(f);
    const add = (b: Box, kind: string) => {
      const f = new Feature(square(b));
      f.set("kind", kind);
      if (b.label) f.set("label", b.label);
      this.boxes.addFeature(f);
    };
    hits.forEach((b) => add(b, "hit"));
    if (query) add(query, "query");
  }
  destroy() { this.map.setTarget(undefined); }
}
