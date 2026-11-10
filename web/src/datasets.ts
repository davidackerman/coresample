// Preset datasets: public, read straight from their hosts (every one answers CORS), each a
// pyramid already published by its owner. View centres are in full-resolution pixels.
import type { GlobeSpec } from "./globeviewer.ts";
import { OmeZarrSource, TileSource, legacySky, legacySurvey, openZoomify, type Source } from "./sources.ts";

export interface Preset {
  id: string;
  label: string;   // name
  detail: string;  // resolution and kind, one line
  kind: string;
  count: number;   // pixels (2-D) or voxels (3-D) at full resolution: the picker sorts by it
  volume?: boolean;
  open: () => Promise<Source>;
  view: { cx: number; cy: number; z?: number; scale: number };
  patch?: number;
  note?: string;
  globe?: GlobeSpec;  // planetary mosaics: also viewable on a globe
  geo?: "equirect" | "mercator" | "sky";  // how pixels map to longitude and latitude (maps of a body); sky: right ascension (360 at the left) and declination, Mercator
  layers?: Layer[];   // more layers over the same space (segmentations, other channels): shown in Neuroglancer
}

/** A layer shown with the data: a Neuroglancer source URL, its kind and a colour. */
export interface Layer { name: string; url: string; kind: "segmentation" | "image"; color: string; visible?: boolean }

// OpenOrganelle's whole-cell organelle segmentations of jrc_hela-2 (N5, same space as the EM)
const HELA_SEG = "n5://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels";
const helaLayers: Layer[] = ([
  ["nucleus", "#4b8bff"], ["mito", "#ff6b3d"], ["er", "#3ddc84"], ["golgi", "#ffd24a"], ["vesicle", "#ff5cc6"],
  ["endo", "#b07cff"], ["lyso", "#ff9f1c"], ["ld", "#9cf"], ["mt-out", "#e5675e"], ["pm", "#cccccc"],
] as const).map(([n, color]) => ({ name: n, url: `${HELA_SEG}/${n}_seg`, kind: "segmentation", color, visible: n === "mito" }));

const SKY_W = 512 * 2 ** 13;  // the whole-sky Legacy Surveys map (Mercator): 360 degrees of right ascension
const skyAt = (ra: number, dec: number) => {
  const r = (dec * Math.PI) / 180;
  return { cx: ((360 - ra) / 360) * SKY_W, cy: ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * SKY_W };
};
// equirectangular and web-mercator placement of a longitude/latitude, for the map-like sources
const equirect = (W: number, H: number, lon: number, lat: number) => ({ cx: ((lon + 180) / 360) * W, cy: ((90 - lat) / 180) * H });
const mercator = (W: number, lon: number, lat: number) => {
  const r = (lat * Math.PI) / 180;
  return { cx: ((lon + 180) / 360) * W, cy: ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * W };
};

const CTX_W = 4194304, CTX_H = 2097152;
const WAC_W = 131072, WAC_H = 65536;
const S2_W = 256 * 2 ** 14;

/** What the page opens on without ?d=: Earth, whose start view is already at a zoom where the
 * first click searches well, in a scene anyone can judge (Mars opens on the whole planet). */
export const DEFAULT_PRESET = "s2";

export const PRESETS: Preset[] = [
  {
    id: "mars-ctx", geo: "equirect", label: "Mars: CTX global mosaic", detail: "5 m/px, Murray Lab / Esri", kind: "planetary map", count: CTX_W * CTX_H,
    open: async () => new TileSource({
      name: "Mars CTX mosaic (Murray Lab)", tileSize: 512, maxZoom: 12, width: CTX_W, height: CTX_H, channels: 1,
      unit: "m", pixelSize: 5.05, attribution: "NASA/JPL/MSSS/Caltech Murray Lab/Esri (OnMars)",
      url: (z, x, y) => `https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX/MapServer/tile/${z}/${y}/${x}`,
      skipZooms: [5, 11],  // Esri's cache: zoom 11 tiles are truncated PNGs, zoom 5 returns error pages in places
    }),
    view: { ...equirect(CTX_W, CTX_H, -65, -8), scale: 1024 },  // the whole planet, Valles Marineris across the middle
    globe: { radii: [3396190, 3396190, 3376200], template: "https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX/MapServer/tile/{z}/{y}/{x}",
             credit: "NASA/JPL/MSSS/Caltech Murray Lab/Esri" },
    note: "Start: the whole planet, Valles Marineris across the middle. A click searches at the zoom on screen: zoom in to a crater, dune field or channel first.",
  },
  {
    id: "moon-wac", geo: "equirect", label: "Moon: LRO WAC global mosaic", detail: "100 m/px, NASA Trek", kind: "planetary map", count: WAC_W * WAC_H,
    open: async () => new TileSource({
      name: "Moon LRO WAC (NASA Trek)", tileSize: 256, maxZoom: 8, width: WAC_W, height: WAC_H, channels: 1,
      unit: "m", pixelSize: 100, attribution: "NASA/GSFC/Arizona State University (NASA Trek)",
      url: (z, x, y) => `https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm/${z}/${y}/${x}.jpg`,
    }),
    view: { ...equirect(WAC_W, WAC_H, -11.4, -43.3), scale: 4 },  // Tycho
    globe: { radii: [1737400, 1737400, 1737400], template: "https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm/{z}/{y}/{x}.jpg",
             credit: "NASA/GSFC/Arizona State University (NASA Trek)" },
    note: "Start: Tycho crater.",
  },
  {
    id: "andromeda", label: "Andromeda: Hubble PHAT mosaic", detail: "Hubble, ESA", kind: "astronomy", count: 1.5e9,
    open: () => openZoomify("https://cdn.esahubble.org/archives/images/zoomable/heic1502a", "Hubble PHAT Andromeda (heic1502a)",
      "NASA, ESA, J. Dalcanton, B.F. Williams, L.C. Johnson (University of Washington), the PHAT team, R. Gendler (CC BY 4.0)"),
    view: { cx: 34768, cy: 11115, scale: 8 },
    note: "Click a star cluster, a dust lane or a background galaxy.",
  },
  {
    id: "legacy-sky", geo: "sky", label: "Sky: DESI Legacy Surveys, the whole survey", detail: "0.31\u2033/px, cutouts as you browse", kind: "astronomy",
    count: SKY_W * SKY_W,
    open: async () => legacySky("ls-dr10"),
    view: { ...skyAt(285.1504, 59.1669), scale: 2 },  // NGC 6750
    note: "Start: NGC 6750. You browse the survey's own map tiles; a search reads cutouts its server draws on request, two at a time (1-28 s each), so it is slow.",
  },
  {
    id: "legacy-ngc6129", label: "Sky: DESI Legacy Surveys near NGC 6129", detail: "0.26\u2033/px, 1.2\u00b0 field", kind: "astronomy", count: 16384 * 16384,
    open: async () => legacySurvey(245.4296, 37.9956, "ls-dr9", 16384, 0.262, "Legacy Surveys DR9 around NGC 6129"),
    view: { cx: 8192, cy: 8192, scale: 4 },
    note: "Click a galaxy to find others like it (elliptical, spiral, edge-on...).",
  },
  {
    id: "webb", label: "Carina: Webb 'Cosmic Cliffs'", detail: "JWST NIRCam, ESA", kind: "astronomy", count: 123e6,
    open: () => openZoomify("https://cdn.esawebb.org/archives/images/zoomable/weic2205a", "JWST Carina (weic2205a)",
      "NASA, ESA, CSA, STScI (CC BY 4.0)"),
    view: { cx: 7287, cy: 4220, scale: 8 },
  },
  {
    id: "s2", geo: "mercator", label: "Earth: Sentinel-2 cloudless 2024", detail: "10 m/px, EOX", kind: "satellite", count: S2_W * S2_W,
    open: async () => new TileSource({
      name: "Sentinel-2 cloudless 2024 (EOX)", tileSize: 256, maxZoom: 14, channels: 3, unit: "m", pixelSize: 9.55,
      attribution: "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2024), CC BY-NC-SA 4.0",
      url: (z, x, y) => `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/${z}/${y}/${x}.jpg`,
    }),
    view: { ...mercator(S2_W, -100.87, 37.97), scale: 8 },  // center-pivot fields, Kansas
    note: "Start: center-pivot irrigation in Kansas. Try a field, a town or a river bend.",
  },
  {
    id: "hela", label: "HeLa cell: FIB-SEM volume", detail: "4 nm voxels, OpenOrganelle", kind: "3-D EM", count: 122e9, volume: true,
    open: () => OmeZarrSource.open("s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8", "jrc_hela-2 EM"),
    view: { cx: 8978, cy: 558, z: 2666, scale: 4 },
    note: "A 3-D volume: scroll through z with the mouse wheel in Neuroglancer. Organelle segmentations are under Layers.",
    layers: helaLayers,
  },
  {
    id: "idr-organoid", label: "Intestinal organoid: EM", detail: "IDR idr0083", kind: "2-D EM", count: 13.5e9,
    open: () => OmeZarrSource.open("https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0083A/9822152.zarr", "IDR idr0083 hSIOs-2 EM"),
    view: { cx: 72192, cy: 46592, scale: 16 },
  },
  {
    id: "idr-lung", label: "Lung tissue: whole-slide image", detail: "colour, IDR idr0073", kind: "histology", count: 347e6,
    open: () => OmeZarrSource.open("https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr", "IDR idr0073 lung08"),
    view: { cx: 10557, cy: 8216, scale: 8 },
  },
];
