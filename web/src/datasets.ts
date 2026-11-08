// Preset datasets: public, read straight from their hosts (every one answers CORS), each a
// pyramid already published by its owner. View centres are in full-resolution pixels.
import type { GlobeSpec } from "./globeviewer.ts";
import { OmeZarrSource, TileSource, legacySurvey, openZoomify, type Source } from "./sources.ts";

export interface Preset {
  id: string;
  label: string;
  kind: string;
  open: () => Promise<Source>;
  view: { cx: number; cy: number; z?: number; scale: number };
  patch?: number;
  note?: string;
  globe?: GlobeSpec;  // planetary mosaics: also viewable on a globe
}

// equirectangular and web-mercator placement of a longitude/latitude, for the map-like sources
const equirect = (W: number, H: number, lon: number, lat: number) => ({ cx: ((lon + 180) / 360) * W, cy: ((90 - lat) / 180) * H });
const mercator = (W: number, lon: number, lat: number) => {
  const r = (lat * Math.PI) / 180;
  return { cx: ((lon + 180) / 360) * W, cy: ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * W };
};

const CTX_W = 4194304, CTX_H = 2097152;
const WAC_W = 131072, WAC_H = 65536;
const S2_W = 256 * 2 ** 14;

export const PRESETS: Preset[] = [
  {
    id: "mars-ctx", label: "Mars: CTX global mosaic, 5 m/px (8.8 trillion pixels)", kind: "planetary map",
    open: async () => new TileSource({
      name: "Mars CTX mosaic (Murray Lab)", tileSize: 512, maxZoom: 12, width: CTX_W, height: CTX_H, channels: 1,
      unit: "m", pixelSize: 5.05, attribution: "NASA/JPL/MSSS/Caltech Murray Lab/Esri (OnMars)",
      url: (z, x, y) => `https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX/MapServer/tile/${z}/${y}/${x}`,
    }),
    view: { ...equirect(CTX_W, CTX_H, 77.45, 18.44), scale: 16 },  // Jezero crater
    globe: { radii: [3396190, 3396190, 3376200], template: "https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX/MapServer/tile/{z}/{y}/{x}",
             credit: "NASA/JPL/MSSS/Caltech Murray Lab/Esri" },
    note: "Start: Jezero crater. Click a crater, dune field or channel.",
  },
  {
    id: "moon-wac", label: "Moon: LRO WAC global mosaic, 100 m/px (8.6 gigapixels)", kind: "planetary map",
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
    id: "andromeda", label: "Andromeda: Hubble PHAT mosaic (1.5 gigapixels)", kind: "astronomy",
    open: () => openZoomify("https://cdn.esahubble.org/archives/images/zoomable/heic1502a", "Hubble PHAT Andromeda (heic1502a)",
      "NASA, ESA, J. Dalcanton, B.F. Williams, L.C. Johnson (University of Washington), the PHAT team, R. Gendler (CC BY 4.0)"),
    view: { cx: 34768, cy: 11115, scale: 8 },
    note: "Click a star cluster, a dust lane or a background galaxy.",
  },
  {
    id: "legacy-ngc6129", label: "Sky: DESI Legacy Surveys, 1.2 degrees around NGC 6129 (268 megapixels)", kind: "astronomy",
    open: async () => legacySurvey(245.4296, 37.9956, "ls-dr9", 16384, 0.262, "Legacy Surveys DR9 around NGC 6129"),
    view: { cx: 8192, cy: 8192, scale: 4 },
    note: "Click a galaxy to find others like it (elliptical, spiral, edge-on...).",
  },
  {
    id: "webb", label: "Webb: Carina 'Cosmic Cliffs' (123 megapixels)", kind: "astronomy",
    open: () => openZoomify("https://cdn.esawebb.org/archives/images/zoomable/weic2205a", "JWST Carina (weic2205a)",
      "NASA, ESA, CSA, STScI (CC BY 4.0)"),
    view: { cx: 7287, cy: 4220, scale: 8 },
  },
  {
    id: "s2", label: "Earth: Sentinel-2 cloudless 2024, 10 m/px (17.6 trillion pixels)", kind: "satellite",
    open: async () => new TileSource({
      name: "Sentinel-2 cloudless 2024 (EOX)", tileSize: 256, maxZoom: 14, channels: 3, unit: "m", pixelSize: 9.55,
      attribution: "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2024), CC BY-NC-SA 4.0",
      url: (z, x, y) => `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/${z}/${y}/${x}.jpg`,
    }),
    view: { ...mercator(S2_W, -100.87, 37.97), scale: 8 },  // center-pivot fields, Kansas
    note: "Start: center-pivot irrigation in Kansas. Try a field, a town or a river bend.",
  },
  {
    id: "hela", label: "HeLa cell: FIB-SEM volume, 4 nm (122 gigavoxels, OpenOrganelle)", kind: "3-D EM",
    open: () => OmeZarrSource.open("s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8", "jrc_hela-2 EM"),
    view: { cx: 8978, cy: 558, z: 2666, scale: 4 },
    note: "A 3-D volume: change the z slice with the slider or [ and ].",
  },
  {
    id: "idr-organoid", label: "Intestinal organoid: EM (13.5 gigapixels, IDR idr0083)", kind: "2-D EM",
    open: () => OmeZarrSource.open("https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0083A/9822152.zarr", "IDR idr0083 hSIOs-2 EM"),
    view: { cx: 72192, cy: 46592, scale: 16 },
  },
  {
    id: "idr-lung", label: "Lung tissue: whole-slide colour image (347 megapixels, IDR)", kind: "histology",
    open: () => OmeZarrSource.open("https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr", "IDR idr0073 lung08"),
    view: { cx: 10557, cy: 8216, scale: 8 },
  },
];
