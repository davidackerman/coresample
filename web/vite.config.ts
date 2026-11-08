// coresample's page reuses chunkmirage's browser pieces (its ONNX Runtime Web wrapper and the
// plugin that serves ORT's WebAssembly runtime beside the page), imported from a chunkmirage
// checkout at CHUNKMIRAGE_WEB (default: a sibling clone, ../../chunkmirage/web), as chunkmirage's
// design notes describe for "another page on the engine".
import { resolve } from "node:path";
import { HttpsProxyAgent } from "https-proxy-agent";
import { defineConfig } from "vite";
import { ortRuntime } from "../../chunkmirage/web/vite-ort";

const chunkmirage = resolve(process.env.CHUNKMIRAGE_WEB ?? resolve(__dirname, "../../chunkmirage/web"));
// Neuroglancer at ./ng/ on the page's own site, so the page can drive it (same-origin iframe).
// The deployed site carries its own build there (.github/workflows/pages.yml); `vite` and
// `vite preview` relay appspot's client under /ng/ instead, as chunkmirage and cellmap-flow do.
// NG_URL relays another build instead, e.g. the site's own (patched, web/scripts/patch_neuroglancer.sh):
// NG_URL=https://davidackerman.github.io/coresample/ng npx vite
// (behind an HTTPS proxy, as in a sandbox, the relay goes through it)
const outbound = process.env.HTTPS_PROXY ? new HttpsProxyAgent(process.env.HTTPS_PROXY) : undefined;
const ng = { "/ng": { target: process.env.NG_URL ?? "https://neuroglancer-demo.appspot.com", changeOrigin: true, agent: outbound, rewrite: (p: string) => p.replace(/^\/ng/, "") } };

export default defineConfig({
  base: "./",
  resolve: {
    alias: {
      "@chunkmirage": resolve(chunkmirage, "src"),
      // one ORT for the page and chunkmirage's wrapper (its turn-taking is per instance)
      "onnxruntime-web/webgpu": resolve(__dirname, "node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs"),
    },
  },
  plugins: [ortRuntime()],
  server: { host: "0.0.0.0", port: 5180, fs: { allow: [__dirname, chunkmirage] }, proxy: ng },
  preview: { host: "0.0.0.0", port: 5180, proxy: ng },
  worker: { format: "es" },
  build: { target: "es2022" },
});
