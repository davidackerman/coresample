// coresample's page reuses chunkmirage's browser pieces (its ONNX Runtime Web wrapper and the
// plugin that serves ORT's WebAssembly runtime beside the page), imported from a chunkmirage
// checkout at CHUNKMIRAGE_WEB (default: a sibling clone, ../../chunkmirage/web), as chunkmirage's
// design notes describe for "another page on the engine".
import { resolve } from "node:path";
import { defineConfig } from "vite";
import { ortRuntime } from "../../chunkmirage/web/vite-ort";

const chunkmirage = resolve(process.env.CHUNKMIRAGE_WEB ?? resolve(__dirname, "../../chunkmirage/web"));

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
  server: { host: "0.0.0.0", port: 5180, fs: { allow: [__dirname, chunkmirage] } },
  preview: { host: "0.0.0.0", port: 5180 },
  worker: { format: "es" },
  build: { target: "es2022" },
});
