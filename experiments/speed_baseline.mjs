// Speed baseline, full run (design Goal 2): for each dataset, 10 searches clicked at different
// places in the preset's view, each in a fresh browser context (empty HTTP cache), on this
// machine's GPU (WebGPU through ANGLE on Vulkan). A search runs until it is done or SECONDS.
// Each search's summary is appended to baseline10.jsonl as it finishes.
// usage: PW_DIR=... node experiments/speed_baseline.mjs <base url> <seconds> <id|query> ...
// (for WebGPU, run where the GPU is visible: not inside an agent sandbox, which hides /dev/nvidia*)
import { createRequire } from "node:module";
import { appendFileSync } from "node:fs";
// PW_DIR: a folder with playwright-core installed (as web/test/pw_*.mjs); its Chromium is found
// through PLAYWRIGHT_BROWSERS_PATH as usual.
if (!process.env.PW_DIR) throw new Error("set PW_DIR to a folder with playwright-core installed");
const require = createRequire(`${process.env.PW_DIR}/package.json`);
const { chromium } = require("playwright-core");

const [base, secs, ...sets] = process.argv.slice(2);
const SECONDS = Number(secs);
const OUT = new URL("./results/speed_baseline/baseline10.jsonl", import.meta.url);
const GPU = ["--no-sandbox", "--disable-dev-shm-usage", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
             "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--use-gl=angle", "--use-angle=vulkan"];
// click positions as fractions of the view (x, y)
const SPOTS = [[0.5, 0.5], [0.35, 0.4], [0.65, 0.4], [0.35, 0.6], [0.65, 0.6], [0.42, 0.3], [0.58, 0.7], [0.25, 0.5], [0.75, 0.5], [0.5, 0.25]];
const num = (s) => (s == null ? null : Number(String(s).replace(/,/g, "")));
const toBytes = (v, u) => v * ({ B: 1, KB: 1e3, kB: 1e3, MB: 1e6, GB: 1e9, TB: 1e12 }[u] ?? 1);

function parse(status, model) {
  const m = (re) => status.match(re);
  const rough = m(/([\d,]+) (?:patches|windows) compared in ([\d,]+) of the ([\d,]+) tiles the model looked over \(of ([\d,]+)\)/);
  const plain = m(/([\d,]+) (?:patches|windows) compared in ([\d,]+) of ([\d,]+) tiles/);
  const bytes = m(/([\d.]+) ?([KMGT]?B) read in (\d+) s/);
  const first = m(/first hits after ([\d.]+) s/);
  const places = m(/picked by the model at level (\d+) \(([\d,]+) places\)/);
  const scan = m(/quick pass over level (\d+) \(([\d.]+) s\)/);
  const area = m(/(\d+) areas? of ([\d.]+) (\w+) x ([\d.]+) (\w+)/);
  const rate = model.match(/([\d,]+) patches\/s/);
  return {
    patches: num(rough?.[1] ?? plain?.[1]), tilesRead: num(rough?.[2] ?? plain?.[2]),
    looked: num(rough?.[3]), total: num(rough?.[4] ?? plain?.[3]),
    bytes: bytes ? toBytes(Number(bytes[1]), bytes[2]) : null, seconds: num(bytes?.[3]),
    firstHit: first ? Number(first[1]) : null, roughPlaces: num(places?.[2]) ?? 0,
    scanSeconds: scan ? Number(scan[2]) : null, areas: num(area?.[1]), rate: num(rate?.[1]), gpu: /GPU/.test(model),
  };
}

const browser = await chromium.launch({ headless: true, args: GPU });
for (const set of sets) {
  const url = set.includes("=") ? `${base}/?${set}` : `${base}/?d=${set}`;
  for (let i = 0; i < SPOTS.length; i++) {
    const ctx = await browser.newContext({ viewport: { width: 1500, height: 900 } });
    const page = await ctx.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message.slice(0, 160)));
    try {
      await page.goto(url);
      await page.waitForTimeout(12000);
      // wait for the model (up to 40 s more) so first-hit times do not include its loading
      for (let w = 0; w < 20 && !/GPU|CPU/.test((await page.textContent("#model")) ?? ""); w++) await page.waitForTimeout(2000);
      const box = await page.locator("#stage").boundingBox();
      const [fx, fy] = SPOTS[i];
      await page.mouse.click(box.x + fx * box.width, box.y + fy * box.height);
      const t0 = Date.now(), samples = [];
      let phase = "";
      while ((Date.now() - t0) / 1000 < SECONDS) {
        await page.waitForTimeout(2000);
        const [ph, status, model] = await Promise.all(["#phase", "#status", "#model"].map((s) => page.textContent(s)));
        phase = ph;
        samples.push({ t: (Date.now() - t0) / 1000, ...parse(status, model) });
        if (/^Done|^Error|went wrong/i.test(ph)) break;
      }
      const status = await page.textContent("#status");
      const last = samples[samples.length - 1] ?? {};
      const rates = samples.map((s) => s.rate).filter((r) => r > 0).sort((a, b) => a - b);
      const rate = rates.length ? rates[Math.floor(rates.length / 2)] : null;
      const runs = (last.patches ?? 0) + (last.roughPlaces ?? 0);
      const rec = {
        set, spot: i, phase, gpu: last.gpu, searchSeconds: last.seconds, firstHit: last.firstHit, scanSeconds: last.scanSeconds,
        tilesRead: last.tilesRead, looked: last.looked, total: last.total, areas: last.areas,
        bytesMB: last.bytes && +(last.bytes / 1e6).toFixed(1), roughPlaces: last.roughPlaces, closePatches: last.patches,
        medianRate: rate, roughShare: runs ? +(last.roughPlaces / runs).toFixed(2) : null,
        modelBusy: rate && last.seconds ? +(runs / rate / last.seconds).toFixed(2) : null,
        status: status.replace(/\n/g, " | ").slice(0, 400), errors,
      };
      appendFileSync(OUT, JSON.stringify(rec) + "\n");
      console.log(set.slice(0, 20), i, rec.phase, "first", rec.firstHit, "busy", rec.modelBusy, "rough", rec.roughShare, "rate", rec.medianRate);
    } catch (e) {
      appendFileSync(OUT, JSON.stringify({ set, spot: i, error: String(e.message).slice(0, 300) }) + "\n");
      console.log(set.slice(0, 20), i, "error", String(e.message).slice(0, 120));
    } finally {
      await ctx.close();
    }
  }
}
await browser.close();
