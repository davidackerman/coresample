// Where the time to first hit goes (design Goal 2 follow-up): the page unchanged, timed from the
// outside. An init script wraps `Worker` to timestamp messages to and from the cheap-scan worker
// and the decode workers, and watches #phase and #hits; Playwright records the network requests;
// CDP gives the main thread's busy time. Splits the wait into: click to scan start, the scan's
// coarse read, the scan worker's compute, scan end to reading, and reading to the first hit.
// usage: node experiments/scan_timing.mjs <base url> <clicks> <id|query> ...  (GPU: outside the agent sandbox)
import { createRequire } from "node:module";
import { appendFileSync } from "node:fs";
// PW_DIR: a folder with playwright-core installed (as web/test/pw_*.mjs); its Chromium is found
// through PLAYWRIGHT_BROWSERS_PATH as usual.
if (!process.env.PW_DIR) throw new Error("set PW_DIR to a folder with playwright-core installed");
const require = createRequire(`${process.env.PW_DIR}/package.json`);
const { chromium } = require("playwright-core");

const [base, clicks, ...sets] = process.argv.slice(2);
const OUT = new URL("./results/speed_baseline/scan_timing.jsonl", import.meta.url);
const GPU = ["--no-sandbox", "--disable-dev-shm-usage", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
             "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--use-gl=angle", "--use-angle=vulkan"];
const SPOTS = [[0.5, 0.5], [0.42, 0.45], [0.58, 0.55]];

const INIT = () => {
  const T = (window.__t = []);
  const W = window.Worker;
  window.Worker = class extends W {
    constructor(url, opts) {
      super(url, opts);
      const u = String(url), name = u.includes("scan") ? "scan" : u.includes("decode") ? "decode" : "other";
      const post = this.postMessage.bind(this);
      this.postMessage = (m, t) => { T.push({ w: name, ev: "post", t: performance.now() }); return post(m, t); };
      this.addEventListener("message", () => T.push({ w: name, ev: "reply", t: performance.now() }));
    }
  };
  addEventListener("DOMContentLoaded", () => {
    const ph = document.getElementById("phase"), hits = document.getElementById("hits");
    new MutationObserver(() => T.push({ w: "phase", ev: ph.textContent, t: performance.now() })).observe(ph, { childList: true, characterData: true, subtree: true });
    new MutationObserver(() => { if (hits.querySelector("canvas, img, .hit")) T.push({ w: "hits", ev: "shown", t: performance.now() }); }).observe(hits, { childList: true, subtree: true });
  });
};

const browser = await chromium.launch({ headless: true, args: GPU });
for (const set of sets) {
  const url = set.includes("=") ? `${base}/?${set}` : `${base}/?d=${set}`;
  for (let i = 0; i < Number(clicks); i++) {
    const ctx = await browser.newContext({ viewport: { width: 1500, height: 900 } });
    await ctx.addInitScript(INIT);
    const page = await ctx.newPage();
    const reqs = [];
    page.on("requestfinished", async (r) => {
      const tm = r.timing(), sz = await r.sizes().catch(() => null);
      reqs.push({ url: r.url(), start: tm.startTime, end: tm.startTime + tm.responseEnd, bytes: sz ? sz.responseBodySize + sz.responseHeadersSize : 0 });
    });
    try {
      await page.goto(url);
      await page.waitForTimeout(12000);
      for (let w = 0; w < 20 && !/GPU|CPU/.test((await page.textContent("#model")) ?? ""); w++) await page.waitForTimeout(2000);
      const cdp = await ctx.newCDPSession(page);
      await cdp.send("Performance.enable");
      const busy = async () => (await cdp.send("Performance.getMetrics")).metrics.find((m) => m.name === "TaskDuration").value;
      const origin = await page.evaluate(() => performance.timeOrigin);
      const box = await page.locator("#stage").boundingBox();
      const [fx, fy] = SPOTS[i % SPOTS.length];
      const tClick = await page.evaluate(() => { window.__t.length = 0; return performance.now(); });
      const busy0 = await busy();
      await page.mouse.click(box.x + fx * box.width, box.y + fy * box.height);
      let busyScan = null;
      for (let k = 0; k < 150; k++) {  // up to 30 s, until the first hit
        await page.waitForTimeout(200);
        const T = await page.evaluate(() => window.__t);
        if (busyScan == null && T.some((e) => e.w === "scan" && e.ev === "reply")) busyScan = await busy();
        if (T.some((e) => e.w === "hits")) break;
      }
      const T = await page.evaluate(() => window.__t);
      const status = (await page.textContent("#status")).replace(/\n/g, " | ");
      const at = (pred) => { const e = T.find(pred); return e ? +((e.t - tClick) / 1000).toFixed(2) : null; };
      const scanStart = at((e) => e.w === "phase" && /scan|quick/i.test(e.ev));
      const scanPost = at((e) => e.w === "scan" && e.ev === "post");
      const scanReply = (() => { const e = [...T].reverse().find((x) => x.w === "scan" && x.ev === "reply"); return e ? +((e.t - tClick) / 1000).toFixed(2) : null; })();  // the last reply: with several scan workers, the join
      const readStart = at((e) => e.w === "phase" && /^Searching|looked over|read$/i.test(e.ev));
      const firstHit = at((e) => e.w === "hits");
      // network during the scan's coarse read (scan start to the worker post)
      const lo = origin + tClick + (scanStart ?? 0) * 1000, hi = origin + tClick + (scanPost ?? 0) * 1000;
      const inScan = reqs.filter((r) => r.end > lo && r.start < hi && !/\.onnx|\/src\/|\/node_modules\/|\/@/.test(r.url));
      let conc = 0;
      if (inScan.length) {
        const ev = inScan.flatMap((r) => [[r.start, 1], [r.end, -1]]).sort((a, b) => a[0] - b[0]);
        let c = 0; for (const [, d] of ev) conc = Math.max(conc, (c += d));
      }
      const decodes = T.filter((e) => e.w === "decode" && e.t - tClick >= (scanStart ?? 0) * 1000 && e.t - tClick <= (scanPost ?? 0) * 1000);
      const rec = {
        set, click: i, scanStart, scanPost, scanReply, readStart, firstHit,
        coarseReadS: scanStart != null && scanPost != null ? +(scanPost - scanStart).toFixed(2) : null,
        scanComputeS: scanPost != null && scanReply != null ? +(scanReply - scanPost).toFixed(2) : null,
        requestsInRead: inScan.length, mbInRead: +(inScan.reduce((s, r) => s + r.bytes, 0) / 1e6).toFixed(2),
        maxConcurrentRequests: conc, decodeMessagesInRead: decodes.filter((e) => e.ev === "post").length,
        mainThreadBusyS: busyScan != null ? +(busyScan - busy0).toFixed(2) : null,
        phases: T.filter((e) => e.w === "phase").map((e) => [+((e.t - tClick) / 1000).toFixed(2), e.ev.slice(0, 40)]),
        status: status.slice(0, 300),
      };
      appendFileSync(OUT, JSON.stringify(rec) + "\n");
      console.log(JSON.stringify({ ...rec, phases: undefined, status: undefined }));
    } catch (e) {
      console.log(set.slice(0, 20), i, "error", String(e.message).slice(0, 160));
    } finally {
      await ctx.close();
    }
  }
}
await browser.close();
