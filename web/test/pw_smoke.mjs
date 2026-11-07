// Headless Chromium smoke test of the built page (Playwright). Opens a preset, waits for the
// view, searches from the centre of the canvas with the smallest settings, waits for hits and
// saves screenshots. usage: node test/pw_smoke.mjs <preset id> <out prefix> [max seconds]
import { createRequire } from "node:module";
const require = createRequire(process.env.PW_DIR + "/package.json");
const { chromium } = require("playwright-core");

const [preset, out, maxS = "150"] = process.argv.slice(2);
const p = new URL(process.env.HTTPS_PROXY);
const browser = await chromium.launch({
  headless: true,
  args: (process.env.PW_ARGS ?? "--no-sandbox --disable-dev-shm-usage").split(" "),
  proxy: { server: `${p.protocol}//${p.host}`, username: decodeURIComponent(p.username), password: decodeURIComponent(p.password), bypass: "127.0.0.1,localhost" },
});
const page = await browser.newPage({ viewport: { width: 1500, height: 900 } });
page.on("console", (m) => { const t = m.text(); if (!/^\[vite\]/.test(t)) console.log(`[console.${m.type()}]`, t.slice(0, 240)); });
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
page.on("crash", () => console.log("[crash] renderer crashed"));
browser.on("disconnected", () => console.log("[disconnected]"));
const t0 = Date.now(), el = (s) => ((Date.now() - t0) / 1000).toFixed(0) + "s";
await page.goto(`http://127.0.0.1:5180/?d=${preset}`);
await page.waitForTimeout(15000);
await page.screenshot({ path: `${out}_view.png` });
console.log(el(), "model:", await page.textContent("#model"));
await page.selectOption("#budget", "0.005");
await page.selectOption("#region", "1");
const box = await page.locator("#view").boundingBox();
await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
let last = "";
while ((Date.now() - t0) / 1000 < Number(maxS)) {
  await page.waitForTimeout(4000);
  const st = (await page.innerText("#status")).replace(/\n/g, " | ");
  if (st !== last) { console.log(el(), st); last = st; }
  if (/paused|done|error/.test(st)) break;
}
console.log(el(), "hits:", await page.locator(".hit").count(), "| model:", await page.textContent("#model"));
await page.screenshot({ path: `${out}_search.png` });
await browser.close();
