// Headless Chromium check of the drawing pad (Playwright): opens a preset, opens the pad,
// draws a ring (dark, then a light highlight on one side), searches for it and saves
// screenshots. usage: node test/pw_sketch.mjs <preset id> <out prefix>
import { createRequire } from "node:module";
const require = createRequire(process.env.PW_DIR + "/package.json");
const { chromium } = require("playwright-core");

const [preset, out] = process.argv.slice(2);
const p = new URL(process.env.HTTPS_PROXY);
const browser = await chromium.launch({
  headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"],
  proxy: { server: `${p.protocol}//${p.host}`, username: decodeURIComponent(p.username), password: decodeURIComponent(p.password), bypass: "127.0.0.1,localhost" },
});
const page = await browser.newPage({ viewport: { width: 1500, height: 900 } });
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
const t0 = Date.now(), el = () => ((Date.now() - t0) / 1000).toFixed(0) + "s";
const status = async () => (await page.innerText("#phase")) + " | " + (await page.innerText("#status")).replace(/\n/g, " | ");
await page.goto(`http://127.0.0.1:5180/?d=${preset}`);
await page.waitForTimeout(Number(process.env.PW_WAIT ?? 12000));
await page.evaluate(() => { document.querySelector("#region").value = "1"; });
await page.click("#drawbtn");
await page.waitForTimeout(2500);
console.log(el(), "pad:", await page.innerText("#padhelp"));
const b = await page.locator("#padcv").boundingBox(), cx = b.x + b.width / 2, cy = b.y + b.height / 2, r = b.width * 0.28;
const ring = async (a0, a1) => {
  await page.mouse.move(cx + r * Math.cos(a0), cy + r * Math.sin(a0));
  await page.mouse.down();
  for (let a = a0; a <= a1; a += 0.15) await page.mouse.move(cx + r * Math.cos(a), cy + r * Math.sin(a));
  await page.mouse.up();
};
await page.click("#padsize button[data-r='5']");
await ring(0, 2 * Math.PI + 0.2);
if (process.env.PW_HIGHLIGHT) { await page.click("#padtone button[data-t='light']"); await ring(-0.3, 1.6); }
await page.locator("#pad").screenshot({ path: `${out}_pad.png` });
await page.click("#padgo");
await page.waitForTimeout(Number(process.env.PW_READ ?? 30000));
console.log(el(), await status());
if (await page.isEnabled("#stop") && (await page.innerText("#stop")) === "Pause") await page.click("#stop");
await page.waitForTimeout(1500);
console.log(el(), await status());
await page.screenshot({ path: `${out}_sketch.png` });
await browser.close();
