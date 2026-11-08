// Headless Chromium check of several examples and the heat (Playwright): opens a preset,
// searches from the centre of the view, shift-clicks a second place and uses "+ example" for a
// third, and saves screenshots. usage: node test/pw_examples.mjs <preset id> <out prefix>
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
page.on("console", (m) => { if (m.type() === "error") console.log("[console.error]", m.text().slice(0, 200)); });
const t0 = Date.now(), el = () => ((Date.now() - t0) / 1000).toFixed(0) + "s";
const status = async () => (await page.innerText("#phase")) + " | " + (await page.innerText("#status")).replace(/\n/g, " | ");
await page.goto(`http://127.0.0.1:5180/?d=${preset}`);
await page.waitForTimeout(Number(process.env.PW_WAIT ?? 12000));
await page.evaluate(() => { document.querySelector("#region").value = "1"; });
const box = await page.locator("#stage").boundingBox(), cx = box.x + box.width / 2, cy = box.y + box.height / 2;
await page.mouse.click(cx, cy);
await page.waitForTimeout(Number(process.env.PW_READ ?? 25000));
console.log(el(), await status());
await page.keyboard.down("Shift");
const [dx1, dy1, dx2, dy2] = (process.env.PW_OFFSETS ?? "180,-120,-200,140").split(",").map(Number);
await page.mouse.click(cx + dx1, cy + dy1);
await page.keyboard.up("Shift");
await page.waitForTimeout(5000);
console.log(el(), "hint:", await page.innerText("#hint"));
console.log(el(), await status());
await page.click("#addex");
await page.mouse.click(cx + dx2, cy + dy2);
await page.waitForTimeout(6000);
console.log(el(), "hint:", await page.innerText("#hint"), "| adding:", await page.evaluate(() => document.querySelector("#addex").classList.contains("on")));
await page.keyboard.press("Escape");
const info = await page.evaluate(() => {
  const s = window.coresample.search, q = document.querySelector("#qcanvas");
  return { examples: s.state.examples, heat: s.heat().n, qcanvas: `${q.width}x${q.height}`, heatlo: document.querySelector("#heatlo").textContent,
           heathi: document.querySelector("#heathi").textContent, overImage: !document.querySelector("#heatimg").hidden };
});
console.log(el(), JSON.stringify(info));
await page.waitForTimeout(8000);
if (await page.isEnabled("#stop") && (await page.innerText("#stop")) === "Pause") await page.click("#stop");
await page.waitForTimeout(1500);
console.log(el(), await status());
await page.screenshot({ path: `${out}_examples.png` });
await page.locator("aside").screenshot({ path: `${out}_aside.png` });
await browser.close();
