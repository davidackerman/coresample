// Drives headless Firefox over WebDriver BiDi: opens the built page on a dataset, waits for the
// first render, clicks the centre (a search), waits for hits, saves screenshots.
// usage: node test/browser_smoke.mjs <bidi ws url> <page url> <out prefix> [seconds to wait]
import { writeFileSync } from "node:fs";

const [ws, page, out, waitS = "90"] = process.argv.slice(2);
const sock = new WebSocket(ws);
let id = 0;
const pending = new Map();
sock.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  else if (m.method === "log.entryAdded") console.log("[page]", m.params.level, m.params.text?.slice(0, 300));
};
const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); sock.send(JSON.stringify({ id: i, method, params })); });
await new Promise((r) => (sock.onopen = r));
console.log("session", (await send("session.new", { capabilities: {} })).result?.sessionId ?? "?");
await send("session.subscribe", { events: ["log.entryAdded"] });
const ctxId = (await send("browsingContext.getTree")).result.contexts[0].context;
await send("browsingContext.setViewport", { context: ctxId, viewport: { width: 1500, height: 900 } });
console.log("navigate", (await send("browsingContext.navigate", { context: ctxId, url: page, wait: "complete" })).type);
const evalJs = async (expression) => (await send("script.evaluate", { expression, target: { context: ctxId }, awaitPromise: true })).result?.result?.value;
const shot = async (name) => {
  const r = await send("browsingContext.captureScreenshot", { context: ctxId });
  writeFileSync(`${out}_${name}.png`, Buffer.from(r.result.data, "base64"));
  console.log("saved", `${out}_${name}.png`);
};
const sleep = (s) => new Promise((r) => setTimeout(r, s * 1000));
await sleep(12);
await shot("view");
console.log("model:", await evalJs("document.getElementById('model').textContent"));
// click the centre of the canvas (a search)
await evalJs(`(() => { const c = document.getElementById('view'), r = c.getBoundingClientRect();
  const o = { clientX: r.left + r.width / 2, clientY: r.top + r.height / 2, pointerId: 1, bubbles: true };
  c.dispatchEvent(new PointerEvent('pointerdown', o)); c.dispatchEvent(new PointerEvent('pointerup', o)); return 1; })()`);
const t0 = Date.now();
let last = "";
while ((Date.now() - t0) / 1000 < Number(waitS)) {
  await sleep(5);
  const st = await evalJs("document.getElementById('status').innerText");
  if (st !== last) { console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s]`, st.replace(/\n/g, " | ")); last = st; }
  if (/paused|done|error/.test(st)) break;
}
console.log("hits:", await evalJs("document.querySelectorAll('.hit').length"));
await shot("search");
await send("session.end");
process.exit(0);
