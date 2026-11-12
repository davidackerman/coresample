// Serves a folder opened from this computer at ./local/<id>/<path>, so the page's readers and
// Neuroglancer (which read data by URL, Neuroglancer in its own frame and worker) can read it like
// any host. Each request is answered by the page that opened the folder (src/local.ts), which
// reads the file where it is; nothing is uploaded. Registered as local-sw.js?coi=1 (src/local.ts
// isolate: the model on the CPU), it also adds the headers that make this site's pages and
// workers cross-origin isolated, for WebAssembly threads. Every other request passes straight through.
const PREFIX = new URL("local/", self.registration.scope).pathname;
const COI = new URL(self.location.href).searchParams.get("coi") === "1";
const owner = new Map();  // folder id -> the id of the window that opened it

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("message", (e) => {
  if (e.data?.coresampleClaim) e.waitUntil(self.clients.claim());  // a page that skipped this worker (shift-reload)
  if (e.data?.coresampleLocal && e.source) owner.set(e.data.coresampleLocal, e.source.id);
  e.ports[0]?.postMessage(true);
});
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (u.origin === location.origin && u.pathname.startsWith(PREFIX)) e.respondWith(serve(e.request, u));
  else if (COI && u.origin === location.origin && (e.request.mode === "navigate" || /^(shared)?worker$/.test(e.request.destination)))
    e.respondWith(isolated(e.request));
});

/** This site's page, frame (Neuroglancer) or worker script with COOP and COEP: credentialless
 * (other sites' resources load as before, without credentials, needing no header of theirs). */
async function isolated(req) {
  const r = await fetch(req);
  if (r.type === "opaqueredirect" || r.status === 0) return r;
  const headers = new Headers(r.headers);
  headers.set("Cross-Origin-Embedder-Policy", "credentialless");
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  return new Response(r.body, { status: r.status, statusText: r.statusText, headers });
}

/** "bytes=a-b", "bytes=a-" or "bytes=-n" (one range; anything else reads the whole file). */
function parseRange(h) {
  const m = /^bytes=(\d*)-(\d*)$/.exec(h ?? "");
  if (!m || (!m[1] && !m[2])) return null;
  if (!m[1]) return { suffix: Number(m[2]) };
  return { start: Number(m[1]), end: m[2] ? Number(m[2]) + 1 : null };
}

function ask(client, msg, ms) {
  return new Promise((res) => {
    const ch = new MessageChannel(), t = setTimeout(() => res(null), ms);
    ch.port1.onmessage = (m) => { clearTimeout(t); res(m.data); };
    client.postMessage(msg, [ch.port2]);
  });
}

async function serve(req, u) {
  const rest = decodeURIComponent(u.pathname.slice(PREFIX.length)), i = rest.indexOf("/");
  const id = i < 0 ? rest : rest.slice(0, i), path = i < 0 ? "" : rest.slice(i + 1);
  const msg = { coresampleLocal: id, path, range: parseRange(req.headers.get("range")), head: req.method === "HEAD" };
  let ans = null;
  const known = owner.has(id) ? await self.clients.get(owner.get(id)) : null;
  if (known) ans = await ask(known, msg, 240000);  // a chunk made from a remote file can take a while
  else {  // this worker restarted (it forgets): ask every window, the one with the folder answers
    const all = await self.clients.matchAll({ type: "window" });
    ans = await new Promise((res) => {
      let left = all.length;
      if (!left) res(null);
      for (const c of all) ask(c, msg, 240000).then((a) => { if (a) { owner.set(id, c.id); res(a); } else if (!--left) res(null); });
    });
  }
  if (!ans) return new Response("this folder is no longer open: open it again", { status: 410 });
  if (ans.status === 404 || ans.status === 500) return new Response(ans.error ?? null, { status: ans.status });
  const headers = { "Accept-Ranges": "bytes", "Cache-Control": "no-store", "Content-Length": String(ans.end - ans.start),
                    "Content-Type": /\.json$|(^|\/)(\.zattrs|\.zarray|\.zgroup|info|attributes\.json)$/.test(path) ? "application/json" : "application/octet-stream" };
  if (ans.status === 416) return new Response(null, { status: 416, headers: { "Content-Range": `bytes */${ans.size}` } });
  if (ans.status === 206) headers["Content-Range"] = `bytes ${ans.start}-${ans.end - 1}/${ans.size}`;
  return new Response(ans.body, { status: ans.status, headers });
}
