// Usage counts with GoatCounter (no cookies, nothing personal): which datasets are opened and
// how many searches start, on the live site only. The page sends GoatCounter's counting
// request itself rather than loading its script, because the script also sends the page's
// query string, which can hold the address of your own data (?url=...). Sent: the dataset id
// (your own data: only its format), a title, whether it is an event, the screen width, and
// the referring site's domain. https://www.goatcounter.com
const CODE = "coresample";  // the site's GoatCounter code: https://coresample.goatcounter.com
const LIVE = "davidackerman.github.io";

function send(path: string, title: string, event: boolean) {
  if (location.hostname !== LIVE || navigator.webdriver) return;  // not local runs or tests
  let ref = "";
  try { ref = document.referrer ? new URL(document.referrer).origin : ""; } catch { /* none */ }
  const q = new URLSearchParams({ p: path, t: title, e: String(event), s: String(screen.width), r: ref, rnd: Math.random().toString(36).slice(2, 7) });
  const url = `https://${CODE}.goatcounter.com/count?${q}`;
  try { if (navigator.sendBeacon?.(url)) return; } catch { /* fall back */ }
  new Image().src = url;
}

/** A page view of a dataset ("/mars-ctx", or "/custom/ome-zarr" for your own). */
export function countOpen(what: string, title: string) { send(`/${what}`, title, false); }
/** An event: a search ("search/mars-ctx") or a drawing searched ("draw/mars-ctx"). */
export function countEvent(path: string) { send(path, path, true); }
