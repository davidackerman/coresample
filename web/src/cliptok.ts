// CLIP's byte-pair tokenizer (OpenAI's simple_tokenizer.py, as open_clip runs it for PE-Core):
// a text lower-cased, split into words, digits and runs of punctuation, each cut into the
// vocabulary's pieces by the merges in models/clip-bpe-merges.txt (the 48,894 the vocabulary is
// built from, most frequent first). Tokens: start, the text's, end, then zeros up to `context`
// (a text too long is cut, its last token the end).

export const CONTEXT = 32;  // PE-Core's text length
const START = 49406, END = 49407;

/** How each byte is written in the vocabulary (GPT-2's bytes_to_unicode): printable ones as
 * themselves, the rest as characters from 256 up; in the vocabulary's order. */
function byteChars(): { order: number[]; ch: string[] } {
  const order: number[] = [];
  for (let b = 33; b < 127; b++) order.push(b);
  for (let b = 161; b < 173; b++) order.push(b);
  for (let b = 174; b < 256; b++) order.push(b);
  const ch: string[] = new Array(256), seen = new Set(order);
  for (const b of order) ch[b] = String.fromCharCode(b);
  let n = 0;
  for (let b = 0; b < 256; b++) if (!seen.has(b)) { order.push(b); ch[b] = String.fromCharCode(256 + n++); }
  return { order, ch };
}

export class ClipTokenizer {
  private ch: string[];
  private ids = new Map<string, number>();   // piece -> token
  private rank = new Map<string, number>();  // "a b" -> how early the pair merges
  private cache = new Map<string, number[]>();

  /** `merges`: the text of clip-bpe-merges.txt (a pair a line). */
  constructor(merges: string) {
    const { order, ch } = byteChars();
    this.ch = ch;
    const vocab = order.map((b) => ch[b]);
    vocab.push(...vocab.map((v) => v + "</w>"));
    merges.split("\n").forEach((line, i) => {
      if (!line) return;
      this.rank.set(line, i);
      vocab.push(line.replace(" ", ""));
    });
    vocab.forEach((v, i) => this.ids.set(v, i));
  }

  /** One word's tokens. */
  private bpe(word: string): number[] {
    const hit = this.cache.get(word);
    if (hit) return hit;
    let parts = Array.from(new TextEncoder().encode(word), (b) => this.ch[b]);
    parts[parts.length - 1] += "</w>";
    while (parts.length > 1) {
      let best = Infinity, at = -1;
      for (let i = 0; i + 1 < parts.length; i++) {
        const r = this.rank.get(`${parts[i]} ${parts[i + 1]}`);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      const a = parts[at], b = parts[at + 1], next: string[] = [];
      for (let i = 0; i < parts.length; i++)
        if (i + 1 < parts.length && parts[i] === a && parts[i + 1] === b) { next.push(a + b); i++; } else next.push(parts[i]);
      parts = next;
    }
    const out = parts.map((p) => this.ids.get(p)!);
    this.cache.set(word, out);
    return out;
  }

  /** The text's tokens, `context` long: start, the text, end, zeros. */
  encode(text: string, context = CONTEXT): Float32Array {
    const clean = text.normalize("NFC").replace(/\s+/g, " ").trim().toLowerCase();
    const out = [START];
    for (const m of clean.matchAll(/<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+/gu))
      out.push(...(m[0] === "<start_of_text>" ? [START] : m[0] === "<end_of_text>" ? [END] : this.bpe(m[0])));
    out.push(END);
    const ids = new Float32Array(context);
    for (let i = 0; i < Math.min(context, out.length); i++) ids[i] = out[i];
    if (out.length > context) ids[context - 1] = END;
    return ids;
  }
}
