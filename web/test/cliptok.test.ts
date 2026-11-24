// The page's CLIP tokenizer against open_clip's (fixtures/clip_tokens.json). Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ClipTokenizer } from "../src/cliptok.ts";

test("texts tokenize as open_clip's tokenizer does", () => {
  const tok = new ClipTokenizer(readFileSync(new URL("../public/models/clip-bpe-merges.txt", import.meta.url), "utf8"));
  const cases = JSON.parse(readFileSync(new URL("./fixtures/clip_tokens.json", import.meta.url), "utf8")) as { text: string; ids: number[] }[];
  assert.ok(cases.length > 10);
  for (const c of cases) assert.deepEqual(Array.from(tok.encode(c.text)), c.ids, JSON.stringify(c.text));
});
