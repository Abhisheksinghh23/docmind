// DocMind browser core: same pipeline as the Python version, runs in a browser or Node.
// Extractive only: every quoted sentence is copied verbatim from a source chunk.
const STOP = new Set("a an the of to in on for and or is are was were be been it its this that these those with as by at from what which who whom how many much when where why do does did can could should would will not no i you we my".split(" "));
export const MIN_RERANK = 0.0;
// A second sentence is quoted only if its score is at least this share of the best sentence's score.
// Stops loosely related filler from riding along with a strong answer. Not tuned on the eval set.
export const SECOND_RATIO = 0.5;
// The cross-encoder was trained on web passages and scores short fragments (resume lines, bullets,
// "CGPA: 8.7 / 10") below 0 even when they are the right sentence. So a sentence at or below 0 is quoted
// only as a LOW-CONFIDENCE best match, and only if it is not hopeless (LOW_FLOOR) and clearly stands out
// from every other candidate (LOW_MARGIN). Both values were chosen by looking at score distributions on the
// project's own sample questions and a few small realistic documents, so they are tuned, not independently proven.
export const LOW_FLOOR = -7;
export const LOW_MARGIN = 3.5;
const SENT = /(?<=[.!?])\s+|\n{2,}/;
const QPREFIX = "Represent this sentence for searching relevant passages: ";

// Line handling: PDFs, resumes and notes break lines without punctuation. A line break joins two lines
// only when it looks like a wrapped sentence; headings, bullets and short lines stay separate units.
const BULLET = /^\s*([-*\u2022\u25aa\u25cf]|\d+[.)])\s/;
function logicalLines(text) {
  const out = [];
  for (const para of text.split(/\n{2,}/)) {
    const lines = para.split("\n").map(l => l.replace(/\s+/g, " ").trim()).filter(Boolean);
    let cur = "";
    for (const ln of lines) {
      if (!cur) { cur = ln; continue; }
      const wrapped = cur.length >= 55 && !/[.!?:;]$/.test(cur) && !BULLET.test(ln) && (/^[a-z]/.test(ln) || cur.length >= 85);
      if (wrapped) cur += " " + ln; else { out.push(cur); cur = ln; }
    }
    if (cur) out.push(cur);
  }
  return out;
}
export function splitSentences(text) {
  text = text.replace(/^#+\s*(.+?)\s*$/gm, (_m, t) => t.replace(/\.+$/, "") + ".");
  const out = [];
  for (const line of logicalLines(text)) for (const s of line.split(/(?<=[.!?])\s+/)) { const t = s.trim(); if (t) out.push(t); }
  return out;
}
export function chunkPage(source, page, text, maxChars = 600, overlap = 1) {
  const sents = splitSentences(text), chunks = [];
  let cur = [], size = 0, n = 0;
  const flush = () => { if (cur.length) { chunks.push({ id: `${source}#p${page}-${n}`, source, page, text: cur.join(" "), sents: cur.slice() }); n++; } };
  for (const s of sents) {
    if (cur.length && size + s.length + 1 > maxChars) {
      flush(); cur = overlap ? cur.slice(-overlap) : []; size = cur.reduce((a, x) => a + x.length + 1, 0);
    }
    cur.push(s); size += s.length + 1;
  }
  if (cur.length && (!chunks.length || cur.length > overlap || overlap === 0)) flush();
  return chunks;
}
const stem = t => (t.length > 3 && t.endsWith("s") && !t.endsWith("ss")) ? t.slice(0, -1) : t;
export const tokenize = text => (text.toLowerCase().match(/[a-z0-9]+/g) || []).filter(t => !STOP.has(t)).map(stem);
const cosine = (a, b) => { let d = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; } return na && nb ? d / Math.sqrt(na * nb) : 0; };

export class Index {
  constructor(models) { this.models = models; this.chunks = []; this.vecs = new Map(); this.k1 = 1.5; this.b = 0.75; this._reindex(); }
  _reindex() {
    this.tf = this.chunks.map(c => { const m = new Map(); for (const t of tokenize(c.text)) m.set(t, (m.get(t) || 0) + 1); return m; });
    this.len = this.tf.map(m => [...m.values()].reduce((a, b) => a + b, 0));
    this.df = new Map(); for (const m of this.tf) for (const t of m.keys()) this.df.set(t, (this.df.get(t) || 0) + 1);
  }
  async add(chunks) {
    this.chunks.push(...chunks); this._reindex();
    const todo = chunks.filter(c => !this.vecs.has(c.id));
    const vs = await this.models.embedDocs(todo.map(c => c.text));
    todo.forEach((c, i) => this.vecs.set(c.id, vs[i]));
  }
  removeSource(source) { this.chunks = this.chunks.filter(c => c.source !== source); this._reindex(); }
  idf(t) { const n = this.chunks.length, df = this.df.get(t) || 0; return Math.log(1 + (n - df + 0.5) / (df + 0.5)); }
  bm25(q, k) {
    const qt = tokenize(q), n = this.chunks.length; if (!qt.length || !n) return [];
    const avg = this.len.reduce((a, b) => a + b, 0) / n || 1, out = [];
    this.tf.forEach((tf, i) => {
      let s = 0;
      for (const t of new Set(qt)) { const f = tf.get(t) || 0; if (!f) continue; s += this.idf(t) * f * (this.k1 + 1) / (f + this.k1 * (1 - this.b + this.b * this.len[i] / avg)); }
      if (s > 0) out.push([this.chunks[i], s]);
    });
    return out.sort((a, b) => b[1] - a[1]).slice(0, k);
  }
  async search(q, k = 3) {
    const lex = this.bm25(q, Math.max(k, 10)), qv = await this.models.embedQuery(q);
    const dense = this.chunks.filter(c => this.vecs.has(c.id)).map(c => [c, cosine(qv, this.vecs.get(c.id))]).sort((a, b) => b[1] - a[1]).slice(0, Math.max(k, 10));
    const fused = new Map(), by = new Map();
    for (const r of [lex, dense]) r.forEach(([c], rank) => { by.set(c.id, c); fused.set(c.id, (fused.get(c.id) || 0) + 1 / (60 + rank + 1)); });
    return [...fused].sort((a, b) => b[1] - a[1]).slice(0, k).map(([id]) => by.get(id));
  }
}
// A low-confidence line must contain a question word that is not just the document's own topic word
// (a word from its file name, e.g. "invoice" in invoice.pdf), otherwise it is only topically related.
function sharesSpecificTerm(qTerms, sentence, source) {
  const topic = new Set(tokenize(String(source).replace(/\.[a-z0-9]+$/i, "")));
  const st = new Set(tokenize(sentence));
  for (const t of qTerms) if (!topic.has(t) && st.has(t)) return true;
  return false;
}
export async function answer(index, question, k = 3, maxSentences = 2) {
  const qTerms = new Set(tokenize(question));
  const hits = await index.search(question, k);
  if (!hits.length || !qTerms.size) return { question, answer: null, abstained: true, citations: [] };
  const qv = await index.models.embedQuery(question), cands = [];
  for (const chunk of hits) {
    const sents = chunk.sents || splitSentences(chunk.text), vs = await index.models.embedDocs(sents);
    sents.forEach((s, i) => {
      const toks = new Set(tokenize(s)); let ov = 0; for (const t of toks) if (qTerms.has(t)) ov++;
      const sim = cosine(qv, vs[i]); cands.push([sim + 0.15 * ov / qTerms.size, s, chunk]);
    });
  }
  cands.sort((a, b) => b[0] - a[0]);
  const uniq = [], seen = new Set();
  for (const [, s, c] of cands.slice(0, 8)) if (!seen.has(s)) { seen.add(s); uniq.push([s, c]); }
  const scores = await index.models.rerank(question, uniq.map(u => u[0]));
  const ranked = uniq.map((u, i) => [scores[i], ...u]).sort((a, b) => b[0] - a[0]);
  let good = ranked.filter(x => x[0] > MIN_RERANK), lowConfidence = false;
  if (good.length) {
    const best = good[0][0];
    good = good.filter(x => x[0] >= SECOND_RATIO * best).slice(0, maxSentences);
  } else if (ranked.length > 1 && ranked[0][0] > LOW_FLOOR && ranked[0][0] - ranked[1][0] >= LOW_MARGIN
             && sharesSpecificTerm(qTerms, ranked[0][1], ranked[0][2].source)) {
    good = [ranked[0]]; lowConfidence = true;
  }
  if (!good.length) return { question, answer: null, abstained: true, citations: [] };
  const citations = good.map(([sc, s, c], i) => ({ n: i + 1, source: c.source, page: c.page, chunkId: c.id, quote: s, score: +sc.toFixed(3) }));
  return { question, answer: citations.map(c => `${c.quote} [${c.n}]`).join(" "), abstained: false, lowConfidence, citations };
}

// Models via Transformers.js (ONNX). `T` is the imported transformers module.
export async function loadModels(T, onProgress) {
  const opt = { progress_callback: onProgress };
  const emb = await T.pipeline("feature-extraction", "Xenova/bge-small-en-v1.5", opt);
  const tok = await T.AutoTokenizer.from_pretrained("Xenova/ms-marco-MiniLM-L-6-v2", opt);
  const ce = await T.AutoModelForSequenceClassification.from_pretrained("Xenova/ms-marco-MiniLM-L-6-v2", opt);
  const embed = async texts => { const out = []; for (let i = 0; i < texts.length; i += 16) { const t = await emb(texts.slice(i, i + 16), { pooling: "cls", normalize: true }); out.push(...t.tolist()); } return out; };
  return {
    embedDocs: embed,
    embedQuery: async q => (await embed([QPREFIX + q]))[0],
    rerank: async (q, texts) => {
      const out = [];
      for (let i = 0; i < texts.length; i += 8) {
        const part = texts.slice(i, i + 8);
        const inp = tok(new Array(part.length).fill(q), { text_pair: part, padding: true, truncation: true });
        const { logits } = await ce(inp); out.push(...logits.tolist().map(r => r[0]));
      }
      return out;
    },
  };
}

// PDF text per page via pdf.js. Returns {pages:[{page,text}], emptyPages:[n], scanned:boolean}.
export async function extractPdf(pdfjs, data) {
  const doc = await pdfjs.getDocument({ data }).promise, pages = [], emptyPages = [];
  for (let n = 1; n <= doc.numPages; n++) {
    const tc = await (await doc.getPage(n)).getTextContent();
    let lastY = null, buf = "";
    for (const i of tc.items) { const y = i.transform ? i.transform[5] : null; if (buf && lastY !== null && y !== null && Math.abs(y - lastY) > 2 && !buf.endsWith("\n")) buf += "\n"; buf += i.str + (i.hasEOL ? "\n" : " "); if (y !== null && i.str.trim()) lastY = y; }
    const text = buf.replace(/[ \t]+/g, " ").replace(/ ?\n ?/g, "\n").trim();
    if (text.replace(/\s/g, "").length >= 10) pages.push({ page: n, text }); else emptyPages.push(n);
  }
  return { pages, emptyPages, scanned: pages.length === 0, numPages: doc.numPages };
}
