# DocMind (browser version) - local prototype

Runs entirely in the browser: Transformers.js (ONNX/WASM) with BAAI bge-small-en-v1.5 embeddings and the ms-marco-MiniLM-L-6-v2 cross-encoder, the same models as the Python version. Extractive, cited answers with abstention. .txt, .md and text PDFs.
Roughly 85-87 MB of uncompressed model and runtime files load on first visit (models 58.6 MB from the Hugging Face CDN, runtime WASM 26.9 MB, file sizes from HEAD requests and local files, not measured network transfer; actual download varies). Versions: @huggingface/transformers 4.3.2, onnxruntime-web 1.31.0-dev.20260914-8d85527a0, pdfjs-dist 6.5.438; models Xenova/bge-small-en-v1.5 and Xenova/ms-marco-MiniLM-L-6-v2 (quantized ONNX). The runtime WASM loads from a pinned jsDelivr URL, not from this repo. PDF text supported; scanned PDFs are refused (no OCR). Documents never leave the browser.
Not deployed. Tested only in headless desktop Chrome via Playwright, on the 4-file sample corpus. No real phone, no Safari/Firefox test.
`node evalnode.mjs` runs the 40-question eval in Node; `python browsertest.py` drives the page in Chrome.

## What it is and is not
Extractive retrieval, not generative: every answer is a sentence copied verbatim from your document with its file and page, and it abstains when no sentence clearly answers. Eval: 40 self-written questions on a 4-file sample corpus (25/30 answers correct, 10/10 correct abstentions). That shows behaviour on the sample, not general accuracy. Scanned PDFs are not supported (no OCR). Not tested on a real phone, Safari or Firefox.

Transformers.js 4.3.2 and the runtime WASM load from pinned jsDelivr URLs, not from this repo.

Open index.html over http(s) (it uses ES modules), for example `python3 -m http.server`. Live demo: GitHub Pages for this repo.

## Answer confidence
Answers come from a cross-encoder trained on web text. Short fragments such as resume lines or bullets often score low, so a best match that is below the confident threshold but clearly stands out is shown with a "Low confidence" label instead of being hidden. Checked on small self-made and small realistic test documents only, so expect wrong low-confidence matches and some missed answers.
