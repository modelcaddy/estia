#!/usr/bin/env python3
"""Ask questions about a folder of notes, with citations (retrieval-augmented
generation).

What it shows:
  - embedding documents with `task: document` and questions with
    `task: query`, so the engine adds the embedding model's own prefixes
  - storing the embedding fingerprint with the index, and sending it back as
    `expect_fingerprint` so the engine refuses to mix vector spaces
  - batching under the 256-inputs-per-request limit, at `background` priority
  - ranking by cosine similarity and answering from the top passages only

Scopes: embed, generate

    export ESTIA_TOKEN=estia_...      # estia token new notes --scopes embed,generate
    python rag.py index sample-notes                 # writes rag-index.json
    python rag.py ask "When do I repot the olive tree?"

Which embeddings route, and why: this uses POST /v1/embeddings through the
OpenAI SDK, because the app already holds an SDK client for chat. Estia's
extra fields (`task`, `expect_fingerprint`, `priority`) go in `extra_body`,
and the fingerprint comes back in `x_estia`. The native POST /engine/embed
takes the same fields as plain JSON (`inputs` instead of `input`) and returns
`vectors` and `fingerprint` at the top level. Use it when you call the engine
without an SDK; the vectors are the same.

The index is a plain JSON file. That is fine for a few thousand passages; past
that, put the vectors in a vector store and keep the fingerprint next to them.
"""

import json
import math
import os
import sys
from pathlib import Path

import openai
from openai import OpenAI

URL = os.environ.get("ESTIA_URL", "http://127.0.0.1:27200").rstrip("/")
TOKEN = os.environ.get("ESTIA_TOKEN", "")
MODEL = os.environ.get("ESTIA_MODEL", "fast")
EMBED_MODEL = "embed"  # the embedding role; the engine picks the model
INDEX = Path(os.environ.get("RAG_INDEX", "rag-index.json"))
BATCH = 64  # the engine accepts at most 256 inputs per request
TOP_K = 3


def explain(e: Exception) -> str:
    if isinstance(e, openai.APIConnectionError):
        return f"cannot reach the engine at {URL}. Is `estia serve` running? Set ESTIA_URL if it listens elsewhere."
    if isinstance(e, openai.APIStatusError):
        message = e.body.get("message") if isinstance(e.body, dict) else e.message
        text = f"the engine answered {e.status_code}: {message}"
        if e.request_id:
            text += f" (request id {e.request_id})"
        if e.status_code == 401:
            text += "\nCheck ESTIA_TOKEN, or mint a token: estia token new notes --scopes embed,generate"
        return text
    return str(e)


def chunks_of(root: Path):
    """Split each .md / .txt file into passages of whole paragraphs."""
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".md", ".txt") or not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        para, start = [], None
        passages = []
        for i, line in enumerate(lines + [""], start=1):
            if line.strip():
                if start is None:
                    start = i
                para.append(line.strip())
            elif para:
                passages.append((start, " ".join(para)))
                para, start = [], None
        # Merge short neighbouring paragraphs (a heading and its first
        # paragraph, say) up to about 400 characters.
        merged = []
        for line_no, text in passages:
            if merged and len(merged[-1][1]) + len(text) < 400:
                merged[-1] = (merged[-1][0], merged[-1][1] + "\n" + text)
            else:
                merged.append((line_no, text))
        for line_no, text in merged:
            yield {"source": str(path.relative_to(root)), "line": line_no, "text": text}


def embed(client: OpenAI, texts: list, task: str, expect: str | None = None, priority: str = "interactive"):
    extra = {"task": task, "priority": priority}
    if expect:
        extra["expect_fingerprint"] = expect
    r = client.embeddings.create(model=EMBED_MODEL, input=texts, encoding_format="float", extra_body=extra)
    info = (r.model_extra or {}).get("x_estia") or {}
    return [d.embedding for d in r.data], info.get("fingerprint"), r.model


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def index(client: OpenAI, folder: str) -> int:
    root = Path(folder)
    if not root.is_dir():
        print(f"{folder} is not a folder", file=sys.stderr)
        return 2
    chunks = list(chunks_of(root))
    if not chunks:
        print(f"no .md or .txt files under {folder}", file=sys.stderr)
        return 2
    fingerprint, model_id = None, None
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        # Indexing is bulk work: background priority lets chat turns go first.
        vectors, fp, model_id = embed(client, [c["text"] for c in batch], "document", expect=fingerprint, priority="background")
        fingerprint = fingerprint or fp
        for c, v in zip(batch, vectors):
            c["vector"] = [round(x, 6) for x in v]
        print(f"embedded {min(i + BATCH, len(chunks))}/{len(chunks)} passages", file=sys.stderr)
    INDEX.write_text(json.dumps({"fingerprint": fingerprint, "model": model_id, "root": str(root), "chunks": chunks}))
    print(f"wrote {INDEX}: {len(chunks)} passages from {folder}, fingerprint {fingerprint}")
    return 0


def ask(client: OpenAI, question: str) -> int:
    if not INDEX.exists():
        print(f"no index at {INDEX}; run: python rag.py index <folder>", file=sys.stderr)
        return 2
    idx = json.loads(INDEX.read_text())
    try:
        # expect_fingerprint: if the engine now runs a different embedding
        # model or backend, it answers 422 instead of returning vectors that
        # cannot be compared with the stored ones.
        [qv], _, _ = embed(client, [question], "query", expect=idx["fingerprint"])
    except openai.UnprocessableEntityError as e:
        print(f"{explain(e)}\nThe index was built with {idx['fingerprint']}. Rebuild it: python rag.py index {idx['root']}", file=sys.stderr)
        return 3

    ranked = sorted(idx["chunks"], key=lambda c: cosine(qv, c["vector"]), reverse=True)[:TOP_K]
    sources = "\n\n".join(f"[{n}] ({c['source']}:{c['line']})\n{c['text']}" for n, c in enumerate(ranked, start=1))
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer the question using only the numbered sources. Cite the sources you used "
                "in square brackets, like [2]. If the sources do not contain the answer, say so.",
            },
            {"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=300,
    )
    print(r.choices[0].message.content.strip())
    print("\nSources:")
    for n, c in enumerate(ranked, start=1):
        print(f"  [{n}] {c['source']}:{c['line']}  (cosine {cosine(qv, c['vector']):.3f})")
    return 0


def main() -> int:
    if not TOKEN:
        print(
            "ESTIA_TOKEN is not set. On the engine's machine run\n"
            "  estia token new notes --scopes embed,generate\n"
            "and export the token it prints: export ESTIA_TOKEN=estia_...",
            file=sys.stderr,
        )
        return 2
    if len(sys.argv) < 3 or sys.argv[1] not in ("index", "ask"):
        print("usage: python rag.py index <folder>\n       python rag.py ask <question>", file=sys.stderr)
        return 2
    client = OpenAI(base_url=f"{URL}/v1", api_key=TOKEN, max_retries=0)
    try:
        if sys.argv[1] == "index":
            return index(client, sys.argv[2])
        return ask(client, " ".join(sys.argv[2:]))
    except openai.OpenAIError as e:
        print(f"error: {explain(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
