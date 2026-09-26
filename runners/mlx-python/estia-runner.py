#!/usr/bin/env python3
"""Resident MLX runner for Estia (protocol v2).

Protocol: one JSON request per line on stdin, one JSON response per line on
stdout. Models are loaded once and kept resident across requests.

Requests:
  {"type":"ping"}                                              -> {"ok": true}
  {"type":"hello"}          -> {"ok":true,"runner":"mlx-python","version":…,"protocol":2,
                                "capabilities":{…see CAPABILITIES below…}}
  {"type":"load","model_path":"...","kind":"generation"|"embedding"}
                            -> {"ok":true,"loaded":true,"ms":N}   (model resident from now on)
  {"type":"unload","model_path":"..."}   -> {"ok":true,"unloaded":bool}
  {"type":"chat","model_path":"...","messages":[{"role":"user","content":"..."},...],
   "tools":[...OpenAI tool schemas...], "cache_key":"conv-1", "format":{...},
   "max_tokens":256,"temperature":0.2}
      -> {"text":"...","meta":{"prompt_tokens":N,"cached_tokens":N,"generation_tokens":N}}
      messages are rendered through the model's own chat template (system, user,
      assistant, tool roles; tools declared natively where the template supports
      it). cache_key keeps the KV cache across turns of one conversation so only
      the new suffix is prefilled. format is accepted and ignored: this runner
      cannot constrain decoding (capabilities.structured is empty).
  {"type":"chat_stream", ...same...}
      -> token lines, then {"type":"meta",...}, then {"done": true}
  {"type":"count_tokens","model_path":"...","text":"..."} -> {"tokens":N}
  {"type":"embed_batch","model_path":"...","inputs":["a","b"]} -> {"embeddings":[[...],[...]]}
  {"type":"embed","model_path":"...","input":"a"}              -> {"embedding":[...]}
  {"type":"generate","model_path":"...","prompt":"...","max_tokens":256,"temperature":0.2} -> {"text":"..."}
  {"type":"generate_stream", ...same fields as generate...}
      -> zero or more {"type":"token","text":"..."} lines as the model decodes,
         then a terminal {"done": true}
      -> OR a single legacy {"text":"..."} line (no streaming API available /
         nothing streamed — the caller treats it as one final chunk)
  {"type":"cancel"}   -> stops the stream in flight (generate_stream or
         chat_stream); that stream ends with {"done": true, "cancelled": true}.
         Never answered on its own, so a cancel that arrives between requests
         is simply dropped. stdin is read on a thread for exactly this: the
         main thread is deep in MLX compute while a stream runs.
On error: {"error":"message"}

Embedding uses mlx-embeddings, the same API as oneshot-runner.py:
`load(path) -> (model, processor)`, then
`generate(model, processor, texts=[...]).text_embeds` — mean-pooled, normalized
sentence vectors (768-dim for nomicai-modernbert-embed-base-6bit).
"""
import sys
import collections
import json
import math
import queue
import re
import threading

# Protect the line protocol: any chatter MLX/loaders print must never land on
# the stdout we use for JSON responses. Redirect the global stdout to stderr and
# emit responses through the saved real stdout only.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def emit(obj):
    _REAL_STDOUT.write(json.dumps(obj) + "\n")
    _REAL_STDOUT.flush()


_EMBED_CACHE = {}
_GEN_CACHE = {}

# Protocol v2 identity. Bump RUNNER_VERSION on any behaviour change a client
# could care about; PROTOCOL is the dialect number from the engine's proto crate.
RUNNER_NAME = "mlx-python"
RUNNER_VERSION = "2.1.0"
PROTOCOL = 2
CAPABILITIES = {
    "generate": True,
    "stream": True,
    "embed": True,
    "cancel": True,
    "load": True,
    "chat": True,
    "tools": True,
    "prompt_cache": True,
    "count_tokens": True,
    # No constrained decoding yet: the engine validates and repairs instead.
    "structured": [],
}

# Prompt (KV) caches by (model_path, cache_key). mlx-vlm's PromptCacheState
# remembers the token ids the cache covers, finds the common prefix with the
# next prompt, trims and reuses — so a conversation that grows turn by turn
# only prefills its new suffix. Bounded: each entry is real memory.
_PROMPT_CACHES = collections.OrderedDict()
_PROMPT_CACHE_MAX = 8

# Set by the stdin reader thread when a {"type":"cancel"} arrives; checked by
# the streaming decode loop between chunks; cleared when the next request
# starts, so a late cancel never leaks into the following generation.
_CANCEL = threading.Event()

# Resolve the embeddings API once (same fallback chain as oneshot-runner.py).
try:
    from mlx_embeddings import load as _embed_load, generate as _embed_generate
except Exception:
    _embed_generate = None
    try:
        from mlx_embeddings.utils import load as _embed_load
    except Exception:
        _embed_load = None


def load_embed(path):
    if _embed_load is None:
        raise RuntimeError("mlx-embeddings is not installed or importable")
    if path not in _EMBED_CACHE:
        _EMBED_CACHE[path] = _embed_load(path)
    return _EMBED_CACHE[path]


def _finite(vec):
    # serde_json (the Rust client) rejects NaN/Infinity, so a non-finite value
    # would break the line protocol. Sanitize to 0.0 (a zero vector scores 0
    # under cosine similarity).
    return [float(x) if math.isfinite(x) else 0.0 for x in vec]


# Architectures that cannot go through `mlx_embeddings.generate(...)`.
#
# `generate()` calls the model with keyword arguments; `gemma3_text` (EmbeddingGemma)
# and `qwen3` take positional `inputs` and raise
#   TypeError: Model.__call__() got an unexpected keyword argument 'input_ids'
# The BERT family (`bert`, `modernbert`, `xlm_roberta`) goes through `generate()`
# normally. Dispatch is on the loaded model's own module name, so a model swap in
# the engine's registry (engine/src/models/embed.rs) needs no change here.
_POSITIONAL_ARCHS = ("gemma3_text", "qwen3")


def _model_arch(model):
    return type(model).__module__.split(".")[-1]


def _embed_one(model, processor, text):
    """One text -> one pooled, finite vector, whichever calling convention applies."""
    if _model_arch(model) not in _POSITIONAL_ARCHS and _embed_generate is not None:
        return _embed_generate(model, processor, texts=[text]).text_embeds.tolist()[0]

    enc = processor(
        [text], return_tensors="mlx", padding=True, truncation=True, max_length=512
    )
    ids = enc["input_ids"]
    mask = enc.get("attention_mask")
    out = model(ids, attention_mask=mask)
    # Positional-call models return the pooled sentence vector as `text_embeds`;
    # some expose it as `pooler_output` instead.
    vec = getattr(out, "text_embeds", None)
    if vec is None:
        vec = getattr(out, "pooler_output")
    vec = vec.tolist()
    # Shape is [1, dims] for a single input, but a model that already squeezed the
    # batch axis returns [dims] — accept both rather than index blindly.
    if vec and isinstance(vec[0], list):
        vec = vec[0]
    return vec


def embed_batch(model_path, inputs):
    # Embed ONE input at a time. Passing many inputs as a single padded batch
    # through the 6-bit ModernBERT model produces NaN rows (mlx-embeddings
    # padding instability on mixed-length batches). We still accept the whole
    # batch in one request (model stays resident, one round-trip) and just loop
    # internally.
    model, processor = load_embed(model_path)
    return [_finite(_embed_one(model, processor, text)) for text in inputs]


def load_gen(path):
    # Gemma 4 (e2b/e4b) are multimodal — load with mlx-vlm (text-only: no image).
    # Same loader as oneshot-runner.py.
    if path not in _GEN_CACHE:
        from mlx_vlm import load
        _GEN_CACHE[path] = load(path)
    return _GEN_CACHE[path]


def _vocab_has(processor, tok):
    t = getattr(processor, "tokenizer", processor)
    try:
        return t.convert_tokens_to_ids(tok) is not None
    except Exception:
        return False


def _manual_chat_prompt(processor, raw_prompt):
    # This Gemma 4 MLX bundle has no usable tokenizer chat_template, but exposes
    # channel/turn tokens (same as oneshot-runner.py). Raw prompts make it echo;
    # this format reliably opens a model response turn.
    if _vocab_has(processor, "<|channel>") and _vocab_has(processor, "<turn|>"):
        return f"<|channel>user\n{raw_prompt}<turn|><|channel>model\n"
    return None


# Hidden-reasoning ("thought"/"analysis") blocks run up to the NEXT channel
# marker (or end of text) — not greedily to EOS, so a trailing real answer is
# preserved. This is the fix for Gemma leaking its planning text ("The user is
# asking… I need to synthesize…") as the answer.
_FINAL_MARKER = re.compile(r"<\|channel\>\s*(?:model|assistant|final)\s*", re.IGNORECASE)
_THOUGHT_BLOCK = re.compile(
    r"<\|channel\>\s*(?:thought|analysis)\b.*?(?=<\|channel\>|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_THINK_BLOCK = re.compile(r"<\|think\>.*?(?=<\|channel\>|<\|/think\>|\Z)", re.DOTALL | re.IGNORECASE)


def _clean_gen(text, raw_prompt=""):
    text = text or ""
    # If the model split into channels, the user-facing answer is whatever
    # follows the LAST final/model/assistant marker; everything before it is
    # hidden reasoning.
    finals = list(_FINAL_MARKER.finditer(text))
    if finals:
        text = text[finals[-1].end():]
    # Drop any residual hidden-reasoning blocks (bounded, not to EOS).
    text = _THOUGHT_BLOCK.sub("", text)
    text = _THINK_BLOCK.sub("", text)
    # Strip leftover marker tokens.
    text = text.replace("<turn|>", "").replace("<|turn>", "")
    text = _FINAL_MARKER.sub("", text)
    text = text.replace("<|channel>", "").replace("<channel|>", "")
    stripped = text.strip()
    prompt = (raw_prompt or "").strip()
    if prompt and stripped.startswith(prompt):
        stripped = stripped[len(prompt):].lstrip(" \n:-")
    return stripped.strip()


def _looks_bad(text):
    # A cleaned answer that still carries channel markers, or is empty, means the
    # candidate prompt format failed — fall through to the next one.
    if not text:
        return True
    low = text.lower()
    return "<|channel>" in low or "<|think>" in low


def generate_text(req):
    from mlx_vlm import apply_chat_template
    from mlx_vlm import generate as mlx_generate
    model, processor = load_gen(req["model_path"])
    raw_prompt = req["prompt"]
    max_tokens = int(req.get("max_tokens") or 256)
    temperature = float(req.get("temperature") or 0.0)

    # Candidate prompt formats, strongest first: manual channel turn, then the
    # model-aware template (often a no-op for these bundles), then the raw prompt.
    candidates = []
    manual = _manual_chat_prompt(processor, raw_prompt)
    if manual:
        candidates.append(manual)
    try:
        candidates.append(apply_chat_template(
            processor, model.config, raw_prompt,
            add_generation_prompt=True, num_images=0, num_audios=0))
    except Exception:
        pass
    candidates.append(raw_prompt)

    def run_once(prompt):
        try:
            out = mlx_generate(model, processor, prompt=prompt, image=None,
                               max_tokens=max_tokens, temperature=temperature, verbose=False)
        except TypeError:
            out = mlx_generate(model, processor, prompt, max_tokens=max_tokens)
        return out if isinstance(out, str) else getattr(out, "text", "")

    last = ""
    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        cleaned = _clean_gen(run_once(cand), raw_prompt)
        if cleaned and not _looks_bad(cleaned):
            return cleaned
        if cleaned:
            last = cleaned
    return last


def _stream_generate_fn():
    # mlx-vlm's streaming API has moved across releases; probe both homes.
    try:
        from mlx_vlm import stream_generate as fn
        return fn
    except Exception:
        try:
            from mlx_vlm.utils import stream_generate as fn
            return fn
        except Exception:
            return None


def _cache_state(model_path, cache_key):
    if not cache_key:
        return None
    try:
        from mlx_vlm.generate.common import PromptCacheState
    except Exception:  # noqa: BLE001
        return None
    key = (model_path, str(cache_key))
    state = _PROMPT_CACHES.pop(key, None) or PromptCacheState()
    _PROMPT_CACHES[key] = state
    while len(_PROMPT_CACHES) > _PROMPT_CACHE_MAX:
        _PROMPT_CACHES.popitem(last=False)
    return state


def _tokenizer_of(processor):
    return getattr(processor, "tokenizer", processor)


def _render_messages(processor, messages, tools):
    """Render an OpenAI-style message list with the model's own chat template.

    Returns (prompt, native): native=True when the tokenizer template did the
    work (and declared tools in the model's own format); False for the manual
    Gemma-turn fallback used when a bundle has no usable template.
    """
    tok = _tokenizer_of(processor)
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    try:
        rendered = tok.apply_chat_template(messages, **kwargs)
        if isinstance(rendered, str) and rendered.strip():
            return rendered, True
    except Exception:  # noqa: BLE001
        pass
    parts = []
    system = None
    if tools:
        system = "You can call these tools by answering with a JSON object " \
                 "{\"tool_call\": {\"name\": ..., \"arguments\": {...}}}:\n" + json.dumps(tools)
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        if role == "system":
            system = (system + "\n" if system else "") + content
            continue
        if role == "assistant":
            role = "model"
        elif role == "tool":
            role = "user"
            content = "[tool result" + (f" for {m.get('tool_call_id')}" if m.get("tool_call_id") else "") + f"]\n{content}"
        if system and role == "user":
            content = f"{system}\n\n{content}"
            system = None
        parts.append(f"<|turn>{role}\n{content}<turn|>\n")
    if system:
        parts.insert(0, f"<|turn>user\n{system}<turn|>\n")
    return "".join(parts) + "<|turn>model\n", False


def _stream_core(model, processor, prompt, raw_prompt, max_tokens, temperature, cache_state):
    """Stream a rendered prompt. Emits token lines and returns (any_emitted,
    cancelled, meta, error). Anti-leak strategy: re-clean the WHOLE accumulated
    buffer each step and emit only the newly revealed cleaned suffix, holding
    back a 24-char tail so a channel marker forming at the end cannot leak.
    Keepalives keep the caller's silence deadline alive while tokens are
    filtered or the prompt is being prefilled.
    """
    stream_fn = _stream_generate_fn()
    if stream_fn is None:
        return False, False, None, "no streaming API in this mlx-vlm"

    def open_stream():
        kwargs = {"max_tokens": max_tokens, "temperature": temperature}
        if cache_state is not None:
            kwargs["prompt_cache_state"] = cache_state
        try:
            return stream_fn(model, processor, prompt, image=None, **kwargs)
        except TypeError:
            kwargs.pop("prompt_cache_state", None)
            return stream_fn(model, processor, prompt, max_tokens=max_tokens, temperature=temperature)

    holdback = 24
    raw_accum = ""
    anchor = ""
    any_emitted = False
    quiet_chunks = 0
    cancelled = False
    last = None
    try:
        for chunk in open_stream():
            if _CANCEL.is_set():
                cancelled = True
                break
            last = chunk
            piece = chunk if isinstance(chunk, str) else getattr(chunk, "text", "")
            if not piece:
                continue
            raw_accum += piece
            cleaned = _clean_gen(raw_accum, raw_prompt)
            emitted_now = False
            if not _looks_bad(cleaned):
                if not cleaned.startswith(anchor):
                    anchor = ""
                safe = max(0, len(cleaned) - holdback)
                if safe > len(anchor):
                    emit({"type": "token", "text": cleaned[len(anchor):safe]})
                    anchor = cleaned[:safe]
                    any_emitted = True
                    emitted_now = True
                    quiet_chunks = 0
            if not emitted_now:
                quiet_chunks += 1
                if quiet_chunks % 25 == 0:
                    emit({"type": "keepalive"})
    except Exception as exc:  # noqa: BLE001
        return any_emitted, _CANCEL.is_set(), None, f"stream failed: {exc}"

    final_cleaned = _clean_gen(raw_accum, raw_prompt)
    if not _looks_bad(final_cleaned):
        if not final_cleaned.startswith(anchor):
            anchor = ""
        if len(final_cleaned) > len(anchor):
            emit({"type": "token", "text": final_cleaned[len(anchor):]})
            any_emitted = True
    meta = None
    if last is not None and not isinstance(last, str):
        meta = {
            "prompt_tokens": getattr(last, "prompt_tokens", None),
            "cached_tokens": getattr(last, "cached_tokens", None),
            "generation_tokens": getattr(last, "generation_tokens", None),
        }
    return any_emitted, cancelled, meta, None


def chat_stream_lines(req):
    model, processor = load_gen(req["model_path"])
    prompt, native = _render_messages(processor, req.get("messages") or [], req.get("tools"))
    max_tokens = int(req.get("max_tokens") or 256)
    temperature = float(req.get("temperature") or 0.0)
    state = _cache_state(req["model_path"], req.get("cache_key"))
    any_emitted, cancelled, meta, error = _stream_core(model, processor, prompt, "", max_tokens, temperature, state)
    if cancelled:
        emit({"done": True, "cancelled": True})
        return
    if error and not any_emitted:
        emit({"error": error})
        return
    if error:
        emit({"error": f"stream failed mid-output: {error}"})
        return
    if meta is not None:
        meta["template"] = "native" if native else "manual"
        emit({"type": "meta", **meta})
    emit({"done": True})


def chat_text(req):
    """Non-streaming chat: same path, tokens collected instead of emitted."""
    collected = []
    real_emit = globals()["emit"]

    def capture(obj):
        if obj.get("type") == "token":
            collected.append(obj["text"])
        elif obj.get("type") in ("keepalive", "meta"):
            pass
        else:
            real_emit(obj)

    globals()["emit"] = capture
    try:
        model, processor = load_gen(req["model_path"])
        prompt, native = _render_messages(processor, req.get("messages") or [], req.get("tools"))
        state = _cache_state(req["model_path"], req.get("cache_key"))
        any_emitted, cancelled, meta, error = _stream_core(
            model, processor, prompt, "", int(req.get("max_tokens") or 256),
            float(req.get("temperature") or 0.0), state)
    finally:
        globals()["emit"] = real_emit
    if error and not collected:
        raise RuntimeError(error)
    if meta is not None:
        meta["template"] = "native" if native else "manual"
    return {"text": "".join(collected), "meta": meta}


def count_tokens(model_path, text):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
    except Exception:  # noqa: BLE001
        _, processor = load_gen(model_path)
        tok = _tokenizer_of(processor)
    return len(tok.encode(text))


def generate_stream_lines(req):
    """Token-streaming generation (multi-line response).

    Emits {"type":"token","text":...} lines as the model decodes, then a
    terminal {"done": true}. The anti-leak strategy is ported from
    oneshot-runner.py's generate_stream: re-clean the WHOLE accumulated
    buffer each step with _clean_gen (hidden thought/analysis channels are only
    strippable once their closing marker arrives) and emit only the newly
    revealed cleaned suffix — so Gemma's hidden-reasoning text never reaches
    the caller. Degrades safely: no streaming API, a pre-token failure, or an
    all-filtered stream falls back to the robust one-shot generate_text and a
    single legacy {"text": full} line.
    """
    stream_fn = _stream_generate_fn()
    if stream_fn is None:
        emit({"text": generate_text(req)})
        return

    model, processor = load_gen(req["model_path"])
    raw_prompt = req["prompt"]
    max_tokens = int(req.get("max_tokens") or 256)
    temperature = float(req.get("temperature") or 0.0)

    prompt = _manual_chat_prompt(processor, raw_prompt)
    if not prompt:
        try:
            from mlx_vlm import apply_chat_template
            prompt = apply_chat_template(
                processor, model.config, raw_prompt,
                add_generation_prompt=True, num_images=0, num_audios=0)
        except Exception:
            prompt = None
    if not prompt:
        prompt = raw_prompt

    def open_stream():
        # Signature has drifted across mlx-vlm releases — modern keyword form
        # first, then the leaner one.
        try:
            return stream_fn(model, processor, prompt, image=None,
                             max_tokens=max_tokens, temperature=temperature)
        except TypeError:
            return stream_fn(model, processor, prompt,
                             max_tokens=max_tokens, temperature=temperature)

    # Hold back the last few chars while streaming: a channel marker forming at
    # the buffer tail ("…<|chan") isn't strippable until complete, and emitting
    # it would leak marker text. 24 covers the longest marker plus slack; the
    # held tail flushes after the stream ends.
    #
    # `anchor` is the prefix of the CURRENT cleaned lineage that has already
    # been emitted — a string, not an integer offset, because _clean_gen can
    # REBASE the whole buffer (a late final-channel marker keeps only the text
    # after it; a completed prompt echo is stripped from the front). An offset
    # into the pre-rebase string would silently drop the answer's prefix; on a
    # rebase (cleaned no longer starts with anchor) we restart the lineage from
    # zero instead — the already-shown tail can't be retracted from a token
    # stream, but nothing of the real answer is ever dropped.
    #
    # Keepalives: while tokens are being filtered (hidden-channel decode) no
    # token lines flow, and the Rust caller's deadline is per-line silence —
    # emit a {"type":"keepalive"} (ignored by the reader) so a healthy child
    # isn't killed mid-think. One is also sent before each in-runner one-shot
    # fallback, which decodes from scratch.
    holdback = 24
    raw_accum = ""
    anchor = ""
    any_emitted = False
    quiet_chunks = 0
    cancelled = False
    try:
        for chunk in open_stream():
            if _CANCEL.is_set():
                # Leaving the loop stops the lazy decode; the tail below is
                # flushed so nothing already decoded is lost.
                cancelled = True
                break
            piece = chunk if isinstance(chunk, str) else getattr(chunk, "text", "")
            if not piece:
                continue
            raw_accum += piece
            cleaned = _clean_gen(raw_accum, raw_prompt)
            emitted_now = False
            if not _looks_bad(cleaned):
                if not cleaned.startswith(anchor):
                    anchor = ""  # cleaner rebased — restart the lineage
                safe = max(0, len(cleaned) - holdback)
                if safe > len(anchor):
                    emit({"type": "token", "text": cleaned[len(anchor):safe]})
                    anchor = cleaned[:safe]
                    any_emitted = True
                    emitted_now = True
                    quiet_chunks = 0
            if not emitted_now:
                quiet_chunks += 1
                if quiet_chunks % 25 == 0:
                    emit({"type": "keepalive"})
    except Exception as exc:  # noqa: BLE001
        if not any_emitted:
            # Nothing shown yet → safe to fall back without duplicating output.
            if _CANCEL.is_set():
                emit({"done": True, "cancelled": True})
                return
            emit({"type": "keepalive"})
            emit({"text": generate_text(req)})
            return
        emit({"error": f"stream failed mid-output: {exc}"})
        return

    # Flush the held-back tail from the final cleaned text.
    final_cleaned = _clean_gen(raw_accum, raw_prompt)
    if not _looks_bad(final_cleaned):
        if not final_cleaned.startswith(anchor):
            anchor = ""  # rebase at the very end — emit the whole final lineage
        if len(final_cleaned) > len(anchor):
            emit({"type": "token", "text": final_cleaned[len(anchor):]})
            any_emitted = True

    if cancelled:
        emit({"done": True, "cancelled": True})
        return

    if not any_emitted:
        # Stream yielded nothing usable (e.g. all hidden-channel) → one-shot.
        emit({"type": "keepalive"})
        emit({"text": generate_text(req)})
        return
    emit({"done": True})


def _load_by_kind(path, kind):
    if kind == "embedding":
        load_embed(path)
    elif kind == "generation":
        load_gen(path)
    else:
        raise ValueError(f"unknown model kind {kind!r} (generation|embedding)")


def _unload(path):
    dropped = False
    for cache in (_EMBED_CACHE, _GEN_CACHE):
        if path in cache:
            del cache[path]
            dropped = True
    if dropped:
        import gc
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass
    return dropped


def handle(req):
    import time as _time
    t = req.get("type")
    if t == "ping":
        return {"ok": True}
    if t == "hello":
        return {
            "ok": True,
            "runner": RUNNER_NAME,
            "version": RUNNER_VERSION,
            "protocol": PROTOCOL,
            "capabilities": CAPABILITIES,
        }
    if t == "load":
        started = _time.monotonic()
        _load_by_kind(req["model_path"], req.get("kind"))
        return {"ok": True, "loaded": True, "ms": int((_time.monotonic() - started) * 1000)}
    if t == "unload":
        return {"ok": True, "unloaded": _unload(req["model_path"])}
    if t == "chat":
        return chat_text(req)
    if t == "chat_stream":
        chat_stream_lines(req)
        return None
    if t == "count_tokens":
        return {"tokens": count_tokens(req["model_path"], req.get("text") or "")}
    if t == "embed_batch":
        return {"embeddings": embed_batch(req["model_path"], req["inputs"])}
    if t == "embed":
        return {"embedding": embed_batch(req["model_path"], [req["input"]])[0]}
    if t == "generate":
        return {"text": generate_text(req)}
    if t == "generate_stream":
        # Multi-line streaming handler emits its own lines (tokens + terminal).
        generate_stream_lines(req)
        return None
    return {"error": f"unknown type {t!r}"}


def _orphan_watchdog():
    # If the parent process dies (crash, SIGKILL, a rebuild) we get reparented to
    # PID 1. The stdin-EOF exit doesn't fire while the main thread is deep in
    # MLX compute, which is how multi-GB zombie runners survived for hours —
    # this daemon thread ends the process regardless of what the main thread
    # is doing.
    import os, threading, time

    def watch():
        while True:
            if os.getppid() == 1:
                os._exit(0)
            time.sleep(10)

    threading.Thread(target=watch, daemon=True).start()


_REQUESTS = queue.Queue()


def _stdin_reader():
    # The only reader of stdin. Cancels are acted on here, immediately, and
    # never queued or answered; everything else is handed to the main thread
    # in order. A line that is not JSON is queued as an error so the reply
    # rhythm (one response per request line) is preserved.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:  # noqa: BLE001
            _REQUESTS.put({"__bad_line__": str(e)})
            continue
        if isinstance(req, dict) and req.get("type") == "cancel":
            _CANCEL.set()
            continue
        _REQUESTS.put(req)
    _REQUESTS.put(None)


def main():
    _orphan_watchdog()
    threading.Thread(target=_stdin_reader, daemon=True).start()
    while True:
        req = _REQUESTS.get()
        if req is None:
            break
        _CANCEL.clear()
        if isinstance(req, dict) and "__bad_line__" in req:
            emit({"error": req["__bad_line__"]})
            continue
        try:
            resp = handle(req)
        except Exception as e:  # noqa: BLE001
            resp = {"error": str(e)}
        if resp is not None:
            emit(resp)


if __name__ == "__main__":
    main()
