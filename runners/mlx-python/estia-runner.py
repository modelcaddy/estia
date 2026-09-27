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
      -> {"text":"...","meta":{"prompt_tokens":N,"cached_tokens":N,"generation_tokens":N,
                                "generation_tps":X,"finish_reason":"stop"|"length"|"tool_calls"}}
      messages are rendered through the model's own chat template (system, user,
      assistant, tool roles; tools declared natively where the template supports
      it). An assistant turn's tool_calls (OpenAI shape, arguments as a JSON
      string) and the tool results that answer them (role "tool" with
      tool_call_id) go through the template too, so the model sees its own calls
      and their results. cache_key keeps the KV cache across turns of one
      conversation so only the new suffix is prefilled (see _Conversation for
      how that survives sliding-window layers and an exact repeat). format is
      accepted and ignored: this runner cannot constrain decoding
      (capabilities.structured is empty). Tool calls stay in the text for the
      client to parse (capabilities.parses_tool_calls is false); finish_reason
      is "tool_calls" when the text holds a call (as server/src/toolcalls.rs
      recognises one), "length" when generation stopped at max_tokens, "stop"
      otherwise. A cancelled stream sends no meta.
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
         main thread is deep in MLX compute while a stream runs. The prompt is
         prefilled in chunks sized to take about a quarter of a second, and
         the cancel is checked between them, so a cancel during a long
         prefill lands within one chunk. Streams send keepalive lines while
         a long prefill runs.
On error: {"error":"message"}

Embedding uses mlx-embeddings, the same API as oneshot-runner.py:
`load(path) -> (model, processor)`, then
`generate(model, processor, texts=[...]).text_embeds` — mean-pooled, normalized
sentence vectors (768-dim for nomicai-modernbert-embed-base-6bit).
"""
import os
import sys
import collections
import json
import math
import queue
import re
import threading
import time

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
RUNNER_VERSION = "2.5.0"
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
    # Tool calls are left in the text; the client parses them.
    "parses_tool_calls": False,
    # Messages may carry images (base64); the vision tower reads them.
    "images": True,
    # Part of every embedding fingerprint this runner's vectors carry.
    "backend": "mlx-python",
}

# Prompt (KV) caches by (model_path, cache_key): one _Conversation each. The
# runner finds the common prefix with the next prompt, trims or restores, and
# prefills only the new suffix. Bounded: each entry is real memory.
_PROMPT_CACHES = collections.OrderedDict()
_PROMPT_CACHE_MAX = 8

# Set by the stdin reader thread when a {"type":"cancel"} arrives; checked
# between prefill chunks and by the streaming decode loop between tokens;
# cleared when the next request starts, so a late cancel never leaks into the
# following generation.
_CANCEL = threading.Event()

# Prefill runs in chunks so a cancel can land between them. A chunk is sized
# to take about _PREFILL_TARGET_S on the model at hand (measured chunk by
# chunk, remembered per model): 0.25 s keeps a cancel well inside half a
# second. Sizes are multiples of 32: on the 4-bit 12B (M1 Pro, 2026-09-27) a
# 48- or 80-token chunk prefilled at 64-75 tok/s against 85-93 for 32, 64,
# 96 or 128, and 16 halved it. 32 is the floor (the 12B lands there, about
# 0.35 s a chunk, some 5-8% slower than 2048-token chunks; E2B loses
# nothing), 2048 the ceiling (mlx-vlm's own step). A model's first chunk is
# the floor, so even the first request on the 12B cancels inside half a
# second; the size grows from the next chunk on.
_PREFILL_TARGET_S = 0.25
_PREFILL_ALIGN = 32
_PREFILL_MIN = 32
_PREFILL_MAX = 2048
_PREFILL_FIRST = 32
_PREFILL_STEP = {}
# A stream sends a keepalive at least this often while it prefills.
_PREFILL_KEEPALIVE_S = 2.0

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
        _apply_memory_governor()
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


_GOVERNED = False


def _env_bytes(name):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        print(f"estia-runner: {name}={raw!r} is not a number of bytes; ignored", file=sys.stderr, flush=True)
        return None
    return n if n > 0 else None


def _apply_memory_governor():
    """Apply the engine's memory caps, once, before the first model loads.

    ESTIA_MLX_WIRED_LIMIT_BYTES caps wired memory. mlx-vlm asks for
    max_recommended_working_set_size (two thirds of RAM) around every
    generation; wired pages cannot be compressed or paged, and on a 16 GB
    fanless Mac 10.7 GB of them stalled the whole machine. mlx-vlm calls
    mx.set_wired_limit through the module at call time, so wrapping it there
    caps every caller, whichever mlx-vlm module it lives in.

    ESTIA_MLX_CACHE_LIMIT_BYTES caps the freed buffers MLX keeps for reuse,
    memory the OS sees as used but no model needs."""
    global _GOVERNED
    if _GOVERNED:
        return
    _GOVERNED = True
    import mlx.core as mx
    wired = _env_bytes("ESTIA_MLX_WIRED_LIMIT_BYTES")
    cache = _env_bytes("ESTIA_MLX_CACHE_LIMIT_BYTES")
    if cache is not None:
        mx.set_cache_limit(cache)
    if wired is not None:
        original = mx.set_wired_limit

        def capped(limit):
            return original(min(int(limit), wired))

        mx.set_wired_limit = capped
    print(f"estia-runner: memory caps: wired {wired or 'mlx default'} bytes, cache {cache or 'mlx default'} bytes",
          file=sys.stderr, flush=True)


def _clear_after_request():
    """On a tight cache cap, hand freed buffers back after every request
    rather than when the cap is reached."""
    limit = _env_bytes("ESTIA_MLX_CACHE_LIMIT_BYTES")
    if limit is None or limit > 512 * 1024 * 1024 or not _GOVERNED:
        return
    import mlx.core as mx
    mx.clear_cache()


def load_gen(path):
    # Gemma 4 is multimodal. Loaded lazily, then only the language model is
    # read in: the audio and vision towers (about a quarter of E2B's weights)
    # stay on disk until an image needs them.
    if path not in _GEN_CACHE:
        _apply_memory_governor()
        import mlx.core as mx
        from mlx_vlm import load
        model, processor = load(path, lazy=True)
        lm = getattr(model, "language_model", None)
        mx.eval((lm if lm is not None else model).parameters())
        _GEN_CACHE[path] = (model, processor)
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


class _Cancelled(Exception):
    """A cancel arrived while the prompt was being prefilled."""


class _Conversation:
    """One conversation's KV cache, the token ids it holds, and a snapshot.

    Invariant: every layer cache in `cache` holds exactly `ids` (their common
    offset is len(ids)). Kept up to date after every generation, including a
    cancelled one, so a retry reuses whatever was already prefilled.

    Why not mlx-vlm's PromptCacheState: it can only trim a cache back to the
    shared prefix while every layer still holds the whole sequence, and
    Gemma 4's sliding-window layers (RotatingKVCache, 512 tokens on E2B/E4B,
    1024 on the 12B) stop doing so once the conversation passes the window.
    A chat's next prompt rarely extends the cached tokens exactly: the 12B's
    generation prompt ends in an empty thought channel that the chat template
    drops from history, so turn 2 diverges a few tokens before turn 1's prompt
    ended. Past the window that meant no reuse at all (0 cached tokens on the
    12B once a conversation passed ~1K tokens). It also refuses a prefix as
    long as the whole prompt, so an exact repeat got 0 cached tokens.

    So the runner keeps the ids itself and takes a snapshot of the
    sliding-window layers at the end of the conversation's history (the
    prompt without its generation prompt): the point every later turn, and a
    regenerate, shares. Full-attention layers are trimmed back to it; the
    sliding ones are put back from the snapshot. A snapshot holds at most one
    window per sliding layer (about 330 MB on the 12B, a few MB on E2B).
    `snap` is (token ids it covers, per-layer states) or None.
    """

    __slots__ = ("cache", "ids", "snap")

    def __init__(self):
        self.cache = None
        self.ids = []
        self.snap = None

    def reset(self):
        self.cache = None
        self.ids = []


def _conversation(model_path, cache_key):
    if not cache_key:
        return None
    key = (model_path, str(cache_key))
    conv = _PROMPT_CACHES.pop(key, None) or _Conversation()
    _PROMPT_CACHES[key] = conv
    while len(_PROMPT_CACHES) > _PROMPT_CACHE_MAX:
        _PROMPT_CACHES.popitem(last=False)
    return conv


def _rotating_cls():
    try:
        from mlx_vlm.models.cache import RotatingKVCache
        return RotatingKVCache
    except Exception:  # noqa: BLE001
        return None


def _cache_len(cache):
    """The number of tokens every layer holds, or None when they disagree."""
    offsets = {int(getattr(c, "offset", -1)) for c in cache}
    if len(offsets) != 1:
        return None
    n = offsets.pop()
    return n if n >= 0 else None


def _trimmable(cache, n):
    """Can every layer drop its last n tokens and still be exact?"""
    if n == 0:
        return True
    for c in cache:
        # RotatingKVCache answers False once it has wrapped; KVCache always True.
        if not getattr(c, "is_trimmable", lambda: False)():
            return False
        if int(getattr(c, "start_position", 0) or 0) != 0:
            return False
    return True


def _snapshot(cache):
    """Per-layer state to rebuild `cache` at its current length, or None.

    Sliding-window layers keep their last window (the only part attention at
    later positions can see), as fresh array objects: MLX copies a buffer on
    write when another array still refers to it, so later in-place updates of
    the live cache never reach the snapshot. Full-attention layers need
    nothing: they are trimmed back on restore."""
    Rot = _rotating_cls()
    states = []
    rotating = False
    for c in cache:
        if Rot is not None and isinstance(c, Rot):
            rotating = True
            if c.keys is None:
                states.append((None, None, int(c.offset), int(c._idx)))
                continue
            k, v, idx = c.keys, c.values, int(c._idx)
            if c.keep == 0 and idx == k.shape[2] and k.shape[2] > c.max_size:
                # Temporal order (a chunked prefill leaves it so): keep the
                # last max_size tokens, a full ring with its write index at
                # the end, exactly what an in-place update expects.
                k, v, idx = k[..., -c.max_size:, :], v[..., -c.max_size:, :], c.max_size
            else:
                k, v = k[...], v[...]
            states.append((k, v, int(c.offset), idx))
        elif getattr(c, "is_trimmable", lambda: False)() and hasattr(c, "trim"):
            states.append(None)
        else:
            return None
    return states if rotating else None


def _restore(cache, n, states):
    for c, s in zip(cache, states):
        if s is None:
            c.trim(int(c.offset) - n)
        else:
            k, v, off, idx = s
            c.keys = None if k is None else k[...]
            c.values = None if v is None else v[...]
            c.offset = off
            c._idx = idx


def _reuse(conv, ids, limit):
    """Bring conv.cache back to the longest prefix of `ids` it can serve, up
    to `limit` tokens (the start of the prompt's tail, always < len(ids), so
    an exact repeat reuses everything before its tail).

    Returns that length (0 = start cold)."""
    cache = conv.cache
    if cache is None:
        return 0
    held = _cache_len(cache)
    if held is None or held != len(conv.ids):
        conv.reset()
        return 0
    p = 0
    for a, b in zip(conv.ids, ids[:limit]):
        if a != b:
            break
        p += 1
    if p > 0 and _trimmable(cache, held - p):
        if held - p:
            for c in cache:
                c.trim(held - p)
        return p
    if conv.snap is not None:
        snap_ids, states = conv.snap
        n = len(snap_ids)
        # The full-attention layers must hold the snapshot's tokens too, which
        # they do when the cached ids share them (p >= n).
        if 0 < n <= p and ids[:n] == snap_ids and len(states) == len(cache):
            _restore(cache, n, states)
            return n
    return 0


def _settle(conv, cache, seq):
    """Record what `cache` holds after a generation, finished or not."""
    held = _cache_len(cache)
    if held is None or held > len(seq):
        conv.reset()
        return
    conv.cache = cache
    conv.ids = list(seq[:held])


def _encode(model, processor, prompt):
    """Token ids exactly as mlx-vlm's stream_generate would compute them."""
    from mlx_vlm.generate import dispatch as D
    add = D.should_add_special_tokens(model.config.model_type, processor)
    inputs = D.prepare_inputs(
        processor, prompts=prompt, add_special_tokens=add,
        image_token_index=getattr(model.config, "image_token_index", None))
    return inputs["input_ids"].flatten().tolist()


def _prefill(model, model_path, cache, ids, start, stop, snap_at=None, on_snapshot=None):
    """Feed ids[start:stop] into `cache` in chunks, checking for a cancel
    between them (raises _Cancelled). The same calls mlx-vlm's own chunked
    prefill makes (generate_step): embeddings, the language model over the
    chunk with the cache, then evaluate only the caches — which also skips
    the KV-shared layers of E2B/E4B, whose outputs no cache needs. A chunk
    boundary falls on `snap_at`, where `on_snapshot` is called."""
    import mlx.core as mx
    from mlx_vlm.generate import generation_stream, wired_limit

    lm = model.language_model
    extra_kw = {"logits_to_keep": 1} if getattr(lm, "supports_logits_to_keep", False) else {}
    step = _PREFILL_STEP.get(model_path, _PREFILL_FIRST)
    pos = start
    last_line = time.monotonic()
    with wired_limit(model, [generation_stream]):
        while pos < stop:
            if _CANCEL.is_set():
                raise _Cancelled()
            end = min(stop, pos + step)
            if snap_at is not None and pos < snap_at < end:
                end = snap_at
            chunk = mx.array([ids[pos:end]])
            started = time.perf_counter()
            with mx.stream(generation_stream):
                emb = model.get_input_embeddings(chunk, None, mask=None)
                kw = {k: v for k, v in emb.to_dict().items() if k != "inputs_embeds" and v is not None}
                kw.update(extra_kw)
                lm(inputs=chunk, inputs_embeds=emb.inputs_embeds, cache=cache, n_to_process=end - pos, **kw)
                mx.eval([c.state for c in cache])
            took = time.perf_counter() - started
            done = end - pos
            pos = end
            if pos == snap_at and on_snapshot is not None:
                on_snapshot(pos)
            if done == step and took > 0:
                # Resize from full chunks only: a short one (cut at the
                # snapshot point or the end) says little about the rate.
                step = int(done * _PREFILL_TARGET_S / took) // _PREFILL_ALIGN * _PREFILL_ALIGN
                step = max(_PREFILL_MIN, min(_PREFILL_MAX, step))
            mx.clear_cache()
            if time.monotonic() - last_line >= _PREFILL_KEEPALIVE_S:
                emit({"type": "keepalive"})
                last_line = time.monotonic()
    _PREFILL_STEP[model_path] = step


def _generate(model, model_path, processor, prompt, max_tokens, temperature, info, conv=None, history=None,
              images=None):
    """Yield mlx-vlm GenerationResults for `prompt`, with prefix reuse from
    `conv` and an interruptible prefill. Fills `info` with prompt_tokens and
    cached_tokens (the whole prompt and its reused part). `history` is the
    prompt without its generation prompt: where the conversation's snapshot
    is taken. Raises _Cancelled for a cancel during prefill. Close the
    generator when stopping early: that records what the cache holds."""
    import mlx.core as mx
    from mlx_vlm.models import cache as cache_mod

    stream_fn = _stream_generate_fn()
    if stream_fn is None:
        raise RuntimeError("no streaming API in this mlx-vlm")
    if images:
        # A turn with images goes through mlx-vlm whole: the vision tower
        # turns each image into embeddings that the text cache cannot hold,
        # so there is no prefix reuse and no chunked prefill (a cancel lands
        # once the prompt is read, when the first token is due).
        if conv is not None:
            conv.reset()
            conv.snap = None
        info["cached_tokens"] = 0
        yield from stream_fn(model, processor, prompt, image=list(images), max_tokens=max_tokens,
                             temperature=temperature)
        return
    try:
        ids = _encode(model, processor, prompt)
        if not ids:
            raise ValueError("empty prompt")
    except Exception as exc:  # noqa: BLE001
        # Tokenising our own way failed: hand the whole prompt to mlx-vlm as
        # before (no reuse, prefill not interruptible).
        print(f"estia-runner: own prefill unavailable ({exc}); plain stream", file=sys.stderr, flush=True)
        if conv is not None:
            conv.reset()
        yield from stream_fn(model, processor, prompt, image=None, max_tokens=max_tokens, temperature=temperature)
        return
    info["prompt_tokens"] = len(ids)

    # The tail: the part of the prompt the final call feeds, which decodes
    # from it. For a chat it is the generation prompt (everything after the
    # history), otherwise the last token. It starts at a point fixed by the
    # prompt alone, where the conversation also keeps its snapshot, so a
    # regenerate rebuilds exactly the cache its first run had there and
    # feeds the same tail: at temperature 0 it reproduces the first answer.
    tail = len(ids) - 1
    if history:
        try:
            hist_ids = _encode(model, processor, history)
            if 0 < len(hist_ids) < len(ids) and ids[:len(hist_ids)] == hist_ids:
                tail = len(hist_ids)
        except Exception:  # noqa: BLE001
            pass

    reuse = _reuse(conv, ids, tail) if conv is not None else 0
    cache = conv.cache if (conv is not None and reuse) else None
    if cache is None:
        reuse = 0
        if conv is not None:
            # Nothing to reuse: free the old cache and snapshot before
            # building the new one, rather than holding both.
            conv.reset()
            conv.snap = None
        cache = cache_mod.make_prompt_cache(model.language_model)
    info["cached_tokens"] = reuse
    snap_at = tail if (conv is not None and reuse < tail) else None

    def take_snapshot(n):
        states = _snapshot(cache)
        if states is not None:
            conv.snap = (list(ids[:n]), states)

    generated = []
    stream = None
    try:
        try:
            _prefill(model, model_path, cache, ids, reuse, tail, snap_at,
                     take_snapshot if snap_at is not None else None)
        except _Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            # This model does not take the chunked calls: prefill the whole
            # prompt through mlx-vlm on a fresh cache, as before.
            print(f"estia-runner: chunked prefill failed ({exc}); plain stream", file=sys.stderr, flush=True)
            if conv is not None:
                conv.reset()
                conv.snap = None
            conv = None
            info["cached_tokens"] = 0
            yield from stream_fn(model, processor, prompt, image=None, max_tokens=max_tokens, temperature=temperature)
            return
        if _CANCEL.is_set():
            raise _Cancelled()
        stream = stream_fn(model, processor, prompt, image=None, input_ids=mx.array([ids[tail:]]),
                           prompt_cache=cache, max_tokens=max_tokens, temperature=temperature)
        for r in stream:
            # Every token handed out has already been fed to the model (mlx-vlm
            # computes the next step before yielding), and so has a stop
            # token, which only the final result carries. A "length" final
            # result repeats the last token.
            reason = getattr(r, "finish_reason", None)
            token = getattr(r, "token", None)
            if token is not None and (reason is None or reason == "stop"):
                generated.append(int(token))
            yield r
    finally:
        if stream is not None:
            stream.close()
        if conv is not None:
            _settle(conv, cache, ids + generated)


def _tokenizer_of(processor):
    return getattr(processor, "tokenizer", processor)


def _call_arguments(raw):
    """A tool call's arguments as a dict when they are a JSON object.

    OpenAI sends them as a JSON string; chat templates (Gemma 4's among them)
    render a mapping in the model's own call syntax but print a string
    verbatim, which would show the model a call it never makes. Anything that
    is not a JSON object is kept as it came."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return raw
        return parsed if isinstance(parsed, dict) else raw
    return {} if raw is None else raw


def _prepare_image(data_b64, index):
    """One message image as a file mlx-vlm can read: decoded, converted to
    RGB, and cropped to its content when the content is a small island on a
    plain ground (a rendered page, a screenshot with margins). Measured on
    Gemma 4 E4B: a 1200 px page with four lines of text at the top read as
    "no picture" about half the time at temperature 0.2; cropped to the text
    it read every time. Returns the path of a temporary PNG."""
    import base64
    import io
    import tempfile
    from PIL import Image, ImageChops

    raw = base64.b64decode(data_b64)
    with Image.open(io.BytesIO(raw)) as im:
        if getattr(im, "is_animated", False):
            im.seek(0)  # a GIF: its first frame
        rgb = im.convert("RGB")
    w, h = rgb.size
    try:
        corner = rgb.getpixel((0, 0))
        bbox = ImageChops.difference(rgb, Image.new("RGB", rgb.size, corner)).getbbox()
        if bbox:
            left, top, right, bottom = bbox
            if (right - left) * (bottom - top) < 0.6 * w * h:
                pad = max(24, int(0.04 * max(w, h)))
                rgb = rgb.crop((max(0, left - pad), max(0, top - pad), min(w, right + pad), min(h, bottom + pad)))
    except Exception:  # noqa: BLE001
        pass  # cropping is an improvement, never a requirement
    out = tempfile.NamedTemporaryFile(prefix=f"estia-image-{index}-", suffix=".png", delete=False)
    out.close()
    rgb.save(out.name, "PNG")
    return out.name


def _message_images(messages):
    """Every image in the conversation, in order, prepared as files. The
    caller deletes them (see _remove_files)."""
    paths = []
    try:
        for m in messages:
            for img in m.get("images") or []:
                paths.append(_prepare_image(img["data"], len(paths)))
    except Exception:
        _remove_files(paths)
        raise
    return paths


def _remove_files(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _template_messages(messages):
    """Messages as chat templates expect them: content always a string,
    assistant tool_calls OpenAI-shaped with their arguments as a mapping, and
    tool results kept with the tool_call_id the template uses to name the
    function they answer."""
    out = []
    for i, m in enumerate(messages):
        msg = {"role": m.get("role", "user"), "content": m.get("content") or ""}
        images = m.get("images") or []
        if images:
            # Content parts, images first: the template writes each image
            # part as <|image|>, which mlx-vlm expands into the image's tokens.
            parts = [{"type": "image"} for _ in images]
            if msg["content"]:
                parts.append({"type": "text", "text": msg["content"]})
            msg["content"] = parts
        for key in ("name", "tool_call_id"):
            if m.get(key):
                msg[key] = m[key]
        calls = []
        for j, c in enumerate(m.get("tool_calls") or []):
            f = c.get("function") or {}
            calls.append({
                "id": c.get("id") or f"call_{i}_{j}",
                "type": "function",
                "function": {"name": f.get("name", ""), "arguments": _call_arguments(f.get("arguments"))},
            })
        if calls:
            msg["tool_calls"] = calls
        out.append(msg)
    return out


def _call_names(messages):
    """tool_call_id -> function name, from the assistant turns' tool_calls."""
    names = {}
    for m in messages:
        for c in m.get("tool_calls") or []:
            if c.get("id"):
                names[c["id"]] = (c.get("function") or {}).get("name", "")
    return names


def _render_messages(processor, messages, tools):
    """Render an OpenAI-style message list with the model's own chat template.

    Returns (prompt, native, history): native=True when the tokenizer template
    did the work (and declared tools in the model's own format); False for the
    manual Gemma-turn fallback used when a bundle has no usable template.
    history is the same messages without the generation prompt (None when it
    cannot be rendered): the prefix every later turn of the conversation
    shares, where the prompt cache takes its snapshot.
    """
    tok = _tokenizer_of(processor)
    messages = _template_messages(messages)
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    try:
        rendered = tok.apply_chat_template(messages, **kwargs)
        if isinstance(rendered, str) and rendered.strip():
            try:
                history = tok.apply_chat_template(messages, **{**kwargs, "add_generation_prompt": False})
                if not isinstance(history, str):
                    history = None
            except Exception:  # noqa: BLE001
                history = None
            return rendered, True, history
    except Exception:  # noqa: BLE001
        pass
    parts = []
    system = None
    if tools:
        system = "You can call these tools by answering with a JSON object " \
                 "{\"tool_call\": {\"name\": ..., \"arguments\": {...}}}:\n" + json.dumps(tools)
    names = _call_names(messages)
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        if isinstance(content, list):
            # Image parts from _template_messages: Gemma's image placeholder,
            # then the text.
            content = "".join("<|image|>" if p.get("type") == "image" else p.get("text", "") for p in content)
        if role == "system":
            system = (system + "\n" if system else "") + content
            continue
        if role == "assistant":
            role = "model"
            # The call in the same JSON shape the fallback asks the model for.
            for c in m.get("tool_calls") or []:
                call = {"tool_call": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
                content = (content + "\n" if content else "") + json.dumps(call)
        elif role == "tool":
            role = "user"
            call_id = m.get("tool_call_id")
            name = m.get("name") or names.get(call_id)
            label = " ".join(x for x in (name, f"({call_id})" if call_id else None) if x)
            content = "[tool result" + (f" for {label}" if label else "") + f"]\n{content}"
        if system and role == "user":
            content = f"{system}\n\n{content}"
            system = None
        parts.append(f"<|turn>{role}\n{content}<turn|>\n")
    if system:
        parts.insert(0, f"<|turn>user\n{system}<turn|>\n")
    history = "".join(parts)
    return history + "<|turn>model\n", False, history


# A tool call in the raw model text, as server/src/toolcalls.rs recognises
# one: Gemma 4's native `<|tool_call>call:name{…}<tool_call|>` (like the
# server, a missing closing marker still counts), or the manual fallback's
# `{"tool_call": {"name": …, "arguments": {…}}}`.
_NATIVE_CALL = re.compile(r"<\|tool_call\>\s*call:\s*[^\s{<]")
_MANUAL_CALL = re.compile(r"\{\s*\"tool_call\"\s*:")


def _has_tool_call(text):
    if not text:
        return False
    if _NATIVE_CALL.search(text):
        return True
    decoder = json.JSONDecoder()
    for m in _MANUAL_CALL.finditer(text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except ValueError:
            continue
        call = obj.get("tool_call") if isinstance(obj, dict) else None
        if isinstance(call, dict) and call.get("name"):
            return True
    return False


def _finish_reason(last, raw_text):
    """OpenAI's finish_reason for a finished generation: "tool_calls" when the
    raw text holds a call, "length" when decoding stopped at max_tokens,
    otherwise "stop" (a stop token)."""
    if _has_tool_call(raw_text):
        return "tool_calls"
    return "length" if getattr(last, "finish_reason", None) == "length" else "stop"


def _stream_core(model, model_path, processor, prompt, raw_prompt, max_tokens, temperature, conv, history=None,
                 images=None):
    """Stream a rendered prompt. Emits token lines and returns (any_emitted,
    cancelled, meta, error). Anti-leak strategy: re-clean the WHOLE accumulated
    buffer each step and emit only the newly revealed cleaned suffix, holding
    back a 24-char tail so a channel marker forming at the end cannot leak.
    Keepalives keep the caller's silence deadline alive while tokens are
    filtered or the prompt is being prefilled. `conv` is the conversation's
    prompt cache (or None); see _generate.
    """
    if _stream_generate_fn() is None:
        return False, False, None, "no streaming API in this mlx-vlm"

    holdback = 24
    raw_accum = ""
    anchor = ""
    any_emitted = False
    quiet_chunks = 0
    cancelled = False
    last = None
    info = {}
    gen = _generate(model, model_path, processor, prompt, max_tokens, temperature, info, conv, history, images)
    try:
        for chunk in gen:
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
    except _Cancelled:
        return any_emitted, True, None, None
    except Exception as exc:  # noqa: BLE001
        return any_emitted, _CANCEL.is_set(), None, f"stream failed: {exc}"
    finally:
        # Records what the prompt cache holds, also after a cancel.
        gen.close()

    final_cleaned = _clean_gen(raw_accum, raw_prompt)
    if not _looks_bad(final_cleaned):
        if not final_cleaned.startswith(anchor):
            anchor = ""
        if len(final_cleaned) > len(anchor):
            emit({"type": "token", "text": final_cleaned[len(anchor):]})
            any_emitted = True
    meta = None
    if last is not None and not isinstance(last, str):
        prompt_tokens = info.get("prompt_tokens")
        meta = {
            "prompt_tokens": prompt_tokens if prompt_tokens is not None else getattr(last, "prompt_tokens", None),
            "cached_tokens": info.get("cached_tokens", getattr(last, "cached_tokens", None)),
            "generation_tokens": getattr(last, "generation_tokens", None),
        }
        # mlx-vlm's GenerationResult carries the decode rate it measured.
        tps = getattr(last, "generation_tps", None)
        if isinstance(tps, (int, float)) and not isinstance(tps, bool) and math.isfinite(tps) and tps > 0:
            meta["generation_tps"] = float(tps)
        if not cancelled:
            meta["finish_reason"] = _finish_reason(last, raw_accum)
    return any_emitted, cancelled, meta, None


# Gemma 4's template leaves the model's turn open after a tool result
# (`...<tool_response|>`), for the answer to follow in the same turn. The
# 4-bit E2B model often ends the turn right there without a word (three of
# five time-zone questions on 2026-09-26, at every temperature tried). Closing
# the turn and opening a new model turn got an answer from the result in all
# three.
_OPEN_TOOL_TURN = "<tool_response|>"
_REOPEN_MODEL_TURN = "<turn|>\n<|turn>model\n"


def _chat_core(model, model_path, processor, prompt, native, history, max_tokens, temperature, conv, images=None):
    """_stream_core, asked once more in a new model turn when the answer after
    a tool result came back empty."""
    result = _stream_core(model, model_path, processor, prompt, "", max_tokens, temperature, conv, history, images)
    any_emitted, cancelled, _meta, error = result
    if native and not (any_emitted or cancelled or error) and prompt.rstrip().endswith(_OPEN_TOOL_TURN):
        print("estia-runner: empty answer after a tool result; asking again in a new model turn", file=sys.stderr, flush=True)
        result = _stream_core(model, model_path, processor, prompt + _REOPEN_MODEL_TURN, "", max_tokens,
                              temperature, conv, history, images)
    return result


def chat_stream_lines(req):
    model_path = req["model_path"]
    model, processor = load_gen(model_path)
    messages = req.get("messages") or []
    prompt, native, history = _render_messages(processor, messages, req.get("tools"))
    max_tokens = int(req.get("max_tokens") or 256)
    temperature = float(req.get("temperature") or 0.0)
    images = _message_images(messages)
    try:
        # No prompt cache for a conversation with images (see _generate).
        conv = None if images else _conversation(model_path, req.get("cache_key"))
        any_emitted, cancelled, meta, error = _chat_core(
            model, model_path, processor, prompt, native, history, max_tokens, temperature, conv, images)
    finally:
        _remove_files(images)
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
        model_path = req["model_path"]
        model, processor = load_gen(model_path)
        messages = req.get("messages") or []
        prompt, native, history = _render_messages(processor, messages, req.get("tools"))
        images = _message_images(messages)
        try:
            conv = None if images else _conversation(model_path, req.get("cache_key"))
            any_emitted, cancelled, meta, error = _chat_core(
                model, model_path, processor, prompt, native, history, int(req.get("max_tokens") or 256),
                float(req.get("temperature") or 0.0), conv, images)
        finally:
            _remove_files(images)
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

    # No cache_key here: a fresh cache each time, but the prefill still runs in
    # cancellable chunks (see _generate).
    gen = _generate(model, req["model_path"], processor, prompt, max_tokens, temperature, {})

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
        for chunk in gen:
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
    except _Cancelled:
        # A cancel during prefill: nothing was decoded, so nothing to flush.
        emit({"done": True, "cancelled": True})
        return
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
    finally:
        gen.close()

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
    # The model's conversation KV caches go with it: they are GPU memory too.
    convs = [k for k in _PROMPT_CACHES if k[0] == path]
    for key in convs:
        del _PROMPT_CACHES[key]
    if dropped or convs:
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
        try:
            _clear_after_request()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
