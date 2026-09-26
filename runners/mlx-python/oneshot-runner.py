#!/usr/bin/env python3
"""One-shot MLX runner (legacy; the resident runner is estia-runner.py).

One request per process: read one JSON request from stdin to EOF, write one
JSON envelope to stdout (or newline-delimited token lines for
generate_stream), exit. Requests: health, generate, generate_stream, embed.
A host drives it through the engine's `OneShot` client
(engine/src/oneshot.rs); the `estia` CLI and server do not use it.

This runner is intentionally tiny: the host owns model selection and download
state; Python only bridges to MLX.

Uses `mlx-vlm` (Vision-Language Models) because the Gemma 4 generation models
(e.g. E4B) are multimodal — their weights carry a `language_model.*` prefix
and a separate vision encoder, which `mlx-lm` cannot load. `mlx-vlm` handles
both multimodal and text-only generation through the same API.
"""

from __future__ import annotations

import json
import importlib.metadata
import importlib.util
import re
import sys
import traceback
from typing import Any


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def emit_line(payload: dict[str, Any]) -> None:
    # Newline-delimited variant for streaming: the Rust side reads one JSON object
    # per line as tokens arrive, so each must be flushed immediately.
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def fail(message: str) -> None:
    emit({"ok": False, "error": message})


def read_request() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("empty request")
    return json.loads(raw)


def _tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def _vocab_has(processor: Any, token: str) -> bool:
    tokenizer = _tokenizer(processor)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        return False
    try:
        return token in get_vocab()
    except Exception:
        return False


def _manual_chat_prompt(processor: Any, raw_prompt: str) -> str | None:
    # Hand-rolled turn format, kept as the first candidate because it is the one
    # with mileage on it. Raw prompts make the model echo or emit hidden
    # thought-channel markers; this reliably opens a model response turn.
    #
    # It is NOT the repo's real template, which renders
    #   <bos><|turn>user\n…<turn|>\n<|turn>model\n
    # — i.e. `<|turn>` opens a turn and `<|channel>` opens a *thought* channel,
    # so this string mislabels the turn as a channel and omits <bos>. Measured
    # A/B over 21 generations on E2B and E4B (extraction, memory ops, reasoning
    # bait, terse prose) found no quality difference and no marker leakage from
    # either, so this stays first and the real template sits behind it as the
    # fallback. Bundles downloaded before chat_template.jinja was added to the
    # download filter have no template at all; there this is the only formatting.
    if _vocab_has(processor, "<|channel>") and _vocab_has(processor, "<turn|>"):
        return f"<|channel>user\n{raw_prompt}<turn|><|channel>model\n"
    return None


def _clean_generated_text(text: str, raw_prompt: str) -> str:
    text = (text or "").replace("<turn|>", "").replace("<|turn>", "")

    # Drop hidden/internal channel output if the model slips into it.
    text = re.sub(r"<\|channel\>\s*(thought|analysis)\b.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|think\>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel\>\s*(model|assistant|final)\s*", "", text)
    text = re.sub(r"<channel\|>", "", text)

    stripped = text.strip()
    prompt = raw_prompt.strip()
    if prompt and stripped.startswith(prompt):
        stripped = stripped[len(prompt):].lstrip(" \n:-")

    return stripped.strip()


def _is_bad_generation(text: str) -> bool:
    if not text.strip():
        return True
    lower = text.lower()
    if "<|channel>" in lower or "<|think|>" in lower or "generationresult(" in lower:
        return True
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4:
        most_common = max(lines.count(line) for line in set(lines))
        if most_common / len(lines) > 0.35:
            return True
    return False


def health() -> None:
    # Avoid importing mlx here. On some machines/environments, importing MLX
    # eagerly initializes Metal and can abort the Python process if no device
    # is available. Health should be cheap and non-crashing; generation will
    # import mlx_vlm in the isolated runner process when the user actually
    # asks for it.
    if importlib.util.find_spec("mlx_vlm") is None:
        emit({
            "ok": True,
            "version": None,
            "mlx_available": False,
            "detail": "mlx-vlm is not installed.",
        })
        return
    if importlib.util.find_spec("mlx_embeddings") is None:
        emit({
            "ok": True,
            "version": None,
            "mlx_available": False,
            "detail": "mlx-embeddings is not installed.",
        })
        return

    try:
        version = importlib.metadata.version("mlx-vlm")
    except Exception:
        version = None

    emit({
        "ok": True,
        "version": version,
        "mlx_available": True,
            "detail": "mlx-vlm and mlx-embeddings packages are installed.",
    })


def generate(req: dict[str, Any]) -> None:
    # mlx-vlm handles multimodal models like Gemma 4 E4B. Passing no image
    # gives us text-only generation through the same API.
    from mlx_vlm import apply_chat_template
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm import load

    model_path = req["model_path"]
    raw_prompt = req["prompt"]
    max_tokens = int(req.get("max_tokens") or 512)
    temperature = float(req.get("temperature") if req.get("temperature") is not None else 0.0)

    model, processor = load(model_path)

    prompt_candidates: list[str] = []

    manual_prompt = _manual_chat_prompt(processor, raw_prompt)
    if manual_prompt:
        prompt_candidates.append(manual_prompt)

    # Second candidate: the repo's own chat template, via mlx-vlm's model-aware
    # helper. Worth its place only when the bundle actually ships one — without
    # `chat_template.jinja` on disk this returns the raw prompt unchanged, and
    # the dedup below then collapses it onto the raw candidate, leaving the
    # manual format above as the only real formatting in the cascade.
    try:
        templated_prompt = apply_chat_template(
            processor,
            model.config,
            raw_prompt,
            add_generation_prompt=True,
            num_images=0,
            num_audios=0,
        )
        prompt_candidates.append(templated_prompt)
    except Exception:
        pass

    prompt_candidates.append(raw_prompt)

    seen = set()
    prompt_candidates = [
        p for p in prompt_candidates if p and not (p in seen or seen.add(p))
    ]

    def run_once(candidate_prompt: str) -> Any:
        # mlx-vlm's `generate` signature has shifted between releases. Try the
        # modern keyword form first, then fall back to positional for older.
        try:
            return mlx_generate(
                model,
                processor,
                prompt=candidate_prompt,
                image=None,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
        except TypeError:
            return mlx_generate(model, processor, candidate_prompt, max_tokens=max_tokens)

    last_output: Any = None
    last_text = ""
    for candidate in prompt_candidates:
        output = run_once(candidate)
        last_output = output
        raw_text = output if isinstance(output, str) else getattr(output, "text", "")
        text = _clean_generated_text(raw_text, raw_prompt)
        last_text = text
        if not _is_bad_generation(text):
            emit({"ok": True, "text": text})
            return

    generated = getattr(last_output, "generation_tokens", None)
    token = getattr(last_output, "token", None)
    if not last_text:
        fail(f"MLX generated no text (generation_tokens={generated}, token={token}).")
        return

    fail(
        "MLX generated unusable text "
        f"(generation_tokens={generated}, token={token}, sample={last_text[:120]!r})."
    )


def generate_stream(req: dict[str, Any]) -> None:
    """Token-streaming generation.

    Emits newline-delimited ``{"type":"token","text":…}`` lines as the model
    decodes, then a terminal ``{"ok":true,"done":true}``. Degrades safely: if this
    mlx-vlm build exposes no streaming API, or streaming raises before any token is
    shown, it falls back to the robust one-shot ``generate`` (which emits a single
    ``{"ok":true,"text":…}`` envelope the Rust reader treats as the final chunk), so
    callers never regress below today's behaviour.
    """
    from mlx_vlm import load

    try:
        from mlx_vlm import apply_chat_template
    except Exception:
        apply_chat_template = None

    stream_fn = None
    try:
        from mlx_vlm import stream_generate as stream_fn  # type: ignore
    except Exception:
        try:
            from mlx_vlm.utils import stream_generate as stream_fn  # type: ignore
        except Exception:
            stream_fn = None

    if stream_fn is None:
        # No streaming API → robust one-shot path (single non-streamed envelope).
        generate(req)
        return

    model_path = req["model_path"]
    raw_prompt = req["prompt"]
    max_tokens = int(req.get("max_tokens") or 512)
    temperature = float(req.get("temperature") if req.get("temperature") is not None else 0.0)

    model, processor = load(model_path)

    prompt = _manual_chat_prompt(processor, raw_prompt)
    if not prompt and apply_chat_template is not None:
        try:
            prompt = apply_chat_template(
                processor,
                model.config,
                raw_prompt,
                add_generation_prompt=True,
                num_images=0,
                num_audios=0,
            )
        except Exception:
            prompt = None
    if not prompt:
        prompt = raw_prompt

    def open_stream():
        # mlx-vlm's stream signature has drifted across releases — try the modern
        # keyword form, then a leaner one, before giving up to the one-shot path.
        try:
            return stream_fn(
                model, processor, prompt,
                image=None, max_tokens=max_tokens, temperature=temperature,
            )
        except TypeError:
            return stream_fn(
                model, processor, prompt,
                max_tokens=max_tokens, temperature=temperature,
            )

    # Hold back the last few chars while streaming: a channel marker forming at
    # the buffer tail ("…<|chan") isn't strippable until complete, and emitting
    # it would leak marker text. 24 covers the longest marker plus slack; the
    # held tail flushes after the stream ends.
    #
    # `anchor` is the prefix of the CURRENT cleaned lineage already emitted — a
    # string, not an integer offset, because the cleaner can REBASE the buffer
    # (e.g. a stripped prompt echo or marker collapse shrinks it beyond the
    # holdback). On a rebase the lineage restarts from zero: the already-shown
    # tail can't be retracted from a token stream, but nothing of the real
    # answer is ever dropped.
    holdback = 24
    raw_accum = ""
    anchor = ""
    any_emitted = False
    try:
        for chunk in open_stream():
            piece = chunk if isinstance(chunk, str) else getattr(chunk, "text", "")
            if not piece:
                continue
            raw_accum += piece
            # Re-clean the whole buffer each step (markers/prompt-echo can span
            # tokens) and emit only the newly revealed, cleaned suffix.
            cleaned = _clean_generated_text(raw_accum, raw_prompt)
            if not cleaned.startswith(anchor):
                anchor = ""  # cleaner rebased — restart the lineage
            safe = max(0, len(cleaned) - holdback)
            if safe > len(anchor):
                emit_line({"type": "token", "text": cleaned[len(anchor):safe]})
                anchor = cleaned[:safe]
                any_emitted = True
    except Exception as exc:
        if not any_emitted:
            # Nothing shown yet → safe to fall back to the robust one-shot path.
            generate(req)
            return
        emit_line({"ok": False, "error": f"MLX stream failed mid-output: {exc}"})
        return

    # Flush the held-back tail from the final cleaned text.
    final_cleaned = _clean_generated_text(raw_accum, raw_prompt)
    if not final_cleaned.startswith(anchor):
        anchor = ""  # rebase at the very end — emit the whole final lineage
    if len(final_cleaned) > len(anchor):
        emit_line({"type": "token", "text": final_cleaned[len(anchor):]})
        any_emitted = True

    if not any_emitted:
        # Stream yielded nothing usable → one-shot fallback.
        generate(req)
        return

    emit_line({"ok": True, "done": True})


def embed(req: dict[str, Any]) -> None:
    try:
        from mlx_embeddings import generate as embed_generate
        from mlx_embeddings import load as embed_load
    except Exception:
        try:
            from mlx_embeddings.utils import load as embed_load
            embed_generate = None
        except Exception as exc:
            fail(f"mlx-embeddings is not installed or importable: {exc}")
            return

    model_path = req["model_path"]
    text = req["input"]
    model, processor = embed_load(model_path)

    # `gemma3_text` (EmbeddingGemma) and `qwen3` take positional `inputs` and raise
    # `TypeError: Model.__call__() got an unexpected keyword argument 'input_ids'`
    # through `generate()`. The BERT family goes through `generate()` normally.
    positional = type(model).__module__.split(".")[-1] in ("gemma3_text", "qwen3")

    try:
        if embed_generate is not None and not positional:
            output = embed_generate(model, processor, texts=[text])
            embeds = output.text_embeds
        else:
            inputs = processor(
                [text],
                return_tensors="mlx",
                padding=True,
                truncation=True,
                max_length=512,
            )
            output = model(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            embeds = getattr(output, "text_embeds", None)
            if embeds is None:
                embeds = output.pooler_output
        vector = embeds[0].tolist()
    except Exception as exc:
        fail(f"MLX embedding failed: {exc}")
        return

    emit({"ok": True, "embedding": vector})


def main() -> int:
    try:
        req = read_request()
        kind = req.get("type")
        if kind == "health":
            health()
        elif kind == "generate":
            generate(req)
        elif kind == "generate_stream":
            generate_stream(req)
        elif kind == "embed":
            embed(req)
        else:
            fail(f"unknown request type: {kind}")
        return 0
    except Exception as exc:
        fail(f"{exc}\n{traceback.format_exc(limit=3)}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
