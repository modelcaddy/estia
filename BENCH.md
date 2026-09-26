# Bench log

Numbers from `estia bench`, `estia chat` and `estia serve`, one machine per
row. A claim about speed is a row here or it is not a claim. The CLI was a
debug build. The runner is the Python MLX runner (`mlx-vlm` /
`mlx-embeddings`), so the binary's build profile does not matter; the model
does.

| Date | Machine | Backend | Model | First token (cold / warm) | Decode | Embed |
|---|---|---|---|---|---|---|
| 2026-09-08 | Apple silicon, 32 GB, macOS 15.6 (Darwin 24.6) | mlx-python (Python 3.12.7, mlx-lm 0.31.3) | gemma4-e2b-it-4bit-mlx, 64 max tokens | 2465 ms / 182 ms | ≈450 chars/s | embeddinggemma-300m-4bit: 32 units × 768 dims in 325 ms (≈98 units/s) |
| 2026-09-09 | same Mac, **while a large `cargo test` build ran in the background** | mlx-python 2.0.0 (protocol v2) | gemma4-e2b-it-4bit-mlx, 64 max tokens | load 7494 ms (v2 `load`), then first generation 2682 ms / warm 387 ms | ≈200 chars/s | 32 units in 841 ms (≈38 units/s); the slowdown is the concurrent compile, not the protocol |
| 2026-09-09 | same Mac, a large `cargo test` build running concurrently | mlx-python 2.1.0, protocol v2 `chat_stream` + `cache_key` | gemma4-e2b-it-4bit-mlx | turn 1: 1038 ms (28 prompt tokens, 0 cached) · turn 2: **371 ms (76 prompt tokens, 57 cached)** | 29 + 8 tokens generated | — |
| 2026-09-09 | same Mac, quiet | server `estia serve` → `/v1/chat/completions`, mlx-python 2.1.0 | gemma4-e2b via role `fast` | first request 5428 ms (spawn + load + prefill) · second turn of the same `user` conversation **422 ms, 31 of 49 prompt tokens cached** · streaming turn 224 ms | 19 / 16 / 11 tokens | `/v1/embeddings`: 2 × 768 |

Same session, other checks that day: a 3.6 GB model spawned and answered a
two-sentence prompt in 4.8 s cold; a schema-enforced JSON digest validated on
the first attempt after the engine stripped the model's code fences (reported
as `repaired: strip_fences_or_preamble`); a cancel issued 1.5 s into a
400-token generation ended the stream at 2.3 s with the runner acknowledging.

The prompt cache row is the one that matters for coding agents: a second turn
that only appends to the conversation prefills its new suffix, not the whole
transcript. Same run: the tokenizer's native chat template rendered system,
user and assistant roles, and a `tools` declaration made Gemma 4 answer with
its own call syntax, `<|tool_call>call:get_weather{city:<|"|>Athens<|"|>}<tool_call|>`,
which the server parses into OpenAI `tool_calls`.

Protocol v2's `load` separates model load from the first generation, so
"first token" after a load is prompt processing plus kernel warm-up, not the
weights coming off disk. Run `estia bench` on a quiet machine before quoting a
number.

"chars/s" is measured on the cleaned, streamed text, so it undercounts raw
token throughput slightly (the runner holds back a 24-char tail while
streaming). The later rows count tokens instead, from the runner's protocol v2
`meta` line.

To reproduce:

```bash
estia bench --model gemma4-e2b
echo "Describe a lighthouse." | estia chat --model gemma4-e2b --system "Be terse." --cache-key c1 --two-turns
```
