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

## 2026-09-27: the 0.4.0 acceptance run, M1 Pro

Two sources, labelled in each table. **Acceptance:** the acceptance run's
performance pass against the installed engine (`estia serve --lan`, MLX
runner 2.2.0), streamed through `/v1/chat/completions` and `/v1/embeddings`
from the same Mac, in the run's quiet passes (one request at a time, no
other client); raw records in its `perf/quiet.jsonl`, `qlong.jsonl` and
`embq.jsonl`.
**Re-test:** the live re-test of the fixes on a scratch engine built from
this tree (MLX runner 2.3.0), same Mac, same models. Machine: Apple M1 Pro,
32 GB, macOS 15.7.3 (Darwin 24.6). Decode rates are the runner's own
`generation_tps` (generation only, prefill excluded).

Generation, warm (acceptance; a 20 to 24 token prompt, three runs each):

| Model | First token, warm | Decode, ~20 tokens | Decode, ~180 tokens | Load (re-test, `load_ms`) |
|---|---|---|---|---|
| gemma4-e2b-it-4bit-mlx | 176–191 ms | 79.8–81.5 tok/s | 74.3 tok/s (179 tokens) | 3786 ms |
| gemma4-e4b-it-4bit-mlx | 322–354 ms | 46.7–47.1 tok/s | 43.9 tok/s (173 tokens) | 6451 ms |
| gemma4-12b-it-qat-4bit-mlx | 669–687 ms | 20.6–20.7 tok/s | 19.5 tok/s (191 tokens) | 4596 ms |

The re-test agrees: 76.9–77.6 tok/s on e2b, 45.2–45.6 on e4b, 19.0–20.9 on
the 12B. Long prompts cost more than these short ones suggest: in the
re-test an 8.5K-token prompt took 5.5 s on e2b and 21 s on e4b before its
first token, and a 1.7K-token first turn on the 12B took 21 s after a 4.6 s
load (its second turn, served from the prompt cache, 1.9 s).

Embeddings (acceptance, one `/v1/embeddings` call per batch, three runs
each; "short" is 256 inputs of 4 to 9 words, "long" 32 inputs of about 356
words):

| Model | Short: 256 inputs | Long: 32 inputs |
|---|---|---|
| embeddinggemma-300m-4bit | 2.40–2.54 s (101–107 inputs/s) | 1.22–1.37 s (23–26 inputs/s) |
| nomicai-modernbert-embed-base-6bit | 1.88–2.30 s (111–136 inputs/s) | 1.18–1.21 s (26–27 inputs/s) |

With generation models busy on the same Mac, the same batches ran at 84–92
and 15 inputs/s (EmbeddingGemma) and 63–78 and 17 inputs/s (nomic).

Memory per loaded model (re-test, `memory_bytes` from `/engine/stats`, the
runner's physical footprint; `footprint(1)` gave the same figures). It is
mostly the weights in Metal buffers, which `ps` does not count: `ps` shows
about 100 MB for the e2b runner.

| Model | Footprint | After |
|---|---|---|
| gemma4-e2b-it-4bit-mlx | 3.96–4.11 GB | 4 short requests (3.96); an 8.5K-token prompt and the other tests (4.11) |
| gemma4-e4b-it-4bit-mlx | 5.76 GB | an 8.5K-token prompt and a cancel |
| gemma4-12b-it-qat-4bit-mlx | 7.94 GB | a 1.7K-token two-turn conversation, its cache included |
| embeddinggemma-300m-4bit | 0.61–0.96 GB | small batches (0.61); a 3 MB batch of 32 long inputs (0.85–0.96) |

All four loaded at once would hold about 18.8 GB of the Mac's 32 GB.
