# Examples

Small programs that use Estia, one idea each. They are meant to be read and
copied. Every one of them was run against a live engine. Those that need a
token exit with a clear message when it is missing, unknown or lacks a scope.

The concepts behind them are explained in
[docs/building-clients.md](../docs/building-clients.md).

| Example | What it shows | Language | Scopes |
|---|---|---|---|
| [curl/quickstart.sh](curl/quickstart.sh) | Every call a client makes, step by step: health, models, chat, the prompt cache, streaming, embeddings, JSON Schema output, tools, errors | shell, curl | `generate`, `embed`, `models:read` |
| [python/chat.py](python/chat.py) | A streaming assistant in the terminal with history, the prompt cache (`user`), cached-token counts from `x_estia`, Ctrl-C to cancel | Python, OpenAI SDK | `generate` |
| [python/no_sdk.py](python/no_sdk.py) | The same chat with the standard library only: raw HTTP and server-sent events | Python, stdlib | `generate` |
| [python/rag.py](python/rag.py) | Questions over a folder of notes with citations: `task` prefixes, fingerprints, batching, cosine ranking | Python, OpenAI SDK | `embed`, `generate` |
| [python/structured.py](python/structured.py) | JSON Schema extraction and the 422 when the output does not fit | Python, OpenAI SDK | `generate` |
| [python/tools.py](python/tools.py) | A tool-calling loop around one local Python function | Python, OpenAI SDK | `generate` |
| [python/pair.py](python/pair.py) | How a device app gets a token: request, wait for approval, save it with mode 0600 | Python, stdlib | none (the token gets what you ask for) |
| [javascript/chat.mjs](javascript/chat.mjs) | Streaming chat, two turns sharing the prompt cache, Ctrl-C to cancel | JavaScript, OpenAI SDK | `generate` |
| [javascript/embed.mjs](javascript/embed.mjs) | Embeddings with `task` and `expect_fingerprint`, ranked by cosine | JavaScript, OpenAI SDK | `embed` |
| [remote_client.rs](../engine/examples/remote_client.rs) | `RemoteEngine` from Rust: pair or use a token, streamed chat, cancel, embeddings, the handle types | Rust | `generate`, `embed` |
| [in_process.rs](../engine/examples/in_process.rs) | `Engine` inside your own process, no server: chat, stream, structured output, embeddings | Rust | none (no server) |

## Before you start

1. An engine is running. On the engine's machine:

   ```bash
   estia setup        # once: runtime, models, first admin token
   estia serve        # http://127.0.0.1:27200
   ```

   The examples work on either backend, MLX or llama.cpp: they ask for roles
   (`fast`, `embed`), never for artifact ids.

2. You have a token.

   On the engine's machine, mint one per app with only the scopes it needs.
   The command prints the token once:

   ```bash
   estia token new myapp --scopes generate,embed,models:read
   ```

   On another device, pair instead. Run `python/pair.py` (or
   `estia pair request --engine http://<engine>:27200`) and approve the
   request on the engine's machine with `estia pair approve <id>`. The engine
   must be serving the LAN (`estia serve --lan`) for other devices to reach it.

3. The environment is set:

   | Variable | Meaning | Default |
   |---|---|---|
   | `ESTIA_URL` | Where the engine listens | `http://127.0.0.1:27200` |
   | `ESTIA_TOKEN` | The bearer token | none; every example except `pair.py` and `in_process.rs` needs it |
   | `ESTIA_MODEL` | The role (or model) to chat with | `fast` |

   ```bash
   export ESTIA_TOKEN=estia_...
   ```

   `in_process.rs` runs no server and reads `ESTIA_DATA_DIR` (default
   `~/Library/Application Support/estia` on macOS) and optionally
   `ESTIA_RUNNER` and `ESTIA_PYTHON` instead.

## Running them

curl (needs only curl):

```bash
./curl/quickstart.sh
```

Python 3.10 or newer:

```bash
cd python
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # only the OpenAI SDK
python chat.py
python rag.py index sample-notes && python rag.py ask "When do I repot the olive tree?"
python structured.py
python tools.py "What time is it in Tokyo?"
python no_sdk.py                         # no pip needed
python pair.py --name "my laptop"        # no pip needed, no token needed
```

JavaScript, Node 18 or newer:

```bash
cd javascript
npm install
node chat.mjs "Suggest a name for a small sailing boat."
node embed.mjs "where is the spare key?"
```

Rust, from the repository root:

```bash
cargo run -p estia-engine --example remote_client
cargo run -p estia-engine --example remote_client -- --pair "my laptop"
cargo run -p estia-engine --example in_process
```

## What the output looks like

From `python/chat.py` against `gemma4-e2b` (role `fast`):

```text
you> My name is Nikos and I keep bees. Say hello.
assistant> Hello, Nikos. It's nice to meet you. I hope your beekeeping is going well.
[41 prompt tokens, 0 from cache, 1717 ms]

you> What do I keep?
assistant> You keep bees. That is what you mentioned.
[78 prompt tokens, 64 from cache, 240 ms]
```

From `python/rag.py` over `sample-notes/`:

```text
$ python rag.py ask "When do I repot the olive tree?"
The next repot for the olive tree is in spring 2028 [1].

Sources:
  [1] plants.txt:1  (cosine 0.548)
  [2] fasolada.md:1  (cosine 0.220)
  [3] home-network.md:1  (cosine 0.088)
```

Model answers vary from run to run.
