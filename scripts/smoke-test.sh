#!/usr/bin/env bash
# Check that a running Estia engine answers the way clients expect.
#
#   scripts/smoke-test.sh [--url URL] [--token TOKEN | --token-file FILE]
#                         [--model ROLE] [--embed ROLE] [--quick]
#
# Each check prints PASS, FAIL or SKIP, how long it took, and what it saw.
# The exit code is 0 when nothing failed, 1 when a check failed, and 2 when
# the script could not start (a bad option, a missing tool, no token).
#
# Needs bash (3.2 or newer), curl and python3. Nothing else. It only reads
# from the engine: it pulls no models and changes no settings. It does run
# the model: the first call loads it, which takes a few seconds.
#
# The token needs the scopes generate, embed and models:read. On the engine's
# machine:
#
#   estia token new smoke --scopes generate,embed,models:read
#
# Environment: ESTIA_URL (default http://127.0.0.1:27200), ESTIA_TOKEN,
# ESTIA_MODEL (default fast). NO_COLOR turns colours off.

set -u

usage() {
  cat <<'EOF'
Usage: scripts/smoke-test.sh [options]

Checks that a running Estia engine answers the way clients expect.

Options:
  --url URL          The engine (default: $ESTIA_URL, else http://127.0.0.1:27200)
  --token TOKEN      A bearer token (default: $ESTIA_TOKEN)
  --token-file FILE  Read the token from the first line of FILE instead
  --model ROLE       The role or model to chat with (default: $ESTIA_MODEL, else fast)
  --embed ROLE       The role or model to embed with (default: embed)
  --quick            Only three checks: health, one chat, one embedding
  -h, --help         Show this help

The token needs the scopes generate, embed and models:read:
  estia token new smoke --scopes generate,embed,models:read

Exit code: 0 when no check failed, 1 when one did, 2 when the script could
not start.
EOF
}

die() {
  printf 'smoke-test: %s\n' "$*" >&2
  exit 2
}

URL="${ESTIA_URL:-http://127.0.0.1:27200}"
TOKEN="${ESTIA_TOKEN:-}"
TOKEN_SOURCE="ESTIA_TOKEN"
MODEL="${ESTIA_MODEL:-fast}"
EMBED="embed"
QUICK=0

while [ $# -gt 0 ]; do
  case "$1" in
    --url) [ $# -ge 2 ] || die "--url needs a value"; URL="$2"; shift 2 ;;
    --url=*) URL="${1#*=}"; shift ;;
    --token) [ $# -ge 2 ] || die "--token needs a value"; TOKEN="$2"; TOKEN_SOURCE="--token"; shift 2 ;;
    --token=*) TOKEN="${1#*=}"; TOKEN_SOURCE="--token"; shift ;;
    --token-file | --token-file=*)
      if [ "$1" = --token-file ]; then
        [ $# -ge 2 ] || die "--token-file needs a value"
        f="$2"; shift 2
      else
        f="${1#*=}"; shift
      fi
      [ -r "$f" ] || die "cannot read the token file $f"
      IFS= read -r TOKEN <"$f" || [ -n "$TOKEN" ] || die "the token file $f is empty"
      TOKEN_SOURCE="$f"
      ;;
    --model) [ $# -ge 2 ] || die "--model needs a value"; MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --embed) [ $# -ge 2 ] || die "--embed needs a value"; EMBED="$2"; shift 2 ;;
    --embed=*) EMBED="${1#*=}"; shift ;;
    --quick) QUICK=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done

# Strip whitespace a copied token often carries.
TOKEN="${TOKEN//[[:space:]]/}"
URL="${URL%/}"
case "$URL" in
  http://* | https://*) ;;
  *) die "--url must start with http:// (got: $URL)" ;;
esac
for v in "$MODEL" "$EMBED"; do
  case "$v" in
    '' | *[!A-Za-z0-9._:-]*) die "a model or role name may hold only letters, digits and . _ : - (got: '$v')" ;;
  esac
done
command -v curl >/dev/null 2>&1 || die "curl is not installed"
command -v python3 >/dev/null 2>&1 || die "python3 is not installed (it reads the JSON answers)"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/estia-smoke.XXXXXX")" || die "cannot make a temporary directory"
trap 'rm -rf "$TMP"' EXIT
chmod 700 "$TMP"
export PYTHONIOENCODING=utf-8

# The token goes to curl through a header file, so it never shows in `ps`.
AUTH="$TMP/auth.header"
if [ -n "$TOKEN" ]; then
  (umask 077 && printf 'Authorization: Bearer %s\n' "$TOKEN" >"$AUTH")
else
  : >"$AUTH"
fi

RUN_ID="smoke-$$-${RANDOM}"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_PASS=$'\033[32m' C_FAIL=$'\033[31m' C_SKIP=$'\033[33m' C_DIM=$'\033[2m' C_OFF=$'\033[0m'
else
  C_PASS='' C_FAIL='' C_SKIP='' C_DIM='' C_OFF=''
fi

# The JSON side: builds request bodies and judges answers. Each judgement
# prints: STATUS <tab> SECONDS <tab> DETAIL <tab> HINT.
PY="$TMP/judge.py"
cat >"$PY" <<'PYEOF'
import json, math, sys

def load(path):
    try:
        with open(path, "rb") as f:
            raw = f.read().decode("utf-8", "replace")
    except OSError:
        return None, ""
    try:
        return json.loads(raw), raw
    except ValueError:
        return None, raw

def headers(path):
    out = {}
    try:
        with open(path, "rb") as f:
            for line in f.read().decode("latin-1").splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    out[k.strip().lower()] = v.strip()
    except OSError:
        pass
    return out

def secs(times):
    total = 0.0
    for t in times.split("+"):
        try:
            total += float(t)
        except ValueError:
            pass
    return "%d ms" % round(total * 1000) if total < 1 else "%.2f s" % total

def clean(s):
    return " ".join(str(s).split())

def short(s, n=70):
    s = clean(s)
    return s if len(s) <= n else s[: n - 1] + "…"

def curl_error(path):
    """curl's own message, without its "curl: (7) " prefix."""
    try:
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    text = clean(text)
    if text.startswith("curl: ("):
        text = text.split(") ", 1)[-1]
    return text

def done(status, times, detail, hint=""):
    print("\t".join([status, secs(times), clean(detail), clean(hint)]))
    sys.exit(0)

def err(body):
    e = body.get("error") if isinstance(body, dict) else None
    if isinstance(e, dict):
        return e.get("message", ""), e.get("type", ""), e.get("request_id", "")
    if isinstance(e, str):
        return e, "", body.get("request_id", "")
    return "", "", ""

SCOPES = "generate, embed and models:read"

def http_failure(times, status, body, raw, err_path, what):
    """Explain a status that is not the one a check wanted."""
    curl_err = curl_error(err_path)
    if status == "000":
        done("FAIL", times, "no answer from the engine: " + (curl_err or "the connection failed"),
             "Is the engine still running? Model calls wait up to 300 s.")
    msg, typ, rid = err(body)
    idpart = " (request id %s)" % rid if rid else ""
    if status == "401":
        done("FAIL", times, "the engine refused the token: %s%s" % (msg or "401", idpart),
             "Check --token, --token-file or ESTIA_TOKEN. To mint one on the engine's machine: "
             "estia token new smoke --scopes generate,embed,models:read")
    if status == "403":
        done("FAIL", times, "refused (403): %s%s" % (msg, idpart),
             "Use a token with the scopes %s: estia token new smoke --scopes generate,embed,models:read" % SCOPES if "scope" in msg else
             "Reach the engine by IP address or <name>.local, or start it with --allow-host <name>.")
    if status == "404" and "model" in msg:
        done("FAIL", times, "no model to answer: %s%s" % (msg, idpart),
             "On the engine's machine, `estia models` shows what is installed and "
             "`estia roles` what each role points to. `estia pull <role>`, such as `estia pull fast`, downloads the model.")
    detail = "%s: HTTP %s %s%s" % (what, status, short(msg or raw, 160), idpart)
    hint = "The engine's log has more under the request id." if rid else ""
    done("FAIL", times, detail, hint)

def health(a):
    times, status, body_path, err_path = a
    body, raw = load(body_path)
    curl_err = curl_error(err_path)
    if status == "000":
        done("FAIL", times, "cannot reach the engine: " + (curl_err or "the connection failed"),
             "Is it running? On its machine `estia status` prints the URL to use; pass it with --url.")
    if status == "403":
        msg, _, _ = err(body)
        done("FAIL", times, "the engine refused the host name in the URL: " + short(msg, 160),
             "Use the engine's IP address or <name>.local, or start it with --allow-host <name>.")
    if status != "200" or not isinstance(body, dict) or body.get("engine") != "estia":
        done("FAIL", times, "something answered, but not Estia's health route (HTTP %s): %s" % (status, short(raw, 100)),
             "Check --url: it should be the engine's address and port, with no path.")
    parts = ["Estia %s" % body.get("version", "?")]
    build = body.get("build")
    if isinstance(build, dict) and build.get("commit"):
        parts[0] += " (%s%s)" % (build["commit"], ", " + build["date"] if build.get("date") else "")
    elif isinstance(build, dict):
        b = " ".join("%s=%s" % (k, v) for k, v in build.items() if v not in (None, ""))
        if b:
            parts.append("build " + b)
    elif build:
        parts.append("build %s" % build)
    backend = body.get("backend", "?")
    for b in body.get("backends") or []:
        if isinstance(b, dict) and b.get("active") and b.get("build"):
            backend += " (llama.cpp %s%s)" % (b["build"], ", " + b["variant"] if b.get("variant") else "")
    parts.append("backend " + backend)
    parts.append("API v%s" % body.get("api_version", "?"))
    parts.append("token required" if body.get("auth_required") else "no token required (--no-auth)")
    loaded = body.get("loaded") or []
    parts.append("loaded: " + (", ".join(loaded) if loaded else "none yet"))
    if body.get("ok") is not True:
        done("FAIL", times, "health says ok=%s: %s" % (body.get("ok"), ", ".join(parts)))
    done("PASS", times, ", ".join(parts))

def health_field(a):
    body, _ = load(a[0])
    v = body.get(a[1]) if isinstance(body, dict) else None
    print("true" if v is True else "false" if v is False else "")

def auth(a):
    times, status, body_path, hdr_path = a
    body, raw = load(body_path)
    msg, typ, rid = err(body)
    hid = headers(hdr_path).get("x-request-id", "")
    if status != "401":
        done("FAIL", times, "a request with no token got HTTP %s, not 401" % status,
             "The engine should refuse /v1/models without a token.")
    if not rid:
        done("FAIL", times, "401, but the error body has no request_id: " + short(raw, 100))
    if hid != rid:
        done("FAIL", times, "401, but X-Request-Id (%s) differs from error.request_id (%s)" % (hid, rid))
    done("PASS", times, "no token: 401 %s \"%s\", request id %s" % (typ, msg, rid))

def models(a):
    times, status, body_path, err_path = a
    body, raw = load(body_path)
    if status != "200":
        http_failure(times, status, body, raw, err_path, "models")
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        done("FAIL", times, "/v1/models returned no models: " + short(raw, 100))
    roles = [m.get("id") for m in data if isinstance(m, dict) and (m.get("x_estia") or {}).get("role")]
    installed = [m.get("id") for m in data if isinstance(m, dict) and (m.get("x_estia") or {}).get("installed")]
    done("PASS", times, "%d entries; roles: %s; installed: %d" % (len(data), ", ".join(roles) or "none", len(installed)))

def chat_body(a):
    model, question, max_tokens = a[0], a[1], int(a[2])
    user = a[3] if len(a) > 3 else ""
    b = {"model": model, "max_tokens": max_tokens, "temperature": 0,
         "messages": [{"role": "user", "content": question}]}
    if user:
        b["user"] = user
    print(json.dumps(b))

def answer_of(body):
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""

def chat(a):
    times, status, body_path, err_path = a
    body, raw = load(body_path)
    if status != "200":
        http_failure(times, status, body, raw, err_path, "chat")
    text = answer_of(body)
    usage = body.get("usage") or {}
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    x = body.get("x_estia") or {}
    if not text.strip():
        done("FAIL", times, "the answer was empty (finish_reason %s)" % ((body.get("choices") or [{}])[0].get("finish_reason")))
    if not isinstance(pt, int) or not isinstance(ct, int) or pt <= 0 or ct <= 0:
        done("FAIL", times, "the answer has no usable token counts: usage=%s" % json.dumps(usage))
    done("PASS", times, "\"%s\" (%s, %d prompt + %d completion tokens)" % (short(text, 60), x.get("family") or body.get("model", "?"), pt, ct))

def stream_body(a):
    print(json.dumps({"model": a[0], "stream": True, "max_tokens": 40, "temperature": 0,
                      "messages": [{"role": "user", "content": "Count from one to five in words."}]}))

def stream(a):
    times, status, body_path, err_path = a
    body, raw = load(body_path)
    if status != "200":
        http_failure(times, status, body, raw, err_path, "stream")
    events = [l[5:].strip() for l in raw.splitlines() if l.startswith("data:")]
    deltas, usage, error = 0, None, None
    text = []
    for e in events:
        if e == "[DONE]":
            continue
        try:
            j = json.loads(e)
        except ValueError:
            continue
        if isinstance(j, dict) and j.get("error"):
            error = j["error"]
        for c in (j.get("choices") or []) if isinstance(j, dict) else []:
            piece = (c.get("delta") or {}).get("content")
            if piece:
                deltas += 1
                text.append(piece)
        if isinstance(j, dict) and j.get("usage"):
            usage = j["usage"]
    if error:
        m = error.get("message", error) if isinstance(error, dict) else error
        r = error.get("request_id", "") if isinstance(error, dict) else ""
        done("FAIL", times, "the stream ended with an error: %s%s" % (short(m, 120), " (request id %s)" % r if r else ""))
    if not events or events[-1] != "[DONE]":
        done("FAIL", times, "the stream did not end with data: [DONE] (%d events)" % len(events))
    if deltas < 1:
        done("FAIL", times, "the stream carried no text (%d events)" % len(events))
    extra = ", %d completion tokens" % usage["completion_tokens"] if usage and "completion_tokens" in usage else ""
    done("PASS", times, "%d text chunks, then [DONE]%s: \"%s\"" % (deltas, extra, short("".join(text), 40)))

def cache_body2(a):
    model, user, turn1 = a
    body, _ = load(turn1)
    print(json.dumps({"model": model, "user": user, "max_tokens": 20, "temperature": 0, "messages": [
        {"role": "user", "content": "My name is Ada and I keep bees. Reply in one short sentence."},
        {"role": "assistant", "content": answer_of(body)},
        {"role": "user", "content": "What do I keep? One word."}]}))

def cached(body):
    usage = body.get("usage") or {}
    v = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if v is None:
        v = (body.get("x_estia") or {}).get("cached_tokens", 0)
    return v or 0, usage.get("prompt_tokens") or 0

def cache(a):
    times, s1, p1, e1, s2, p2, e2 = a
    b1, r1 = load(p1)
    if s1 != "200":
        http_failure(times, s1, b1, r1, e1, "turn 1")
    b2, r2 = load(p2)
    if s2 != "200":
        http_failure(times, s2, b2, r2, e2, "turn 2")
    c1, _ = cached(b1)
    c2, t2 = cached(b2)
    if c2 > 0:
        done("PASS", times, "turn 2: %d of %d prompt tokens came from the cache" % (c2, t2))
    if c1 == 0:
        done("SKIP", times, "the engine reported 0 cached tokens on both turns, so the cache cannot be checked",
             "Both built-in backends cache by `user`; a runner without prompt-cache support reports 0. "
             "With a built-in backend, treat this as a failure.")
    done("FAIL", times, "turn 2 reused nothing (cached_tokens 0 of %d) though turn 1 reported %d" % (t2, c1))

def schema_body(a):
    print(json.dumps({"model": a[0], "temperature": 0, "max_tokens": 80,
        "messages": [{"role": "user", "content": "Where is the Acropolis? Give the city and the country."}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "place", "schema": {
            "type": "object", "additionalProperties": False, "required": ["city", "country"],
            "properties": {"city": {"type": "string"}, "country": {"type": "string"}}}}}}))

def schema(a):
    times, status, body_path, err_path = a
    body, raw = load(body_path)
    if status == "422":
        msg, _, rid = err(body)
        done("FAIL", times, "the model's answer did not fit the schema after a retry (request id %s)" % rid,
             "A small model can miss; run it again. If it keeps failing, the engine's log has the attempts.")
    if status != "200":
        http_failure(times, status, body, raw, err_path, "json schema")
    text = answer_of(body)
    try:
        j = json.loads(text)
    except ValueError:
        done("FAIL", times, "the answer is not JSON: \"%s\"" % short(text, 80))
    if not isinstance(j, dict) or not isinstance(j.get("city"), str) or not isinstance(j.get("country"), str):
        done("FAIL", times, "the JSON does not match the schema: %s" % short(text, 80))
    x = body.get("x_estia") or {}
    note = " (repaired by the engine)" if x.get("repaired") else ""
    done("PASS", times, "%s%s" % (json.dumps(j, ensure_ascii=False), note))

def embed_body(a):
    print(json.dumps({"model": a[0], "task": "document",
                      "input": ["The spare key is in the blue tin.", "Water the lemon tree twice a week."]}))

def embed(a):
    times, status, body_path, err_path, fp_out = a
    body, raw = load(body_path)
    if status != "200":
        http_failure(times, status, body, raw, err_path, "embed")
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or len(data) != 2:
        done("FAIL", times, "expected 2 vectors, got: " + short(raw, 100))
    vecs = [d.get("embedding") for d in data]
    if not all(isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v) for v in vecs):
        done("FAIL", times, "the vectors are not lists of numbers")
    x = body.get("x_estia") or {}
    dims, fp = x.get("dims"), x.get("fingerprint")
    lens = sorted(set(len(v) for v in vecs))
    if len(lens) != 1 or (dims is not None and dims != lens[0]):
        done("FAIL", times, "vector lengths %s do not match x_estia.dims %s" % (lens, dims))
    norms = [math.sqrt(sum(x * x for x in v)) for v in vecs]
    if any(abs(n - 1) > 1e-3 for n in norms):
        done("FAIL", times, "the vectors are not unit length (norms %s)" % ", ".join("%.4f" % n for n in norms),
             "Estia returns L2-normalised vectors, so cosine similarity is a dot product.")
    if not fp:
        done("FAIL", times, "the answer has no x_estia.fingerprint")
    with open(fp_out, "w") as f:
        f.write(fp)
    done("PASS", times, "2 vectors of %d dims, unit length, fingerprint %s" % (lens[0], fp))

def mismatch_body(a):
    print(json.dumps({"model": a[0], "input": "x", "expect_fingerprint": "not-a-model@not-a-backend"}))

def expect_status(a):
    times, status, body_path, hdr_path, want, what, err_path = a
    body, raw = load(body_path)
    curl_err = curl_error(err_path)
    msg, typ, rid = err(body)
    if status == "000":
        done("FAIL", times, "no answer from the engine: " + (curl_err or "the connection failed"))
    if status != want:
        m = short(msg or raw, 120)
        hint = ""
        if what.startswith("Host") and status == "200":
            hint = "The engine answers any Host name, so a web page could reach it by DNS rebinding. Is it running with --allow-host '*'?"
        done("FAIL", times, "%s got HTTP %s, expected %s%s" % (what, status, want, ": " + m if m else ""), hint)
    if not rid:
        done("FAIL", times, "%s got HTTP %s, but the error body has no request_id: %s" % (what, status, short(raw, 100)))
    done("PASS", times, "%s got %s %s" % (what, status, typ))

def request_id(a):
    times, status, hdr_path, sent = a
    got = headers(hdr_path).get("x-request-id", "")
    if status == "000":
        done("FAIL", times, "no answer from the engine")
    if got != sent:
        done("FAIL", times, "sent X-Request-Id %s, got back %s" % (sent, got or "nothing"))
    done("PASS", times, "sent X-Request-Id %s, got the same id back" % sent)

globals()[sys.argv[1]](sys.argv[2:])
PYEOF

PASSED=0
FAILED=0
SKIPPED=0

# report STATUS TIME NAME DETAIL [HINT]
report() {
  local st="$1" tm="$2" name="$3" detail="$4" hint="${5:-}" colour=""
  case "$st" in
    PASS) colour="$C_PASS"; PASSED=$((PASSED + 1)) ;;
    FAIL) colour="$C_FAIL"; FAILED=$((FAILED + 1)) ;;
    *) colour="$C_SKIP"; SKIPPED=$((SKIPPED + 1)) ;;
  esac
  printf '  %s%-4s%s  %-17s %s%8s%s  %s\n' "$colour" "$st" "$C_OFF" "$name" "$C_DIM" "$tm" "$C_OFF" "$detail"
  if [ -n "$hint" ]; then
    printf '%36s%s%s%s\n' '' "$C_DIM" "$hint" "$C_OFF"
  fi
}

# judge NAME CHECK ARGS...: run the Python judgement and report it.
LAST_STATUS=""
judge() {
  local name="$1" line st tm detail hint
  shift
  line="$(python3 "$PY" "$@")" || line="$(printf 'FAIL\t-\tthe check itself crashed (python3 %s)\t' "$1")"
  IFS=$'\t' read -r st tm detail hint <<EOF
$line
EOF
  LAST_STATUS="$st"
  report "$st" "$tm" "$name" "$detail" "$hint"
}

# req NAME METHOD PATH [BODYFILE] [EXTRA CURL ARGS...]: one HTTP call.
# Sets CODE (000 when curl failed), TIME (seconds), and writes
# $TMP/NAME.body, $TMP/NAME.head and $TMP/NAME.err.
req() {
  local name="$1" method="$2" path="$3" data="${4:-}" out
  shift 3
  [ $# -gt 0 ] && shift
  local max=20
  case "$path" in /v1/chat/* | /v1/embeddings) max=300 ;; esac
  if [ -n "$data" ]; then
    set -- -H 'Content-Type: application/json' --data-binary "@$data" "$@"
  fi
  # Tag each request, so the engine's log lines for it are easy to find. The
  # auth check leaves it out to see the engine make an id of its own.
  case "$name" in
    auth | rid) ;;
    *) set -- -H "X-Request-Id: $RUN_ID-$name" "$@" ;;
  esac
  out="$(curl -sS -X "$method" "$URL$path" --connect-timeout 5 --max-time "$max" \
    -o "$TMP/$name.body" -D "$TMP/$name.head" -w '%{http_code} %{time_total}' "$@" 2>"$TMP/$name.err")" || true
  CODE="${out%% *}"
  TIME="${out##* }"
  [ -n "$CODE" ] || CODE=000
  [ -n "$TIME" ] && [ "$TIME" != "$out" ] || TIME=0
  : >>"$TMP/$name.body"
}

TOKEN_REFUSED=0
# Token checks stop after the engine refuses the token once: the rest would
# fail for the same reason.
token_ok() {
  if [ "$TOKEN_REFUSED" = 1 ]; then
    report SKIP - "$1" "skipped: the engine refused the token (see above)"
    return 1
  fi
  return 0
}
note_refusal() { [ "$CODE" = 401 ] && TOKEN_REFUSED=1; return 0; }

summary() {
  local total=$((PASSED + FAILED + SKIPPED))
  echo
  if [ "$FAILED" -gt 0 ]; then
    printf '%s%d of %d checks failed%s, %d passed, %d skipped.\n' "$C_FAIL" "$FAILED" "$total" "$C_OFF" "$PASSED" "$SKIPPED"
    echo "Each FAIL line says what went wrong, and the line under it what to try."
  else
    printf '%sAll checks passed%s: %d passed, %d skipped.\n' "$C_PASS" "$C_OFF" "$PASSED" "$SKIPPED"
  fi
  echo "The engine's log lines for this run carry $RUN_ID (docs/logging.md says where the log is)."
}

finish() {
  summary
  [ "$FAILED" -eq 0 ]
  exit $?
}

echo "Estia smoke test"
echo "  engine  $URL"
echo "  model   $MODEL (chat), $EMBED (embeddings)"
if [ -n "$TOKEN" ]; then
  echo "  token   from $TOKEN_SOURCE"
else
  echo "  token   none"
fi
[ "$QUICK" = 1 ] && echo "  mode    quick: health, one chat, one embedding"
echo "  ids     requests carry X-Request-Id $RUN_ID-<check>"
echo
echo "The first check that uses a model loads it, which can take several seconds."
echo

# 1. Health: the one route that needs no token.
req health GET /engine/health
judge health health "$TIME" "$CODE" "$TMP/health.body" "$TMP/health.err"
if [ "$LAST_STATUS" != PASS ]; then
  echo
  echo "Stopped: the other checks need the engine's health route to answer first."
  finish
fi
AUTH_REQUIRED="$(python3 "$PY" health_field "$TMP/health.body" auth_required)"

if [ -z "$TOKEN" ] && [ "$AUTH_REQUIRED" != false ]; then
  echo
  echo "This engine needs a token, and none was given. Pass --token or --token-file, or set"
  echo "ESTIA_TOKEN. On the engine's machine:"
  echo "  estia token new smoke --scopes generate,embed,models:read"
  exit 2
fi

# 2. Auth: a request without a token is refused, with a request id.
if [ "$QUICK" = 0 ]; then
  if [ "$AUTH_REQUIRED" = false ]; then
    report SKIP - auth "the engine runs with --no-auth, so it accepts requests with no token"
  else
    req auth GET /v1/models
    judge auth auth "$TIME" "$CODE" "$TMP/auth.body" "$TMP/auth.head"
  fi
fi

# 3. Models.
if [ "$QUICK" = 0 ] && token_ok models; then
  req models GET /v1/models "" -H "@$AUTH"
  note_refusal
  judge models models "$TIME" "$CODE" "$TMP/models.body" "$TMP/models.err"
fi

# 4. One chat completion.
if token_ok chat; then
  python3 "$PY" chat_body "$MODEL" "Name one sea in Europe. Answer in one short sentence." 40 >"$TMP/chat.json"
  req chat POST /v1/chat/completions "$TMP/chat.json" -H "@$AUTH"
  note_refusal
  judge chat chat "$TIME" "$CODE" "$TMP/chat.body" "$TMP/chat.err"
fi

# 5. Streaming.
if [ "$QUICK" = 0 ] && token_ok stream; then
  python3 "$PY" stream_body "$MODEL" >"$TMP/stream.json"
  req stream POST /v1/chat/completions "$TMP/stream.json" -N -H "@$AUTH"
  note_refusal
  judge stream stream "$TIME" "$CODE" "$TMP/stream.body" "$TMP/stream.err"
fi

# 6. Prompt cache: the second turn of one conversation (same `user`) should
# reuse the first turn's prompt.
if [ "$QUICK" = 0 ] && token_ok "prompt cache"; then
  python3 "$PY" chat_body "$MODEL" "My name is Ada and I keep bees. Reply in one short sentence." 30 "$RUN_ID-cache" >"$TMP/cache1.json"
  req cache1 POST /v1/chat/completions "$TMP/cache1.json" -H "@$AUTH"
  note_refusal
  C1="$CODE" T1="$TIME"
  C2=000 T2=0
  : >"$TMP/cache2.body"
  : >"$TMP/cache2.err"
  if [ "$C1" = 200 ]; then
    python3 "$PY" cache_body2 "$MODEL" "$RUN_ID-cache" "$TMP/cache1.body" >"$TMP/cache2.json"
    req cache2 POST /v1/chat/completions "$TMP/cache2.json" -H "@$AUTH"
    C2="$CODE" T2="$TIME"
  fi
  judge "prompt cache" cache "$T1+$T2" "$C1" "$TMP/cache1.body" "$TMP/cache1.err" "$C2" "$TMP/cache2.body" "$TMP/cache2.err"
fi

# 7. JSON Schema output.
if [ "$QUICK" = 0 ] && token_ok "json schema"; then
  python3 "$PY" schema_body "$MODEL" >"$TMP/schema.json"
  req schema POST /v1/chat/completions "$TMP/schema.json" -H "@$AUTH"
  note_refusal
  judge "json schema" schema "$TIME" "$CODE" "$TMP/schema.body" "$TMP/schema.err"
fi

# 8. Embeddings.
if token_ok embeddings; then
  python3 "$PY" embed_body "$EMBED" >"$TMP/embed.json"
  req embed POST /v1/embeddings "$TMP/embed.json" -H "@$AUTH"
  note_refusal
  judge embeddings embed "$TIME" "$CODE" "$TMP/embed.body" "$TMP/embed.err" "$TMP/fingerprint"
fi

if [ "$QUICK" = 1 ]; then
  finish
fi

# 9. A wrong expect_fingerprint is refused with 422 before any vector is made.
if token_ok "fingerprint check"; then
  python3 "$PY" mismatch_body "$EMBED" >"$TMP/mismatch.json"
  req mismatch POST /v1/embeddings "$TMP/mismatch.json" -H "@$AUTH"
  note_refusal
  judge "fingerprint check" expect_status "$TIME" "$CODE" "$TMP/mismatch.body" "$TMP/mismatch.head" 422 "a wrong expect_fingerprint" "$TMP/mismatch.err"
fi

# 10. A path that matches no route is a JSON 404, token or not.
req notfound GET /v1/no-such-route "" -H "@$AUTH"
judge "unknown route" expect_status "$TIME" "$CODE" "$TMP/notfound.body" "$TMP/notfound.head" 404 "GET /v1/no-such-route" "$TMP/notfound.err"

# 11. The DNS-rebinding guard: a Host name the engine does not know is 403.
req host GET /engine/health "" -H 'Host: evil.example'
judge "host check" expect_status "$TIME" "$CODE" "$TMP/host.body" "$TMP/host.head" 403 "Host: evil.example" "$TMP/host.err"

# 12. The engine keeps a request id the client sends. /v1/models is logged at
# info level (health polls are debug), so this id shows in a default log.
req rid GET /v1/models "" -H "@$AUTH" -H "X-Request-Id: $RUN_ID-rid"
judge "request id" request_id "$TIME" "$CODE" "$TMP/rid.head" "$RUN_ID-rid"

finish
