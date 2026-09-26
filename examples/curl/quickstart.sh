#!/usr/bin/env bash
# Estia from the command line: the calls a client makes, one step at a time.
#
# Scopes: generate, embed, models:read
#
#   estia token new quickstart --scopes generate,embed,models:read
#   ESTIA_TOKEN=estia_... ./quickstart.sh
#
# Reads ESTIA_URL (default http://127.0.0.1:27200), ESTIA_TOKEN and
# ESTIA_MODEL (default fast). Needs only curl. Long JSON is shortened with
# "..." so each step fits on screen.

set -u

URL="${ESTIA_URL:-http://127.0.0.1:27200}"
URL="${URL%/}"
TOKEN="${ESTIA_TOKEN:-}"
MODEL="${ESTIA_MODEL:-fast}"
CONV="quickstart-$$" # one conversation id: the prompt-cache key
OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

if [ -z "$TOKEN" ]; then
  echo "ESTIA_TOKEN is not set. On the engine's machine run" >&2
  echo "  estia token new quickstart --scopes generate,embed,models:read" >&2
  echo "and export the token it prints: export ESTIA_TOKEN=estia_..." >&2
  exit 2
fi

step() { printf '\n== %s\n' "$*"; }
say() { printf '   %s\n' "$*"; }

# call METHOD PATH [JSON]: the body goes to $OUT, the HTTP status to $STATUS.
call() {
  if [ $# -ge 3 ]; then
    STATUS=$(curl -sS -o "$OUT" -w '%{http_code}' -X "$1" "$URL$2" \
      -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d "$3") || STATUS=000
  else
    STATUS=$(curl -sS -o "$OUT" -w '%{http_code}' -X "$1" "$URL$2" \
      -H "Authorization: Bearer $TOKEN") || STATUS=000
  fi
}

# show [N]: the body, cut to N characters (default 700).
show() {
  local n="${1:-700}"
  if [ "$(wc -c <"$OUT")" -gt "$n" ]; then
    head -c "$n" "$OUT"
    printf ' ...\n'
  else
    cat "$OUT"
    echo
  fi
}

# ends: the first and last characters of the body, for long vector lists.
ends() {
  head -c 160 "$OUT"
  printf ' ... '
  tail -c 190 "$OUT"
  echo
}

step "1. Health: GET /engine/health needs no token. Is the engine up, which API version?"
call GET /engine/health
if [ "$STATUS" != 200 ]; then
  echo "cannot reach the engine at $URL (status $STATUS). Is 'estia serve' running? Set ESTIA_URL if it listens elsewhere." >&2
  exit 1
fi
show

step "2. Models: GET /v1/models checks the token (scope models:read). Roles come first."
call GET /v1/models
case "$STATUS" in
  200) show 500 ;;
  401)
    show
    echo "The engine refused the token. Check ESTIA_TOKEN, or mint one with 'estia token new'." >&2
    exit 1
    ;;
  403)
    show
    echo "The token lacks a scope. This script needs generate, embed and models:read." >&2
    exit 1
    ;;
  *)
    show
    exit 1
    ;;
esac

step "3. Chat: POST /v1/chat/completions with model \"$MODEL\" (a role) and user \"$CONV\" (the cache key)."
say "The first request to a model loads it, which takes a few seconds."
call POST /v1/chat/completions "{\"model\": \"$MODEL\", \"user\": \"$CONV\", \"max_tokens\": 60,
  \"messages\": [{\"role\": \"user\", \"content\": \"Name one sea in Europe. One sentence.\"}]}"
show 900
[ "$STATUS" = 200 ] || exit 1

step "4. The same conversation, one turn longer. Only the new turn is prefilled: see cached_tokens."
call POST /v1/chat/completions "{\"model\": \"$MODEL\", \"user\": \"$CONV\", \"max_tokens\": 60,
  \"messages\": [{\"role\": \"user\", \"content\": \"Name one sea in Europe. One sentence.\"},
                 {\"role\": \"assistant\", \"content\": \"The Aegean Sea.\"},
                 {\"role\": \"user\", \"content\": \"Which countries border it?\"}]}"
show 900

step "5. Streaming: \"stream\": true returns server-sent events, one 'data:' line per piece, then 'data: [DONE]'."
say "The last chunk before [DONE] carries usage and x_estia. Closing the connection cancels the generation."
curl -sS -N "$URL/v1/chat/completions" -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"model\": \"$MODEL\", \"stream\": true, \"max_tokens\": 40,
       \"messages\": [{\"role\": \"user\", \"content\": \"Count from one to five in words.\"}]}" |
  grep '^data:'

step "6. Embeddings: POST /v1/embeddings. task \"document\" for what you store, \"query\" for what you search with."
say "Keep x_estia.fingerprint with your vectors."
call POST /v1/embeddings '{"model": "embed", "task": "document", "input": ["The spare key is in the blue tin.", "Water the lemon tree twice a week."]}'
ends
FP=$(grep -o '"fingerprint":"[^"]*"' "$OUT" | head -n 1 | cut -d '"' -f 4)
say "fingerprint: ${FP:-(none in the response)}"

step "7. The native route: POST /engine/embed returns vectors, fingerprint and dims at the top level."
say "expect_fingerprint makes the engine refuse (422) if it would produce different vectors."
call POST /engine/embed "{\"inputs\": [\"where is the spare key?\"], \"task\": \"query\", \"expect_fingerprint\": \"$FP\"}"
show 260

step "8. JSON Schema output: response_format json_schema. The engine validates, repairs and retries once."
say "The model does not see the schema; describe the shape in the prompt too."
call POST /v1/chat/completions "{\"model\": \"$MODEL\", \"temperature\": 0, \"max_tokens\": 120,
  \"messages\": [{\"role\": \"system\", \"content\": \"Answer with JSON only: {\\\"city\\\": string, \\\"country\\\": string}\"},
                 {\"role\": \"user\", \"content\": \"Where is the Acropolis?\"}],
  \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"place\", \"schema\":
    {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"country\": {\"type\": \"string\"}},
     \"required\": [\"city\", \"country\"]}}}}"
show 900

step "9. Tools: declare a function; the model's call comes back as tool_calls (finish_reason \"tool_calls\")."
call POST /v1/chat/completions "{\"model\": \"$MODEL\", \"temperature\": 0, \"max_tokens\": 80,
  \"messages\": [{\"role\": \"user\", \"content\": \"What time is it in Tokyo?\"}],
  \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"current_time\",
    \"description\": \"Current local time in an IANA time zone\",
    \"parameters\": {\"type\": \"object\", \"properties\": {\"timezone\": {\"type\": \"string\"}}, \"required\": [\"timezone\"]}}}]}"
show 900

step "10. Errors share one shape: {\"error\": {\"message\", \"type\", \"code\", \"request_id\"}}."
say "No token (401):"
STATUS=$(curl -sS -o "$OUT" -w '%{http_code}' "$URL/v1/models")
printf '   %s ' "$STATUS"
show
say "Unknown model (404):"
call POST /v1/chat/completions '{"model": "no-such-role", "messages": [{"role": "user", "content": "hi"}]}'
printf '   %s ' "$STATUS"
show
say "More than 256 embedding inputs (400):"
INPUTS=$(printf '"x",%.0s' $(seq 1 257))
call POST /v1/embeddings "{\"model\": \"embed\", \"input\": [${INPUTS%,}]}"
printf '   %s ' "$STATUS"
show
say "Fingerprint mismatch (422):"
call POST /v1/embeddings '{"model": "embed", "input": "x", "expect_fingerprint": "embeddinggemma-300m-4bit@some-other-backend"}'
printf '   %s ' "$STATUS"
show
say "Schema that no answer can meet (422 after one retry):"
call POST /v1/chat/completions "{\"model\": \"$MODEL\", \"max_tokens\": 30,
  \"messages\": [{\"role\": \"user\", \"content\": \"Give me a number as {\\\"n\\\": number}.\"}],
  \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"n\", \"schema\":
    {\"type\": \"object\", \"properties\": {\"n\": {\"type\": \"integer\", \"minimum\": 1000, \"maximum\": 999}}, \"required\": [\"n\"]}}}}"
printf '   %s ' "$STATUS"
show

step "11. Request ids: every response has an X-Request-Id header, and error bodies repeat it as error.request_id."
say "Send your own id (1 to 64 letters, digits, '.', '_', ':' or '-') and the engine keeps it."
say "Log it with every error you show or report, so the operator can find the request."
RID="quickstart-$$-11"
curl -sS -D - -o "$OUT" "$URL/v1/models" -H "X-Request-Id: $RID" -H 'Authorization: Bearer estia_not_a_real_token' |
  tr -d '\r' | grep -i '^x-request-id' | sed 's/^/   /'
show
say "The engine logged this 401 with request_id=$RID. docs/logging.md says where the log is."

echo
echo "Done. Every call above works the same from any HTTP client. See docs/building-clients.md."
