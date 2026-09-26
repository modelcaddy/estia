// Streaming chat against Estia with the OpenAI SDK for JavaScript. Node 18+.
//
// What it shows:
//   - baseURL + apiKey is all the SDK needs
//   - streaming, and reading Estia's facts (`x_estia`, cached tokens) from the
//     last chunk
//   - two turns with the same `user`, so the second turn reuses the prompt
//     cache and prefills only what is new
//   - cancelling with Ctrl-C: aborting the request closes the connection and
//     the engine stops generating
//
// Scopes: generate
//
//   export ESTIA_TOKEN=estia_...      # estia token new web-app --scopes generate
//   node chat.mjs "Suggest a name for a small sailing boat."

import OpenAI from 'openai';
import { randomUUID } from 'node:crypto';

const URL = (process.env.ESTIA_URL ?? 'http://127.0.0.1:27200').replace(/\/+$/, '');
const TOKEN = process.env.ESTIA_TOKEN ?? '';
const MODEL = process.env.ESTIA_MODEL ?? 'fast'; // a role; the engine picks the model

if (!TOKEN) {
  console.error(
    'ESTIA_TOKEN is not set. On the engine\'s machine run\n' +
      '  estia token new web-app --scopes generate\n' +
      'and export the token it prints: export ESTIA_TOKEN=estia_...',
  );
  process.exit(2);
}

function explain(e) {
  if (e instanceof OpenAI.APIConnectionError) {
    return `cannot reach the engine at ${URL}. Is \`estia serve\` running? Set ESTIA_URL if it listens elsewhere.`;
  }
  if (e instanceof OpenAI.APIError && e.status) {
    const message = e.error?.message ?? e.message;
    let text = `the engine answered ${e.status}: ${message}`;
    // The engine's X-Request-Id; the same id is on its log lines for this request.
    const requestId = e.requestID;
    if (requestId) text += ` (request id ${requestId})`;
    if (e.status === 401) text += '\nCheck ESTIA_TOKEN, or mint a token: estia token new web-app --scopes generate';
    return text;
  }
  return String(e?.message ?? e);
}

// maxRetries: 0 — a local engine that refused a request will refuse it again.
const client = new OpenAI({ baseURL: `${URL}/v1`, apiKey: TOKEN, maxRetries: 0 });

// One id per conversation: the engine's prompt-cache key, scoped to this token.
const conversation = `js-${randomUUID()}`;
const messages = [{ role: 'system', content: 'You are a concise assistant. Answer in one or two sentences.' }];

async function turn(text) {
  messages.push({ role: 'user', content: text });
  console.log(`\nyou> ${text}`);
  const stream = await client.chat.completions.create({
    model: MODEL,
    messages,
    stream: true,
    user: conversation,
    max_tokens: 400,
  });
  process.stdout.write('assistant> ');
  // Ctrl-C aborts the request. The SDK closes the connection and the engine
  // cancels the generation. Depending on the SDK version the loop below
  // either throws APIUserAbortError or just ends, so check the signal too.
  const onInt = () => stream.controller.abort();
  process.once('SIGINT', onInt);
  let reply = '';
  let usage;
  let extra;
  try {
    for await (const chunk of stream) {
      const piece = chunk.choices?.[0]?.delta?.content;
      if (piece) {
        reply += piece;
        process.stdout.write(piece);
      }
      if (chunk.usage) usage = chunk.usage;
      if (chunk.x_estia) extra = chunk.x_estia;
    }
  } catch (e) {
    if (!(e instanceof OpenAI.APIUserAbortError)) throw e;
  } finally {
    process.off('SIGINT', onInt);
  }
  if (stream.controller.signal.aborted) {
    console.log('\n[cancelled]');
    process.exit(130);
  }
  messages.push({ role: 'assistant', content: reply });
  console.log(
    `\n[${usage?.prompt_tokens} prompt tokens, ${usage?.prompt_tokens_details?.cached_tokens ?? 0} from cache, ${extra?.ms} ms]`,
  );
}

try {
  await turn(process.argv.slice(2).join(' ') || 'Suggest a name for a small sailing boat.');
  await turn('Give me one more, shorter.');
} catch (e) {
  console.error(`\nerror: ${explain(e)}`);
  process.exit(1);
}
