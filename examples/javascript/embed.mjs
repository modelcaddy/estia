// Embeddings from JavaScript: rank a few passages against a question. Node 18+.
//
// What it shows:
//   - Estia's extra request fields (`task`, `expect_fingerprint`) passed next
//     to the standard ones; the SDK sends them through unchanged
//   - `task: 'document'` for what you store and `task: 'query'` for what you
//     search with, so the engine adds the embedding model's own prefixes
//   - the fingerprint in `x_estia`: store it with your vectors and send it
//     back as `expect_fingerprint`
//   - `encoding_format: 'float'`: Estia always returns float arrays; saying so
//     stops the SDK from asking for base64 and trying to decode it
//
// Scopes: embed
//
//   export ESTIA_TOKEN=estia_...      # estia token new search --scopes embed
//   node embed.mjs "where is the spare key?"

import OpenAI from 'openai';

const URL = (process.env.ESTIA_URL ?? 'http://127.0.0.1:27200').replace(/\/+$/, '');
const TOKEN = process.env.ESTIA_TOKEN ?? '';

if (!TOKEN) {
  console.error(
    'ESTIA_TOKEN is not set. On the engine\'s machine run\n' +
      '  estia token new search --scopes embed\n' +
      'and export the token it prints: export ESTIA_TOKEN=estia_...',
  );
  process.exit(2);
}

const client = new OpenAI({ baseURL: `${URL}/v1`, apiKey: TOKEN, maxRetries: 0 });

const passages = [
  'The spare house key is in the blue tin on the top shelf of the garage.',
  'Water the lemon tree twice a week in summer.',
  'The router admin page is at 192.168.1.1.',
  'Fasolada needs the beans soaked overnight.',
];
const question = process.argv.slice(2).join(' ') || 'where is the spare key?';

function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

try {
  const docs = await client.embeddings.create({
    model: 'embed',
    input: passages, // at most 256 inputs per request
    encoding_format: 'float',
    task: 'document',
  });
  const fingerprint = docs.x_estia?.fingerprint;
  console.log(`${docs.data.length} passages, ${docs.x_estia?.dims} dims, fingerprint ${fingerprint}`);

  const query = await client.embeddings.create({
    model: 'embed',
    input: question,
    encoding_format: 'float',
    task: 'query',
    expect_fingerprint: fingerprint, // 422 if the engine's model or backend changed
  });
  const q = query.data[0].embedding;

  console.log(`\nquestion: ${question}`);
  passages
    .map((text, i) => ({ text, score: cosine(q, docs.data[i].embedding) }))
    .sort((a, b) => b.score - a.score)
    .forEach(({ text, score }) => console.log(`  ${score.toFixed(3)}  ${text}`));
} catch (e) {
  if (e instanceof OpenAI.APIConnectionError) {
    console.error(`error: cannot reach the engine at ${URL}. Is \`estia serve\` running?`);
  } else if (e instanceof OpenAI.APIError && e.status) {
    const requestId = e.requestID; // the engine's X-Request-Id
    console.error(`error: the engine answered ${e.status}: ${e.error?.message ?? e.message}${requestId ? ` (request id ${requestId})` : ''}`);
    if (e.status === 401) console.error('Check ESTIA_TOKEN, or mint a token: estia token new search --scopes embed');
  } else {
    console.error(`error: ${e?.message ?? e}`);
  }
  process.exit(1);
}
