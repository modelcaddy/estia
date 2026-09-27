"""Ask a model about a picture on disk.

    python vision.py invoice.png "What is the total?"

Reads ESTIA_URL (default http://127.0.0.1:27200) and ESTIA_TOKEN. The token
needs the `generate` scope; the `vision` role (Gemma 4 E4B by default) reads
the image.
"""

import base64
import mimetypes
import os
import sys

from openai import APIStatusError, OpenAI


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 2
    path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "Describe this picture. If it contains text, transcribe it."
    token = os.environ.get("ESTIA_TOKEN")
    if not token:
        print("set ESTIA_TOKEN to a token with the generate scope (estia token new vision --scopes generate)")
        return 2
    base = os.environ.get("ESTIA_URL", "http://127.0.0.1:27200").rstrip("/")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    client = OpenAI(base_url=f"{base}/v1", api_key=token)
    try:
        r = client.chat.completions.create(
            model=os.environ.get("ESTIA_MODEL", "vision"),
            # Reading a picture is transcription: greedy decoding reads text
            # far more reliably than sampling.
            temperature=0,
            max_tokens=400,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            ]}],
        )
    except APIStatusError as e:
        print(f"the engine answered {e.status_code}: {e.message} (request id {e.request_id})")
        return 1
    print(r.choices[0].message.content)
    x = getattr(r, "x_estia", None) or (r.model_extra or {}).get("x_estia", {})
    print(f"\n[{r.model}, {r.usage.prompt_tokens} prompt tokens, {x.get('ms')} ms]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
