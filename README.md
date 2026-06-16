# std-qwen3-asr

> ⚠️ **Experimental — for protocol testing.** This is an experimental Standard ASR engine plugin, published to exercise and validate the [Standard ASR](https://github.com/standard-voice/standard_asr) interface. Expect breaking changes; it is not production-ready.

A [Standard ASR](https://github.com/standard-voice/standard_asr) engine plugin for
**Qwen3-ASR served over a vLLM backend**, with batch **and** streaming
transcription. Install it alongside `standard-asr` and any app that speaks the
Standard ASR protocol can transcribe through a Qwen3-ASR deployment — no app
code changes.

- **Package:** `std-qwen3-asr` · **import module:** `std_qwen3_asr`
- **Model keys:** `qwen3-asr/flash`, `qwen3-asr/1.7b`, `qwen3-asr/0.6b`
- **Adapter, not a fork.** This is a thin HTTP/WebSocket *client*. It never
  imports vLLM, torch, or Qwen's inference code — those run on the remote host.
  That keeps the plugin's dependency + license footprint tiny (Apache-2.0
  adapter) and isolated from the model runtime (Standard ASR goal G.4.2).

> **vLLM is CUDA-first.** Serving Qwen3-ASR requires an NVIDIA GPU host. This
> plugin runs anywhere (it is just a client), but the *server* it talks to needs
> CUDA. See [What you need to run it](#what-you-need-to-run-it) and
> [`VERIFICATION.md`](VERIFICATION.md).

## What it provides

| Capability | Status | Notes |
|---|---|---|
| Batch (`transcribe`) | yes | Multipart upload to the OpenAI-compatible `/v1/audio/transcriptions` (vLLM) or chat-completions (DashScope). |
| Streaming (`start_transcription`) | yes | Incremental PCM frames over the vLLM **Realtime** WebSocket (default) or SSE; whole-input + streaming-output too. |
| Language override (`language`) | yes | BCP-47 -> Qwen language. Best-effort on vLLM (upstream language forcing is currently unreliable — see findings). |
| Context biasing (`prompt`) | yes | Qwen3-ASR's headline feature: free-text context mapped from the portable `prompt` channel (Standard ASR spec §5.3). |
| Inverse text normalization | yes | Via `Qwen3ASRParams(enable_itn=True)` (Chinese/English only upstream). |
| Emotion annotation | yes (DashScope) | Surfaced in `result.extra["emotion"]`. |
| Word/segment timestamps | no | The hosted REST response and the streaming path return text only; declared **unsupported** (fail-closed) rather than faked. |
| `phrase_hints` | no | Qwen biases via free-text `context`, not a structured term list — pass terms inside `prompt`. |

### Presets

| Key | Backend (default) | Model | Model license |
|---|---|---|---|
| `qwen3-asr/flash` | DashScope (cloud) | `qwen3-asr-flash` | Proprietary hosted service |
| `qwen3-asr/1.7b` | vLLM (self-hosted) | `Qwen/Qwen3-ASR-1.7B` | Apache-2.0 |
| `qwen3-asr/0.6b` | vLLM (self-hosted) | `Qwen/Qwen3-ASR-0.6B` | Apache-2.0 |

Each preset is a separate entry point (Standard ASR spec IC.7: model selection is
the preset you choose, never an init `model` field). The default backend follows
the preset (cloud Flash -> DashScope; open weights -> vLLM) but is overridable via
config `backend=...`.

## Install

> **Not yet published to PyPI** — install from GitHub:

```bash
uv pip install git+https://github.com/standard-voice/std-qwen3-asr
# Optional: high-quality audio decode/resample for the batch path
uv pip install "std-qwen3-asr[audio] @ git+https://github.com/standard-voice/std-qwen3-asr.git"
```

This package depends only on `standard-asr` (from GitHub `main`), `httpx`, and
`websockets` — no vLLM, no torch. Once published to PyPI this becomes
`uv pip install std-qwen3-asr`.

## Quick start

### Discover it

```bash
standard-asr list
#  - qwen3-asr/flash  engine=qwen3-asr  model=flash
#  - qwen3-asr/1.7b   engine=qwen3-asr  model=1.7b
#  - qwen3-asr/0.6b   engine=qwen3-asr  model=0.6b
```

### Batch transcription (vLLM)

```python
from standard_asr import discover_models, RuntimeParams

asr = discover_models().create(
    "qwen3-asr/1.7b",
    base_url="http://your-gpu-host:8000/v1",   # your vLLM server
)
result = asr.transcribe(
    "meeting.wav",
    params=RuntimeParams(
        language="en",                            # BCP-47; best-effort on vLLM
        prompt="Acme Corp, Q3 OKRs, Kubernetes",  # Qwen context biasing
    ),
)
print(result.text)
```

### Streaming transcription (vLLM Realtime)

```python
import asyncio
from standard_asr import discover_models
from standard_asr.audio_format import AudioFormat

async def main():
    asr = discover_models().create("qwen3-asr/1.7b", base_url="http://your-gpu-host:8000/v1")
    fmt = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)
    async with asr.start_transcription(audio_format=fmt) as session:
        session.feed(microphone_pcm16_chunks())   # your async/sync byte source
        async for event in session:
            if event.type == "partial":
                print("...", event.text)           # cumulative text so far
            elif event.type == "final":
                print("ok", event.text)

asyncio.run(main())
```

Streaming notes (see [`docs/STANDARD_ASR_FINDINGS.md`](docs/STANDARD_ASR_FINDINGS.md)
for the full rationale):

- The wire **must** be 16 kHz mono `pcm_s16le` (the vLLM Realtime requirement);
  an off-rate session fails loudly at establishment.
- Qwen3-ASR streams **append-only token deltas**. The adapter accumulates them
  into cumulative `partial` text (Standard ASR requires cumulative/replace, never
  deltas on the wire) and emits one `final` at the end.
- **`stable_until` is always `0`** and `word_stability` is declared `false`: the
  stream carries no per-token timestamps or right-context, so no prefix can
  honestly be frozen. (The Standard ASR spec names Qwen3-ASR streaming as exactly
  this case.) Simple subtitle apps ignore `stable_until` anyway; voice-assistant
  apps must not expect a stable prefix from this engine.

### Hosted DashScope (Qwen3-ASR-Flash)

```python
asr = discover_models().create(
    "qwen3-asr/flash",
    api_key="sk-...",          # or env STANDARD_ASR_QWEN3_ASR__API_KEY / DASHSCOPE_API_KEY
)
result = asr.transcribe("speech.mp3", params=RuntimeParams(language="zh"))
print(result.text, result.detected_language, result.extra.get("emotion"))
```

## Configuration

All fields are settable as `create(...)` kwargs or via
`STANDARD_ASR_QWEN3_ASR__<FIELD>` environment variables (Standard ASR IC.4; note
the double underscore).

| Field | Default | Meaning |
|---|---|---|
| `backend` | preset-specific | `"vllm"` or `"dashscope"`. |
| `base_url` | backend default | vLLM OpenAI-compatible root, or DashScope compatible-mode root. |
| `api_key` | `None` | `SecretStr`. Required for DashScope; optional for an unauthenticated vLLM server. |
| `default_language` | `"auto"` | BCP-47 or `"auto"` (Qwen multilingual auto-detect). |
| `stream_transport` | `"realtime"` | vLLM streaming transport: `"realtime"` (WebSocket) or `"sse"`. |
| `connect_timeout` / `read_timeout` | `10` / `60` | Seconds. |
| `verify_tls` | `True` | Set `False` only for a self-signed local dev server. |

Engine-specific decode knobs live in `Qwen3ASRParams` (the typed `provider_params`
escape hatch): `enable_itn`, `temperature`, `top_p`, `max_completion_tokens`,
`emotion`.

**Security:** `api_key` is a `SecretStr` (never logged/echoed as plaintext).
`base_url` is validated to be HTTP(S); a non-loopback DashScope `base_url` must be
HTTPS.

## What you need to run it

| You have... | Then... |
|---|---|
| A CUDA GPU host | `pip install "vllm[audio]"` then `vllm serve Qwen/Qwen3-ASR-1.7B`; point `base_url` at it. Full streaming + batch. |
| A DashScope API key | Use the `qwen3-asr/flash` preset; no GPU needed (it's a cloud call). |
| Neither (e.g. an Apple-Silicon Mac) | You can still develop and test against the bundled mock server: `uv run python examples/stream_against_mock.py`. vLLM cannot serve here (CUDA-first). |

### Launch a vLLM server (CUDA host)

```bash
pip install "vllm[audio]"
vllm serve Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000
# Qwen also ships a thin wrapper: qwen-asr-serve Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000
```

Then verify from this package (on any machine that can reach the host):

```bash
STD_QWEN3_VLLM_URL=http://<gpu-host>:8000/v1 \
    uv run python examples/stream_against_vllm.py /path/to/16k-mono.wav
```

## Examples

- [`examples/stream_against_mock.py`](examples/stream_against_mock.py) — GPU-free
  streaming demo against the bundled fake vLLM server.
- [`examples/stream_against_vllm.py`](examples/stream_against_vllm.py) — the real
  vLLM Realtime path (needs a CUDA host).

## Development

```bash
uv sync --all-extras --group dev
uv run pytest                       # 94 tests, 100% coverage (enforced)
uv run ruff check src tests examples && uv run ruff format --check src tests
uv run pyright src tests
uv run standard-asr compliance run qwen3-asr/1.7b
```

Tests run entirely offline: an in-process fake server reproduces the vLLM
(HTTP + Realtime WebSocket) and DashScope wire shapes, so the full batch +
streaming paths are exercised without a GPU. See
[`VERIFICATION.md`](VERIFICATION.md) for exactly what was run and what requires a
CUDA host.

## Licensing (honest boundary, Standard ASR G.4.2)

- **This adapter** is Apache-2.0 (`LICENSE`).
- **The model runtime is vLLM**, run as a separate remote server — it is not a
  dependency of this package. Its license and the model weights' license are the
  operator's concern: the open-weight `Qwen/Qwen3-ASR-{0.6B,1.7B}` checkpoints are
  Apache-2.0; the hosted `qwen3-asr-flash` is a proprietary Alibaba Cloud service.
- This package adds **no model weights** and pulls in **no GPU/ML dependencies**.

## Links

- Standard ASR: https://github.com/standard-voice/standard_asr
- Qwen3-ASR (upstream): https://github.com/QwenLM/Qwen3-ASR
- vLLM Qwen3-ASR recipe: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html
