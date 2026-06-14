# VERIFICATION

Reproducible report of what was actually run for `std-qwen3-asr`, separating
**verified on this M5 Max (no GPU)** from **requires a CUDA host** for the live
vLLM path. Honesty over impressiveness: every "verified" line below is backed by
output captured on this machine; every "requires a CUDA host" line is marked as
not run here, with the exact commands to run it elsewhere.

## STEP 1 — Adapter vs. fork: **thin adapter** (verdict)

The plugin is a **thin HTTP/WebSocket client**, not a fork of Qwen3-ASR.

**Why.** vLLM serves Qwen3-ASR behind an OpenAI-compatible HTTP API
(`/v1/audio/transcriptions`) and a Realtime WebSocket (`/v1/realtime`). DashScope
exposes Qwen3-ASR-Flash behind an OpenAI-compatible chat-completions endpoint.
All capabilities this plugin needs (batch transcription, incremental streaming,
language selection, the `context` biasing feature) are reachable over those wire
protocols. There is **no reason to fork** upstream: forking would pull vLLM/torch
and the model into the plugin's dependency and license surface, violating Standard
ASR's dependency/license isolation goal (G.4.2). The plugin therefore imports
**only** `standard-asr`, `httpx`, and `websockets`; vLLM is the *remote runtime*,
reached purely over the wire. The user's "vLLM backend" requirement is satisfied
by a thin client over vLLM's server API — confirmed.

## Environment (this machine)

```
$ uname -mns
Darwin tims-MacBook-Pro.local arm64        # Apple M5 Max, macOS, NO CUDA GPU

$ uv run python --version
Python 3.12.11                              # pinned via .python-version -> 3.12

$ uv run python -c "import vllm"
ModuleNotFoundError: No module named 'vllm' # vLLM is CUDA-first; not installed/usable here

$ printenv | grep -iE 'dashscope|openai_api'
  (none set)                                # no cloud API key available
```

**Consequence (an expected validation-phase blocker, handled head-on):** a real
vLLM server cannot run on this machine (vLLM is CUDA-first / GPU-only for
practical serving), and there is no DashScope/OpenAI key, so **no live model
inference was possible here**. The adapter's correctness is instead proven
end-to-end against a **mock vLLM/DashScope server** over real sockets (below), and
the real vLLM path is provided as copy-pasteable commands for a CUDA host.

---

## Verified on this M5 Max (mock server, real transports)

### 1. Full test suite — 94 tests, 100% coverage (enforced)

```
$ uv run pytest
...
Name    Stmts   Miss Branch BrPart  Cover   Missing
---------------------------------------------------
TOTAL     645      0    168      0   100%
Required test coverage of 100% reached. Total coverage: 100.00%
94 passed in 16.81s
```

The suite includes **integration tests against a real in-process server**
(`tests/fake_server.py`): a `uvicorn` FastAPI app for the HTTP/SSE surfaces and a
real `websockets` server for the vLLM Realtime path, each on a random localhost
port. The adapter connects to genuine `http://127.0.0.1:<port>` / `ws://...` URLs
over real sockets, so the true transport, request-building, SSE/WebSocket framing,
event mapping, gating, session lifecycle, and sync-bridge code all execute. No
GPU and no network egress are involved; a real model is not — the fake echoes a
scripted transcript and reflects the request back so tests assert the wire
contract.

What the integration tests cover (representative):
- **Batch / vLLM** — multipart upload, language (`zh-CN` -> `zh`), `prompt` ->
  context, `enable_itn`, bearer auth, backend-500 -> portable `TranscriptionError`.
- **Batch / DashScope** — chat-completions body shape (`input_audio` + system
  context), `annotations` -> `detected_language` + `extra["emotion"]`.
- **Streaming / vLLM Realtime (WebSocket)** — cumulative `partial` accumulation,
  single `seg-0`, `stable_until=0`, one `final` then `done`, session-param
  forwarding (`language`/`instructions`/`enable_itn`/`pcm16`), audio bytes
  received, server-abrupt-close handling, server error-event -> `engine_error`.
- **Streaming / SSE** — vLLM `transcription.chunk` and DashScope chat deltas.
- **Whole-input + streaming-output** — the real 48 kHz stereo reference clip,
  decoded/resampled by the standard layer, streamed.
- **Compliance** — `check_entrypoints`, `check_streaming_param_gating`,
  `check_provider_params_swap_safety`, `check_sync_bridge`, and
  `check_event_sequence` on a recorded real stream.

### 2. Real reference audio through full negotiation

The reference clip
`standard_asr/reference/standard_asr_test_audio_english.m4a` (48 kHz **stereo**,
~57 s) flows through the standard layer's real decode + resample + downmix to
16 kHz mono and is uploaded by the adapter (tests
`test_real_audio_negotiation_through_fake` and `test_whole_input_streaming_output`
in `tests/`). This proves the adapter's audio handling and upload work on a real
file on this machine. The transcript text returned is the mock's scripted value
(no model runs here).

### 3. Discovery + compliance CLI (all three presets)

```
$ uv run standard-asr models list
 - qwen3-asr/0.6b   engine=qwen3-asr  model=0.6b
 - qwen3-asr/1.7b   engine=qwen3-asr  model=1.7b
 - qwen3-asr/flash  engine=qwen3-asr  model=flash

$ uv run standard-asr compliance run qwen3-asr/1.7b
[OK] Entry point compliance checks passed.
[INFO] Streaming event-sequence is not run here; cover it with
       standard_asr.compliance.check_event_sequence in your tests ...
[OK] Compliance run passed.
```

(`qwen3-asr/flash` and `qwen3-asr/0.6b` print the same `[OK]`.)

```
$ uv run standard-asr doctor
Standard ASR doctor (Python 3.12)
Installed plugins:
  - qwen3-asr/0.6b [std-qwen3-asr] numpy None
  - qwen3-asr/1.7b [std-qwen3-asr] numpy None
  - qwen3-asr/flash [std-qwen3-asr] numpy None
No dependency conflicts detected.
```

### 4. Streaming demo against the mock server (captured output)

```
$ uv run python examples/stream_against_mock.py
--- streaming events ---
partial  seg=seg-0 stable_until=0 text='Hello from Qwen3-'
partial  seg=seg-0 stable_until=0 text='Hello from Qwen3-ASR '
partial  seg=seg-0 stable_until=0 text='Hello from Qwen3-ASR streaming.'
final    seg=seg-0 stable_until=0 text='Hello from Qwen3-ASR streaming.'
done     type='done' ...
--- reduced result ---
'Hello from Qwen3-ASR streaming.'
```

This is the exact Standard ASR event mapping the adapter produces: cumulative
`partial` text (not deltas), one deterministic `seg-0`, `stable_until=0`
throughout, a single `final`, then the base-appended `done`, and a reduced
result equal to the concatenated stream. (Some intermediate partials may be
coalesced by the standard layer's backpressure rule — expected, spec §6.4.)

### 5. Lint, types, build

```
$ uv run ruff check src tests examples
All checks passed!

$ uv run pyright src tests
0 errors, 0 warnings, 0 informations         # src is STRICT; tests are standard-level

$ uv build
Successfully built dist/std_qwen3_asr-0.1.0.tar.gz
Successfully built dist/std_qwen3_asr-0.1.0-py3-none-any.whl
```

### 6. Cloud (DashScope) inference — NOT run (no key)

The DashScope backend is fully implemented and unit/integration-tested against
the mock (request shape, `asr_options`, annotations, SSE streaming). **No real
DashScope transcript was captured** because no `DASHSCOPE_API_KEY` is set in this
environment. With a key it is a one-liner:

```bash
STANDARD_ASR_QWEN3_ASR__API_KEY=sk-... \
    uv run python -c "
from standard_asr import discover_models
asr = discover_models().create('qwen3-asr/flash')
print(asr.transcribe('reference/clip.mp3').text)
"
```

---

## Requires a CUDA host (real vLLM) — NOT run here; exact commands

vLLM cannot serve Qwen3-ASR on this Apple-Silicon machine. On an NVIDIA CUDA host
the live path is:

### (a) Launch the vLLM server (on the GPU host)

```bash
pip install "vllm[audio]"
vllm serve Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000
# or the Qwen wrapper, which forwards vllm args:
# qwen-asr-serve Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000
```

The server exposes `POST /v1/audio/transcriptions` (batch, multipart) and
`WS /v1/realtime` (streaming) — the two surfaces this adapter drives.

### (b) Point the plugin at it and verify (from any machine that can reach it)

Batch:

```bash
uv run python -c "
from standard_asr import discover_models, RuntimeParams
asr = discover_models().create('qwen3-asr/1.7b', base_url='http://<gpu-host>:8000/v1')
r = asr.transcribe('reference/clip.wav', params=RuntimeParams(language='en', prompt='Standard ASR, vLLM'))
print(r.text)
"
```

Streaming (Realtime WebSocket), using the provided example:

```bash
# Convert the reference clip to the required 16 kHz mono PCM16 first:
ffmpeg -i standard_asr_test_audio_english.m4a -ar 16000 -ac 1 -c:a pcm_s16le ref16k.wav

STD_QWEN3_VLLM_URL=http://<gpu-host>:8000/v1 \
    uv run python examples/stream_against_vllm.py ref16k.wav
```

Expected: live `partial ... <growing text>` lines followed by a `final` and the
full transcript — the same event shape verified against the mock in section 4,
but with real Qwen3-ASR output.

### Known upstream caveats to expect on the live path

- **Language forcing is unreliable** on vLLM / `qwen-asr-serve` today
  (Qwen3-ASR issue #93, vLLM #35767). The adapter still accepts and forwards
  `language` (the Standard ASR contract); treat it as best-effort on vLLM.
- The vLLM **Realtime segment size is hardcoded to ~5.0 s** with no server flag.
  This does not change the adapter's event mapping (we emit one `seg-0` stream),
  but it bounds latency/segmentation server-side.
- **Streaming returns no timestamps** (Qwen3-ASR). The adapter declares
  `streaming.timestamps = none` and `word_stability = false` accordingly.

---

## Summary

| Path | Status on this M5 Max |
|---|---|
| Mock-server batch + streaming, full event mapping | **Verified** (94 tests, 100% cov; captured demo) |
| Real reference audio through standard negotiation | **Verified** (uploaded; mock transcript) |
| Discovery, compliance CLI, doctor, build, lint, types | **Verified** |
| Real vLLM inference (batch + Realtime streaming) | **Not run** — needs a CUDA host; commands above |
| Real DashScope cloud transcript | **Not run** — needs `DASHSCOPE_API_KEY`; command above |
