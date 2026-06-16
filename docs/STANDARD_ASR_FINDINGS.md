# Standard ASR findings — adapting Qwen3-ASR (vLLM backend)

Every problem, blocker, DX gap, and design issue hit while building this plugin
against Standard ASR `main`. The goal is to stress-test the
protocol's developer experience, especially the **streaming** path. For each item:
what happened, why it mattered, and a concrete suggested improvement.

Overall the protocol held up very well: a real, two-backend, streaming-capable
HTTP adapter with 100% coverage was buildable in one pass, and the compliance
suite + spec answered most questions. The streaming model in particular is
well-thought-out — `stable_until`, cumulative/replace, and the segment lifecycle
mapped onto Qwen3-ASR's append-only token stream cleanly, and the spec even names
Qwen3-ASR streaming as the canonical `stable_until=0` case. The findings below are
mostly papercuts and HTTP/streaming-backend ergonomics, not design flaws.

Severity legend: **[blocker]** stopped progress until worked around ·
**[papercut]** cost time / confusion · **[enhancement]** would help future
HTTP/streaming adapters.

---

## A. Streaming

### A1. [enhancement] No author-facing way to emit a diagnostic from inside `_produce()`

**What happened.** Qwen3-ASR streaming reports a detected language and (on
DashScope) an emotion. I wanted to surface "language was auto-detected as X" or
"emotion=happy" as a structured `Diagnostic` on the live session, the way the
batch path can attach diagnostics to a `TranscriptionResult`. But
`session.diagnostics()` returns `[*self._initial_diagnostics, *self._guard.diagnostics]`
— the only ways diagnostics enter a session are (a) the base seeding gating/
language/conversion diagnostics at `start_transcription`, and (b) the lifecycle
guard. There is **no public method** like `self.emit_diagnostic(...)` for the
adapter to add its own during `_produce()`.

**Why it mattered.** Streaming adapters routinely learn things mid-stream worth
telling the app (a best-effort language fallback, a degraded segment, a provider
warning). The batch path has a first-class channel for this (`result.diagnostics`);
the streaming path's equivalent is closed to authors. I worked around it by
putting engine-specific info in `event.extra` and `event.detected_language`, but
those are not the diagnostics channel and `extra` is scrubbed on `error` events.

**Suggested improvement.** Add a protected `TranscriptionSession._emit_diagnostic(
diagnostic: Diagnostic)` (or accept author diagnostics yielded as a sentinel) that
appends to the same bounded, de-duplicated list `diagnostics()` reads, subject to
the existing `diagnostics_truncated` cap. Document it in `adapting_engine.md`
beside the reconnect helpers.

### A2. [papercut] vLLM has TWO streaming surfaces, and "the streaming path" is genuinely ambiguous

**What happened.** Researching how to stream Qwen3-ASR over vLLM surfaced two
distinct, incompatible transports: (1) SSE on `POST /v1/audio/transcriptions`
(`stream=true`, `transcription.chunk` deltas) and (2) a **WebSocket Realtime API**
at `/v1/realtime` (`transcription.delta` / `transcription.done`). Qwen3-ASR's
genuine incremental streaming is the WebSocket one; the SSE one is the generic
vLLM mechanism (validated mainly against Whisper) and is effectively
whole-input-then-stream-output. The spec's incremental model (`audio_format` +
`send_audio`) maps to the WebSocket; the SSE one is closer to the whole-input
`audio=...` path. Deciding which to call "streaming" and how to expose both
through one Standard ASR capability set took real thought.

**Why it mattered.** This is exactly the fragmentation Standard ASR exists to
hide, and the protocol *did* let me hide it (I default to Realtime, expose
`stream_transport="sse"` as a config knob, and route both through the same
session). But there was no guidance in the docs for "one engine, two streaming
transports with different input semantics."

**Suggested improvement.** A short `adapting_engine.md` note on multi-transport
engines: which Standard ASR capability/path each transport maps to
(`streaming_input` for incremental WS, `streaming_output`/whole-input for
SSE-after-upload), and that exposing the choice as a `provider`/init knob is the
expected pattern.

### A3. [papercut] Backpressure coalescing of partials is correct but under-advertised — it broke my first tests

**What happened.** My first streaming test asserted the exact sequence of
`partial` texts (`["the ", "the quick ", ...]`). It failed: the delivered
partials were `["the quick ", ...]` — earlier ones were dropped. This is the
spec's §6.4 backpressure rule (consecutive same-segment partials coalesce, keep
latest) working as designed, because the mock WS server emits all deltas faster
than the consumer drains them.

**Why it mattered.** It's the right behavior, but it's a sharp edge for *both*
adapter authors writing tests and app developers reasoning about what they'll
receive. The natural mental model ("I'll see every partial I emit") is wrong under
load, and nothing in `adapting_engine.md` flags it.

**Suggested improvement.** Call this out explicitly in `adapting_engine.md` and
the streaming §ST narrative: "partials are lossy under backpressure; only the
latest per segment is guaranteed; assert invariants (monotonic growth, final
text), not exact partial counts." A `check_event_sequence`-adjacent helper that
asserts "partials are growing prefixes of the final" would be a nice testing aid.

### A4. [enhancement] `stable_until=0` + `word_stability=false` is the spec-named case but easy to get subtly wrong

**What happened.** Qwen3-ASR streaming has no per-token timestamps / right-context,
so every event must carry `stable_until=0` and `word_stability` must be declared
`false`. The spec (ST §4.2 / capability list §9) names Qwen3-ASR streaming
*explicitly* as this case — excellent. The subtlety: there is a *third* coupled
thing, `streaming.timestamps`, which must also be `none`, and a *fourth*,
`audio_processed_until`, which I must **omit** (not fabricate). Getting all four
consistent (`stable_until=0`, `word_stability=false`, `timestamps=none`, no
cursor) is an implicit invariant scattered across §4.2, §4.4, §9.

**Why it mattered.** These four are really one decision ("this engine has no
time/stability information"), but they live in four places, and a compliant-
looking engine could declare `word_stability=false` yet still emit a fabricated
`stable_until>0` or cursor.

**Suggested improvement.** Either (a) a documented "no-timestamps streaming
profile" in `adapting_engine.md` listing the four coupled settings together, or
(b) a compliance check: if `word_stability=false`, assert no event in a recorded
stream carries `stable_until>0`; if `streaming.timestamps=none`, assert no
`audio_processed_until`/`start`/`end`. That turns an implicit invariant into a
verifiable one.

### A5. [papercut] `start_transcription(audio=...)` whole-input path hands me `prepared_audio` as an array, but the backend wants encoded/PCM bytes

**What happened.** For the whole-input + streaming-output path, the base
negotiates `audio=...` and hands `_start_transcription` a `PreparedAudio`. My
backends consume encoded bytes (batch) or raw PCM frames (streaming), so I had to
convert the `PreparedAudio` (array, or encoded bytes/path) into PCM16 myself,
re-deriving sample rate and handling the encoded-bytes case via
`load_audio_from_bytes`. The batch `_transcribe` path has the same shape problem.

**Why it mattered.** Every HTTP/cloud adapter (OpenAI, DashScope, vLLM, AWS) has
to turn `PreparedAudio` into "encoded bytes to upload" or "PCM to stream." Each
re-implements the same array->WAV / decode-bytes glue. The standard already owns a
canonical WAV encoder (it uses it for the array->encoded-file negotiation cell);
authors just can't reach it.

**Suggested improvement.** Expose a small helper on `PreparedAudio` or the
standard layer: `prepared.to_wav_bytes()` and/or `prepared.to_pcm16(sample_rate)`
that reuses the canonical quantization (the spec's clip+round_half+int16) so every
HTTP adapter produces byte-identical uploads without copying the routine. (I wrote
`_audio.float32_to_pcm16` / `wrap_pcm16_wav` to match the spec; this should be
shared, not re-derived per plugin.)

---

## B. Capabilities & properties

### B1. [papercut] `streaming_input` / `streaming_output` are top-level siblings, not under `streaming.*` — silent fail-closed bit me

**What happened.** I declared a full `StreamingCapabilities(...)` subtree
(`emits_partials`, `re_segments`, etc.) and assumed that meant "streaming
supported." But `engine.supports("streaming_input")` returned `False`:
`streaming_input` and `streaming_output` are **top-level** `FlagCap` fields on
`DeclaredCapabilities` (siblings of `batch`/`streaming`), defaulting to
`supported=False`. I had to set them explicitly. The compliance CLI passed
*without* them (it doesn't open a streaming session unless asked), so the gap was
silent until I queried `supports()` directly.

**Why it mattered.** It's a fail-closed footgun: a fully-populated `streaming`
subtree reads as "streaming declared" to a human, but the orthogonal axes that
actually gate `start_transcription` are elsewhere and default off. An author can
ship a "streaming" engine where every `start_transcription` call fails closed.

**Suggested improvement.** A compliance warning: "engine declares a non-empty
`streaming` subtree but `streaming_input`/`streaming_output` are both false
(no streaming path is reachable)." It mirrors the existing helpful warning for
`streaming_input` declared without `wire_encodings`.

### B2. [papercut] `word_timestamps` must be omitted entirely to mean "unsupported" — declaring the field with `granularities=[]` is a different, confusing state

**What happened.** Qwen's hosted/streaming responses return text only, so I want
"no word timestamps." The fail-closed way is to **omit** `word_timestamps` from
the capability tree. But `WordTimestampsCap` defaults to
`supported=False, granularities=[]`, and the canonical JSON shows
`"word_timestamps": {"supported": false, "granularities": []}` whether I declared
it or not. It took a careful read of the spec's "declare every granularity you can
deliver" guidance to be confident that omitting it (vs declaring an empty one) is
correct and that an empty `granularities` won't be misread as "supports the
feature with zero granularities."

**Why it mattered.** The "omit = unsupported" rule (R1) is clear, but
`word_timestamps` (and `phrase_hints`) have non-trivial defaults that serialize
identically to an explicit "unsupported" declaration, so it's easy to second-guess
whether you've actually opted out.

**Suggested improvement.** A one-line note in `adapting_engine.md`: "to declare a
guidance/timestamp channel unsupported, simply omit it; the default node
serializes as `supported:false` and is treated as absent." Reassurance removes the
second-guessing.

### B3. [enhancement] `selectable_languages` forces a `default_language`, but `CredentialsConfigMixin` doesn't include one — the failure surfaces only at first transcribe/compliance

**What happened.** I built my config on `CredentialsConfigMixin` (for
`api_key`/`base_url`). Because I declare `selectable_languages`, the standard
*requires* a `default_language` (IC.6 / LANG R1) — but that field lives on
`LanguageConfigMixin`, which I hadn't included. The result: `__init__` succeeded,
but `compliance entrypoints` and every transcribe failed with
`language_config_invalid`. The fix (add `LanguageConfigMixin`, default
`"auto"`) was easy once the compliance error pointed at it.

**Why it mattered.** The coupling "declared a language axis => MUST also mix in
`LanguageConfigMixin`/set `default_language`" is implicit. The compliance suite
*did* catch it with a clear message (great), but only at run time, not at
declaration time.

**Suggested improvement.** Either document the coupling prominently in
`adapting_engine.md` ("if your Properties set `selectable_languages`, your config
MUST provide `default_language` — mix in `LanguageConfigMixin`"), or have the
example/minimal engine in the docs show the language mixin so the pattern is
copied by default.

---

## C. HTTP / cloud-backend shape

### C1. [blocker→workaround] `ConfigError` / `TranscriptionError` accept no structured fields, unlike `UnsupportedFeatureError`

**What happened.** Following the pattern of `UnsupportedFeatureError(msg,
param=..., mode=..., hint=...)`, I wrote `ConfigError("bad base_url",
param="base_url")`. At runtime it raised `TypeError: ConfigError() takes no
keyword arguments` — `ConfigError`/`TranscriptionError` are bare
`StandardASRError`/`ValueError` subclasses with no `param`/`hint`. (pyright also
flagged it.) I removed the kwargs and folded the field name into the message.

**Why it mattered.** It's an inconsistency in the error API: some standard
exceptions carry structured, machine-readable context (`param`, `mode`, `hint`)
and some don't, and the ones that don't are exactly the ones most likely to want
`param=` (config field errors, runtime failures). An adapter author naturally
assumes symmetry and gets a `TypeError`.

**Suggested improvement.** Give `ConfigError` and `TranscriptionError` the same
optional structured fields (`param`, `hint`, and for `TranscriptionError` maybe
`retriable`) as `UnsupportedFeatureError`, even if unused internally. Symmetry
makes the error surface learnable and lets the server/`auto-UI` show the offending
field.

### C2. [papercut] Validators that raise `ConfigError` get wrapped into `pydantic.ValidationError`, so callers can't catch `ConfigError`

**What happened.** My pydantic `field_validator`/`model_validator` raise
`ConfigError` (the semantically correct type). But because they run inside
pydantic construction, the error surfaces to the caller as
`pydantic_core.ValidationError`, not `ConfigError`. My tests had to expect
`ValidationError`. An app that does `try: create(...) except ConfigError` would
miss config errors raised by validators.

**Why it mattered.** The spec maps both `ConfigError` and `ValidationError` to
HTTP 422, so the *server* is fine. But for **in-process** app code (the primary
Standard ASR audience), "catch `ConfigError` for bad config" silently doesn't
cover validator-raised config errors — a portability papercut.

**Suggested improvement.** Document the recommended pattern (validate in a
`model_validator` and let pydantic wrap, or validate post-construction and raise
`ConfigError` directly), and state clearly which exception in-process callers
should catch for construction-time config errors. Possibly have
`Config.from_env` re-wrap a `ValidationError` whose root cause is a `ConfigError`
back into `ConfigError`.

### C3. [enhancement] No shared SSE-parsing or "OpenAI-compatible chat/transcription" helper for the (many) cloud/vLLM adapters

**What happened.** I wrote SSE line parsing twice (vLLM `transcription.chunk` and
DashScope chat deltas), plus the multipart/transcription request shape and the
chat-completions request shape, plus typed-JSON extraction helpers to keep pyright
strict happy on `json.loads(...) -> Any`. Every OpenAI-compatible cloud/vLLM
adapter (OpenAI, DashScope, vLLM, Groq, Together, ...) will reimplement the same
SSE framing and the same `{"text": ...}` / `choices[0].delta.content` parsing.

**Why it mattered.** The plugin is supposed to be thin plumbing, but the
"OpenAI-compatible audio over HTTP" plumbing is non-trivial and is going to be
copy-pasted across the ecosystem, with subtle divergences (e.g. who strips
`data: [DONE]`, how `verbose_json` segments map).

**Suggested improvement.** An optional `standard-asr[http]` (or a tiny companion
package) with: an SSE line iterator, an OpenAI-transcription request/response
mapper, and a chat-completions audio mapper. Not core, but it would make
HTTP-backed plugins (a large fraction of the ecosystem) genuinely thin.

### C4. [papercut] The sync batch path forces every cloud adapter to invent its own "run this coroutine without an ambient loop" + client-lifecycle dance

**What happened.** My backends are async-first (httpx/websockets). The streaming
path is async and the base owns the loop, so that's clean. But the **batch**
`_transcribe` is synchronous, so I had to run the async client on a fresh event
loop and — to avoid leaking the httpx client's sockets when that loop closes —
close the client inside the same loop and recreate it next call. Getting this
right (no `ResourceWarning`, works under `transcribe_async`'s `asyncio.to_thread`,
no ambient-loop reuse) took a couple of iterations.

**Why it mattered.** Streaming gets a beautiful async-first base with a managed
sync bridge (`SyncSession`). Batch does not: an async-backed engine must hand-roll
the sync/async boundary for `transcribe`, and the loop/client lifecycle is exactly
the kind of footgun the streaming side already solved centrally.

**Suggested improvement.** Mirror the streaming sync-bridge convenience for batch:
let an engine implement an `async def _transcribe_async(...)` hook and have
`EngineBase` run it on a managed loop for the sync `transcribe` (the inverse of
the current `transcribe_async = to_thread(transcribe)` default). Async-first cloud
adapters would then never touch event-loop lifecycle.

---

## D. Smaller notes

- **[papercut] `load_audio_from_bytes` signature.** It's `(data, target_sr=...,
  target_channels=...)` returning a bare resampled `ndarray` — not the
  `(array, sr)` tuple I first assumed from the name. Minor, but a docstring example
  in `adapting_engine.md`'s "Audio you receive" section would prevent the guess.
- **[papercut] `PreparedAudio` is a dataclass, not a pydantic model**, so it has
  `.kind`/`.array`/`.data`/... as dataclass fields (no `model_fields`,
  no `prepared.kind` enum convenience like a `match` helper). Fine once known;
  the docs show `prepared.array`/`prepared.path` but not that it's a plain
  dataclass.
- **[positive] The compliance suite is genuinely good.** `check_entrypoints`,
  `check_streaming_param_gating`, `check_provider_params_swap_safety`,
  `check_sync_bridge`, and `check_event_sequence` caught real issues (the missing
  `default_language`, and they gave me confidence the streaming event stream is
  conformant). The "wire `check_event_sequence` into your own tests with a
  recorded stream" guidance is the right division of labor.
- **[positive] Spec specificity.** The spec naming Qwen3-ASR streaming as the
  `stable_until=0` case (ST §4.2), and the Qwen3 `context -> prompt` mapping
  example (Runtime §5.3), removed all ambiguity for the two hardest decisions in
  this adapter. More worked examples like these per engine archetype (cloud-chat,
  vLLM-transcription, realtime-WS) would accelerate future adapters.
- **[papercut] `uv` build warns on the PEP 639 license classifier.** Using
  `License :: OSI Approved :: Apache Software License` triggers a deprecation
  warning under the `uv` build backend; the cookbook templates don't show the
  modern `license = "Apache-2.0"` + `license-files = [...]` form. Updating the
  templates would save every plugin author the same warning.
