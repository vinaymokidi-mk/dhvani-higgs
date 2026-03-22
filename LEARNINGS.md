# Dhvani — Hackathon Learnings (March 20-22, 2026)
## Boson x Eigen AI Hackathon

---

## Architecture Evolution

### v1 → v2 → v3

| Version | ASR | Brain | TTS | Pattern |
|---------|-----|-------|-----|---------|
| v1 | Higgs ASR 3 (cloud) | GPT-OSS (cloud) | Higgs TTS 2.5 | Sequential, 1 chunk at a time |
| v2 | Higgs ASR 3 (cloud) | Gemini 3.1 Flash Lite (cloud) | Higgs TTS 2.5 | 2-chunk batch |
| v3 | faster-whisper base.en (local ARM64) | Gemini 3.1 Flash Lite (cloud) | Higgs TTS 2.5 | 3-queue concurrent pipeline |

### Final Architecture (v3)

```
Audio chunks → [ASR Stage] → text_q → [Brain Stage] → tts_q → [TTS Stage] → WebSocket → Browser
```

Three async stages run concurrently via `asyncio.gather()`:
- While TTS speaks chunk N, Brain translates N+1, ASR transcribes N+2
- Bounded queues (`asyncio.Queue(maxsize=6)`) prevent memory runaway when TTS is slower than ASR+Brain

---

## Key Performance Numbers

| Stage | v1 (Higgs ASR) | v3 (faster-whisper ARM64) |
|-------|----------------|---------------------------|
| ASR | 1000ms OR 5-8000ms (bimodal) | 190ms flat |
| Brain | GPT-OSS 500ms → 429 rate limits | Gemini 700-1300ms, no limits |
| TTS | 3200ms | 3200ms (unchanged) |

---

## Critical Discoveries

### 1. Rosetta 2 Was Destroying faster-whisper Performance
**Symptom**: faster-whisper taking 30+ seconds per chunk on M1 Pro.
**Root cause**: Anaconda Python (`/opt/anaconda3/bin/python3`) is x86_64, runs under Rosetta 2 emulation.
**Tell**: `This system does not support SSE4.2` warning in faster-whisper output.
**Fix**: Rebuild venv with native ARM64 Python from Homebrew (`/opt/homebrew/bin/python3.13`).
**Result**: 190ms flat per chunk — 150x speedup.

**Lesson**: Always check `python3 -c "import platform; print(platform.machine())"` before benchmarking ML models. Should say `arm64` on M1/M2/M3.

### 2. Cloud ASR Has Bimodal Latency — Local Is More Predictable
**Observation**: Higgs ASR 3 (Eigen API) had two modes:
- Fast path: ~1000ms (warm)
- Slow path: 5-8000ms (cold start / load)

For a real-time dubbing demo, this bimodal behavior destroys sync.
**Lesson**: Local inference with flat latency beats cloud API with variable latency for latency-sensitive pipelines.

### 3. GPT-OSS Rate Limits Kill Every 3rd Chunk
**Symptom**: Translation skipped (SKIP logged) roughly every 2-3 chunks.
**Root cause**: GPT-OSS model has aggressive rate limits; 429 after burst.
**Fix**: Move to Gemini 3.1 Flash Lite as primary translator — no rate limits observed.
**Lesson**: Free-tier rate limits are fatal for any streaming pipeline. Test burst behavior before committing.

### 4. Gemini Returns Markdown in Translations
**Symptom**: Translation output had `**word**` bolding.
**Fix**: `re.sub(r'\*+', '', result).strip("\"'\`_ \n")`
**Lesson**: Always strip markdown from LLM outputs when the text feeds into TTS.

### 5. Timestamp-Aware Audio Sync Is Required for Streaming Dubbing
**Problem**: Processing takes 3.4s for a 3s audio chunk → audio always ~4-6s behind video.
**Fix**:
- Backend sends `start_s` / `end_s` with each chunk
- Frontend checks `ytPlayer.getCurrentTime()` vs chunk timestamps
- Late (video past `end_s + 0.3s`): discard silently
- Early (video ahead of `start_s`): poll every 50ms until video catches up
- On time: play immediately

**Lesson**: Sync state must live in the frontend against the video clock, not the processing clock.

### 6. TTS Hallucination — 57s Audio for a 3s Chunk
**Symptom**: Chunk 8 for Japanese returned 2,764,844 bytes (~57 seconds of audio). Was English monologue.
**Root cause**: Higgs TTS 2.5 with voice clone occasionally hallucinates English when the target language phoneme space differs drastically from the reference voice.
**Guard (optional)**: `if len(tts_raw) > 500_000: skip` — but issue resolved itself without this.
**Lesson**: Add a size/duration sanity check on TTS output before playing.

### 7. Parakeet TDT 0.6b-v2 Requires Python 3.10+ (Blocked on 3.9)
**Why we tried it**: NVIDIA Parakeet TDT 0.6b-v2 claims SOTA ASR at 0.6B params.
**Blocker chain**:
1. Parakeet requires NeMo 2.x
2. NeMo 2.x not on PyPI — only NeMo 1.x (max 1.23.0)
3. NeMo 1.23.0 fails with: `TypeError: __init__() got an unexpected keyword argument 'use_bias'` (ConformerEncoder)
4. NeMo 2.x requires Python 3.10+
**Conclusion**: Not viable for Python 3.9. Switch to faster-whisper instead.

### 8. OpenMP Conflict (faster-whisper + numpy)
**Error**: `OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized`
**Fix**: `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` at top of main.py
**Lesson**: Add this before any imports that use numpy or CTranslate2 on macOS.

### 9. Disk Cache Prevents Re-Processing on Resume
**Pattern**: `.cache/{video_id}_{lang}_{idx}.json` — stores transcript + translation per chunk.
**Benefit**: If WebSocket disconnects and reconnects, already-processed chunks serve instantly.
**Lesson**: Cheap cache = huge demo resilience for live demos.

### 10. Voice Clone Reference Selection
**Pattern**: Pick the loudest chunk from first 5 chunks (RMS threshold 0.01) as voice reference.
**Lesson**: Don't use the first chunk — it may be silence or intro music. Use loudest of first N.

---

## API Reference

### Higgs TTS 2.5 (Eigen)
- **Voice clone mode**: `multipart/form-data` POST, returns WAV file
- **Non-clone mode**: SSE streaming, returns PCM16 raw bytes
- **Endpoint**: via Boson AI base URL + API key

### Gemini 3.1 Flash Lite Preview
- **Model ID**: `gemini-3.1-flash-lite-preview`
- **Rate limits**: None observed during hackathon (free tier)
- **Async client**: `client.aio.models.generate_content(...)`
- **SDK**: `google-genai` package

### faster-whisper
- **Model**: `base.en` (English-only, fastest)
- **Settings**: `beam_size=1`, `language="en"`, `condition_on_previous_text=False`
- **Latency**: 190ms on ARM64 M1 Pro
- **Compute**: `device="cpu"`, `compute_type="int8"`

---

## Dependency Fixes

```
# Required versions (ARM64 venv, Python 3.13)
setuptools==69.5.1       # Fixes: No module named 'pkg_resources'
huggingface_hub==0.20.3  # Fixes: cannot import name 'ModelFilter'
pydantic>=2.9.0          # Fixes: cannot import name 'model_validator'
```

---

## What Would Make This Production-Ready

1. **CJK translation validation**: Check if output contains CJK unicode range before accepting
2. **TTS duration guard**: If `len(tts_raw) > 500_000`, log and skip (hallucination detection)
3. **Adaptive queue depth**: Increase `maxsize` if video is fast-forwarded
4. **Prefetch**: Start ASR+Brain for next chunk while current TTS plays
5. **Better voice clone**: Use a 10-15s clean reference clip, not the first loud chunk
6. **Word-level timestamps**: faster-whisper supports `word_timestamps=True` — use for karaoke-style on-screen text

---

## Final Stack

```
Browser (YouTube IFrame API)
  ↕ WebSocket (ws://)
FastAPI (Python 3.13, ARM64)
  ├── ASR: faster-whisper base.en (local, 190ms flat)
  ├── Brain: Gemini 3.1 Flash Lite Preview (cloud, 700-1300ms)
  │     └── Fallback: Higgs AST → GPT-OSS
  └── TTS: Higgs TTS 2.5 via Boson AI (cloud, 3200ms)
        └── Voice clone from loudest of first 5 chunks
```
