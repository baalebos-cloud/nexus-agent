# Project Nexus-Agent — Build Status & Setup

This archive is a working snapshot of the architecture built so far. It is
**not a finished product** — see "Known gaps" below before you assume
anything works end-to-end with live services.

## What's verified working (31/31 automated tests passing)

- **Sandboxed Execution Engine** (`src/sandbox/`) — full E2B-backed code/
  shell/git-clone execution, Redis session registry, TTL reaper, policy
  enforcement. Tested with mocked E2B/Redis — never run against a live E2B
  account.
- **Video OSINT Pipeline** (`src/osint/`) — scope-gated face detection +
  embedding (DeepFace) + vector search (Qdrant), with a hard 0.7
  cosine-similarity confidence threshold enforced structurally. Face
  detection now works with **zero configuration** — defaults to a Haar
  cascade bundled inside `opencv-python-headless` (no download needed),
  verified against a real test image and a real generated video, not
  mocks. YuNet remains available as an optional higher-accuracy upgrade.
  Embedding/Qdrant search still tested with mocked Qdrant only.
- **Orchestrator** (`src/orchestrator/`) — LangGraph state machine wiring
  both of the above as LLM tool calls. Verified to compile and execute
  against real `langgraph`/`langchain-core`, using a fake LLM (no real
  model calls made).
- **LLM client wiring** (`src/api/llm.py`) — `build_chat_llm()` builds a
  real `ChatOpenAI` client from `.env` config, works with either Groq or
  self-hosted vLLM (both expose OpenAI-compatible endpoints). Verified to
  return `None` gracefully when unconfigured and a correctly-configured
  client when it is — never tested against a live Groq/vLLM endpoint.
- **API** (`src/api/`) — FastAPI app with `/health`, `/chat`,
  `/sandbox/sessions/{id}`, `/osint/analyze-video`. Verified booting through
  its real startup/shutdown lifecycle against a genuinely running local
  Redis instance, with correct graceful-degradation (503) when the chat LLM
  or OSINT model aren't configured.

## Known gaps — nothing below this line has been run against a real service

1. **No live E2B account has ever been used.** `src/sandbox/e2b_backend.py`
   is untested against the real E2B API.
2. **No face detector model file is bundled** (Git LFS binaries aren't
   fetchable from the build environment this was created in). See
   `src/osint/face_detector_loader.py`'s docstring for the download command.
3. **No live Groq/vLLM endpoint has been used.** `src/api/llm.py` is wired
   and unit-tested, but never called against a real inference endpoint —
   fill in `STREAMING_LLM_BASE_URL`/`STREAMING_LLM_API_KEY` in `.env` and
   restart the app to activate `/chat`.
4. **Voice stack — connected to a real LiveKit room, three real bugs found and fixed.**
   Two live test sessions surfaced three genuine issues, all now fixed
   and covered by regression tests using real SDK types (not mocks):
   (1) spoken input passed a raw `livekit.agents.llm.ChatMessage` straight
   into the LangGraph state, which langchain's message coercion rejected —
   `.text_content` is now extracted first; (2) typed chat-box input (e.g.
   LiveKit Cloud's console) went through a different default path
   requiring `AgentSession`'s own `llm`, never configured since this
   architecture routes everything through LangGraph — a custom
   `text_input_cb` now routes typed input through the same
   `generate_reply_text()` path as spoken input; (3) **every turn was
   provisioning a brand-new E2B sandbox instead of reusing one for the
   whole call** — confirmed live, with 3 different `sandbox_id`s logged
   across a single conversation. `NexusVoiceAgent` now persists
   `sandbox_session_id` across turns so the orchestrator's existing
   session-reuse logic actually gets used, instead of wasting billable
   sandbox-hours on every single utterance. Also migrated off the
   deprecated `RoomInputOptions`/`RoomOutputOptions` params (SDK warning,
   confirmed) to the current `RoomOptions`. The worker has successfully
   registered with a real LiveKit Cloud project, received real job
   dispatches, and processed real spoken turns without the earlier
   crashes — full audio-out confirmation (do you actually hear the reply)
   is the one thing not yet explicitly confirmed back to me. Run with:
   ```bash
   python3 -m livekit.agents start src/voice/worker.py --dev
   ```
   Then connect via https://agents-playground.livekit.io or your LiveKit
   Cloud project's Console.
5. **Backend-failure hardening — implemented and regression-tested.** A
   live Redis outage during a real call was observed causing
   `on_user_turn_completed` to raise unhandled — the caller heard nothing
   and had no way to know the agent had failed versus just being slow.
   Fixed: any exception from `generate_reply_text` (Redis down, E2B
   unreachable, LLM erroring, etc.) now produces a spoken error message
   ("Sorry, I hit a technical problem processing that...") instead of
   silence, on both the voice and text-input paths. Proven by
   deliberately reverting the fix and confirming the regression test
   reproduces the exact live traceback (same call stack:
   `provision_node` → `ensure_session_for_thread` → `create_session`).
6. **Fine-tuning pipeline — data layer built and fully tested; the actual
   training run has not been executed anywhere, including here.**
   `src/finetune/` implements the data side of QLoRA fine-tuning per
   PROJECT_CONTEXT.md's Model Layer: `data_schema.py` (Pydantic schemas
   for training conversations, matching the real tool-calling shape the
   live agent already produces, with a structural safeguard requiring
   explicit `reviewed` tagging before any OSINT-capture transcript can
   become training data), `formatting.py` (real Qwen2.5 ChatML +
   tool-call tag formatting, and an alternate Llama 3.3 header-based
   formatter), `dataset_prep.py` (load/validate/split/write), and
   `capture.py` (turns real `/chat` and voice conversations into training
   data automatically — see below). All genuinely tested — 32 tests
   across the fine-tuning module, plus a real (not mocked) tokenization
   dry-run using a tokenizer trained from scratch, offline, proving the
   formatted ChatML text is correctly structured and its special tokens
   (`<|im_start|>`, `<|im_end|>`) survive tokenization intact.
   `train_qlora.py` is a complete, real training script against
   documented `transformers`/`peft`/`trl`/`bitsandbytes` APIs — its CLI
   argument parsing is verified working, but the actual training path was
   **never executed**: installing the required torch/CUDA/bitsandbytes
   stack (torch alone is a 555MB wheel plus several GB of CUDA
   dependencies) risked exhausting this build environment's ~5GB of free
   disk entirely, and a 70B-parameter QLoRA run needs real GPU
   infrastructure this environment doesn't have regardless. Run
   `python -m src.finetune.train_qlora --help` to confirm the CLI works
   on your machine, then install the commented-out heavy dependencies in
   `requirements.txt` on your actual training infra before a real run.

## Live-usage capture: real conversations become training data

`src/finetune/capture.py` bridges live `/chat` and voice usage to the
fine-tuning dataset — without it, building a training set means
hand-authoring synthetic examples indefinitely. Wired into both
`src/api/routes.py`'s `/chat` endpoint and `src/voice/worker.py`.

**Off by default** (`CAPTURE_TRAINING_DATA=false` in `.env`) — capturing
conversation transcripts has real privacy implications, so this is
explicit opt-in, never on by default. When enabled, every completed
conversation is written as a validated `TrainingExample` to
`TRAINING_CAPTURE_PATH`.

**Structural OSINT safeguard**: any conversation that invoked an OSINT
tool is **never** auto-written as valid training data — `data_schema.py`'s
`TrainingExample` already refuses to construct a
`source="live_capture_osint"` example without a `reviewed` tag, and no
automated code path can add that tag. Instead, OSINT-touched conversations
are quarantined as unvalidated raw JSON to
`TRAINING_CAPTURE_PENDING_REVIEW_PATH`. A human reviews that file and
calls `capture.promote_reviewed_capture()` with the approved entry IDs to
deliberately promote specific conversations into the training set — this
is the *only* path by which OSINT-touched data can become training data,
and it requires an explicit human decision every time.

Capture never raises — a failure here must never break the actual
chat/voice response the user is waiting on; failures are logged and
swallowed. Verified end-to-end (not just unit-tested) via
`tests/api/test_chat_capture_integration.py`, which hits the real
`/chat` route with a fake LLM and confirms a genuine `TrainingExample`
gets written to disk.

## Resolved: OSINT identity-matching — now proven end-to-end, zero mocks

The full chain (video → face detection → DeepFace embedding → Qdrant
indexing → Qdrant vector search) was run for real, with nothing mocked:
a real generated test video, a real DeepFace model download and
inference, and a real Qdrant instance (in-memory mode). It correctly
identified a self-match (similarity=1.00) and correctly rejected a random
different vector (0.01, below the 0.7 threshold). This run caught and
fixed a real bug: `qdrant_client.search()` doesn't exist in current
`qdrant-client` versions (renamed to `.query_points()`, which wraps hits
in a `.points` attribute rather than returning a list directly) — a bug
the entire mocked test suite had missed, because an unspec'd `AsyncMock()`
silently accepts calls to methods that don't exist on the real object at
all. Tests were updated to use `AsyncMock(spec=AsyncQdrantClient)` so this
class of bug can't hide again, and a permanent, non-mocked integration
test was added (`tests/osint/test_full_pipeline_integration.py`, opt-in
via `RUN_LIVE_TESTS=1` since it downloads real model weights).

Also confirmed live: DeepFace's model weights are hosted as genuine
GitHub Release binary assets (`serengil/deepface_models`), not Git LFS —
unlike the YuNet model, these download reliably (verified: ~95MB at
~180MB/s in testing) with no manual step required.

Separately confirmed and fixed: `deepface` requires `tf-keras` against
TensorFlow 2.16+ (which defaults to Keras 3, incompatible with deepface's
`retina-face` dependency) — `import deepface` failed outright without it.
Added to `requirements.txt`.

Also confirmed and fixed: `AsyncQdrantClient.search()` doesn't exist in
current `qdrant-client` — renamed to `.query_points()` (which wraps hits
in a `.points` attribute). This bug had been completely invisible to the
existing mocked test suite, because an unspec'd `AsyncMock()` silently
accepts calls to methods that don't exist on the real object at all —
fixed by using `AsyncMock(spec=AsyncQdrantClient)` throughout, so this
class of bug can't hide again.

The actual `/osint/analyze-video` HTTP route (not just the underlying
FaceRecognizer/VideoProcessor classes) is now also verified end-to-end,
zero mocks: a real video, a real Qdrant client (in-memory transport,
since no standalone Qdrant binary was obtainable in this build
environment — GitHub API rate-limited on every attempt), and confirmation
of both the empty-index case (correctly reports "no match," never
fabricates one) and the populated-index case (correctly finds a
pre-indexed face). See `tests/api/test_osint_route_live.py`.



```bash
cp .env.example .env
# fill in .env with real values as you get them

# bring up infra — just redis + qdrant, both small official images with no
# build step. You do NOT need to build the API's own Docker image to test
# locally; running uvicorn directly (below) is faster and avoids a slow,
# 500MB+ tensorflow download inside a container build.
docker compose up -d redis qdrant

# install dependencies — DO NOT run plain `pip install -r requirements.txt`
# and stop there. deepface pulls in plain `opencv-python` as a transitive
# dependency, which conflicts with opencv-python-headless (both write to
# the same cv2/ import namespace and corrupt each other, producing
# confusing "module 'cv2' has no attribute X" errors). setup.sh installs
# requirements.txt AND removes/repairs the conflicting install afterward —
# it's not optional, it's the actual fix for a real, verified issue. Note:
# tensorflow (a deepface dependency) is a 500MB+ download — on a slow
# connection this step can take a while; setup.sh sets generous pip
# timeouts/retries, but be patient on the first run.
bash setup.sh

# run tests (Redis must be reachable; OSINT/live-service tests skip
# gracefully if their dependency isn't up)
pytest tests/ -v

# run the API locally (without Docker)
uvicorn src.api.server:app --reload
```

If you ever see `AttributeError: module 'cv2' has no attribute 'X'` after
installing new dependencies, re-run `bash setup.sh` — something reinstalled
the conflicting `opencv-python` package again.

### Building the API's own Docker image (optional)

Only needed if you want the API itself running inside a container (e.g.
for a production-style `docker compose up` with no local Python
environment at all). This rebuilds the same large dependency set
(including tensorflow) inside the image, so it's slow on a constrained
connection — budget real time for the first build, or skip this
entirely and just run `uvicorn` locally as shown above.

```bash
docker compose build api
docker compose up -d
```

### Face detector model (optional — the pipeline works without it)

The OSINT pipeline uses a bundled Haar cascade detector by default,
requiring **zero setup**. If you want the more accurate YuNet detector
(better on angled/non-frontal faces) as an upgrade, download it yourself —
it's a Git LFS binary, unreachable from the sandboxed environment this
project was built in:

```bash
mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

If that returns a tiny (~130 byte) file instead of ~228KB, you got a Git LFS
pointer, not the binary — use `git lfs pull` against a clone of
`opencv/opencv_zoo` instead. Once present at
`models/face_detection_yunet_2023mar.onnx`, the app picks it up
automatically on next startup — no code or config changes needed.

## Project structure

```
src/
├── sandbox/        Sandboxed Execution Engine (E2B-backed)
│   ├── models.py          Pydantic schemas
│   ├── manager.py         SandboxManager ABC — orchestrator's only import
│   ├── policy.py          Timeout/output/operation enforcement
│   ├── session_registry.py Redis-backed session CRUD
│   ├── e2b_backend.py      E2B implementation
│   └── reaper.py           TTL background cleanup task
├── osint/           Video OSINT Pipeline
│   ├── models.py           AnalysisScope, FaceEmbedding, IdentityMatchResult
│   ├── access_control.py   Scope + identity verification gate
│   ├── video_processor.py  Frame extraction + face detection
│   ├── face_recognizer.py  DeepFace embedding + Qdrant search
│   └── face_detector_loader.py  YuNet ONNX model loader
├── orchestrator/    LangGraph agent core
│   ├── state.py            AgentState schema
│   ├── agent.py             build_agent_graph() — the state machine
│   ├── sandbox_tools.py     LangGraph tools wrapping SandboxManager
│   └── osint_tools.py       LangGraph tools wrapping the OSINT pipeline
├── voice/           Real-time voice (LiveKit) — scaffolded, not functional
│   └── worker.py
└── api/             FastAPI application
    ├── config.py            Centralized Settings (env var source of truth)
    ├── routes.py            /chat, /sandbox, /osint endpoints
    └── server.py            App + lifespan wiring

tests/               28 tests, mirroring src/ structure
```

## Security/safety design notes worth knowing before you extend this

- **Sandbox**: `SandboxManager` is the *only* execution surface the
  orchestrator can reach — no direct `git clone`/`eval`/shell access exists
  outside it, and everything routes through `policy.py` for
  timeout/output enforcement.
- **OSINT**: identity-matching cannot run without an `AnalysisScope`
  (purpose, requester, justification, audit log). The LLM can never
  construct or choose this scope itself — it's always issued by
  request-handling code (`src/api/routes.py`), not agent-selectable. Match
  results below 0.7 cosine similarity are structurally prevented from being
  reported as confirmed identities (`IdentityMatchResult.to_agent_summary()`
  is the only sanctioned path from a match to text).
