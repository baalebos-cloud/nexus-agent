"""
FastAPI application entrypoint.

Owns process-lifetime wiring for the sandbox subsystem: builds the Redis
client, the SessionRegistry, the E2BSandboxManager, and starts/stops the
SandboxReaper background task. Route modules (not yet built) should depend
on `app.state.sandbox_manager` rather than constructing their own.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from src.api.config import get_settings
from src.api.llm import build_chat_llm
from src.api.routes import router as api_router
from src.osint.face_detection_backend import (
    FaceDetectionBackend,
    YuNetDetector,
    load_default_backend,
)
from src.osint.face_detector_loader import FaceDetectorModelMissingError, load_face_detector
from src.osint.face_recognizer import FaceRecognizer
from src.osint.video_processor import VideoProcessor
from src.sandbox.e2b_backend import E2BSandboxManager
from src.sandbox.models import SandboxPolicy
from src.sandbox.reaper import SandboxReaper
from src.sandbox.session_registry import SessionRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    redis_client = aioredis.from_url(settings.redis_url)
    registry = SessionRegistry(redis_client)

    policy = SandboxPolicy(
        default_timeout_seconds=settings.sandbox_default_timeout_seconds,
        max_timeout_seconds=settings.sandbox_max_timeout_seconds,
        max_output_chars=settings.sandbox_max_output_chars,
        allow_shell=settings.sandbox_allow_shell,
        allow_clone_repo=settings.sandbox_allow_clone_repo,
    )

    sandbox_manager = E2BSandboxManager(
        registry=registry,
        policy=policy,
        e2b_api_key=settings.e2b_api_key,
    )

    reaper = SandboxReaper(
        manager=sandbox_manager,
        interval_seconds=settings.sandbox_reaper_interval_seconds,
    )

    app.state.settings = settings
    app.state.redis_client = redis_client
    app.state.sandbox_manager = sandbox_manager

    # OSINT pipeline — always available now. Uses the bundled Haar cascade
    # detector by default (zero configuration, ships with
    # opencv-python-headless); automatically upgrades to the more accurate
    # YuNet detector if that model file happens to be present at
    # settings.face_detector_model_path. See src/osint/face_detection_backend.py.
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    face_detector_backend: FaceDetectionBackend
    try:
        yunet_detector = load_face_detector(settings.face_detector_model_path)
        face_detector_backend = YuNetDetector(yunet_detector)
        logger.info(
            "OSINT: using YuNet detector (higher accuracy) from %s",
            settings.face_detector_model_path,
        )
    except FaceDetectorModelMissingError:
        face_detector_backend = load_default_backend()
        logger.info(
            "OSINT: using bundled Haar cascade detector (default, zero-config). "
            "For higher accuracy, see src/osint/face_detector_loader.py to add YuNet."
        )

    app.state.video_processor = VideoProcessor(face_detector=face_detector_backend)
    app.state.face_recognizer = FaceRecognizer(qdrant_client=qdrant_client)
    try:
        await app.state.face_recognizer.ensure_collection()
        logger.info("OSINT pipeline initialized.")
    except Exception as exc:  # noqa: BLE001
        # Qdrant unreachable at startup — degrade gracefully rather than
        # crash the whole app; /osint routes will surface real errors per
        # request instead.
        logger.warning("OSINT pipeline degraded — Qdrant unavailable: %s", exc)

    # Chat LLM — None if STREAMING_LLM_BASE_URL isn't set in .env yet;
    # /chat returns 503 until then rather than failing app startup.
    # See src/api/llm.py for the Groq/vLLM wiring.
    app.state.chat_llm = build_chat_llm(settings)
    if app.state.chat_llm is not None:
        logger.info(
            "Chat LLM configured: base_url=%s model=%s",
            settings.streaming_llm_base_url,
            settings.streaming_llm_model,
        )

    await reaper.start()
    logger.info("Nexus-Agent API startup complete.")

    try:
        yield
    finally:
        await reaper.stop()
        await redis_client.aclose()
        logger.info("Nexus-Agent API shutdown complete.")


app = FastAPI(title="Nexus-Agent API", lifespan=lifespan)
app.include_router(api_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
