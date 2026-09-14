"""
Centralized application settings.

Per the earlier design decision: backend classes (e2b_backend.py) never read
os.environ directly — they accept credentials as constructor arguments. This
module is the single place environment variables are read, so secret
sourcing stays auditable and swappable (e.g. for a secrets manager later)
without touching sandbox/orchestrator code.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # E2B
    e2b_api_key: str | None = Field(default=None, description="E2B Sandbox API key.")

    # Voice stack (LiveKit pipeline)
    deepgram_api_key: str | None = Field(default=None, description="Deepgram STT API key.")
    cartesia_api_key: str | None = Field(default=None, description="Cartesia TTS API key.")
    livekit_url: str | None = Field(default=None, description="LiveKit server WebSocket URL.")
    livekit_api_key: str | None = Field(default=None)
    livekit_api_secret: str | None = Field(default=None)
    streaming_llm_base_url: str | None = Field(
        default=None, description="vLLM or Groq endpoint for low-latency streaming inference."
    )
    streaming_llm_api_key: str | None = Field(default=None)
    streaming_llm_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Model name/ID as expected by the configured endpoint "
        "(Groq model name, or the --served-model-name used to launch vLLM). "
        "Must support tool/function calling — check the provider's model "
        "list for a 'tools' capability flag before picking one.",
    )

    # OSINT face detector — see src/osint/face_detector_loader.py. Not bundled
    # in the repo; download separately (path below is the expected default).
    face_detector_model_path: str = Field(
        default="models/face_detection_yunet_2023mar.onnx",
        description="Path to a YuNet ONNX face-detection model.",
    )

    # Redis (matches docker-compose.yml service `redis`)
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Qdrant (matches docker-compose.yml service `qdrant`)
    qdrant_url: str = Field(default="http://localhost:6333")

    # Sandbox policy defaults
    sandbox_default_timeout_seconds: int = Field(default=30)
    sandbox_max_timeout_seconds: int = Field(default=120)
    sandbox_max_output_chars: int = Field(default=8_000)
    sandbox_allow_shell: bool = Field(default=True)
    sandbox_allow_clone_repo: bool = Field(default=True)
    sandbox_default_ttl_seconds: int = Field(default=600)
    sandbox_reaper_interval_seconds: int = Field(default=60)

    # Fine-tuning: live-usage capture (src/finetune/capture.py). Default
    # OFF — capturing conversation transcripts has real privacy
    # implications, so this must be explicit opt-in, never on by default.
    capture_training_data: bool = Field(
        default=False,
        description="If true, /chat and voice conversations are captured as "
        "fine-tuning training data (src/finetune/capture.py). OFF by default.",
    )
    training_capture_path: str = Field(default="data/finetune/captured.jsonl")
    training_capture_pending_review_path: str = Field(
        default="data/finetune/pending_osint_review.jsonl",
        description="OSINT-touched conversations are quarantined here, never "
        "auto-written to training_capture_path — see capture.py's docstring.",
    )


def get_settings() -> Settings:
    """Factory rather than a module-level singleton, so tests can construct
    isolated Settings instances without env-var leakage between cases."""
    return Settings()
