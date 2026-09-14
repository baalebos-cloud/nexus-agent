"""
API routes for Nexus-Agent.

Kept thin: routes validate input, pull dependencies off `app.state`
(wired in server.py's lifespan), and delegate to the orchestrator/sandbox/
osint modules. No business logic lives here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.api.config import get_settings
from src.finetune.capture import capture_conversation
from src.orchestrator.agent import build_agent_graph
from src.osint.models import AnalysisPurpose, AnalysisScope

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Chat / agent invocation
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    thread_id: str = Field(..., description="Stable ID for this conversation/task.")
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    sandbox_session_id: str | None = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    sandbox_manager = request.app.state.sandbox_manager
    llm = getattr(request.app.state, "chat_llm", None)
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="No chat LLM configured on app.state.chat_llm — wire a BaseChatModel "
            "instance in server.py's lifespan before this endpoint can serve requests.",
        )

    graph = build_agent_graph(llm=llm, sandbox_manager=sandbox_manager)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=body.message)],
            "thread_id": body.thread_id,
            "sandbox_session_id": None,
            "osint_scope": None,
            "is_complete": False,
            "scratch": {},
        }
    )

    # Fire-and-forget-safe: capture_conversation never raises and is a
    # true no-op unless explicitly enabled via Settings.capture_training_data.
    settings = get_settings()
    await capture_conversation(
        messages=result["messages"],
        enabled=settings.capture_training_data,
        output_path=settings.training_capture_path,
        pending_review_path=settings.training_capture_pending_review_path,
    )

    return ChatResponse(
        thread_id=body.thread_id,
        reply=result["messages"][-1].content,
        sandbox_session_id=result.get("sandbox_session_id"),
    )


# ---------------------------------------------------------------------------
# Sandbox session inspection (read-only; creation happens implicitly via /chat)
# ---------------------------------------------------------------------------


class SandboxSessionResponse(BaseModel):
    session_id: str
    status: str
    runtime: str
    created_at: str


@router.get("/sandbox/sessions/{session_id}", response_model=SandboxSessionResponse)
async def get_sandbox_session(request: Request, session_id: str) -> SandboxSessionResponse:
    from src.sandbox.manager import SessionNotFoundError

    sandbox_manager = request.app.state.sandbox_manager
    try:
        session = await sandbox_manager.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return SandboxSessionResponse(
        session_id=session.session_id,
        status=session.status.value,
        runtime=session.runtime.value,
        created_at=session.created_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# OSINT analysis — explicitly scoped; this is the ONLY entrypoint that can
# construct an AnalysisScope from request data, and it requires the caller
# to state purpose/justification up front (audited, per access_control.py).
# ---------------------------------------------------------------------------


class AnalyzeVideoRequest(BaseModel):
    video_path: str = Field(..., description="Server-accessible path to the video file.")
    purpose: AnalysisPurpose
    requested_by: str = Field(..., min_length=1)
    justification: str = Field(..., min_length=10)
    case_reference: str | None = None


class AnalyzeVideoResponse(BaseModel):
    scope_id: str
    summary: str


@router.post("/osint/analyze-video", response_model=AnalyzeVideoResponse)
async def analyze_video(request: Request, body: AnalyzeVideoRequest) -> AnalyzeVideoResponse:
    # Validate the scope FIRST — a malformed/unauthorized request should
    # fail with 400 regardless of whether the pipeline happens to be
    # configured, so misuse is caught the same way in every environment.
    try:
        scope = AnalysisScope(
            purpose=body.purpose,
            requested_by=body.requested_by,
            justification=body.justification,
            case_reference=body.case_reference,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    video_processor = getattr(request.app.state, "video_processor", None)
    face_recognizer = getattr(request.app.state, "face_recognizer", None)
    if video_processor is None or face_recognizer is None:
        raise HTTPException(
            status_code=503,
            detail="OSINT pipeline not configured on app.state — requires a loaded "
            "face detector (see src/osint/face_detector_loader.py) and an "
            "initialized Qdrant collection.",
        )

    from src.orchestrator.osint_tools import build_osint_tools

    tools = build_osint_tools(
        video_processor=video_processor, face_recognizer=face_recognizer, scope=scope
    )
    analyze_tool = tools[0]
    summary = await analyze_tool.coroutine(video_path=body.video_path)

    return AnalyzeVideoResponse(scope_id=scope.scope_id, summary=summary)
