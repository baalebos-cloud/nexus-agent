"""
LangGraph tool node for Video OSINT analysis.

Mirrors src/orchestrator/sandbox_tools.py's shape: the LLM sees clean,
narrow function-calling schemas, never AnalysisScope/FaceRecognizer
internals directly. Unlike the sandbox tools, this tool node does NOT let
the LLM freely invoke identity search — `AnalysisScope` is constructed by
the orchestrator from the authenticated request context, not from LLM
-chosen arguments, so the model can't self-authorize a surveillance query
by picking a favorable `purpose` string.
"""

from __future__ import annotations

import logging

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.osint.access_control import ScopeError
from src.osint.face_recognizer import FaceRecognitionError, FaceRecognizer
from src.osint.models import AnalysisScope
from src.osint.video_processor import VideoProcessingError, VideoProcessor

logger = logging.getLogger(__name__)


class AnalyzeVideoInput(BaseModel):
    video_path: str = Field(..., description="Path to the video file to analyze for faces.")


def build_osint_tools(
    video_processor: VideoProcessor,
    face_recognizer: FaceRecognizer,
    scope: AnalysisScope,
) -> list[StructuredTool]:
    """
    Build OSINT tools bound to a single, orchestrator-issued AnalysisScope.

    `scope` is captured by closure and never exposed as an LLM-settable
    argument. If the orchestrator has no valid scope for this task (e.g. no
    authenticated request context justifies an OSINT operation), it should
    simply not call this builder — the agent then has no OSINT tools
    available at all, which is the correct default.
    """

    async def analyze_video_for_faces(video_path: str) -> str:
        try:
            crops = await video_processor.extract_face_crops(video_path)
        except VideoProcessingError as exc:
            return f"VIDEO PROCESSING FAILED: {exc}"

        if not crops:
            return "No faces detected in the video."

        summaries: list[str] = []
        for crop in crops:
            try:
                embedding = await face_recognizer.extract_embedding(
                    scope=scope,
                    image_path=crop.image_path,
                    source_video_id=video_path,
                    source_frame_timestamp_ms=crop.frame_timestamp_ms,
                    detection_confidence=crop.detection_confidence,
                )
                match_result = await face_recognizer.search_identity(scope, embedding)
            except (ScopeError, FaceRecognitionError) as exc:
                summaries.append(f"[frame {crop.frame_timestamp_ms}ms] ANALYSIS FAILED: {exc}")
                continue

            summaries.append(
                f"[frame {crop.frame_timestamp_ms}ms] {match_result.to_agent_summary()}"
            )

        return "\n".join(summaries)

    return [
        StructuredTool.from_function(
            coroutine=analyze_video_for_faces,
            name="osint_analyze_video_for_faces",
            description=(
                "Analyze a video file for faces: detects faces, extracts embeddings, and "
                "searches for candidate identity matches. Every result is reported with an "
                "explicit confidence framing — matches below the confidence threshold are "
                "reported as unconfirmed, never as a verified identity."
            ),
            args_schema=AnalyzeVideoInput,
        )
    ]
