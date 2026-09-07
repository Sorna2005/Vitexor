"""
schemas.py

Phase 10 - FastAPI request and response schemas.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    """Request body for the /review endpoint."""

    code: str = Field(
        ...,
        min_length=1,
        description="Source code to review.",
    )

    language: str = Field(
        default="python",
        min_length=1,
        description="Programming language of the submitted code.",
    )


class ReviewResponse(BaseModel):
    """Response returned by the /review endpoint."""

    success: bool

    summary: str = ""
    issues: list[Any] = Field(default_factory=list)
    security: list[Any] = Field(default_factory=list)
    performance: list[Any] = Field(default_factory=list)
    maintainability: list[Any] = Field(default_factory=list)
    improvements: list[Any] = Field(default_factory=list)

    improved_code: str = ""
    final_verdict: str = ""
    formatted_text: str = ""

    validation_errors: list[Any] = Field(default_factory=list)
    validation_warnings: list[Any] = Field(default_factory=list)

    error: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response returned by the /health endpoint."""

    retrieval_ok: bool
    gemini_configured: bool
    healthy: bool

    retrieval_info: dict[str, Any] | None = None
    retrieval_error: str | None = None
    gemini_error: str | None = None