"""
main.py

Phase 10 - FastAPI application for the AI Code Reviewer.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from src.app.review_service import health_check, review_code
from src.app.schemas import (
    HealthResponse,
    ReviewRequest,
    ReviewResponse,
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Vitexor AI",
    description=(
        "AI-powered code review API using retrieval, CodeBERT, "
        "FAISS, and Gemini."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
def get_health() -> HealthResponse:
    """
    Check whether the retrieval system and Gemini configuration
    are available.
    """

    result = health_check()

    return HealthResponse(**result)


@app.post(
    "/review",
    response_model=ReviewResponse,
    tags=["Code Review"],
)
def review(request: ReviewRequest) -> ReviewResponse:
    """
    Review submitted source code.
    """

    try:
        result = review_code(
            code=request.code,
            language=request.language,
        )

        return ReviewResponse(**result.to_dict())

    except Exception as exc:
        logger.exception("Unexpected API error.")

        raise HTTPException(
            status_code=500,
            detail=f"Unexpected review error: {exc}",
        ) from exc


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    """Basic API information."""

    return {
        "name": "AI Code Reviewer",
        "status": "running",
        "docs": "/docs",
    }