"""
review_service.py

Phase 9 - single end-to-end orchestration layer for the AI Code Reviewer.

This is the ONE public entry point the API (Phase 10) and UI (Phase 11)
call. It wires together every piece already built in earlier phases
without duplicating their logic:

    USER CODE
      -> input validation                (ReviewGenerator.validate_code)
      -> CodeBERT query embedding        (src.embeddings, via FaissSearcher)
      -> FAISS retrieval                 (src.retrieval.RetrievalPipeline)
      -> filter/rerank                   (src.retrieval.reranker, inside FaissSearcher)
      -> context building                (src.review_engine.ContextBuilder)
      -> prompt building                 (src.review_engine.PromptBuilder)
      -> Gemini                          (src.llm.GeminiClient)
      -> response parsing                (src.llm.response_parser)
      -> validation                      (src.review_engine.validators)
      -> formatting                      (src.review_engine.formatter)
      -> FINAL REVIEW
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.review_engine.review_generator import ReviewGenerator
from src.review_engine.context_builder import ContextBuilder
from src.review_engine.prompt_builder import PromptBuilder
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.llm.gemini_client import create_gemini_client_from_settings
from src.llm.response_parser import parse_review
from src.review_engine.validators import (
    validate_parsed_review,
    parsed_review_to_formatter_dict,
)
from src.review_engine.formatter import format_review

logger = logging.getLogger(__name__)


###############################################################################
# Result
###############################################################################


@dataclass(slots=True)
class ReviewServiceResult:
    """
    Final structured result returned by review_code().
    """

    success: bool

    summary: str = ""
    issues: list = field(default_factory=list)
    security: list = field(default_factory=list)
    performance: list = field(default_factory=list)
    maintainability: list = field(default_factory=list)
    improvements: list = field(default_factory=list)
    improved_code: str = ""
    final_verdict: str = ""

    formatted_text: str = ""

    validation_errors: list = field(default_factory=list)
    validation_warnings: list = field(default_factory=list)

    error: Optional[str] = None

    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "summary": self.summary,
            "issues": self.issues,
            "security": self.security,
            "performance": self.performance,
            "maintainability": self.maintainability,
            "improvements": self.improvements,
            "improved_code": self.improved_code,
            "final_verdict": self.final_verdict,
            "formatted_text": self.formatted_text,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "error": self.error,
            "metadata": self.metadata,
        }


###############################################################################
# Lazily-constructed singleton generator
###############################################################################

_generator: ReviewGenerator | None = None


def _get_generator() -> ReviewGenerator:

    global _generator

    if _generator is not None:
        return _generator

    logger.info("Initializing ReviewGenerator (first call)...")

    llm_client = create_gemini_client_from_settings()

    _generator = ReviewGenerator(
        llm_client=llm_client,
        retrieval_pipeline=RetrievalPipeline(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
    )

    logger.info("ReviewGenerator ready.")

    return _generator


def reset_generator() -> None:
    global _generator
    _generator = None


###############################################################################
# Public Entry Point
###############################################################################


def review_code(
    code: str,
    language: str = "python",
) -> ReviewServiceResult:
    """
    Run the complete AI code review pipeline on a single snippet.
    """

    start = time.perf_counter()

    if not isinstance(code, str):
        return ReviewServiceResult(
            success=False,
            error="code must be a string.",
        )

    if not code.strip():
        return ReviewServiceResult(
            success=False,
            error="code cannot be empty.",
        )

    try:
        generator = _get_generator()

    except RuntimeError as exc:
        logger.error("Generator initialization failed: %s", exc)
        return ReviewServiceResult(
            success=False,
            error=str(exc),
        )

    except Exception as exc:
        logger.exception("Unexpected error initializing generator.")
        return ReviewServiceResult(
            success=False,
            error=f"Failed to initialize review pipeline: {exc}",
        )

    try:
        review_result = generator.review(code)

    except (TypeError, ValueError) as exc:
        return ReviewServiceResult(
            success=False,
            error=str(exc),
        )

    except Exception as exc:
        logger.exception("Review pipeline failed.")
        return ReviewServiceResult(
            success=False,
            error=(
                "The review pipeline failed (retrieval or Gemini "
                f"request error): {exc}"
            ),
        )

    raw_response = review_result.llm_response.response

    if not raw_response or not raw_response.strip():
        return ReviewServiceResult(
            success=False,
            error="Gemini returned an empty response.",
            metadata=review_result.metadata,
        )

    try:
        parsed = parse_review(raw_response)

    except Exception as exc:
        logger.exception("Failed to parse LLM response.")
        return ReviewServiceResult(
            success=False,
            error=f"Failed to parse LLM response: {exc}",
            metadata=review_result.metadata,
        )

    validation = validate_parsed_review(parsed)

    formatter_dict = parsed_review_to_formatter_dict(parsed)

    try:
        formatted_text = format_review(formatter_dict)

    except Exception as exc:
        logger.exception("Formatting failed.")
        formatted_text = ""

    total_time = time.perf_counter() - start

    metadata = dict(review_result.metadata)
    metadata["total_service_time"] = total_time
    metadata["language"] = language

    return ReviewServiceResult(
        success=validation.is_valid,
        summary=parsed.summary,
        issues=parsed.issues,
        security=parsed.security,
        performance=parsed.performance,
        maintainability=parsed.maintainability,
        improvements=parsed.improvements,
        improved_code=parsed.improved_code,
        final_verdict=parsed.verdict,
        formatted_text=formatted_text,
        validation_errors=validation.errors,
        validation_warnings=validation.warnings,
        error=(
            None
            if validation.is_valid
            else (
                "LLM response was missing required sections: "
                + "; ".join(validation.errors)
            )
        ),
        metadata=metadata,
    )


###############################################################################
# Health Check
###############################################################################


def health_check() -> dict[str, Any]:
    """
    Lightweight health check for the /health API endpoint.
    """

    result: dict[str, Any] = {
        "retrieval_ok": False,
        "gemini_configured": False,
    }

    try:
        pipeline = RetrievalPipeline()
        result["retrieval_ok"] = pipeline.health_check()
        result["retrieval_info"] = pipeline.info()

    except Exception as exc:
        result["retrieval_error"] = str(exc)

    try:
        from src.review_engine.config import settings
        result["gemini_configured"] = bool(
            settings.gemini_api_key
            and settings.gemini_api_key.strip()
        )

    except Exception as exc:
        result["gemini_error"] = str(exc)

    result["healthy"] = (
        result["retrieval_ok"]
        and result["gemini_configured"]
    )

    return result


###############################################################################
# CLI Test
###############################################################################

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("=" * 70)
    print("HEALTH CHECK")
    print("=" * 70)
    print(health_check())

    print()
    print("=" * 70)
    print("REVIEW: divide by zero")
    print("=" * 70)

    sample = """
def divide(a, b):
    return a / b
""".strip()

    result = review_code(sample, language="python")

    print("success:", result.success)
    print("error:", result.error)
    print()
    print(result.formatted_text)
    print()
    print("metadata:", result.metadata)


###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "ReviewServiceResult",
    "review_code",
    "health_check",
    "reset_generator",
]
