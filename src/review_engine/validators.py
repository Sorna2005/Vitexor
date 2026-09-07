"""
validators.py

Enterprise validation utilities for the AI Code Reviewer.

This module contains reusable validation functions used throughout the
application to validate files, directories, configuration values,
retrieval parameters, prompts, and AI responses.
"""

from __future__ import annotations

###############################################################################
# Standard Library Imports
###############################################################################

from pathlib import Path
from typing import Any, List as _List
from dataclasses import dataclass, field as _field

###############################################################################
# Local Imports
###############################################################################

from .exceptions import (
    FileValidationError,
    InputValidationError,
)

###############################################################################
# File Validation
###############################################################################


def validate_file_exists(file_path: Path) -> Path:
    """
    Validate that a file exists.

    Parameters
    ----------
    file_path : Path
        Path to the file.

    Returns
    -------
    Path
        Validated file path.

    Raises
    ------
    FileValidationError
    """

    if not file_path.exists():
        raise FileValidationError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise FileValidationError(
            f"Expected a file but received: {file_path}"
        )

    return file_path


###############################################################################
# Directory Validation
###############################################################################


def validate_directory(directory: Path) -> Path:
    """
    Validate directory existence.

    Parameters
    ----------
    directory : Path

    Returns
    -------
    Path
    """

    if not directory.exists():
        raise FileValidationError(
            f"Directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise FileValidationError(
            f"Expected directory: {directory}"
        )

    return directory


###############################################################################
# String Validation
###############################################################################


def validate_non_empty_string(
    value: str,
    field_name: str,
) -> str:
    """
    Validate non-empty string.
    """

    if not isinstance(value, str):
        raise InputValidationError(
            f"{field_name} must be a string."
        )

    value = value.strip()

    if not value:
        raise InputValidationError(
            f"{field_name} cannot be empty."
        )

    return value


###############################################################################
# Integer Validation
###############################################################################


def validate_positive_integer(
    value: int,
    field_name: str,
) -> int:
    """
    Validate positive integer.
    """

    if not isinstance(value, int):
        raise InputValidationError(
            f"{field_name} must be an integer."
        )

    if value <= 0:
        raise InputValidationError(
            f"{field_name} must be greater than zero."
        )

    return value


###############################################################################
# Float Validation
###############################################################################


def validate_probability(
    value: float,
    field_name: str,
) -> float:
    """
    Validate probability value.

    Must be between 0 and 1.
    """

    if value < 0 or value > 1:
        raise InputValidationError(
            f"{field_name} must be between 0 and 1."
        )

    return value


###############################################################################
# Temperature Validation
###############################################################################


def validate_temperature(
    temperature: float,
) -> float:
    """
    Validate LLM temperature.
    """

    if temperature < 0 or temperature > 2:
        raise InputValidationError(
            "Temperature must be between 0 and 2."
        )

    return temperature


###############################################################################
# Top-K Validation
###############################################################################


def validate_top_k(
    top_k: int,
) -> int:
    """
    Validate retrieval top-k value.
    """

    if top_k <= 0:
        raise InputValidationError(
            "top_k must be greater than zero."
        )

    return top_k


###############################################################################
# Prompt Validation
###############################################################################


def validate_prompt(
    prompt: str,
) -> str:
    """
    Validate generated prompt.
    """

    prompt = validate_non_empty_string(
        prompt,
        "Prompt",
    )

    if len(prompt) < 30:
        raise InputValidationError(
            "Prompt is too short."
        )

    return prompt


###############################################################################
# API Response Validation
###############################################################################


def validate_llm_response(
    response: str,
) -> str:
    """
    Validate LLM response.
    """

    response = validate_non_empty_string(
        response,
        "LLM Response",
    )

    return response


###############################################################################
# Generic Validation
###############################################################################


def validate_not_none(
    value: Any,
    field_name: str,
) -> Any:
    """
    Validate object is not None.
    """

    if value is None:
        raise InputValidationError(
            f"{field_name} cannot be None."
        )

    return value


###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "validate_file_exists",
    "validate_directory",
    "validate_non_empty_string",
    "validate_positive_integer",
    "validate_probability",
    "validate_temperature",
    "validate_top_k",
    "validate_prompt",
    "validate_llm_response",
    "validate_not_none",
    "ReviewValidationResult",
    "validate_parsed_review",
    "parsed_review_to_formatter_dict",
]


###############################################################################
# Parsed Review Validation (Phase 7)
###############################################################################
#
# Validates the structured output of src.llm.response_parser.parse_review()
# (a ParsedReview). Required fields per the project spec: summary, issues,
# security, performance, verdict. Optional fields (strengths,
# maintainability, improvements, improved_code) must never crash the
# application even when missing or malformed - the LLM is not guaranteed
# to follow the requested format exactly.


@dataclass(slots=True)
class ReviewValidationResult:
    """
    Result of validating a ParsedReview.

    is_valid is True only if every REQUIRED field is present and
    non-empty. errors lists exactly what's wrong, in plain language,
    so callers (API layer, UI, tests) can surface something useful
    instead of a stack trace.
    """

    is_valid: bool
    errors: _List[str] = _field(default_factory=list)
    warnings: _List[str] = _field(default_factory=list)


def validate_parsed_review(review: Any) -> ReviewValidationResult:
    """
    Validate a parsed LLM code review.

    Required (missing/empty -> error, is_valid becomes False):
        summary, issues, security, performance, verdict

    Optional (missing/empty -> warning only, does not fail validation):
        strengths, maintainability, improvements, improved_code

    Accepts either a ParsedReview instance or a plain dict with the
    same field names, since malformed LLM output may need to be
    validated before it's even wrapped in a dataclass.
    """

    errors: list[str] = []
    warnings: list[str] = []

    def get(name: str, default: Any = None) -> Any:
        if review is None:
            return default
        if isinstance(review, dict):
            return review.get(name, default)
        return getattr(review, name, default)

    # ---- required: summary ----
    summary = get("summary", "")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("summary is missing or empty.")

    # ---- required: verdict ----
    verdict = get("verdict", "")
    if not isinstance(verdict, str) or not verdict.strip():
        errors.append("verdict (Final Verdict) is missing or empty.")

    # ---- required: issues (list, may be empty list = "no issues found") ----
    issues = get("issues", None)
    if issues is None or not isinstance(issues, list):
        errors.append("issues is missing or not a list.")

    # ---- required: security (list, may be empty list = "no findings") ----
    security = get("security", None)
    if security is None or not isinstance(security, list):
        errors.append("security is missing or not a list.")

    # ---- required: performance (list, may be empty list = "no findings") ----
    performance = get("performance", None)
    if performance is None or not isinstance(performance, list):
        errors.append("performance is missing or not a list.")

    # ---- optional: maintainability ----
    maintainability = get("maintainability", None)
    if maintainability is None or not isinstance(maintainability, list):
        warnings.append(
            "maintainability is missing or not a list (optional)."
        )

    # ---- optional: improvements (Suggested Improvements) ----
    improvements = get("improvements", None)
    if improvements is None or not isinstance(improvements, list):
        warnings.append(
            "improvements is missing or not a list (optional)."
        )

    # ---- optional: improved_code ----
    improved_code = get("improved_code", "")
    if not isinstance(improved_code, str) or not improved_code.strip():
        warnings.append(
            "improved_code is missing or empty (optional - the model "
            "may have legitimately found no code change necessary)."
        )

    # ---- optional: strengths ----
    strengths = get("strengths", None)
    if strengths is None or not isinstance(strengths, list):
        warnings.append(
            "strengths is missing or not a list (optional)."
        )

    return ReviewValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
    )


###############################################################################
# ParsedReview -> ReviewFormatter Adapter (Phase 7/8 bridge)
###############################################################################
#
# ReviewFormatter (src.review_engine.formatter) expects a dict/object
# with keys: summary, issues, performance, security, best_practices,
# improved_code, verdict. ParsedReview (src.llm.response_parser) uses a
# different field vocabulary (security/performance/maintainability are
# lists of bullet strings, no "best_practices" key at all). This
# adapter converts one to the other WITHOUT modifying either class,
# so both keep their own, already-tested, independent shapes.


def _bullets_to_text(items: _List[str]) -> str:
    if not items:
        return "None."
    return "\n".join(f"- {item}" for item in items)


def parsed_review_to_formatter_dict(review: Any) -> dict[str, Any]:
    """
    Convert a ParsedReview (or equivalent dict) into the dict shape
    ReviewFormatter.format() / .to_dict() expect.

    maintainability findings are folded into best_practices, since the
    project's target final output format has a single "BEST PRACTICES"
    section rather than separate maintainability/strengths sections.
    """

    def get(name: str, default: Any = None) -> Any:
        if review is None:
            return default
        if isinstance(review, dict):
            return review.get(name, default)
        return getattr(review, name, default)

    issues = get("issues", []) or []
    security = get("security", []) or []
    performance = get("performance", []) or []
    maintainability = get("maintainability", []) or []
    improvements = get("improvements", []) or []

    best_practices_parts = []
    if maintainability:
        best_practices_parts.append(_bullets_to_text(maintainability))
    if improvements:
        best_practices_parts.append(_bullets_to_text(improvements))

    best_practices = (
        "\n\n".join(best_practices_parts)
        if best_practices_parts
        else "None."
    )

    return {
        "summary": get("summary", "") or "",
        "issues": issues,
        "performance": _bullets_to_text(performance),
        "security": _bullets_to_text(security),
        "best_practices": best_practices,
        "improved_code": get("improved_code", "") or "",
        "verdict": get("verdict", "") or "",
    }