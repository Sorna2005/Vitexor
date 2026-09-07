"""
Global constants used throughout the AI Code Reviewer.

This module centralizes all immutable project-wide constants to avoid
magic numbers and hardcoded values across the codebase.
"""

from __future__ import annotations

###############################################################################
# Project Metadata
###############################################################################

PROJECT_NAME = "AI Code Reviewer"

PROJECT_VERSION = "1.0.0"

PROJECT_AUTHOR = "Sornalatha"

PROJECT_DESCRIPTION = (
    "Enterprise AI-powered Code Review System using "
    "CodeBERT, FAISS and Gemini."
)

###############################################################################
# Supported Programming Languages
###############################################################################

SUPPORTED_LANGUAGES = (
    "python",
    "java",
    "javascript",
    "typescript",
    "c",
    "cpp",
    "csharp",
    "go",
    "php",
)

###############################################################################
# Gemini Models
###############################################################################

SUPPORTED_GEMINI_MODELS = (
    "gemini-3.6-flash",
    "gemini-2.5-pro",
)

DEFAULT_MODEL = "gemini-3.6-flash"

###############################################################################
# Embedding Models
###############################################################################

CODEBERT_MODEL = "microsoft/codebert-base"

###############################################################################
# Retrieval
###############################################################################

DEFAULT_TOP_K = 5

MAX_TOP_K = 20

SIMILARITY_THRESHOLD = 0.65

###############################################################################
# Generation
###############################################################################

DEFAULT_TEMPERATURE = 0.2

DEFAULT_TOP_P = 0.95

DEFAULT_MAX_OUTPUT_TOKENS = 4096

MAX_CONTEXT_CHARACTERS = 30000

###############################################################################
# File Extensions
###############################################################################

PYTHON_EXTENSION = ".py"

JSON_EXTENSION = ".json"

PARQUET_EXTENSION = ".parquet"

TEXT_EXTENSION = ".txt"

LOG_EXTENSION = ".log"

###############################################################################
# Output Files
###############################################################################

REVIEW_OUTPUT_FILE = "review.json"

MARKDOWN_REPORT = "review.md"

HTML_REPORT = "review.html"

###############################################################################
# Logging
###############################################################################

DEFAULT_LOGGER_NAME = "AI_Code_Reviewer"

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

###############################################################################
# Prompt
###############################################################################

SYSTEM_PROMPT_TITLE = (
    "Senior Software Engineer and Security Code Reviewer"
)

###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "PROJECT_NAME",
    "PROJECT_VERSION",
    "PROJECT_AUTHOR",
    "PROJECT_DESCRIPTION",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_GEMINI_MODELS",
    "DEFAULT_MODEL",
    "CODEBERT_MODEL",
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "SIMILARITY_THRESHOLD",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "MAX_CONTEXT_CHARACTERS",
    "PYTHON_EXTENSION",
    "JSON_EXTENSION",
    "PARQUET_EXTENSION",
    "TEXT_EXTENSION",
    "LOG_EXTENSION",
    "REVIEW_OUTPUT_FILE",
    "MARKDOWN_REPORT",
    "HTML_REPORT",
    "DEFAULT_LOGGER_NAME",
    "LOG_FORMAT",
    "DATE_FORMAT",
    "SYSTEM_PROMPT_TITLE",
]