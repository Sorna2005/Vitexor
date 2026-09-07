"""
LLM Package
===========

This package provides a unified interface for interacting with
Large Language Models (LLMs).

Modules
-------
base_client
    Abstract base class for all LLM providers.

gemini_client
    Google Gemini implementation.

prompts
    Prompt templates used for code review.

response_parser
    Parses and structures LLM responses.
"""

from .base_client import (
    BaseLLMClient,
    LLMResponse,
)

from .gemini_client import (
    GeminiClient,
)

from .prompts import (
    CodeReviewPromptBuilder,
)

from .response_parser import (
    ParsedReview,
    ReviewResponseParser,
)

__version__ = "1.0.0"

__author__ = "Sornalatha"

__all__ = [
    "BaseLLMClient",
    "LLMResponse",
    "GeminiClient",
    "CodeReviewPromptBuilder",
    "ParsedReview",
    "ReviewResponseParser",
]