"""
exceptions.py

Enterprise exception hierarchy for the AI Code Reviewer.

This module defines all custom exceptions used across the application.
Using custom exceptions improves readability, debugging, maintainability,
and allows fine-grained exception handling.
"""

from __future__ import annotations

###############################################################################
# Base Exception
###############################################################################


class ReviewEngineError(Exception):
    """
    Base exception for the AI Code Reviewer.

    All custom exceptions should inherit from this class.
    """

    default_message = "An unexpected Review Engine error occurred."

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)


###############################################################################
# Configuration Exceptions
###############################################################################


class ConfigurationError(ReviewEngineError):
    """
    Raised when configuration loading fails.
    """

    default_message = "Invalid application configuration."


class EnvironmentVariableError(ConfigurationError):
    """
    Raised when required environment variables are missing.
    """

    default_message = "Required environment variable is missing."


class InvalidPathError(ConfigurationError):
    """
    Raised when a configured project path is invalid.
    """

    default_message = "Configured project path is invalid."


###############################################################################
# Retrieval Exceptions
###############################################################################


class RetrievalError(ReviewEngineError):
    """
    Base retrieval exception.
    """

    default_message = "Retrieval operation failed."


class EmbeddingError(RetrievalError):
    """
    Raised during embedding generation.
    """

    default_message = "Failed to generate embeddings."


class FAISSIndexError(RetrievalError):
    """
    Raised when FAISS index operations fail.
    """

    default_message = "FAISS index operation failed."


class SimilaritySearchError(RetrievalError):
    """
    Raised when similarity search fails.
    """

    default_message = "Similarity search failed."


class DatasetError(RetrievalError):
    """
    Raised while loading datasets.
    """

    default_message = "Dataset operation failed."


###############################################################################
# Prompt Exceptions
###############################################################################


class PromptError(ReviewEngineError):
    """
    Base prompt generation exception.
    """

    default_message = "Prompt generation failed."


class PromptValidationError(PromptError):
    """
    Raised when a prompt is invalid.
    """

    default_message = "Prompt validation failed."


###############################################################################
# LLM Exceptions
###############################################################################


class LLMError(ReviewEngineError):
    """
    Base LLM exception.
    """

    default_message = "Language model operation failed."


class GeminiAPIError(LLMError):
    """
    Raised when Gemini API fails.
    """

    default_message = "Gemini API request failed."


class RateLimitError(LLMError):
    """
    Raised when rate limits are exceeded.
    """

    default_message = "Gemini API rate limit exceeded."


class AuthenticationError(LLMError):
    """
    Raised when authentication fails.
    """

    default_message = "Gemini API authentication failed."


class ResponseParsingError(LLMError):
    """
    Raised when parsing Gemini responses fails.
    """

    default_message = "Unable to parse Gemini response."


###############################################################################
# Validation Exceptions
###############################################################################


class ValidationError(ReviewEngineError):
    """
    Base validation exception.
    """

    default_message = "Validation failed."


class InputValidationError(ValidationError):
    """
    Raised for invalid user input.
    """

    default_message = "Invalid input provided."


class FileValidationError(ValidationError):
    """
    Raised when validating uploaded files.
    """

    default_message = "File validation failed."


###############################################################################
# Review Exceptions
###############################################################################


class ReviewGenerationError(ReviewEngineError):
    """
    Raised during AI review generation.
    """

    default_message = "Unable to generate review."


class ReviewFormattingError(ReviewEngineError):
    """
    Raised while formatting review output.
    """

    default_message = "Review formatting failed."


###############################################################################
# Output Exceptions
###############################################################################


class ExportError(ReviewEngineError):
    """
    Raised when exporting reports.
    """

    default_message = "Failed to export review."


class SerializationError(ExportError):
    """
    Raised during serialization.
    """

    default_message = "Serialization failed."


###############################################################################
# Logger Exceptions
###############################################################################


class LoggerConfigurationError(ReviewEngineError):
    """
    Raised when logger initialization fails.
    """

    default_message = "Logger configuration failed."


###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "ReviewEngineError",
    "ConfigurationError",
    "EnvironmentVariableError",
    "InvalidPathError",
    "RetrievalError",
    "EmbeddingError",
    "FAISSIndexError",
    "SimilaritySearchError",
    "DatasetError",
    "PromptError",
    "PromptValidationError",
    "LLMError",
    "GeminiAPIError",
    "RateLimitError",
    "AuthenticationError",
    "ResponseParsingError",
    "ValidationError",
    "InputValidationError",
    "FileValidationError",
    "ReviewGenerationError",
    "ReviewFormattingError",
    "ExportError",
    "SerializationError",
    "LoggerConfigurationError",
]