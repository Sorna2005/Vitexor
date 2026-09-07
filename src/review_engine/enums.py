"""
enums.py

Enterprise enumerations used throughout the AI Code Reviewer.

This module defines strongly typed enumerations that eliminate magic
strings, improve readability, simplify validation, and provide a
consistent interface across the application.
"""

from __future__ import annotations

###############################################################################
# Standard Library Imports
###############################################################################

from enum import Enum


###############################################################################
# Base Enum
###############################################################################


class StringEnum(str, Enum):
    """
    Base class for all string enumerations.

    Provides string compatibility while maintaining Enum safety.
    """

    def __str__(self) -> str:
        return self.value


###############################################################################
# LLM Provider
###############################################################################


class LLMProvider(StringEnum):
    """
    Supported Large Language Model providers.
    """

    GEMINI = "gemini"


###############################################################################
# Gemini Models
###############################################################################


class GeminiModel(StringEnum):
    """
    Supported Gemini models.
    """

    GEMINI_25_FLASH = "gemini-3.6-flash"

    GEMINI_25_PRO = "gemini-2.5-pro"


###############################################################################
# Embedding Models
###############################################################################


class EmbeddingModel(StringEnum):
    """
    Supported embedding models.
    """

    CODEBERT = "microsoft/codebert-base"


###############################################################################
# Review Severity
###############################################################################


class ReviewSeverity(StringEnum):
    """
    Severity levels assigned to detected issues.
    """

    LOW = "Low"

    MEDIUM = "Medium"

    HIGH = "High"

    CRITICAL = "Critical"


###############################################################################
# Review Status
###############################################################################


class ReviewStatus(StringEnum):
    """
    Status of review generation.
    """

    SUCCESS = "Success"

    FAILED = "Failed"

    PARTIAL = "Partial"


###############################################################################
# Review Category
###############################################################################


class ReviewCategory(StringEnum):
    """
    Categories used in AI review.
    """

    BUG = "Bug"

    SECURITY = "Security"

    PERFORMANCE = "Performance"

    READABILITY = "Readability"

    MAINTAINABILITY = "Maintainability"

    STYLE = "Style"

    DOCUMENTATION = "Documentation"

    REFACTORING = "Refactoring"

    BEST_PRACTICE = "Best Practice"


###############################################################################
# Programming Languages
###############################################################################


class ProgrammingLanguage(StringEnum):
    """
    Supported programming languages.
    """

    PYTHON = "python"

    JAVA = "java"

    JAVASCRIPT = "javascript"

    TYPESCRIPT = "typescript"

    C = "c"

    CPP = "cpp"

    CSHARP = "csharp"

    GO = "go"

    PHP = "php"


###############################################################################
# Output Format
###############################################################################


class OutputFormat(StringEnum):
    """
    Supported output formats.
    """

    JSON = "json"

    MARKDOWN = "markdown"

    HTML = "html"

    TEXT = "text"


###############################################################################
# Retrieval Method
###############################################################################


class RetrievalMethod(StringEnum):
    """
    Supported retrieval mechanisms.
    """

    FAISS = "faiss"

    COSINE = "cosine_similarity"


###############################################################################
# Log Level
###############################################################################


class LogLevel(StringEnum):
    """
    Logging levels.
    """

    DEBUG = "DEBUG"

    INFO = "INFO"

    WARNING = "WARNING"

    ERROR = "ERROR"

    CRITICAL = "CRITICAL"


###############################################################################
# File Type
###############################################################################


class FileType(StringEnum):
    """
    Supported project file types.
    """

    PYTHON = ".py"

    JSON = ".json"

    PARQUET = ".parquet"

    LOG = ".log"

    TEXT = ".txt"


###############################################################################
# Environment
###############################################################################


class Environment(StringEnum):
    """
    Runtime environments.
    """

    DEVELOPMENT = "development"

    TESTING = "testing"

    PRODUCTION = "production"


###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "StringEnum",
    "LLMProvider",
    "GeminiModel",
    "EmbeddingModel",
    "ReviewSeverity",
    "ReviewStatus",
    "ReviewCategory",
    "ProgrammingLanguage",
    "OutputFormat",
    "RetrievalMethod",
    "LogLevel",
    "FileType",
    "Environment",
]