"""
review_engine

Enterprise Review Engine package for the AI Code Reviewer.

This package provides the complete business logic required for
retrieval-augmented AI code review using CodeBERT, FAISS, and Gemini.

Modules
-------
config
    Centralized application configuration.

constants
    Global constants used across the application.

enums
    Enumerations for configuration, review severity,
    output formats, logging, and providers.

exceptions
    Custom exception hierarchy.

validators
    Validation utilities.

models
    Shared Pydantic data models.

logger
    Centralized logging configuration.

context_builder
    Builds contextual information for Retrieval-Augmented Generation.

prompt_builder
    Generates optimized prompts for Gemini.

review_generator
    Generates structured AI code reviews.

formatter
    Formats generated reviews into multiple output formats.
"""

from __future__ import annotations

###############################################################################
# Package Metadata
###############################################################################

__title__ = "review_engine"

__description__ = (
    "Enterprise AI Code Review Engine using "
    "CodeBERT + FAISS + Gemini."
)

__version__ = "1.0.0"

__author__ = "Sornalatha"

__license__ = "MIT"

###############################################################################
# Configuration
###############################################################################

from .config import (
    settings,
    BASE_DIR,
    SRC_DIR,
    DATASETS_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURES_DIR,
    EMBEDDINGS_DIR,
    LABELS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    FAISS_INDEX_DIR,
    get_gemini_api_key,
    get_model_name,
    get_temperature,
    get_max_output_tokens,
    get_top_k_results,
    get_top_p,
    get_log_level,
    validate_configuration,
    create_required_directories,
    get_environment_summary,
)

###############################################################################
# Core Components
###############################################################################

from .context_builder import ContextBuilder
from .prompt_builder import PromptBuilder
from .review_generator import ReviewGenerator
from .formatter import ReviewFormatter

###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "__title__",
    "__description__",
    "__version__",
    "__author__",
    "__license__",
    "settings",
    "BASE_DIR",
    "SRC_DIR",
    "DATASETS_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "FEATURES_DIR",
    "EMBEDDINGS_DIR",
    "LABELS_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "LOGS_DIR",
    "REPORTS_DIR",
    "FAISS_INDEX_DIR",
    "get_gemini_api_key",
    "get_model_name",
    "get_temperature",
    "get_max_output_tokens",
    "get_top_k_results",
    "get_top_p",
    "get_log_level",
    "validate_configuration",
    "create_required_directories",
    "get_environment_summary",
    "ContextBuilder",
    "PromptBuilder",
    "ReviewGenerator",
    "ReviewFormatter",
]