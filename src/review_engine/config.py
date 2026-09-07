"""
Configuration module for the AI Code Reviewer.

This module centralizes all application settings, including:
- Gemini API configuration
- Retrieval settings
- Prompt settings
- Logging configuration
- Review generation parameters

Environment variables are loaded automatically from the project's
`.env` file using Pydantic Settings.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------
# Base Project Settings
# --------------------------------------------------

class Settings(BaseSettings):
    """
    Global configuration settings for the AI Code Reviewer.

    Values are automatically loaded from the .env file.
    """

    # ---------- Gemini ----------
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    model_name: str = Field(default="gemini-3.6-flash", alias="MODEL_NAME")

    # ---------- Generation ----------
    temperature: float = Field(default=0.2, alias="TEMPERATURE")
    max_output_tokens: int = Field(default=2048, alias="MAX_OUTPUT_TOKENS")

    # ---------- Retrieval ----------
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")

    # ---------- Sampling ----------
    top_p: float = Field(default=0.95, alias="TOP_P")

    # ---------- Logging ----------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

SRC_DIR = BASE_DIR / "src"

DATASETS_DIR = BASE_DIR / "datasets"

RAW_DATA_DIR = DATASETS_DIR / "raw"

PROCESSED_DATA_DIR = DATASETS_DIR / "processed"

FEATURES_DIR = DATASETS_DIR / "featured"

EMBEDDINGS_DIR = DATASETS_DIR / "embeddings"

LABELS_DIR = DATASETS_DIR / "labels"

MODELS_DIR = BASE_DIR / "models"

OUTPUTS_DIR = BASE_DIR / "outputs"

LOGS_DIR = OUTPUTS_DIR / "logs"

REPORTS_DIR = OUTPUTS_DIR / "reports"

FAISS_INDEX_DIR = BASE_DIR / "faiss_indexes"


# --------------------------------------------------
# Create Required Directories
# --------------------------------------------------

for directory in [
    MODELS_DIR,
    OUTPUTS_DIR,
    LOGS_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Global Settings Instance
# --------------------------------------------------

settings = Settings()


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def get_gemini_api_key() -> str:
    """Return the Gemini API key."""
    return settings.gemini_api_key


def get_model_name() -> str:
    """Return the configured Gemini model."""
    return settings.model_name


def get_temperature() -> float:
    """Return the generation temperature."""
    return settings.temperature


def get_max_output_tokens() -> int:
    """Return the maximum output tokens."""
    return settings.max_output_tokens


def get_top_k_results() -> int:
    """Return the number of retrieval results."""
    return settings.top_k_results


def get_top_p() -> float:
    """Return the top-p sampling value."""
    return settings.top_p


def get_log_level() -> str:
    """Return the configured logging level."""
    return settings.log_level


# --------------------------------------------------
# Exports
# --------------------------------------------------

__all__ = [
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
]
# --------------------------------------------------
# Configuration Validation
# --------------------------------------------------

def validate_configuration() -> None:
    """
    Validate the application configuration.

    Raises
    ------
    ValueError
        If any configuration value is invalid.
    """

    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Please configure it in the .env file."
        )

    if settings.temperature < 0 or settings.temperature > 2:
        raise ValueError(
            "TEMPERATURE must be between 0.0 and 2.0."
        )

    if settings.top_p <= 0 or settings.top_p > 1:
        raise ValueError(
            "TOP_P must be between 0 and 1."
        )

    if settings.max_output_tokens <= 0:
        raise ValueError(
            "MAX_OUTPUT_TOKENS must be greater than zero."
        )

    if settings.top_k_results <= 0:
        raise ValueError(
            "TOP_K_RESULTS must be greater than zero."
        )


# --------------------------------------------------
# Path Helper Functions
# --------------------------------------------------

def get_dataset_path() -> Path:
    """
    Returns
    -------
    Path
        datasets directory.
    """
    return DATASETS_DIR


def get_raw_dataset_path() -> Path:
    """
    Returns
    -------
    Path
        raw dataset directory.
    """
    return RAW_DATA_DIR


def get_processed_dataset_path() -> Path:
    """
    Returns
    -------
    Path
        processed dataset directory.
    """
    return PROCESSED_DATA_DIR


def get_embeddings_path() -> Path:
    """
    Returns
    -------
    Path
        embeddings directory.
    """
    return EMBEDDINGS_DIR


def get_labels_path() -> Path:
    """
    Returns
    -------
    Path
        labels directory.
    """
    return LABELS_DIR


def get_models_path() -> Path:
    """
    Returns
    -------
    Path
        models directory.
    """
    return MODELS_DIR


def get_outputs_path() -> Path:
    """
    Returns
    -------
    Path
        outputs directory.
    """
    return OUTPUTS_DIR


def get_logs_path() -> Path:
    """
    Returns
    -------
    Path
        logs directory.
    """
    return LOGS_DIR


def get_reports_path() -> Path:
    """
    Returns
    -------
    Path
        reports directory.
    """
    return REPORTS_DIR


def get_faiss_index_path() -> Path:
    """
    Returns
    -------
    Path
        FAISS index directory.
    """
    return FAISS_INDEX_DIR


# --------------------------------------------------
# Directory Utilities
# --------------------------------------------------

def create_required_directories() -> None:
    """
    Create all required project directories.
    """

    directories = [
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
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def directory_exists(directory: Path) -> bool:
    """
    Check whether a directory exists.

    Parameters
    ----------
    directory : Path

    Returns
    -------
    bool
    """
    return directory.exists()


# --------------------------------------------------
# Environment Utilities
# --------------------------------------------------

def environment_loaded() -> bool:
    """
    Returns
    -------
    bool
        True if Gemini API Key exists.
    """
    return bool(settings.gemini_api_key.strip())


def get_environment_summary() -> dict:
    """
    Returns
    -------
    dict
        Configuration summary.
    """

    return {
        "model": settings.model_name,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k_results": settings.top_k_results,
        "max_output_tokens": settings.max_output_tokens,
        "log_level": settings.log_level,
        "environment_loaded": environment_loaded(),
    }


# --------------------------------------------------
# Initialize Configuration
# --------------------------------------------------

create_required_directories()

# NOTE: validate_configuration() is intentionally NOT called here.
#
# It used to run unconditionally at import time, which meant simply
# importing anything in this package (including code that never
# touches Gemini, e.g. ContextBuilder) crashed with
# "GEMINI_API_KEY is missing" for anyone without a .env file yet.
#
# Validation now happens lazily, at the point something actually
# needs the Gemini API key (e.g. when constructing a GeminiClient),
# so retrieval/context-building/testing code can be imported and run
# without requiring Gemini credentials to exist.


__all__.extend(
    [
        "validate_configuration",
        "create_required_directories",
        "directory_exists",
        "environment_loaded",
        "get_environment_summary",
        "get_dataset_path",
        "get_raw_dataset_path",
        "get_processed_dataset_path",
        "get_embeddings_path",
        "get_labels_path",
        "get_models_path",
        "get_outputs_path",
        "get_logs_path",
        "get_reports_path",
        "get_faiss_index_path",
    ]
)