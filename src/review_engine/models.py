"""
models.py

Enterprise data models for the AI Code Reviewer.

This module contains strongly typed Pydantic models shared across the
application. These models provide validation, serialization,
deserialization, and structured communication between modules.
"""

from __future__ import annotations

###############################################################################
# Standard Library Imports
###############################################################################

from datetime import datetime
from typing import Any

###############################################################################
# Third Party Imports
###############################################################################

from pydantic import BaseModel, Field, ConfigDict

###############################################################################
# Review Issue
###############################################################################


class ReviewIssue(BaseModel):
    """
    Represents a single issue detected during AI code review.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    title: str = Field(...)

    description: str = Field(...)

    category: str = Field(...)

    severity: str = Field(...)

    recommendation: str = Field(...)

    line_number: int | None = Field(default=None)


###############################################################################
# Retrieved Context
###############################################################################


class RetrievedContext(BaseModel):
    """
    Represents one retrieved document from FAISS.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    document_id: str

    similarity_score: float

    source_file: str

    code: str

    metadata: dict[str, Any] = Field(default_factory=dict)


###############################################################################
# Prompt Request
###############################################################################


class PromptRequest(BaseModel):
    """
    Prompt generation request.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    source_code: str

    language: str

    retrieved_context: list[RetrievedContext]

    additional_instruction: str | None = None


###############################################################################
# Prompt Response
###############################################################################


class PromptResponse(BaseModel):
    """
    Generated prompt.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    prompt: str

    token_estimate: int


###############################################################################
# Review Result
###############################################################################


class ReviewResult(BaseModel):
    """
    AI review result.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    summary: str

    risk_level: str

    issues: list[ReviewIssue]

    improved_code: str

    execution_time: float


###############################################################################
# Retrieval Result
###############################################################################


class RetrievalResult(BaseModel):
    """
    Result returned from FAISS retrieval.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    query: str

    retrieved_documents: list[RetrievedContext]

    total_documents: int

    retrieval_time: float


###############################################################################
# Gemini Response
###############################################################################


class GeminiResponse(BaseModel):
    """
    Parsed Gemini response.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    model_name: str

    response: str

    prompt_tokens: int | None = None

    completion_tokens: int | None = None

    total_tokens: int | None = None


###############################################################################
# Review Report
###############################################################################


class ReviewReport(BaseModel):
    """
    Final report returned by the application.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    project_name: str

    generated_at: datetime = Field(
        default_factory=datetime.utcnow
    )

    review: ReviewResult

    retrieval: RetrievalResult

    llm: GeminiResponse


###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "ReviewIssue",
    "RetrievedContext",
    "PromptRequest",
    "PromptResponse",
    "ReviewResult",
    "RetrievalResult",
    "GeminiResponse",
    "ReviewReport",
]
###############################################################################
# Code Metadata
###############################################################################

class CodeMetadata(BaseModel):
    """
    Metadata describing the submitted source code.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    filename: str

    language: str

    file_extension: str

    total_lines: int

    total_characters: int

    total_functions: int | None = None

    total_classes: int | None = None


###############################################################################
# Embedding Information
###############################################################################


class EmbeddingInformation(BaseModel):
    """
    Information about generated embeddings.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    embedding_model: str

    embedding_dimension: int

    vector_count: int

    generated_time: datetime = Field(
        default_factory=datetime.utcnow
    )


###############################################################################
# Retrieval Statistics
###############################################################################


class RetrievalStatistics(BaseModel):
    """
    Statistics collected during retrieval.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    searched_documents: int

    retrieved_documents: int

    similarity_threshold: float

    average_similarity: float

    retrieval_time: float


###############################################################################
# Prompt Statistics
###############################################################################


class PromptStatistics(BaseModel):
    """
    Statistics about the generated prompt.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    prompt_length: int

    estimated_tokens: int

    context_documents: int

    context_characters: int


###############################################################################
# Review Metrics
###############################################################################


class ReviewMetrics(BaseModel):
    """
    Metrics collected after review generation.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    total_issues: int

    low_severity: int

    medium_severity: int

    high_severity: int

    critical_severity: int

    execution_time: float


###############################################################################
# Application Metadata
###############################################################################


class ApplicationMetadata(BaseModel):
    """
    Metadata describing the application execution.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    project_name: str

    project_version: str

    generated_at: datetime = Field(
        default_factory=datetime.utcnow
    )

    llm_model: str

    embedding_model: str


###############################################################################
# Pipeline Result
###############################################################################


class PipelineResult(BaseModel):
    """
    Complete output generated by the review pipeline.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    metadata: ApplicationMetadata

    code_metadata: CodeMetadata

    retrieval_statistics: RetrievalStatistics

    prompt_statistics: PromptStatistics

    review_metrics: ReviewMetrics

    report: ReviewReport

###############################################################################
# Health Status
###############################################################################


class HealthStatus(BaseModel):
    """
    Represents the health status of the application.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    status: str

    timestamp: datetime = Field(
        default_factory=datetime.utcnow
    )

    services: dict[str, bool]

    message: str


###############################################################################
# Model Configuration
###############################################################################


class ModelConfiguration(BaseModel):
    """
    Configuration details of the active LLM.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    provider: str

    model_name: str

    temperature: float

    max_output_tokens: int

    top_p: float

    top_k: int


###############################################################################
# Embedding Configuration
###############################################################################


class EmbeddingConfiguration(BaseModel):
    """
    Configuration for embedding generation.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    model_name: str

    embedding_dimension: int

    normalize_embeddings: bool


###############################################################################
# Execution Summary
###############################################################################


class ExecutionSummary(BaseModel):
    """
    Summary of the entire review pipeline execution.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    started_at: datetime

    completed_at: datetime

    total_execution_time: float

    successful: bool

    error_message: str | None = None


###############################################################################
# Application Settings
###############################################################################


class ApplicationSettings(BaseModel):
    """
    Runtime settings used by the application.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    debug: bool

    environment: str

    log_level: str

    cache_enabled: bool

    retrieval_top_k: int

    similarity_threshold: float


###############################################################################
# Complete Application State
###############################################################################


class ApplicationState(BaseModel):
    """
    Represents the complete state of the AI Code Reviewer.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    settings: ApplicationSettings

    model_configuration: ModelConfiguration

    embedding_configuration: EmbeddingConfiguration

    execution_summary: ExecutionSummary

    pipeline_result: PipelineResult


###############################################################################
# Update Public Exports
###############################################################################

__all__.extend(
    [
        "CodeMetadata",
        "EmbeddingInformation",
        "RetrievalStatistics",
        "PromptStatistics",
        "ReviewMetrics",
        "ApplicationMetadata",
        "PipelineResult",
        "HealthStatus",
        "ModelConfiguration",
        "EmbeddingConfiguration",
        "ExecutionSummary",
        "ApplicationSettings",
        "ApplicationState",
    ]
)