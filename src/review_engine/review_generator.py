"""
review_generator.py
===================

Enterprise Review Generator for the AI Code Reviewer.

Responsibilities
----------------
1. Execute the complete AI review workflow.
2. Retrieve similar code using FAISS.
3. Build optimized context.
4. Construct the LLM prompt.
5. Send the prompt to the configured LLM.
6. Return a structured review response.
7. Measure execution time.
8. Provide detailed logging and metadata.
"""

from __future__ import annotations

###############################################################################
# Standard Library Imports
###############################################################################

import logging
import time
from dataclasses import dataclass
from typing import Any

###############################################################################
# Local Imports
###############################################################################

from src.review_engine.context_builder import (
    ContextBuilder,
    ContextResult,
)

from src.review_engine.prompt_builder import (
    PromptBuilder,
    PromptResult,
)

from src.retrieval.retrieval_pipeline import (
    RetrievalPipeline,
    RetrievalResult,
)

from src.llm.base_client import (
    BaseLLMClient,
    LLMResponse,
)

###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)

###############################################################################
# Review Result
###############################################################################


@dataclass(slots=True)
class ReviewResult:
    """
    Final review object returned by the ReviewGenerator.
    """

    query: str

    review: str

    retrieval: RetrievalResult

    context: ContextResult

    prompt: PromptResult

    llm_response: LLMResponse

    execution_time: float

    metadata: dict[str, Any]


###############################################################################
# Review Generator
###############################################################################


class ReviewGenerator:
    """
    Enterprise AI Review Generator.

    Workflow

    User Code
        │
        ▼
    RetrievalPipeline
        │
        ▼
    ContextBuilder
        │
        ▼
    PromptBuilder
        │
        ▼
    Gemini/OpenAI/Claude
        │
        ▼
    ReviewResult
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        retrieval_pipeline: RetrievalPipeline | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        """
        Initialize Review Generator.
        """

        self.llm_client = llm_client

        self.retrieval_pipeline = (
            retrieval_pipeline
            if retrieval_pipeline
            else RetrievalPipeline()
        )

        self.context_builder = (
            context_builder
            if context_builder
            else ContextBuilder()
        )

        self.prompt_builder = (
            prompt_builder
            if prompt_builder
            else PromptBuilder()
        )

        logger.info(
            "ReviewGenerator initialized successfully."
        )

    ###########################################################################
    # Validation
    ###########################################################################

    @staticmethod
    def validate_code(
        code: str,
    ) -> None:
        """
        Validate input source code.
        """

        if not isinstance(code, str):

            raise TypeError(
                "Source code must be a string."
            )

        if not code.strip():

            raise ValueError(
                "Source code cannot be empty."
            )

    ###########################################################################
    # Step 1
    ###########################################################################

    def retrieve_context(
        self,
        code: str,
    ) -> tuple[
        RetrievalResult,
        ContextResult,
    ]:
        """
        Retrieve similar code and build context.
        """

        logger.info(
            "Retrieving similar code..."
        )

        retrieval_result = (
            self.retrieval_pipeline.retrieve(
                query=code,
            )
        )

        context_result = (
            self.context_builder.build(
                retrieval_result.results
            )
        )

        logger.info(
            "Retrieved %d snippets.",
            context_result.snippets_used,
        )

        return (
            retrieval_result,
            context_result,
        )

    ###########################################################################
    # Step 2
    ###########################################################################

    def build_prompt(
        self,
        code: str,
        context: ContextResult,
    ) -> PromptResult:
        """
        Build the final prompt.
        """

        logger.info(
            "Building prompt..."
        )

        return self.prompt_builder.build(
            code=code,
            context=context.context,
        )

    ###########################################################################
    # Step 3
    ###########################################################################

    def generate_review(
        self,
        prompt: PromptResult,
    ) -> LLMResponse:
        """
        Generate AI review using the configured LLM.
        """

        logger.info(
            "Sending prompt to LLM..."
        )

        return self.llm_client.generate_response(
            prompt.prompt,
        )
    ###########################################################################
    # Complete Review Pipeline
    ###########################################################################

    def review(
        self,
        code: str,
    ) -> ReviewResult:
        """
        Execute the complete AI review workflow.

        Workflow
        --------
        Validate Input
            ↓
        Retrieve Similar Code
            ↓
        Build Context
            ↓
        Build Prompt
            ↓
        Generate AI Review
            ↓
        Build ReviewResult
        """

        self.validate_code(code)

        logger.info(
            "Starting AI code review..."
        )

        start_time = time.perf_counter()

        try:

            ###################################################################
            # Retrieve Similar Code
            ###################################################################

            retrieval_result, context_result = (
                self.retrieve_context(code)
            )

            ###################################################################
            # Build Prompt
            ###################################################################

            prompt_result = self.build_prompt(
                code=code,
                context=context_result,
            )

            ###################################################################
            # Generate AI Review
            ###################################################################

            llm_response = self.generate_review(
                prompt_result,
            )

            ###################################################################
            # Total Execution Time
            ###################################################################

            execution_time = (
                time.perf_counter()
                - start_time
            )

            ###################################################################
            # Metadata
            ###################################################################

            metadata = {
                "retrieved_snippets":
                    context_result.snippets_used,

                "context_size":
                    context_result.total_characters,

                "prompt_size":
                    prompt_result.total_characters,

                "retrieval_time":
                    retrieval_result.retrieval_time,

                "llm_latency":
                    llm_response.latency,

                "total_execution_time":
                    execution_time,

                "model":
                    llm_response.model_name,
            }

            logger.info(
                "AI review completed in %.3f seconds.",
                execution_time,
            )

            ###################################################################
            # Build Final Result
            ###################################################################

            return ReviewResult(
                query=code,
                review=llm_response.response,
                retrieval=retrieval_result,
                context=context_result,
                prompt=prompt_result,
                llm_response=llm_response,
                execution_time=execution_time,
                metadata=metadata,
            )

        except Exception as exc:

            logger.exception(
                "AI review failed: %s",
                exc,
            )

            raise

    ###########################################################################
    # Batch Review
    ###########################################################################

    def batch_review(
        self,
        codes: list[str],
    ) -> list[ReviewResult]:
        """
        Review multiple source code files.

        Parameters
        ----------
        codes : list[str]

        Returns
        -------
        list[ReviewResult]
        """

        logger.info(
            "Starting batch review (%d files)...",
            len(codes),
        )

        reviews: list[ReviewResult] = []

        for index, code in enumerate(
            codes,
            start=1,
        ):

            logger.info(
                "Processing file %d/%d",
                index,
                len(codes),
            )

            try:

                reviews.append(
                    self.review(code)
                )

            except Exception:

                logger.exception(
                    "Failed reviewing file %d.",
                    index,
                )

        logger.info(
            "Batch review completed."
        )

        return reviews

    ###########################################################################
    # Quick Review
    ###########################################################################

    def quick_review(
        self,
        code: str,
    ) -> str:
        """
        Return only the review text.
        """

        return self.review(
            code
        ).review

    ###########################################################################
    # Prompt Preview
    ###########################################################################

    def preview_prompt(
        self,
        code: str,
    ) -> str:
        """
        Preview the generated prompt without calling the LLM.
        """

        retrieval_result, context_result = (
            self.retrieve_context(code)
        )

        prompt_result = self.build_prompt(
            code=code,
            context=context_result,
        )

        return prompt_result.prompt

    ###########################################################################
    # Context Preview
    ###########################################################################

    def preview_context(
        self,
        code: str,
    ) -> str:
        """
        Preview retrieved context only.
        """

        _, context_result = (
            self.retrieve_context(code)
        )

        return context_result.context
    ###########################################################################
    # Health Check
    ###########################################################################

    def health_check(
        self,
    ) -> bool:
        """
        Verify that all review generator components are operational.

        Returns
        -------
        bool
        """

        try:

            retrieval_ok = (
                self.retrieval_pipeline.health_check()
            )

            llm_ok = (
                self.llm_client.health_check()
            )

            return retrieval_ok and llm_ok

        except Exception:

            logger.exception(
                "ReviewGenerator health check failed."
            )

            return False

    ###########################################################################
    # Information
    ###########################################################################

    def info(
        self,
    ) -> dict[str, Any]:
        """
        Return information about the ReviewGenerator.
        """

        return {
            "llm": self.llm_client.info(),
            "retrieval": self.retrieval_pipeline.info(),
            "context_builder": self.context_builder.info(),
            "prompt_builder": self.prompt_builder.info(),
        }

    ###########################################################################
    # Reload Components
    ###########################################################################

    def reload(
        self,
    ) -> None:
        """
        Reload internal resources.
        """

        logger.info(
            "Reloading ReviewGenerator resources..."
        )

        self.retrieval_pipeline.reload()

        logger.info(
            "Resources reloaded successfully."
        )

    ###########################################################################
    # Utility Methods
    ###########################################################################

    def __repr__(
        self,
    ) -> str:
        """
        Developer representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"llm={self.llm_client.model_name}, "
            f"retrieval={self.retrieval_pipeline.__class__.__name__})"
        )

    ###########################################################################

    def __str__(
        self,
    ) -> str:
        """
        User-friendly representation.
        """

        return (
            f"ReviewGenerator("
            f"model={self.llm_client.model_name})"
        )


###############################################################################
# Convenience Function
###############################################################################


def review_code(
    code: str,
    llm_client: BaseLLMClient,
) -> ReviewResult:
    """
    Review source code using the default ReviewGenerator.

    Parameters
    ----------
    code : str

    llm_client : BaseLLMClient

    Returns
    -------
    ReviewResult
    """

    generator = ReviewGenerator(
        llm_client=llm_client,
    )

    return generator.review(
        code=code,
    )


###############################################################################
# CLI Example
###############################################################################

if __name__ == "__main__":

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    class DummyLLM(BaseLLMClient):
        """
        Minimal implementation for local testing.
        """

        def __init__(self) -> None:
            super().__init__(
                model_name="Dummy-LLM"
            )

        def connect(self) -> None:
            pass

        def generate(
            self,
            prompt: str,
        ) -> LLMResponse:

            return LLMResponse(
                response="Dummy AI review generated successfully.",
                model_name=self.model_name,
            )

        def health_check(
            self,
        ) -> bool:
            return True

    sample_code = """
def divide(a, b):
    return a / b
"""

    llm = DummyLLM()

    generator = ReviewGenerator(
        llm_client=llm,
    )

    print("=" * 80)
    print("Generator Information")
    print("=" * 80)
    print(generator.info())

    print("\nHealth Check")
    print(generator.health_check())

    print("\nPrompt Preview")
    print("-" * 80)
    print(generator.preview_prompt(sample_code))

    print("\nRunning Review")
    print("-" * 80)

    result = generator.review(
        sample_code
    )

    print(result.review)

    print("\nMetadata")
    print(result.metadata)

    print("\nExecution Time")
    print(f"{result.execution_time:.3f} seconds")


###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "ReviewResult",
    "ReviewGenerator",
    "review_code",
]