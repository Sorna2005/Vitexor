"""
base_client.py
==============

Abstract base class for all Large Language Model (LLM) providers.

Responsibilities
----------------
* Provide a common interface for every LLM provider.
* Validate prompts before inference.
* Standardize responses across providers.
* Measure inference latency.
* Support future providers including:
    - Google Gemini
    - OpenAI GPT
    - Anthropic Claude
    - Azure OpenAI
    - Ollama
"""

from __future__ import annotations

###############################################################################
# Standard Library Imports
###############################################################################

import logging
import time
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)

###############################################################################
# Response Object
###############################################################################


@dataclass(slots=True)
class LLMResponse:
    """
    Standard response returned by every LLM implementation.
    """

    response: str

    model_name: str

    prompt_tokens: int | None = None

    completion_tokens: int | None = None

    total_tokens: int | None = None

    latency: float = 0.0

    finish_reason: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    ###########################################################################

    @property
    def success(
        self,
    ) -> bool:
        """
        Indicates whether a valid response was produced.
        """

        return bool(self.response.strip())

    ###########################################################################

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert response to dictionary.
        """

        return {
            "response": self.response,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency": self.latency,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
        }


###############################################################################
# Base Client
###############################################################################


class BaseLLMClient(ABC):
    """
    Abstract interface implemented by all LLM providers.

    Every provider must implement:

    • connect()
    • generate()
    • health_check()
    """

    ###########################################################################

    def __init__(
        self,
        model_name: str,
    ) -> None:

        self.model_name = model_name

        logger.info(
            "%s initialized with model '%s'.",
            self.__class__.__name__,
            self.model_name,
        )

    ###########################################################################
    # Validation
    ###########################################################################

    @staticmethod
    def validate_prompt(
        prompt: str,
    ) -> None:
        """
        Validate prompt before sending it to an LLM.
        """

        if not isinstance(prompt, str):

            raise TypeError(
                "Prompt must be a string."
            )

        if not prompt.strip():

            raise ValueError(
                "Prompt cannot be empty."
            )

    ###########################################################################
    # Response Generation Wrapper
    ###########################################################################

    def generate_response(
        self,
        prompt: str,
    ) -> LLMResponse:
        """
        Wrapper around generate().

        Performs:

        - prompt validation
        - latency measurement
        - response standardization
        """

        self.validate_prompt(
            prompt
        )

        logger.info(
            "Generating response using %s...",
            self.model_name,
        )

        start_time = time.perf_counter()

        response = self.generate(
            prompt
        )

        response.latency = (
            time.perf_counter()
            - start_time
        )

        if response.total_tokens is None:

            prompt_tokens = (
                response.prompt_tokens
                if response.prompt_tokens is not None
                else self.count_prompt_tokens(prompt)
            )

            completion_tokens = (
                response.completion_tokens
                if response.completion_tokens is not None
                else self.count_response_tokens(
                    response.response
                )
            )

            response.prompt_tokens = prompt_tokens
            response.completion_tokens = completion_tokens
            response.total_tokens = (
                prompt_tokens
                + completion_tokens
            )

        logger.info(
            "Response generated successfully in %.3f seconds.",
            response.latency,
        )

        return response

    ###########################################################################
    # Abstract Methods
    ###########################################################################

    @abstractmethod
    def connect(
        self,
    ) -> None:
        """
        Initialize the provider.
        """

    ###########################################################################

    @abstractmethod
    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        """
        Generate a response from the provider.
        """

    ###########################################################################

    @abstractmethod
    def health_check(
        self,
    ) -> bool:
        """
        Verify that the provider is operational.
        """

    ###########################################################################
    # Provider Information
    ###########################################################################

    def info(
        self,
    ) -> dict[str, str]:
        """
        Return provider information.
        """

        return {
            "provider": self.__class__.__name__,
            "model": self.model_name,
        }

    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}')"
        )
    ###########################################################################
    # Token Utilities
    ###########################################################################

    @staticmethod
    def estimate_tokens(
        text: str,
    ) -> int:
        """
        Estimate the number of tokens.

        Note
        ----
        This is an approximation (≈4 characters/token).
        """

        if not text:
            return 0

        return max(1, len(text) // 4)

    ###########################################################################

    def count_prompt_tokens(
        self,
        prompt: str,
    ) -> int:
        """
        Estimate prompt token count.
        """

        self.validate_prompt(prompt)

        return self.estimate_tokens(prompt)

    ###########################################################################

    def count_response_tokens(
        self,
        response: str,
    ) -> int:
        """
        Estimate response token count.
        """

        return self.estimate_tokens(response)

    ###########################################################################
    # Lifecycle Methods
    ###########################################################################

    def close(
        self,
    ) -> None:
        """
        Close provider resources.

        Override if the provider maintains
        persistent sessions.
        """

        logger.info(
            "%s closed.",
            self.__class__.__name__,
        )

    ###########################################################################

    def __enter__(
        self,
    ) -> "BaseLLMClient":
        """
        Context manager entry.
        """

        self.connect()

        return self

    ###########################################################################

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """
        Context manager exit.
        """

        self.close()

    ###########################################################################
    # Provider Capabilities
    ###########################################################################

    def supports_streaming(
        self,
    ) -> bool:
        """
        Whether the provider supports streaming.
        """

        return False

    ###########################################################################

    def supports_system_prompt(
        self,
    ) -> bool:
        """
        Whether native system prompts are supported.
        """

        return True

    ###########################################################################

    def supports_json_mode(
        self,
    ) -> bool:
        """
        Whether structured JSON output is supported.
        """

        return False

    ###########################################################################

    def supports_function_calling(
        self,
    ) -> bool:
        """
        Whether tool/function calling is supported.
        """

        return False

    ###########################################################################

    def capabilities(
        self,
    ) -> dict[str, object]:
        """
        Return provider capabilities.
        """

        return {
            "provider": self.__class__.__name__,
            "model": self.model_name,
            "streaming": self.supports_streaming(),
            "system_prompt": self.supports_system_prompt(),
            "json_mode": self.supports_json_mode(),
            "function_calling": self.supports_function_calling(),
        }

    ###########################################################################
    # Utility Methods
    ###########################################################################

    def __str__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}"
            f"(model='{self.model_name}')"
        )


###############################################################################
# CLI Example
###############################################################################

if __name__ == "__main__":

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
        Dummy implementation for testing.
        """

        def __init__(
            self,
        ) -> None:

            super().__init__(
                model_name="dummy-model"
            )

        def connect(
            self,
        ) -> None:

            logger.info(
                "Connected to Dummy LLM."
            )

        def generate(
            self,
            prompt: str,
        ) -> LLMResponse:

            prompt_tokens = (
                self.count_prompt_tokens(
                    prompt
                )
            )

            response = (
                "This is a dummy response."
            )

            completion_tokens = (
                self.count_response_tokens(
                    response
                )
            )

            return LLMResponse(
                response=response,
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=(
                    prompt_tokens
                    + completion_tokens
                ),
                finish_reason="completed",
            )

        def health_check(
            self,
        ) -> bool:

            return True


    with DummyLLM() as client:

        result = client.generate_response(
            "Explain binary search."
        )

        print("=" * 80)
        print("Provider Information")
        print("=" * 80)
        print(client.info())

        print("\nCapabilities")
        print("=" * 80)
        print(client.capabilities())

        print("\nLLM Response")
        print("=" * 80)
        print(result.response)

        print("\nToken Usage")
        print("=" * 80)
        print(
            f"Prompt Tokens     : {result.prompt_tokens}"
        )
        print(
            f"Completion Tokens : {result.completion_tokens}"
        )
        print(
            f"Total Tokens      : {result.total_tokens}"
        )

        print(
            f"\nLatency           : "
            f"{result.latency:.3f} sec"
        )


###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "LLMResponse",
    "BaseLLMClient",
]