"""
Production-ready Google Gemini client implementation.

Responsibilities
----------------
* Connect to Google Gemini.
* Generate code reviews.
* Handle retries.
* Handle API failures.
* Support configurable generation parameters.
* Return standardized LLMResponse objects.
"""

from __future__ import annotations

###############################################################################
# Standard Library Imports
###############################################################################

import logging
import time
from typing import Any

###############################################################################
# Third-Party Imports
###############################################################################

import google.generativeai as genai

from google.api_core.exceptions import (
    DeadlineExceeded,
    GoogleAPIError,
    ResourceExhausted,
    ServiceUnavailable,
)

###############################################################################
# Local Imports
###############################################################################

from .base_client import (
    BaseLLMClient,
    LLMResponse,
)

###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)


###############################################################################
# Gemini Client
###############################################################################


class GeminiClient(BaseLLMClient):
    """
    Google Gemini implementation.

    Parameters
    ----------
    api_key : str
        Google Gemini API key.

    model_name : str
        Gemini model name.

    temperature : float
        Controls randomness of the generated response.

    top_p : float
        Nucleus sampling parameter.

    top_k : int
        Limits the number of candidate tokens.

    max_output_tokens : int
        Maximum number of tokens Gemini can generate.

    max_retries : int
        Maximum number of attempts for temporary API failures.
    """

    ###########################################################################
    # Initialization
    ###########################################################################

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.6-flash",
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 2048,
        max_retries: int = 2,
    ) -> None:

        super().__init__(
            model_name=model_name,
        )

        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        # Reduced from 4096 to 2048.
        # Code reviews normally do not require 4096 generated tokens.
        self.max_output_tokens = max_output_tokens

        # Reduced from 3 to 2 to avoid unnecessarily long retry cycles.
        self.max_retries = max_retries

        self._client = None

        self.connect()

    ###########################################################################
    # Connection
    ###########################################################################

    def connect(
        self,
    ) -> None:
        """
        Configure the Gemini SDK and initialize the model.
        """

        logger.info(
            "Connecting to Google Gemini model '%s'...",
            self.model_name,
        )

        genai.configure(
            api_key=self.api_key,
        )

        self._client = genai.GenerativeModel(
            model_name=self.model_name,
        )

        logger.info(
            "Gemini connection initialized successfully."
        )

    ###########################################################################
    # Client Property
    ###########################################################################

    @property
    def client(
        self,
    ) -> genai.GenerativeModel:
        """
        Return the initialized Gemini client.
        """

        if self._client is None:
            raise RuntimeError(
                "Gemini client is not initialized."
            )

        return self._client

    ###########################################################################
    # Generation Configuration
    ###########################################################################

    def _generation_config(
        self,
    ) -> genai.GenerationConfig:
        """
        Build Gemini generation configuration.
        """

        return genai.GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
        )

    ###########################################################################
    # Response Extraction
    ###########################################################################

    def _extract_text(
        self,
        response: Any,
    ) -> str:
        """
        Safely extract generated text from Gemini response.
        """

        if hasattr(response, "text"):

            return response.text.strip()

        if (
            hasattr(response, "candidates")
            and response.candidates
        ):

            candidate = response.candidates[0]

            if (
                hasattr(candidate, "content")
                and candidate.content.parts
            ):

                return "".join(
                    part.text
                    for part in candidate.content.parts
                    if hasattr(part, "text")
                ).strip()

        return ""

    ###########################################################################
    # Retry Delay
    ###########################################################################

    def _retry_delay(
        self,
        attempt: int,
    ) -> None:
        """
        Apply a short exponential backoff before retrying.
        """

        # 1st retry = 1 second
        # 2nd retry = 2 seconds
        delay = 2 ** attempt

        logger.warning(
            "Retrying Gemini request in %s seconds...",
            delay,
        )

        time.sleep(delay)

    ###########################################################################
    # Gemini Invocation
    ###########################################################################

    def _invoke(
        self,
        prompt: str,
    ) -> LLMResponse:
        """
        Send one request to Gemini and standardize the response.
        """

        request_start = time.perf_counter()

        response = self.client.generate_content(
            prompt,
            generation_config=self._generation_config(),
        )

        request_latency = (
            time.perf_counter() - request_start
        )

        logger.info(
            "Gemini API request completed in %.3f seconds.",
            request_latency,
        )

        #######################################################################
        # Extract Response Text
        #######################################################################

        text = self._extract_text(response)

        if not text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        #######################################################################
        # Token Usage
        #######################################################################

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if usage is not None:

            prompt_tokens = getattr(
                usage,
                "prompt_token_count",
                None,
            )

            completion_tokens = getattr(
                usage,
                "candidates_token_count",
                None,
            )

            total_tokens = getattr(
                usage,
                "total_token_count",
                None,
            )

        #######################################################################
        # Finish Reason
        #######################################################################

        finish_reason = None

        if (
            hasattr(response, "candidates")
            and response.candidates
        ):

            finish_reason = str(
                getattr(
                    response.candidates[0],
                    "finish_reason",
                    None,
                )
            )

        #######################################################################
        # Return Standardized Response
        #######################################################################

        return LLMResponse(
            response=text,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            metadata={
                "provider": "Google Gemini",
                "request_latency": request_latency,
                "max_output_tokens": self.max_output_tokens,
            },
        )

    ###########################################################################
    # Generate
    ###########################################################################

    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        """
        Generate a response using Gemini.

        Temporary API failures are retried automatically.
        """

        self.validate_prompt(prompt)

        last_exception = None

        for attempt in range(
            self.max_retries,
        ):

            try:

                logger.info(
                    "Gemini request attempt %s/%s",
                    attempt + 1,
                    self.max_retries,
                )

                return self._invoke(
                    prompt,
                )

            ###################################################################
            # Temporary API Errors
            ###################################################################

            except (
                ResourceExhausted,
                ServiceUnavailable,
                DeadlineExceeded,
            ) as exc:

                last_exception = exc

                logger.warning(
                    "Temporary Gemini error on attempt %s: %s",
                    attempt + 1,
                    exc,
                )

                if (
                    attempt
                    < self.max_retries - 1
                ):

                    self._retry_delay(
                        attempt,
                    )

            ###################################################################
            # Google API Errors
            ###################################################################

            except GoogleAPIError as exc:

                logger.exception(
                    "Google Gemini API error."
                )

                raise RuntimeError(
                    "Gemini API request failed."
                ) from exc

            ###################################################################
            # Unexpected Errors
            ###################################################################

            except Exception as exc:

                logger.exception(
                    "Unexpected Gemini client error."
                )

                raise RuntimeError(
                    "Unexpected Gemini client error."
                ) from exc

        raise RuntimeError(
            "Maximum Gemini retry attempts exceeded."
        ) from last_exception

    ###########################################################################
    # Health Check
    ###########################################################################

    def health_check(
        self,
    ) -> bool:
        """
        Verify that Gemini is reachable.
        """

        try:

            self.generate(
                "Reply with only the word: OK"
            )

            logger.info(
                "Gemini health check passed."
            )

            return True

        except Exception as exc:

            logger.exception(
                "Gemini health check failed."
            )

            logger.debug(
                "Health check error: %s",
                exc,
            )

            return False

    ###########################################################################
    # Close
    ###########################################################################

    def close(
        self,
    ) -> None:
        """
        Release the Gemini client reference.
        """

        logger.info(
            "Closing Gemini client."
        )

        self._client = None

    ###########################################################################
    # Provider Capabilities
    ###########################################################################

    def supports_streaming(
        self,
    ) -> bool:
        """
        Indicates whether Gemini supports streaming.
        """

        return True

    ###########################################################################

    def supports_json_mode(
        self,
    ) -> bool:
        """
        Indicates whether structured JSON output is supported
        by this client implementation.
        """

        return False

    ###########################################################################

    def supports_function_calling(
        self,
    ) -> bool:
        """
        Indicates whether Gemini supports function calling.
        """

        return True

    ###########################################################################
    # Information
    ###########################################################################

    def info(
        self,
    ) -> dict[str, Any]:
        """
        Return information about the current Gemini client.
        """

        return {
            "provider": "Google Gemini",
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "max_retries": self.max_retries,
        }

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"max_output_tokens={self.max_output_tokens}, "
            f"max_retries={self.max_retries})"
        )


###############################################################################
# Factory
###############################################################################


def create_gemini_client_from_settings() -> "GeminiClient":
    """
    Build GeminiClient from application settings.
    """

    from src.review_engine.config import settings

    if (
        not settings.gemini_api_key
        or not settings.gemini_api_key.strip()
    ):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file "
            "(GEMINI_API_KEY=your-key-here) "
            "before making a Gemini request."
        )

    return GeminiClient(
        api_key=settings.gemini_api_key,
        model_name=settings.model_name,
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_output_tokens=settings.max_output_tokens,
    )


###############################################################################
# CLI Example
###############################################################################


if __name__ == "__main__":

    import os

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise EnvironmentError(
            "GEMINI_API_KEY environment variable "
            "is not set."
        )

    client = GeminiClient(
        api_key=api_key,
    )

    print("=" * 80)
    print("Client Information")
    print("=" * 80)
    print(client.info())

    print("\nHealth Check")
    print("=" * 80)
    print(client.health_check())

    print("\nGenerating Response...")
    print("=" * 80)

    result = client.generate_response(
        """
        Review this Python function:

        def add(a, b):
            return a + b
        """
    )

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

    client.close()


###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "GeminiClient",
    "create_gemini_client_from_settings",
]