"""
context_builder.py
==================

Builds an optimized context from retrieval results for the LLM.

Responsibilities
----------------
* Remove duplicate code snippets.
* Filter by similarity threshold.
* Rank snippets.
* Respect context size budget.
* Generate a clean context string for prompt construction.
"""

from __future__ import annotations

###############################################################################
# Standard Library Imports
###############################################################################

import logging
from dataclasses import dataclass
from typing import Iterable

###############################################################################
# Third Party Imports
###############################################################################

import pandas as pd

###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)

###############################################################################
# Defaults
###############################################################################

DEFAULT_MAX_SNIPPETS = 5

DEFAULT_MAX_CONTEXT_CHARS = 12000

DEFAULT_MIN_SIMILARITY = 0.60

###############################################################################
# Context Result
###############################################################################


@dataclass(slots=True)
class ContextResult:
    """
    Final context returned to the prompt builder.
    """

    context: str

    snippets_used: int

    total_characters: int

    average_similarity: float

    metadata: list[dict]


###############################################################################
# Context Builder
###############################################################################


class ContextBuilder:
    """
    Builds the context supplied to the LLM.
    """

    def __init__(
        self,
        max_snippets: int = DEFAULT_MAX_SNIPPETS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> None:

        self.max_snippets = max_snippets

        self.max_context_chars = max_context_chars

        self.min_similarity = min_similarity

        logger.info(
            "ContextBuilder initialized."
        )

    ###########################################################################
    # Validation
    ###########################################################################

    @staticmethod
    def validate(
        results: pd.DataFrame,
    ) -> None:

        if not isinstance(results, pd.DataFrame):

            raise TypeError(
                "results must be a pandas DataFrame."
            )

    ###########################################################################
    # Remove Duplicates
    ###########################################################################

    def remove_duplicates(
        self,
        results: pd.DataFrame,
    ) -> pd.DataFrame:

        self.validate(results)

        if results.empty:
            return results

        if "func_code_string" not in results.columns:
            return results

        deduplicated = (
            results
            .drop_duplicates(
                subset="func_code_string"
            )
            .reset_index(drop=True)
        )

        logger.info(
            "Removed %d duplicate snippets.",
            len(results) - len(deduplicated),
        )

        return deduplicated

    ###########################################################################
    # Similarity Filter
    ###########################################################################

    def filter_similarity(
        self,
        results: pd.DataFrame,
    ) -> pd.DataFrame:

        self.validate(results)

        if results.empty:
            return results

        if "similarity_score" not in results.columns:
            return results

        filtered = (
            results[
                results["similarity_score"]
                >= self.min_similarity
            ]
            .reset_index(drop=True)
        )

        logger.info(
            "Retained %d snippets after similarity filtering.",
            len(filtered),
        )

        return filtered

    ###########################################################################
    # Ranking
    ###########################################################################

    def rank(
        self,
        results: pd.DataFrame,
    ) -> pd.DataFrame:

        self.validate(results)

        if results.empty:
            return results

        if "similarity_score" not in results.columns:
            return results

        ranked = (
            results
            .sort_values(
                by="similarity_score",
                ascending=False,
            )
            .head(self.max_snippets)
            .reset_index(drop=True)
        )

        logger.info(
            "Selected top %d snippets.",
            len(ranked),
        )

        return ranked

    ###########################################################################
    # Context Section
    ###########################################################################

    @staticmethod
    def build_section(
        row: pd.Series,
        rank: int,
    ) -> str:

        similarity = (
            float(
                row.get(
                    "similarity_score",
                    0.0,
                )
            )
            * 100
        )

        function = row.get(
            "func_name",
            "Unknown",
        )

        project = row.get(
            "project_name",
            row.get(
                "repository",
                "Unknown",
            ),
        )

        documentation = row.get(
            "func_documentation_string",
            "",
        )

        documentation = (
            documentation.strip()
            if isinstance(documentation, str)
            else ""
        )

        code = row.get(
            "func_code_string",
            "",
        )

        code = (
            code.strip()
            if isinstance(code, str)
            else ""
        )

        header = (
            f"### Retrieved Snippet {rank}\n\n"
            f"Similarity : {similarity:.2f}%\n"
            f"Function   : {function}\n"
            f"Project    : {project}\n"
        )

        doc_section = (
            f"\nDocumentation: {documentation}\n"
            if documentation
            else ""
        )

        # Only emit a code block if source code is actually available.
        # An empty ```python ``` block would be misleading filler.
        code_section = (
            f"\n```python\n{code}\n```\n"
            if code
            else "\n(source code not available for this snippet)\n"
        )

        return header + doc_section + code_section
    ###########################################################################
    # Build Context
    ###########################################################################

    def build(
        self,
        results: pd.DataFrame,
    ) -> ContextResult:
        """
        Build the final LLM context.

        Parameters
        ----------
        results : pd.DataFrame

        Returns
        -------
        ContextResult
        """

        self.validate(results)

        if results.empty:

            logger.warning(
                "No retrieval results available."
            )

            return ContextResult(
                context="",
                snippets_used=0,
                total_characters=0,
                average_similarity=0.0,
                metadata=[],
            )

        #######################################################################
        # Pipeline
        #######################################################################

        results = self.remove_duplicates(results)

        results = self.filter_similarity(results)

        results = self.rank(results)

        #######################################################################
        # Build Context
        #######################################################################

        context_parts: list[str] = []

        metadata: list[dict] = []

        current_size = 0

        similarity_sum = 0.0

        snippets_used = 0

        for index, (_, row) in enumerate(
            results.iterrows(),
            start=1,
        ):

            section = self.build_section(
                row=row,
                rank=index,
            )

            section_size = len(section)

            ###################################################################
            # Respect Context Budget
            ###################################################################

            if (
                current_size + section_size
                > self.max_context_chars
            ):

                logger.info(
                    "Context limit reached (%d characters).",
                    self.max_context_chars,
                )

                break

            context_parts.append(section)

            current_size += section_size

            snippets_used += 1

            similarity_sum += float(
                row.get(
                    "similarity_score",
                    0.0,
                )
            )

            metadata.append(
                {
                    "rank": index,
                    "function": row.get(
                        "func_name",
                        "Unknown",
                    ),
                    "project": row.get(
                        "project_name",
                        row.get(
                            "repository",
                            "Unknown",
                        ),
                    ),
                    "similarity": float(
                        row.get(
                            "similarity_score",
                            0.0,
                        )
                    ),
                    "documentation": row.get(
                        "func_documentation_string",
                        "",
                    ),
                }
            )

        #######################################################################
        # Statistics
        #######################################################################

        average_similarity = (
            similarity_sum / snippets_used
            if snippets_used
            else 0.0
        )

        context = "\n\n".join(
            context_parts
        )

        logger.info(
            "Built context with %d snippets (%d characters).",
            snippets_used,
            len(context),
        )

        return ContextResult(
            context=context,
            snippets_used=snippets_used,
            total_characters=len(context),
            average_similarity=average_similarity,
            metadata=metadata,
        )

    ###########################################################################
    # Utility Methods
    ###########################################################################

    def info(
        self,
    ) -> dict[str, float | int]:

        return {
            "max_snippets": self.max_snippets,
            "max_context_chars": self.max_context_chars,
            "min_similarity": self.min_similarity,
        }

    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}("
            f"max_snippets={self.max_snippets}, "
            f"max_context_chars={self.max_context_chars}, "
            f"min_similarity={self.min_similarity})"
        )


###############################################################################
# Convenience Function
###############################################################################


def build_context(
    results: pd.DataFrame,
    max_snippets: int = DEFAULT_MAX_SNIPPETS,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> ContextResult:
    """
    Build context using default configuration.
    """

    builder = ContextBuilder(
        max_snippets=max_snippets,
        max_context_chars=max_context_chars,
        min_similarity=min_similarity,
    )

    return builder.build(
        results
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

    sample = pd.DataFrame(
        {
            "func_name": [
                "factorial",
                "sum_numbers",
            ],
            "project_name": [
                "Demo Project",
                "Demo Project",
            ],
            "similarity_score": [
                0.96,
                0.91,
            ],
            "func_code_string": [
                (
                    "def factorial(n):\n"
                    "    if n == 0:\n"
                    "        return 1\n"
                    "    return n * factorial(n - 1)"
                ),
                (
                    "def sum_numbers(a, b):\n"
                    "    return a + b"
                ),
            ],
        }
    )

    result = build_context(sample)

    print("=" * 80)
    print(result.context)
    print("=" * 80)

    print("\nMetadata")
    print(result.metadata)

    print("\nCharacters")
    print(result.total_characters)

    print("\nAverage Similarity")
    print(f"{result.average_similarity:.2%}")


###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "ContextResult",
    "ContextBuilder",
    "build_context",
    "DEFAULT_MAX_SNIPPETS",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_MIN_SIMILARITY",
]