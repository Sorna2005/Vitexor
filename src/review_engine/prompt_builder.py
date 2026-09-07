"""
prompt_builder.py

Constructs high-quality prompts for the AI Code Reviewer.

Responsibilities
----------------
- Build a structured prompt for the LLM.
- Combine:
    - System Instructions
    - Retrieved Context
    - User Code
    - Review Guidelines
    - Expected Output Format
- Produce a deterministic prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# PROMPT SECTIONS
# ============================================================

SYSTEM_PROMPT = """
You are an Expert Senior Software Engineer.

You specialize in:

- Python
- Java
- JavaScript
- C#
- C++
- Go
- Rust
- SQL

Your responsibilities are:

1. Detect bugs.
2. Detect logical errors.
3. Detect security vulnerabilities.
4. Detect performance bottlenecks.
5. Detect code smells.
6. Recommend best practices.
7. Suggest optimizations.
8. Improve readability.
9. Improve maintainability.
10. Produce production-ready code.

Use the retrieved reference code whenever it is helpful.

Never invent APIs.

Only provide technically correct recommendations.
""".strip()


REVIEW_GUIDELINES = """
Review the submitted code carefully.

Analyze:

1. Correctness
2. Readability
3. Performance
4. Maintainability
5. Security
6. Complexity
7. Error Handling
8. Edge Cases
9. Naming Conventions
10. Language Best Practices

If there are no issues, explicitly state that.

If issues exist, explain:

- Why they occur.
- Their impact.
- How to fix them.
""".strip()


OUTPUT_FORMAT = """
Return your answer using EXACTLY the following markdown section
headers, in this exact order, with the exact text shown (## plus
the exact header name). Do not rename, reorder, merge, or omit any
section, even if it is empty - write "None." under an empty section
instead of omitting it.

## Overall Summary

Short overview of the code and the review.

## Strengths

- One bullet per strength. Write "None." if there are none.

## Issues

One bullet per issue. Put the whole issue on a single bullet line,
in this form:
- [Severity: High|Medium|Low] Issue description. Explanation: why it
  matters. Recommendation: how to fix it.

## Security

- One bullet per security finding, same single-line style as Issues.
  Write "None." if there are none.

## Performance

- One bullet per performance finding, same single-line style.
  Write "None." if there are none.

## Maintainability

- One bullet per maintainability finding, same single-line style.
  Write "None." if there are none.

## Suggested Improvements

- One bullet per suggested improvement.

## Improved Code

Provide the improved implementation when appropriate, in a fenced
code block using the appropriate language tag. Write "None." if no
code change is warranted.

## Final Verdict

One concise concluding paragraph.
""".strip()


# ============================================================
# PROMPT RESULT
# ============================================================

@dataclass(slots=True)
class PromptResult:
    """
    Final prompt object returned by PromptBuilder.
    """

    prompt: str
    total_characters: int
    context_characters: int
    code_characters: int

    def __repr__(self) -> str:
        return (
            "PromptResult("
            f"total_characters={self.total_characters}, "
            f"context_characters={self.context_characters}, "
            f"code_characters={self.code_characters}"
            ")"
        )


# ============================================================
# PROMPT BUILDER
# ============================================================

class PromptBuilder:
    """
    Builds structured prompts for the AI Code Reviewer.
    """

    def __init__(self, include_context: bool = True) -> None:
        """
        Initialize PromptBuilder.

        Parameters
        ----------
        include_context : bool
            Whether retrieved reference code should be included.
        """

        self.include_context = include_context

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(
        self,
        code: str,
        context: str = "",
    ) -> None:
        """
        Validate code and retrieved context.
        """

        if not isinstance(code, str):
            raise TypeError(
                "code must be a string"
            )

        if not isinstance(context, str):
            raise TypeError(
                "context must be a string"
            )

        if not code.strip():
            raise ValueError(
                "Source code cannot be empty."
            )

    # ========================================================
    # BUILD CONTEXT BLOCK
    # ========================================================

    def build_context_block(
        self,
        context: str,
    ) -> str:
        """
        Build the retrieved-context section.
        """

        if not context.strip():
            return (
                "No retrieved reference code was provided."
            )

        return (
            "The following code was retrieved from the "
            "reference dataset.\n\n"
            "Use it only when it is relevant to the "
            "submitted source code.\n\n"
            "---------------- RETRIEVED CODE ----------------\n"
            f"{context.strip()}\n"
            "-------------- END RETRIEVED CODE --------------"
        )

    # ========================================================
    # BUILD SOURCE CODE BLOCK
    # ========================================================

    def build_code_block(
        self,
        code: str,
    ) -> str:
        """
        Build the source-code section.
        """

        return (
            "The following source code must be reviewed.\n\n"
            "---------------- SOURCE CODE ----------------\n"
            f"{code.strip()}\n"
            "-------------- END SOURCE CODE --------------"
        )

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    def build(
        self,
        code: str,
        context: str = "",
    ) -> PromptResult:
        """
        Build the final LLM prompt.

        Parameters
        ----------
        code : str
            Source code to review.

        context : str
            Retrieved reference code.

        Returns
        -------
        PromptResult
            Structured result containing the final prompt.
        """

        self.validate(
            code=code,
            context=context,
        )

        prompt_parts: list[str] = []

        # ----------------------------------------------------
        # SYSTEM PROMPT
        # ----------------------------------------------------

        prompt_parts.append(
            "# SYSTEM\n\n"
            + SYSTEM_PROMPT
        )

        # ----------------------------------------------------
        # RETRIEVED CONTEXT
        # ----------------------------------------------------

        if self.include_context:
            prompt_parts.append(
                "# RETRIEVED CONTEXT\n\n"
                + self.build_context_block(context)
            )

        # ----------------------------------------------------
        # SOURCE CODE
        # ----------------------------------------------------

        prompt_parts.append(
            "# SOURCE CODE\n\n"
            + self.build_code_block(code)
        )

        # ----------------------------------------------------
        # REVIEW GUIDELINES
        # ----------------------------------------------------

        prompt_parts.append(
            "# REVIEW GUIDELINES\n\n"
            + REVIEW_GUIDELINES
        )

        # ----------------------------------------------------
        # OUTPUT FORMAT
        # ----------------------------------------------------

        prompt_parts.append(
            "# OUTPUT FORMAT\n\n"
            + OUTPUT_FORMAT
        )

        # ----------------------------------------------------
        # FINAL PROMPT
        # ----------------------------------------------------

        prompt = "\n\n".join(prompt_parts)

        logger.info(
            "Prompt built successfully (%d characters).",
            len(prompt),
        )

        return PromptResult(
            prompt=prompt,
            total_characters=len(prompt),
            context_characters=len(context),
            code_characters=len(code),
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    def statistics(
        self,
        code: str,
        context: str = "",
    ) -> dict[str, int]:
        """
        Return prompt statistics.
        """

        result = self.build(
            code=code,
            context=context,
        )

        return {
            "total_characters": result.total_characters,
            "context_characters": result.context_characters,
            "code_characters": result.code_characters,
            "system_prompt_characters": len(SYSTEM_PROMPT),
            "review_guidelines_characters": len(
                REVIEW_GUIDELINES
            ),
            "output_format_characters": len(
                OUTPUT_FORMAT
            ),
        }

    # ========================================================
    # INFORMATION
    # ========================================================

    def info(self) -> dict[str, bool]:
        """
        Return PromptBuilder configuration.
        """

        return {
            "include_context": self.include_context,
        }

    # ========================================================
    # STRING REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        """
        Developer representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"include_context={self.include_context})"
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def build_prompt(
    code: str,
    context: str = "",
    include_context: bool = True,
) -> PromptResult:
    """
    Build a prompt using the default PromptBuilder.
    """

    builder = PromptBuilder(
        include_context=include_context,
    )

    return builder.build(
        code=code,
        context=context,
    )


# ============================================================
# CLI TEST
# ============================================================

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

    sample_context = """
### Retrieved Snippet 1

Similarity: 96.42%

Function: factorial

```python
def factorial(n):
    if n == 0:
        return 1

    return n * factorial(n - 1)

""".strip()

    sample_code = """

def factorial(n):

if n == 0:
    return 1

return n * factorial(n - 1)

""".strip()

    print("=" * 70)
    print("PROMPT BUILDER TEST")
    print("=" * 70)

    result = build_prompt(
        code=sample_code,
        context=sample_context,
        include_context=True,
    )

    print()
    print("Prompt generated successfully.")
    print("Total characters:", result.total_characters)
    print("Context characters:", result.context_characters)
    print("Code characters:", result.code_characters)

    print()
    print("=" * 70)
    print("PROMPT")
    print("=" * 70)

    print(result.prompt)

    print()
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)

    builder = PromptBuilder(
        include_context=True
    )

    print(
        builder.statistics(
            code=sample_code,
            context=sample_context,
        )
    )

    print()
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    print(builder.info())