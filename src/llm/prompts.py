"""
prompts.py
==========

Enterprise prompt templates for the AI Code Reviewer.

Responsibilities
----------------
* Define system instructions.
* Define review guidelines.
* Build consistent prompts.
* Support multiple programming languages.
* Produce deterministic prompts for LLM inference.
"""

from __future__ import annotations

###############################################################################
# Standard Library Imports
###############################################################################

from dataclasses import dataclass
from textwrap import dedent

###############################################################################
# System Prompt
###############################################################################

SYSTEM_PROMPT = dedent(
    """
    You are an expert Senior Software Engineer and Code Reviewer.

    Your responsibilities include:

    • Reviewing code for correctness.
    • Detecting bugs and logical issues.
    • Identifying security vulnerabilities.
    • Finding performance bottlenecks.
    • Suggesting best coding practices.
    • Evaluating readability and maintainability.
    • Following clean architecture principles.
    • Following SOLID principles.
    • Following language-specific conventions.

    Your review must always be:

    - Professional
    - Accurate
    - Constructive
    - Concise
    - Actionable

    Never invent issues.

    Only report issues that actually exist.

    If no issues exist,
    clearly mention that the code looks good.

    Do not rewrite the entire program unless requested.

    Prefer minimal improvements over unnecessary changes.
    """
)

###############################################################################
# Review Guidelines
###############################################################################

REVIEW_GUIDELINES = dedent(
    """
    Review the submitted code using the following checklist.

    1. Correctness
       - Logic errors
       - Runtime errors
       - Missing edge cases

    2. Readability
       - Naming conventions
       - Code clarity
       - Documentation

    3. Maintainability
       - Modular design
       - Code duplication
       - Separation of concerns

    4. Performance
       - Inefficient loops
       - Expensive operations
       - Memory usage

    5. Security
       - Injection attacks
       - Hardcoded secrets
       - Unsafe operations
       - Input validation

    6. Best Practices
       - Language conventions
       - Error handling
       - Logging
       - Testing considerations

    7. Suggestions
       - Refactoring ideas
       - Simpler implementation
       - Better APIs
    """
)

###############################################################################
# Output Format
###############################################################################

OUTPUT_FORMAT = dedent(
    """
    Return the review using the following structure.

    ## Overall Summary

    Brief summary.

    ## Strengths

    - Item

    ## Issues

    - Severity
    - Description
    - Recommendation

    ## Security

    - Findings

    ## Performance

    - Findings

    ## Maintainability

    - Findings

    ## Suggested Improvements

    - Improvement 1
    - Improvement 2

    ## Final Verdict

    One concise paragraph.
    """
)

###############################################################################
# Prompt Result
###############################################################################


@dataclass(slots=True)
class PromptResult:
    """
    Stores the generated prompt.
    """

    prompt: str

    language: str

    context_length: int

    code_length: int

    def __len__(
        self,
    ) -> int:

        return len(self.prompt)

    def preview(
        self,
        characters: int = 500,
    ) -> str:

        return self.prompt[:characters]

###############################################################################
# Prompt Builder
###############################################################################


class CodeReviewPromptBuilder:
    """
    Builds production-ready prompts for LLM inference.
    """

    ###########################################################################

    def build_context_block(
        self,
        context: str,
    ) -> str:
        """
        Format retrieved knowledge.
        """

        if not context.strip():

            return "No external context provided."

        return dedent(
            f"""
            ## Retrieved Context

            {context}
            """
        ).strip()

    ###########################################################################

    def build_code_block(
        self,
        code: str,
        language: str,
    ) -> str:
        """
        Format source code.
        """

        return dedent(
            f"""
            ## Source Code

            ```{language}
            {code}
            ```
            """
        ).strip()

    ###########################################################################

    def build(
        self,
        code: str,
        language: str,
        context: str = "",
    ) -> PromptResult:
        """
        Build the final prompt.
        """

        if not code.strip():

            raise ValueError(
                "Code cannot be empty."
            )

        context_block = self.build_context_block(
            context
        )

        code_block = self.build_code_block(
            code,
            language,
        )
        prompt = "\n\n".join(
            [
                SYSTEM_PROMPT.strip(),
                REVIEW_GUIDELINES.strip(),
                context_block,
                code_block,
                OUTPUT_FORMAT.strip(),
            ]
        )

        return PromptResult(
            prompt=prompt,
            language=language,
            context_length=len(context),
            code_length=len(code),
        )

    ###########################################################################

    def statistics(
        self,
        result: PromptResult,
    ) -> dict[str, int]:
        """
        Return prompt statistics.
        """

        return {
            "prompt_length": len(result.prompt),
            "code_length": result.code_length,
            "context_length": result.context_length,
        }

    ###########################################################################

    def info(
        self,
    ) -> dict[str, str]:
        """
        Return builder information.
        """

        return {
            "builder": self.__class__.__name__,
            "purpose": "Enterprise Code Review Prompt Builder",
        }

    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}("
            "system_prompt=True, "
            "review_guidelines=True, "
            "output_format=True)"
        )


###############################################################################
# Convenience Function
###############################################################################


def build_review_prompt(
    code: str,
    language: str,
    context: str = "",
) -> PromptResult:
    """
    Build a review prompt using the default prompt builder.

    Parameters
    ----------
    code : str
        Source code to review.

    language : str
        Programming language.

    context : str, optional
        Retrieved documentation or coding guidelines.

    Returns
    -------
    PromptResult
    """

    builder = CodeReviewPromptBuilder()

    return builder.build(
        code=code,
        language=language,
        context=context,
    )


###############################################################################
# CLI Example
###############################################################################

if __name__ == "__main__":

    sample_code = """
def divide(a, b):
    return a / b
"""

    sample_context = """
Always validate user inputs.
Handle exceptions gracefully.
Avoid division-by-zero errors.
"""

    builder = CodeReviewPromptBuilder()

    result = builder.build(
        code=sample_code,
        language="python",
        context=sample_context,
    )

    print("=" * 80)
    print("Prompt Statistics")
    print("=" * 80)
    print(builder.statistics(result))

    print("\nPrompt Preview")
    print("=" * 80)
    print(result.preview(1000))


###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "SYSTEM_PROMPT",
    "REVIEW_GUIDELINES",
    "OUTPUT_FORMAT",
    "PromptResult",
    "CodeReviewPromptBuilder",
    "build_review_prompt",
]