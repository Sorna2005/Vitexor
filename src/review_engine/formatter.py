"""
formatter.py

Formats the final AI Code Review response.

Responsibilities
----------------
- Convert validated review data into a readable response.
- Provide consistent sections.
- Handle missing sections safely.
- Support both dictionary-based and object-based responses.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# FORMATTER
# ============================================================

class ReviewFormatter:
    """
    Formats AI-generated code review results.
    """

    def __init__(
        self,
        include_sections: bool = True,
    ) -> None:
        """
        Initialize the formatter.

        Parameters
        ----------
        include_sections : bool
            Whether section headings should be included.
        """

        self.include_sections = include_sections

    # ========================================================
    # INTERNAL VALUE HELPER
    # ========================================================

    @staticmethod
    def _get_value(
        data: Any,
        key: str,
        default: Any = "",
    ) -> Any:
        """
        Safely retrieve a value from either a dictionary
        or an object.
        """

        if data is None:
            return default

        if isinstance(data, dict):
            return data.get(key, default)

        return getattr(data, key, default)

    # ========================================================
    # TEXT CLEANING
    # ========================================================

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:
        """
        Convert a value into clean text.
        """

        if value is None:
            return ""

        if isinstance(value, str):
            return value.strip()

        return str(value).strip()

    # ========================================================
    # SECTION BUILDER
    # ========================================================

    def _section(
        self,
        title: str,
        content: Any,
    ) -> str:
        """
        Build a formatted review section.
        """

        text = self._clean_text(content)

        if not text:
            text = "No information provided."

        if not self.include_sections:
            return text

        return (
            f"{title}\n"
            f"{'-' * len(title)}\n"
            f"{text}"
        )

    # ========================================================
    # ISSUE FORMATTER
    # ========================================================

    def format_issues(
        self,
        issues: Any,
    ) -> str:
        """
        Format detected issues.

        Supports:

        - list of dictionaries
        - list of objects
        - strings
        - single dictionary
        """

        if not issues:
            return "No issues detected."

        if isinstance(issues, dict):
            issues = [issues]

        if isinstance(issues, str):
            return issues.strip()

        formatted = []

        for index, issue in enumerate(
            issues,
            start=1,
        ):

            if isinstance(issue, str):

                formatted.append(
                    f"{index}. {issue.strip()}"
                )

                continue

            issue_name = self._clean_text(
                self._get_value(
                    issue,
                    "issue",
                    self._get_value(
                        issue,
                        "title",
                        "Issue",
                    ),
                )
            )

            severity = self._clean_text(
                self._get_value(
                    issue,
                    "severity",
                    "Unknown",
                )
            )

            explanation = self._clean_text(
                self._get_value(
                    issue,
                    "explanation",
                    self._get_value(
                        issue,
                        "description",
                        "",
                    ),
                )
            )

            recommendation = self._clean_text(
                self._get_value(
                    issue,
                    "recommendation",
                    self._get_value(
                        issue,
                        "suggestion",
                        "",
                    ),
                )
            )

            block = [
                f"{index}. {issue_name}",
                f"Severity: {severity}",
            ]

            if explanation:
                block.append(
                    f"Explanation: {explanation}"
                )

            if recommendation:
                block.append(
                    f"Recommendation: {recommendation}"
                )

            formatted.append(
                "\n".join(block)
            )

        return "\n\n".join(formatted)

    # ========================================================
    # MAIN FORMAT METHOD
    # ========================================================

    def format(
        self,
        review: Any,
    ) -> str:
        """
        Format a complete AI code review.

        Parameters
        ----------
        review : Any
            Validated review response.

        Returns
        -------
        str
            Human-readable formatted review.
        """

        summary = self._get_value(
            review,
            "summary",
            "",
        )

        issues = self._get_value(
            review,
            "issues",
            [],
        )

        performance = self._get_value(
            review,
            "performance",
            self._get_value(
                review,
                "performance_improvements",
                "",
            ),
        )

        security = self._get_value(
            review,
            "security",
            self._get_value(
                review,
                "security_review",
                "",
            ),
        )

        best_practices = self._get_value(
            review,
            "best_practices",
            "",
        )

        improved_code = self._get_value(
            review,
            "improved_code",
            "",
        )

        verdict = self._get_value(
            review,
            "verdict",
            self._get_value(
                review,
                "final_verdict",
                "",
            ),
        )

        # ----------------------------------------------------
        # Build sections
        # ----------------------------------------------------

        sections = []

        sections.append(
            self._section(
                "SUMMARY",
                summary,
            )
        )

        sections.append(
            self._section(
                "ISSUES",
                self.format_issues(issues),
            )
        )

        sections.append(
            self._section(
                "PERFORMANCE IMPROVEMENTS",
                performance,
            )
        )

        sections.append(
            self._section(
                "SECURITY REVIEW",
                security,
            )
        )

        sections.append(
            self._section(
                "BEST PRACTICES",
                best_practices,
            )
        )

        if improved_code:
            sections.append(
                self._section(
                    "IMPROVED CODE",
                    improved_code,
                )
            )

        sections.append(
            self._section(
                "FINAL VERDICT",
                verdict,
            )
        )

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        header = (
            "=" * 70
            + "\n"
            + "AI CODE REVIEW"
            + "\n"
            + "=" * 70
        )

        return (
            header
            + "\n\n"
            + "\n\n".join(sections)
        )

    # ========================================================
    # SHORT FORMAT
    # ========================================================

    def format_short(
        self,
        review: Any,
    ) -> str:
        """
        Return a shorter review response.
        """

        summary = self._clean_text(
            self._get_value(
                review,
                "summary",
                "",
            )
        )

        issues = self._get_value(
            review,
            "issues",
            [],
        )

        issue_count = (
            len(issues)
            if isinstance(issues, list)
            else 0
        )

        return (
            "AI CODE REVIEW\n"
            "===============\n\n"
            f"Summary: {summary or 'No summary provided.'}\n"
            f"Issues detected: {issue_count}"
        )

    # ========================================================
    # DICTIONARY OUTPUT
    # ========================================================

    def to_dict(
        self,
        review: Any,
    ) -> dict[str, Any]:
        """
        Convert review data into a standardized dictionary.
        """

        return {
            "summary": self._clean_text(
                self._get_value(
                    review,
                    "summary",
                    "",
                )
            ),
            "issues": self._get_value(
                review,
                "issues",
                [],
            ),
            "performance": self._clean_text(
                self._get_value(
                    review,
                    "performance",
                    "",
                )
            ),
            "security": self._clean_text(
                self._get_value(
                    review,
                    "security",
                    "",
                )
            ),
            "best_practices": self._clean_text(
                self._get_value(
                    review,
                    "best_practices",
                    "",
                )
            ),
            "improved_code": self._clean_text(
                self._get_value(
                    review,
                    "improved_code",
                    "",
                )
            ),
            "verdict": self._clean_text(
                self._get_value(
                    review,
                    "verdict",
                    self._get_value(
                        review,
                        "final_verdict",
                        "",
                    ),
                )
            ),
        }

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def info(self) -> dict[str, bool]:
        """
        Return formatter configuration.
        """

        return {
            "include_sections": self.include_sections,
        }

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        """
        Developer representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"include_sections="
            f"{self.include_sections})"
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def format_review(
    review: Any,
    include_sections: bool = True,
) -> str:
    """
    Format a review using the default formatter.
    """

    formatter = ReviewFormatter(
        include_sections=include_sections,
    )

    return formatter.format(review)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_review = {
        "summary": (
            "The function works correctly, "
            "but several improvements are recommended."
        ),

        "issues": [
            {
                "issue": "Missing input validation",
                "severity": "Medium",
                "explanation": (
                    "The function does not validate "
                    "the input before processing it."
                ),
                "recommendation": (
                    "Validate the input before "
                    "performing the operation."
                ),
            },
            {
                "issue": "Potential performance issue",
                "severity": "Low",
                "explanation": (
                    "The function performs repeated "
                    "operations inside a loop."
                ),
                "recommendation": (
                    "Consider caching repeated results."
                ),
            },
        ],

        "performance": (
            "Consider reducing repeated calculations "
            "inside loops."
        ),

        "security": (
            "No critical security vulnerabilities "
            "were identified."
        ),

        "best_practices": (
            "Use descriptive variable names and "
            "add input validation."
        ),

        "improved_code": """
def example(value):
    if value is None:
        raise ValueError("value cannot be None")

    return value
""".strip(),
    }

    print("=" * 70)
    print("FORMATTER TEST")
    print("=" * 70)

    formatter = ReviewFormatter()

    formatted_review = formatter.format(
        sample_review
    )

    print(formatted_review)

    print()
    print("=" * 70)
    print("SHORT FORMAT")
    print("=" * 70)

    print(
        formatter.format_short(
            sample_review
        )
    )

    print()
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    print(formatter.info())

    print()
    print("=" * 70)
    print("DICTIONARY")
    print("=" * 70)

    print(
        formatter.to_dict(
            sample_review
        )
    )