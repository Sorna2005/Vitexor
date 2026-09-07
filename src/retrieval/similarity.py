"""
similarity.py
=============

Enterprise formatter for displaying FAISS similarity search results.

Features
--------
* Pretty console output
* Markdown output
* HTML output
* JSON export
* Summary statistics
* Configurable preview length
* Result validation
* Enterprise logging
"""

from __future__ import annotations

###############################################################################
# Standard Library Imports
###############################################################################

import json
import logging
from pathlib import Path
from typing import Any

###############################################################################
# Third Party Imports
###############################################################################

import pandas as pd

###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)

###############################################################################
# Constants
###############################################################################

DEFAULT_PREVIEW_LENGTH = 300

SEPARATOR = "=" * 90

SUB_SEPARATOR = "-" * 90

###############################################################################
# Validation
###############################################################################


def validate_results(
    results: pd.DataFrame,
) -> None:
    """
    Validate search results.

    Parameters
    ----------
    results : pd.DataFrame

    Raises
    ------
    TypeError
    """

    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "results must be a pandas DataFrame."
        )


###############################################################################
# Helper Functions
###############################################################################


def truncate(
    text: str,
    max_length: int = DEFAULT_PREVIEW_LENGTH,
) -> str:
    """
    Truncate long source code.

    Parameters
    ----------
    text : str

    max_length : int

    Returns
    -------
    str
    """

    text = str(text).strip()

    if len(text) <= max_length:
        return text

    return (
        text[:max_length].rstrip()
        + "\n..."
    )


###############################################################################
# Statistics
###############################################################################


def summary_statistics(
    results: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute retrieval statistics.

    Returns
    -------
    dict
    """

    validate_results(results)

    if results.empty:

        return {
            "matches": 0,
            "average_similarity": 0.0,
            "best_similarity": 0.0,
        }

    similarities = (
        results["similarity_score"]
        .astype(float)
    )

    return {
        "matches": len(results),
        "average_similarity": similarities.mean(),
        "best_similarity": similarities.max(),
    }


###############################################################################
# Formatter
###############################################################################


def format_results(
    results: pd.DataFrame,
    max_preview_length: int = DEFAULT_PREVIEW_LENGTH,
) -> str:
    """
    Format retrieval results for console output.

    Parameters
    ----------
    results : pd.DataFrame

    max_preview_length : int

    Returns
    -------
    str
    """

    validate_results(results)

    if results.empty:

        return "\nNo similar functions found."

    stats = summary_statistics(
        results
    )

    output: list[str] = []

    output.append(SEPARATOR)
    output.append(
        "               TOP SIMILAR CODE SNIPPETS"
    )
    output.append(SEPARATOR)

    output.append(
        f"Matches            : {stats['matches']}"
    )

    output.append(
        f"Average Similarity : "
        f"{stats['average_similarity'] * 100:.2f}%"
    )

    output.append(
        f"Best Similarity    : "
        f"{stats['best_similarity'] * 100:.2f}%"
    )

    output.append(SEPARATOR)

    for rank, (_, row) in enumerate(
        results.iterrows(),
        start=1,
    ):

        similarity = (
            float(
                row.get(
                    "similarity_score",
                    0.0,
                )
            )
            * 100
        )

        function_name = row.get(
            "func_name",
            "Unknown",
        )

        source = row.get(
            "project_name",
            row.get(
                "repository",
                row.get(
                    "file_name",
                    "Unknown",
                ),
            ),
        )

        code = row.get(
            "func_code_string",
            "",
        )

        output.append("")
        output.append(f"Rank #{rank}")
        output.append(SUB_SEPARATOR)

        output.append(
            f"Similarity : {similarity:.2f}%"
        )

        output.append(
            f"Function   : {function_name}"
        )

        output.append(
            f"Source     : {source}"
        )

        output.append("")
        output.append("Code Preview")
        output.append(SUB_SEPARATOR)

        output.append(
            truncate(
                code,
                max_preview_length,
            )
        )

        output.append(SEPARATOR)

    return "\n".join(output)


###############################################################################
# Markdown Formatter
###############################################################################


def to_markdown(
    results: pd.DataFrame,
) -> str:
    """
    Convert search results to Markdown.

    Useful for Streamlit and GitHub reports.
    """

    validate_results(results)

    if results.empty:

        return "No similar code snippets found."

    markdown = []

    markdown.append("# Similar Code Snippets\n")

    for index, (_, row) in enumerate(
        results.iterrows(),
        start=1,
    ):

        markdown.append(
            f"## {index}. {row.get('func_name', 'Unknown')}"
        )

        markdown.append(
            f"**Similarity:** "
            f"{row['similarity_score'] * 100:.2f}%"
        )

        markdown.append(
            f"**Source:** "
            f"{row.get('project_name', 'Unknown')}"
        )

        markdown.append("```python")

        markdown.append(
            row.get(
                "func_code_string",
                "",
            )
        )

        markdown.append("```")

    return "\n".join(markdown)
###############################################################################
# HTML Formatter
###############################################################################


def to_html(
    results: pd.DataFrame,
) -> str:
    """
    Convert search results to an HTML table.

    Parameters
    ----------
    results : pd.DataFrame

    Returns
    -------
    str
        HTML representation of the search results.
    """

    validate_results(results)

    if results.empty:
        return "<p>No similar functions found.</p>"

    html: list[str] = []

    html.append("<h2>Similar Code Snippets</h2>")
    html.append("<table border='1' cellspacing='0' cellpadding='6'>")

    html.append(
        "<tr>"
        "<th>Rank</th>"
        "<th>Similarity</th>"
        "<th>Function</th>"
        "<th>Source</th>"
        "<th>Code</th>"
        "</tr>"
    )

    for rank, (_, row) in enumerate(
        results.iterrows(),
        start=1,
    ):

        html.append("<tr>")

        html.append(f"<td>{rank}</td>")

        html.append(
            f"<td>{row['similarity_score'] * 100:.2f}%</td>"
        )

        html.append(
            f"<td>{row.get('func_name', 'Unknown')}</td>"
        )

        html.append(
            f"<td>{row.get('project_name', 'Unknown')}</td>"
        )

        html.append(
            "<td><pre>"
            f"{truncate(row.get('func_code_string', ''))}"
            "</pre></td>"
        )

        html.append("</tr>")

    html.append("</table>")

    return "\n".join(html)


###############################################################################
# JSON Formatter
###############################################################################


def to_json(
    results: pd.DataFrame,
    indent: int = 4,
) -> str:
    """
    Convert results to JSON.

    Parameters
    ----------
    results : pd.DataFrame

    indent : int

    Returns
    -------
    str
    """

    validate_results(results)

    return json.dumps(
        results.to_dict(
            orient="records",
        ),
        indent=indent,
        default=str,
    )


###############################################################################
# Save Utilities
###############################################################################


def save_text(
    results: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save formatted text report.
    """

    output_path = Path(output_path)

    output_path.write_text(
        format_results(results),
        encoding="utf-8",
    )

    logger.info(
        "Text report saved to '%s'",
        output_path,
    )


def save_markdown(
    results: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save Markdown report.
    """

    output_path = Path(output_path)

    output_path.write_text(
        to_markdown(results),
        encoding="utf-8",
    )

    logger.info(
        "Markdown report saved to '%s'",
        output_path,
    )


def save_html(
    results: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save HTML report.
    """

    output_path = Path(output_path)

    output_path.write_text(
        to_html(results),
        encoding="utf-8",
    )

    logger.info(
        "HTML report saved to '%s'",
        output_path,
    )


def save_json(
    results: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save JSON report.
    """

    output_path = Path(output_path)

    output_path.write_text(
        to_json(results),
        encoding="utf-8",
    )

    logger.info(
        "JSON report saved to '%s'",
        output_path,
    )


###############################################################################
# Printing
###############################################################################


def print_results(
    results: pd.DataFrame,
    max_preview_length: int = DEFAULT_PREVIEW_LENGTH,
) -> None:
    """
    Print formatted similarity results.
    """

    try:

        print(
            format_results(
                results,
                max_preview_length=max_preview_length,
            )
        )

    except Exception:

        logger.exception(
            "Unable to display search results."
        )

        raise


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
                "add",
                "multiply",
            ],
            "project_name": [
                "demo",
                "demo",
            ],
            "similarity_score": [
                0.981,
                0.942,
            ],
            "func_code_string": [
                "def add(a, b):\n    return a + b",
                "def multiply(a, b):\n    return a * b",
            ],
        }
    )

    print_results(sample)

    print("\nMarkdown Preview\n")
    print(to_markdown(sample))

    print("\nJSON Preview\n")
    print(to_json(sample))


###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "DEFAULT_PREVIEW_LENGTH",
    "validate_results",
    "truncate",
    "summary_statistics",
    "format_results",
    "to_markdown",
    "to_html",
    "to_json",
    "save_text",
    "save_markdown",
    "save_html",
    "save_json",
    "print_results",
]