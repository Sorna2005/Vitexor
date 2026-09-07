"""
reranker.py

Second-stage reranking for retrieved code candidates.

Uses lightweight code-aware signals:
- FAISS similarity
- programming/function name similarity
- token overlap
- structural similarity
"""

from __future__ import annotations

import re

import pandas as pd


PYTHON_KEYWORDS = {
    "def",
    "return",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "try",
    "except",
    "class",
    "import",
    "from",
    "with",
    "lambda",
    "yield",
    "raise",
}


def tokenize_code(text: str) -> set[str]:
    """
    Extract simple identifier/code tokens.
    """

    if not isinstance(text, str):
        return set()

    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*",
        text.lower(),
    )

    return set(tokens)


def token_overlap(
    query_tokens: set[str],
    candidate_tokens: set[str],
) -> float:

    if not query_tokens or not candidate_tokens:
        return 0.0

    intersection = (
        query_tokens
        & candidate_tokens
    )

    union = (
        query_tokens
        | candidate_tokens
    )

    if not union:
        return 0.0

    return len(intersection) / len(union)


def structural_similarity(
    query_code: str,
    candidate_code: str,
) -> float:

    query_tokens = tokenize_code(
        query_code
    )

    candidate_tokens = tokenize_code(
        candidate_code
    )

    query_keywords = (
        query_tokens
        & PYTHON_KEYWORDS
    )

    candidate_keywords = (
        candidate_tokens
        & PYTHON_KEYWORDS
    )

    if not query_keywords:
        return 0.0

    intersection = (
        query_keywords
        & candidate_keywords
    )

    return (
        len(intersection)
        / len(query_keywords)
    )


def rerank_results(
    query_code: str,
    results: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Rerank FAISS candidates using lightweight
    code-aware signals.
    """

    if results.empty:
        return results.copy()

    ranked = results.copy()

    query_tokens = tokenize_code(
        query_code
    )

    # --------------------------------------------------------
    # Candidate token overlap
    # --------------------------------------------------------

    overlaps = []

    structural_scores = []

    for _, row in ranked.iterrows():

        candidate_code = row.get(
            "func_code_string",
            "",
        )

        candidate_tokens = tokenize_code(
            candidate_code
        )

        overlaps.append(
            token_overlap(
                query_tokens,
                candidate_tokens,
            )
        )

        structural_scores.append(
            structural_similarity(
                query_code,
                candidate_code,
            )
        )

    ranked[
        "token_overlap"
    ] = overlaps

    ranked[
        "structural_similarity"
    ] = structural_scores

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    ranked[
        "rerank_score"
    ] = (
        0.70
        * ranked["similarity_score"].astype(
            float
        )
        + 0.20
        * ranked["token_overlap"]
        + 0.10
        * ranked["structural_similarity"]
    )

    ranked = ranked.sort_values(
        "rerank_score",
        ascending=False,
    )

    ranked = ranked.head(
        top_k
    )

    return ranked.reset_index(
        drop=True
    )