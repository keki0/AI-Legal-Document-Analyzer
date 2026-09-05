"""Offline diagnostic for the documented retrieval failure.

Explains *why* the baseline confuses two clauses, using only token overlap.
Needs no model download, so it runs anywhere and is reproducible in a viva.

    python scripts/lexical_diagnostic.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS  # noqa: E402
from src.document_processing import load_and_segment  # noqa: E402

QUERY = "Who owns the work produced by the contractor?"
STOPWORDS = {
    "who", "the", "by", "a", "an", "of", "to", "and", "is", "in", "for",
    "this", "that", "or", "what", "which", "shall", "will", "any", "be",
}


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def main() -> int:
    _, clauses = load_and_segment(PATHS.sample_pdf)
    query_tokens = tokens(QUERY) - STOPWORDS

    print(f"Query: {QUERY}")
    print(f"Content tokens: {sorted(query_tokens)}\n")
    print(f"{'clause':<32}{'body overlap':<28}{'title overlap'}")
    print("-" * 92)
    for clause in clauses:
        body = sorted(query_tokens & tokens(clause.text))
        title = sorted(query_tokens & tokens(clause.title))
        print(f"{clause.title[:30]:<32}{str(body):<28}{title}")

    print("\nOccurrences of 'own*' across the document:")
    for clause in clauses:
        hits = re.findall(r"\bown\w*", clause.text.lower())
        if hits:
            print(f"  {clause.title:<32}{hits}")

    print(
        "\nDiagnosis: the query's key verb 'owns' never appears in the "
        "Intellectual Property clause, but 'own' does appear in Independent "
        "Contractor Status in an unrelated sense ('their own work schedule'). "
        "Both clauses also contain 'contractor' and 'work'. Body-only "
        "embeddings therefore have almost nothing to separate them, and the "
        "distractor carries the surface form of the query term."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
