"""Phase 1 retrieval spike.

Compares the baseline retriever against the improved one on the failure
recorded in ``docs/baseline.md``::

    Query:     "Who owns the work produced by the contractor?"
    Baseline:  Independent Contractor Status  (0.5407)
    Expected:  Intellectual Property

Two changes are made together in the improved configuration — title-augmented
chunks and an asymmetric retrieval model — so the script also runs a 2x2
ablation to show which one is doing the work. A combined result that "fixes"
the query tells you nothing about why.

    python scripts/retrieval_spike.py

Downloads two Sentence-Transformers models on first run (roughly 500 MB
combined) and then runs on CPU.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS  # noqa: E402
from src.document_processing import load_and_segment  # noqa: E402
from src.retrieval import (  # noqa: E402
    BASELINE_MODEL,
    IMPROVED_MODEL,
    ClauseRetriever,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

FAILING_QUERY = "Who owns the work produced by the contractor?"
EXPECTED_TITLE = "Intellectual Property"

# Additional probes. These are sanity checks for the spike, not the
# evaluation set -- that gets built properly in Phase 2.
PROBES = [
    ("Who owns the work produced by the contractor?", "Intellectual Property"),
    ("What happens if the contractor wants to end the agreement?", "Term and Termination"),
    ("What information must the contractor keep secret?", "Confidentiality"),
    ("How much and when will the contractor be paid?", "Payment Terms"),
    ("Which state's law applies to this agreement?", "Miscellaneous"),
]


def report(name: str, retriever: ClauseRetriever, query: str, expected: str) -> bool:
    results = retriever.search(query, top_k=3)
    hit = results[0].title == expected
    print(f"\n{name}")
    print(f"  model            : {retriever.model_name}")
    print(f"  title-augmented  : {retriever.title_augmented}")
    for result in results:
        marker = " <-- expected" if result.title == expected else ""
        print(f"  {result}{marker}")
    print(f"  top-1 correct    : {'YES' if hit else 'NO'}")
    return hit


def main() -> int:
    document, clauses = load_and_segment(PATHS.sample_pdf)
    print("=" * 68)
    print("PHASE 1 RETRIEVAL SPIKE")
    print("=" * 68)
    print(f"document : {document.path.name}")
    print(f"clauses  : {len(clauses)}")
    print(f"query    : {FAILING_QUERY!r}")
    print(f"expected : {EXPECTED_TITLE!r}")

    # --- 2x2 ablation: which change actually fixes it? -------------------
    print("\n" + "-" * 68)
    print("ABLATION")
    print("-" * 68)

    configurations = [
        ("A. BASELINE (MiniLM, body only)", BASELINE_MODEL, False),
        ("B. MiniLM + title-augmented chunks", BASELINE_MODEL, True),
        ("C. mpnet-QA, body only", IMPROVED_MODEL, False),
        ("D. IMPROVED (mpnet-QA + titles)", IMPROVED_MODEL, True),
    ]

    outcomes: dict[str, bool] = {}
    retrievers: dict[str, ClauseRetriever] = {}
    for label, model_name, augmented in configurations:
        retriever = ClauseRetriever(
            clauses, model_name=model_name, title_augmented=augmented
        )
        retrievers[label] = retriever
        outcomes[label] = report(label, retriever, FAILING_QUERY, EXPECTED_TITLE)

    # --- broader probe set on baseline vs improved -----------------------
    print("\n" + "-" * 68)
    print("PROBE SET — top-1 accuracy (sanity check, not the Phase 2 eval set)")
    print("-" * 68)
    baseline = retrievers["A. BASELINE (MiniLM, body only)"]
    improved = retrievers["D. IMPROVED (mpnet-QA + titles)"]

    base_hits = improved_hits = 0
    for query, expected in PROBES:
        base_top = baseline.search(query, top_k=1)[0]
        imp_top = improved.search(query, top_k=1)[0]
        base_ok = base_top.title == expected
        imp_ok = imp_top.title == expected
        base_hits += base_ok
        improved_hits += imp_ok
        print(f"\n  Q: {query}")
        print(f"     expected : {expected}")
        print(f"     baseline : {base_top.title:<32} {'OK' if base_ok else 'MISS'}")
        print(f"     improved : {imp_top.title:<32} {'OK' if imp_ok else 'MISS'}")

    total = len(PROBES)
    print("\n" + "-" * 68)
    print("SUMMARY")
    print("-" * 68)
    for label, hit in outcomes.items():
        print(f"  {label:<40} {'PASS' if hit else 'FAIL'}")
    print(f"\n  probe top-1  baseline : {base_hits}/{total}")
    print(f"  probe top-1  improved : {improved_hits}/{total}")

    regression_passed = outcomes["D. IMPROVED (mpnet-QA + titles)"]
    print(
        f"\n  REGRESSION TEST ({EXPECTED_TITLE}): "
        f"{'PASS' if regression_passed else 'FAIL'}"
    )
    return 0 if regression_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
