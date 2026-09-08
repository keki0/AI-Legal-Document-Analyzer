"""Phase 6: evaluate retrieval on the existing 29-query set.

    python scripts/evaluate_retrieval.py

Downloads two Sentence-Transformers models on first run, then runs on CPU.

WHAT COUNTS AS CORRECT
----------------------
``data/eval/retrieval_queries.json`` pairs each query with an ``expected``
clause title. A retrieval is correct when the retrieved clause's title matches
that string exactly. Titles are produced by ``src.document_processing`` and a
Phase 2 test already asserts every ``expected`` value corresponds to a clause
the segmenter actually emits, so the ground truth cannot silently drift.

The labels are hand-written by the project authors, not expert annotations.
They encode which clause answers each question; they are not a benchmark.

The flag-rule diagnostic at the end measures the Phase 6 context-selection
rule rather than assuming it helps. Note that the rule does not reorder
retrieval -- it can only add clauses to a context -- so it is reported
separately and never folded into Hit@k.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS, set_seed  # noqa: E402
from src.document_processing import load_and_segment  # noqa: E402
from src.rag import RAGContextBuilder  # noqa: E402
from src.retrieval import build_baseline, build_improved  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

RESULTS = PATHS.root / "evaluation" / "results"
QUERIES = PATHS.data_eval / "retrieval_queries.json"
MAX_K = 5


def load_queries() -> list[dict]:
    if not QUERIES.exists():
        raise SystemExit(f"Missing {QUERIES}. It is created in Phase 1.")
    return json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]


def score_retriever(retriever, queries: list[dict]) -> dict:
    """Hit@1/3/5 and MRR over one document's queries."""
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_total = 0.0
    per_query: list[dict] = []

    for item in queries:
        results = retriever.search(item["query"], top_k=MAX_K)
        titles = [r.title for r in results]
        expected = item["expected"]

        position = titles.index(expected) + 1 if expected in titles else None
        for k in hits:
            if position is not None and position <= k:
                hits[k] += 1
        reciprocal_total += 1.0 / position if position else 0.0

        per_query.append({
            "query": item["query"],
            "expected": expected,
            "type": item["type"],
            "retrieved_top1": titles[0] if titles else None,
            "rank_of_expected": position,
            "top1_correct": position == 1,
            "top1_score": round(results[0].score, 4) if results else None,
        })

    n = len(queries)
    return {
        "n_queries": n,
        "hit_at_1": hits[1] / n if n else 0.0,
        "hit_at_3": hits[3] / n if n else 0.0,
        "hit_at_5": hits[5] / n if n else 0.0,
        "mrr": reciprocal_total / n if n else 0.0,
        "per_query": per_query,
    }


def merge(parts: list[dict]) -> dict:
    """Combine per-document results into a corpus-level score."""
    per_query = [q for part in parts for q in part["per_query"]]
    n = len(per_query)
    if not n:
        return {"n_queries": 0}
    return {
        "n_queries": n,
        "hit_at_1": sum(q["rank_of_expected"] == 1 for q in per_query) / n,
        "hit_at_3": sum(
            q["rank_of_expected"] is not None and q["rank_of_expected"] <= 3
            for q in per_query) / n,
        "hit_at_5": sum(q["rank_of_expected"] is not None for q in per_query) / n,
        "mrr": sum(1.0 / q["rank_of_expected"] if q["rank_of_expected"] else 0.0
                   for q in per_query) / n,
        "per_query": per_query,
    }


def by_query_type(result: dict) -> dict:
    """Break Hit@1 down by the query categories recorded in Phase 1."""
    grouped: dict[str, list[bool]] = defaultdict(list)
    for query in result["per_query"]:
        grouped[query["type"]].append(query["rank_of_expected"] == 1)
    return {
        name: {"n": len(v), "hit_at_1": sum(v) / len(v)}
        for name, v in sorted(grouped.items())
    }


def flag_rule_diagnostic(documents: dict, queries: list[dict]) -> dict:
    """Measure the Phase 6 flag-support rule instead of assuming it helps.

    Reports how often it fires and -- the number that matters -- how often it
    recovers a gold clause that semantic top-k missed.
    """
    from src.importance import FlagDetector

    detector = FlagDetector()
    fired = 0
    recovered = 0
    added_total = 0

    for name, (clauses, retriever) in documents.items():
        # Minimal stand-ins carrying only flags; no classifier is needed to
        # test a selection rule that keys on flags alone.
        class _FlagOnly:
            def __init__(self, clause):
                self.clause_id = clause.clause_id
                self.category = None
                self.salience = None
                self.calibrated_confidence = None
                self.flags = detector.detect(clause.text)
                self.extractions = []

        analyses = [_FlagOnly(c) for c in clauses]
        builder = RAGContextBuilder(
            retriever, analyses=analyses, top_k=3, include_flagged=True
        )
        plain = RAGContextBuilder(retriever, analyses=analyses, top_k=3)

        for item in (q for q in queries if q["document"] == name):
            with_flags = builder.select(item["query"])
            without = plain.select(item["query"])
            added = [c for c in with_flags if c.selection_reason == "flag_supported"]
            if added:
                fired += 1
                added_total += len(added)
            base_titles = {c.title for c in without}
            if item["expected"] not in base_titles and any(
                c.title == item["expected"] for c in added
            ):
                recovered += 1

    return {
        "queries_where_rule_fired": fired,
        "total_clauses_added": added_total,
        "gold_clauses_recovered_from_outside_top3": recovered,
        "note": (
            "The rule adds clauses to a context; it never reorders retrieval, "
            "so it cannot change Hit@k. 'Recovered' counts cases where the "
            "expected clause was absent from semantic top-3 but admitted by "
            "the flag rule."
        ),
    }


def report(name: str, result: dict) -> None:
    print(f"\n{name}")
    print(f"  n         : {result['n_queries']}")
    print(f"  Hit@1     : {result['hit_at_1']:.4f}  "
          f"({round(result['hit_at_1'] * result['n_queries'])}/{result['n_queries']})")
    print(f"  Hit@3     : {result['hit_at_3']:.4f}")
    print(f"  Hit@5     : {result['hit_at_5']:.4f}")
    print(f"  MRR       : {result['mrr']:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="evaluate only the improved retriever")
    args = parser.parse_args()
    set_seed()

    queries = load_queries()
    names = sorted({q["document"] for q in queries})

    print("=" * 74)
    print("PHASE 6 — RETRIEVAL EVALUATION")
    print("=" * 74)
    print(f"queries   : {len(queries)}")
    print(f"documents : {len(names)}")
    print("correct   : retrieved clause title == expected title (exact match)")

    baseline_parts, improved_parts = [], []
    improved_docs = {}

    for name in names:
        path = PATHS.data_raw / name
        if not path.exists():
            print(f"\nSKIPPING {name}: not found at {path}")
            continue
        _, clauses = load_and_segment(path)
        subset = [q for q in queries if q["document"] == name]
        print(f"\n--- {name}: {len(clauses)} clauses, {len(subset)} queries ---")

        improved = build_improved(clauses)
        improved_docs[name] = (clauses, improved)
        improved_parts.append(score_retriever(improved, subset))

        if not args.skip_baseline:
            baseline_parts.append(score_retriever(build_baseline(clauses), subset))

    improved_all = merge(improved_parts)
    payload = {
        "n_queries": len(queries),
        "documents": names,
        "correctness_definition": "retrieved clause title == expected title",
        "improved_mpnet_qa": {k: v for k, v in improved_all.items() if k != "per_query"},
        "improved_by_query_type": by_query_type(improved_all),
        "per_query": improved_all["per_query"],
    }

    if baseline_parts:
        baseline_all = merge(baseline_parts)
        payload["baseline_minilm"] = {
            k: v for k, v in baseline_all.items() if k != "per_query"
        }
        report("BASELINE — all-MiniLM-L6-v2, body-only", baseline_all)
    report("IMPROVED — multi-qa-mpnet-base-dot-v1, title-augmented", improved_all)

    print("\nHit@1 BY QUERY TYPE (improved)")
    print(f"  {'type':<18}{'n':>5}{'Hit@1':>10}")
    for name, stats in payload["improved_by_query_type"].items():
        print(f"  {name:<18}{stats['n']:>5}{stats['hit_at_1']:>10.4f}")

    print("\nQUERIES WHERE THE EXPECTED CLAUSE WAS NOT RETRIEVED IN TOP-5")
    misses = [q for q in improved_all["per_query"] if q["rank_of_expected"] is None]
    if not misses:
        print("  none")
    for miss in misses:
        print(f"  {miss['query']}")
        print(f"     expected: {miss['expected']}  |  got: {miss['retrieved_top1']}")

    # The documented Phase 1 regression case.
    regression = next(
        (q for q in improved_all["per_query"] if q["type"] == "regression"), None
    )
    if regression:
        print("\nREGRESSION CASE (documented baseline failure)")
        print(f"  query    : {regression['query']}")
        print(f"  expected : {regression['expected']}")
        print(f"  top-1    : {regression['retrieved_top1']}")
        print(f"  result   : {'PASS' if regression['top1_correct'] else 'FAIL'}")
        payload["regression_case"] = regression

    if improved_docs:
        print("\nFLAG-SUPPORT RULE DIAGNOSTIC (Phase 6 secondary selection rule)")
        diagnostic = flag_rule_diagnostic(improved_docs, queries)
        for key, value in diagnostic.items():
            if key != "note":
                print(f"  {key:<44}{value}")
        print(f"  {diagnostic['note']}")
        payload["flag_rule_diagnostic"] = diagnostic

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "retrieval_metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
