"""Phase 8: run the complete pipeline over both evaluation documents.

    python scripts/evaluate_pipeline.py
    python scripts/evaluate_pipeline.py --limit 2     # quick check

Requires the fine-tuned Legal-BERT in models/, plus MPNet-QA and FLAN-T5 from
Hugging Face. If the classifier is absent the pipeline runs in degraded mode
and says so in the output rather than failing.

WHAT THIS DOES AND DOES NOT MEASURE
-----------------------------------
This is an INTEGRATION check, not an accuracy benchmark. It answers: does a
PDF flow through every stage, do clause identifiers survive each hop, and does
the pipeline fail safely?

**No end-to-end accuracy metric is computed, because there is no ground truth
for it.** Component quality is measured in the phase that owns it:
classification in Phase 3, retrieval in Phase 6, generation in Phase 7. Those
results are not recomputed here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS, set_seed  # noqa: E402
from src.generation import GenerationTask  # noqa: E402
from src.pipeline import LegalDocumentPipeline  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

RESULTS = PATHS.root / "evaluation" / "results"
CONTRACT = "sample-independent-contractor-agreement.pdf"
POLICY = "synthetic-unnumbered-policy.pdf"

# Representative queries: one per topic the documents actually cover. Kept
# deliberately small -- padding this list would inflate the counts without
# testing anything new.
QUERIES: dict[str, list[dict]] = {
    CONTRACT: [
        {"query": "Who owns the work produced by the contractor?",
         "task": GenerationTask.CLAUSE_EXPLANATION, "topic": "intellectual property"},
        {"query": "How much notice is needed to end the agreement?",
         "task": GenerationTask.CLAUSE_EXPLANATION, "topic": "termination / notice"},
        {"query": "What information must the contractor keep confidential?",
         "task": GenerationTask.CLAUSE_EXPLANATION, "topic": "confidentiality"},
        {"query": "How and when is the contractor paid?",
         "task": GenerationTask.CLAUSE_EXPLANATION, "topic": "payment"},
        {"query": "Which law governs the agreement and how are disputes resolved?",
         "task": GenerationTask.CLAUSE_EXPLANATION,
         "topic": "governing law / dispute resolution"},
        {"query": "In simple terms, who pays if a third party makes a claim?",
         "task": GenerationTask.SIMPLE_EXPLANATION, "topic": "liability / indemnity"},
        {"query": "Summarize the payment and termination terms.",
         "task": GenerationTask.QUICK_SUMMARY, "topic": "summary"},
    ],
    POLICY: [
        {"query": "Who else can see my personal data?",
         "task": GenerationTask.CLAUSE_EXPLANATION, "topic": "data sharing"},
        {"query": "Does my subscription renew automatically?",
         "task": GenerationTask.CLAUSE_EXPLANATION, "topic": "automatic renewal"},
        {"query": "How are disputes resolved under this policy?",
         "task": GenerationTask.CLAUSE_EXPLANATION, "topic": "dispute resolution"},
        {"query": "In simple terms, how long is my data kept?",
         "task": GenerationTask.SIMPLE_EXPLANATION, "topic": "data retention"},
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="max queries per document")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    set_seed()

    started = time.perf_counter()
    print("=" * 74)
    print("PHASE 8 — END-TO-END PIPELINE EVALUATION")
    print("=" * 74)

    pipeline = LegalDocumentPipeline.from_defaults(top_k=args.top_k)
    if pipeline.classifier is None:
        print("WARNING: no classifier found; running in DEGRADED mode.")
        print(f"         expected at {PATHS.classifier_dir}")
    if pipeline.scaler is None:
        print("WARNING: no temperature.json; confidence will be uncalibrated.")

    documents: list[dict] = []
    all_results: list[dict] = []
    failures: list[dict] = []

    for name, queries in QUERIES.items():
        path = PATHS.data_raw / name
        if not path.exists():
            print(f"\nSKIPPING {name}: not found")
            failures.append({"document": name, "error": "file not found"})
            continue

        if args.limit:
            queries = queries[: args.limit]

        print(f"\n{'-' * 74}\n{name}\n{'-' * 74}")
        analysis = pipeline.analyze_document(path)
        summary = analysis.summary()
        print(f"pages {summary['pages']} | clauses {summary['clauses']} | "
              f"classified {summary['classified']} | flagged {summary['with_flags']} | "
              f"extractions {summary['extracted_items']}")
        if analysis.stage_errors:
            print(f"stage errors: {analysis.stage_errors}")
            failures.extend(
                {"document": name, "error": e} for e in analysis.stage_errors
            )

        print("\ntop clauses by salience:")
        for item in analysis.top_salient(3):
            flags = ", ".join(item.flag_labels) or "-"
            print(f"  {item.salience:.3f}  {item.title[:34]:<36}{item.category[:26]:<28}{flags}")

        summary["clauses_detail"] = [
            {
                "clause_id": a.clause_id, "title": a.title, "page": a.page,
                "category": a.category,
                "calibrated_confidence": round(a.calibrated_confidence, 4),
                "salience": round(a.salience, 4),
                "flags": a.flag_labels,
                "n_extractions": len(a.extractions),
            }
            for a in analysis.analyses
        ]
        documents.append(summary)

        print("\nqueries:")
        for item in queries:
            result = pipeline.ask(analysis, item["query"], task=item["task"])
            record = result.to_dict()
            record["topic"] = item["topic"]
            all_results.append(record)

            if result.error:
                print(f"  ERROR  {item['query'][:50]} -> {result.error}")
                failures.append({"document": name, "query": item["query"],
                                 "error": result.error})
                continue

            status = "grounded" if result.is_grounded else "EMPTY"
            print(f"  [{status}] {item['query'][:52]}")
            print(f"      retrieved: {result.retrieved_titles}")
            print(f"      -> {(result.response.text if result.response else '')[:100]}")

    elapsed = time.perf_counter() - started

    grounded = sum(1 for r in all_results if r["is_grounded"])
    empty = sum(1 for r in all_results if r["empty_context"])
    generated = sum(1 for r in all_results if r["generated_text"])
    retrieved_ok = sum(1 for r in all_results if r["retrieved_clause_ids"])

    summary_payload = {
        "documents_processed": len(documents),
        "total_clauses": sum(d["clauses"] for d in documents),
        "clauses_classified": sum(d["classified"] for d in documents),
        "clauses_with_flags": sum(d["with_flags"] for d in documents),
        "clauses_with_extractions": sum(d["with_extractions"] for d in documents),
        "total_extracted_items": sum(d["extracted_items"] for d in documents),
        "queries_evaluated": len(all_results),
        "successful_retrievals": retrieved_ok,
        "generated_responses": generated,
        "grounded_responses": grounded,
        "empty_context_responses": empty,
        "pipeline_failures": len(failures),
        "degraded_mode": any(d["degraded"] for d in documents),
        "total_runtime_seconds": round(elapsed, 1),
        "note": (
            "Integration counts only. No end-to-end accuracy metric is "
            "computed because no ground truth exists for the combined "
            "pipeline. Component quality is measured in Phases 3, 6 and 7."
        ),
    }

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for key, value in summary_payload.items():
        if key != "note":
            print(f"  {key:<32}{value}")
    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"  {failure}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "final_pipeline_results.json").write_text(
        json.dumps({"documents": documents, "queries": all_results,
                    "failures": failures}, indent=2),
        encoding="utf-8",
    )
    (RESULTS / "final_pipeline_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    print(f"\nsaved -> {RESULTS / 'final_pipeline_results.json'}")
    print(f"saved -> {RESULTS / 'final_pipeline_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
