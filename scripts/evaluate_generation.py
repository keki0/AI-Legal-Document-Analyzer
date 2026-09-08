"""Phase 7: evaluate grounded generation on the hand-written query set.

    python scripts/evaluate_generation.py
    python scripts/evaluate_generation.py --limit 4      # quick smoke run
    python scripts/evaluate_generation.py --no-bertscore

Downloads FLAN-T5 and the MPNet-QA retriever on first run. CPU inference with
beam search is slow; use --limit first to confirm the pipeline works before
committing to the full set.

WHAT IS AND IS NOT MEASURED
---------------------------
ROUGE and BERTScore compare generated text against 23 author-written reference
answers. They measure similarity to those references. They do not measure
legal correctness, and the set is far too small to support a claim beyond an
indicative comparison.

The grounding diagnostic is surface-level: key-term coverage, and numeric
tokens appearing in the output but not in the context. A fluent fabrication
containing no numbers would pass it. It is a diagnostic, not a hallucination
detector.

CONTEXT COMPARISON
------------------
Each example is generated twice: once from the full Phase 6 RAG context
(top-k clauses with titles, categories and citations) and once from a reduced
baseline context (top-1 clause text only, no metadata). Same model, same
decoding, same prompt template -- only the context differs. This isolates
whether the richer grounded context changes output quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS, set_seed  # noqa: E402
from src.document_processing import load_and_segment  # noqa: E402
from src.generation import (  # noqa: E402
    GenerationPipeline,
    GenerationSettings,
    GenerationTask,
    check_grounding,
)
from src.rag import RAGContext, RAGContextBuilder, RetrievedClause  # noqa: E402
from src.retrieval import build_improved  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

RESULTS = PATHS.root / "evaluation" / "results"
QUERIES = PATHS.data_eval / "generation_queries.json"


def load_queries() -> list[dict]:
    if not QUERIES.exists():
        raise SystemExit(f"Missing {QUERIES}.")
    return json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]


def top1_context(context: RAGContext) -> RAGContext:
    """Reduced baseline: top-1 clause text only, no titles or metadata.

    Same retrieval, so the comparison isolates context richness rather than
    retrieval quality.
    """
    if context.is_empty:
        return context
    first = context.retrieved[0]
    stripped = RetrievedClause(
        rank=first.rank, score=first.score, clause_id=first.clause_id,
        title=first.title, text=first.text, page=first.page,
        original_char_count=first.original_char_count,
    )
    return RAGContext(
        query=context.query,
        retrieved=[stripped],
        formatted_context=first.text,   # bare text, no citation header
        char_count=len(first.text),
        token_estimate=context.token_estimate,
        budget_chars=context.budget_chars,
        within_budget=True,
    )


def rouge_scores(predictions: list[str], references: list[str]) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    totals = defaultdict(float)
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        for name, score in scores.items():
            totals[name] += score.fmeasure
    n = max(len(predictions), 1)
    return {name: round(total / n, 4) for name, total in totals.items()}


def bert_scores(predictions: list[str], references: list[str]) -> dict | None:
    try:
        from bert_score import score
    except ImportError:
        logging.warning("bert_score not installed; skipping.")
        return None
    precision, recall, f1 = score(
        predictions, references, lang="en", rescale_with_baseline=False, verbose=False
    )
    return {
        "precision": round(float(precision.mean()), 4),
        "recall": round(float(recall.mean()), 4),
        "f1": round(float(f1.mean()), 4),
    }


def summarise(records: list[dict], key: str) -> dict:
    """Aggregate one context variant's records."""
    predictions = [r[key]["text"] for r in records]
    references = [r["reference"] for r in records]
    coverage = [r[key]["key_point_coverage"] for r in records]
    unsupported = sum(len(r[key]["unsupported_numbers"]) for r in records)
    with_unsupported = sum(1 for r in records if r[key]["unsupported_numbers"])
    correct_sources = sum(1 for r in records if r[key]["source_titles_correct"])

    return {
        "n": len(records),
        "rouge": rouge_scores(predictions, references),
        "key_point_coverage": round(sum(coverage) / max(len(coverage), 1), 4),
        "examples_with_unsupported_numbers": with_unsupported,
        "total_unsupported_numeric_tokens": unsupported,
        "examples_citing_all_expected_sources": correct_sources,
        "mean_output_chars": round(
            sum(len(p) for p in predictions) / max(len(predictions), 1), 1
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/flan-t5-base")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip the top-1 context comparison")
    args = parser.parse_args()
    set_seed()

    queries = load_queries()
    if args.limit:
        queries = queries[: args.limit]

    settings = GenerationSettings(model_name=args.model)
    pipeline = GenerationPipeline(settings)

    print("=" * 74)
    print("PHASE 7 — GROUNDED GENERATION EVALUATION")
    print("=" * 74)
    print(f"model    : {args.model}")
    print(f"device   : {pipeline.device}")
    print(f"examples : {len(queries)}")
    print(f"decoding : beams={settings.num_beams} sample={settings.do_sample}")

    # Build one retriever per document; retrieval is Phase 6's job.
    documents: dict[str, tuple] = {}
    for name in sorted({q["document"] for q in queries}):
        path = PATHS.data_raw / name
        if not path.exists():
            raise SystemExit(f"Missing evaluation document: {path}")
        _, clauses = load_and_segment(path)
        documents[name] = (clauses, build_improved(clauses))
        print(f"  {name}: {len(clauses)} clauses")

    records: list[dict] = []
    start = time.perf_counter()

    for index, item in enumerate(queries, start=1):
        clauses, retriever = documents[item["document"]]
        builder = RAGContextBuilder(retriever, top_k=args.top_k)
        context = builder.build(item["query"])

        variants = {"rag": context}
        if not args.no_baseline:
            variants["top1"] = top1_context(context)

        record = {
            "query": item["query"],
            "task": item["task"],
            "document": item["document"],
            "reference": item["reference"],
            "expected_sources": item["expected_sources"],
            "retrieved_titles": [c.title for c in context.retrieved],
        }

        for label, variant in variants.items():
            clause = variant.retrieved[0] if (
                item["task"] == GenerationTask.CLAUSE_EXPLANATION and variant.retrieved
            ) else None
            response = pipeline.run(
                item["task"], variant, clause=clause,
                key_points=item["key_points"],
            )
            grounding = response.grounding or check_grounding(response.text, "")
            record[label] = {
                "text": response.text,
                "source_clause_ids": response.source_clause_ids,
                "source_titles": response.source_titles,
                "source_titles_correct": all(
                    s in response.source_titles for s in item["expected_sources"]
                ),
                "key_point_coverage": grounding.key_point_coverage or 0.0,
                "covered_key_points": grounding.covered_key_points,
                "missing_key_points": grounding.missing_key_points,
                "unsupported_numbers": grounding.unsupported_numbers,
                "is_grounded": response.is_grounded,
            }

        records.append(record)
        print(f"[{index}/{len(queries)}] {item['task']:<20} {item['query'][:44]}")
        print(f"    -> {record['rag']['text'][:110]}")

    elapsed = time.perf_counter() - start

    payload = {
        "model": args.model,
        "device": pipeline.device,
        "prompt_version": __import__("src.generation", fromlist=["x"]).PROMPT_VERSION,
        "generation_settings": settings.to_dict(),
        "n_examples": len(records),
        "top_k": args.top_k,
        "total_seconds": round(elapsed, 1),
        "seconds_per_example": round(elapsed / max(len(records), 1), 2),
        "rag_context": summarise(records, "rag"),
        "per_example": records,
        "caveat": (
            "ROUGE and BERTScore measure similarity to 23 author-written "
            "reference answers. They do not measure legal correctness. The "
            "grounding check is a surface diagnostic, not a hallucination "
            "detector."
        ),
    }

    if not args.no_baseline:
        payload["top1_context"] = summarise(records, "top1")

    if not args.no_bertscore:
        print("\ncomputing BERTScore ...")
        references = [r["reference"] for r in records]
        payload["rag_context"]["bertscore"] = bert_scores(
            [r["rag"]["text"] for r in records], references
        )
        if not args.no_baseline:
            payload["top1_context"]["bertscore"] = bert_scores(
                [r["top1"]["text"] for r in records], references
            )

    # Per-task breakdown for the richer context.
    by_task: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    payload["rag_by_task"] = {
        task: summarise(group, "rag") for task, group in sorted(by_task.items())
    }

    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    for label in ("rag_context", "top1_context"):
        block = payload.get(label)
        if not block:
            continue
        print(f"\n{label}  (n={block['n']})")
        print(f"  ROUGE-1                     {block['rouge']['rouge1']:.4f}")
        print(f"  ROUGE-2                     {block['rouge']['rouge2']:.4f}")
        print(f"  ROUGE-L                     {block['rouge']['rougeL']:.4f}")
        if block.get("bertscore"):
            print(f"  BERTScore F1                {block['bertscore']['f1']:.4f}")
        print(f"  key-point coverage          {block['key_point_coverage']:.4f}")
        print(f"  examples w/ unsupported nums {block['examples_with_unsupported_numbers']}"
              f"/{block['n']}")
        print(f"  cited all expected sources  {block['examples_citing_all_expected_sources']}"
              f"/{block['n']}")
        print(f"  mean output chars           {block['mean_output_chars']}")

    print("\nBY TASK (RAG context)")
    print(f"  {'task':<22}{'n':>4}{'R-1':>9}{'R-L':>9}{'coverage':>11}")
    for task, block in payload["rag_by_task"].items():
        print(f"  {task:<22}{block['n']:>4}{block['rouge']['rouge1']:>9.4f}"
              f"{block['rouge']['rougeL']:>9.4f}{block['key_point_coverage']:>11.4f}")

    print(f"\ntiming: {elapsed:.1f}s total, "
          f"{payload['seconds_per_example']:.2f}s per example")
    print(f"\n{payload['caveat']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "generation_metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
