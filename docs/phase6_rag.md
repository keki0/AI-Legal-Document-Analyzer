# Phase 6 — Semantic Retrieval & Lightweight RAG

## 1. Objective

Turn a user question plus an analysed document into a **grounded, budgeted,
provenance-carrying context** that Phase 7 can hand to FLAN-T5.

**Phase 6 implements retrieval and grounded context construction.
Natural-language generation is deferred to Phase 7.** No text is generated
here, so no claim is made about answer quality — there are no answers yet.

## 2. Existing retrieval baseline

Phase 1 established, on a five-query probe:

| Configuration | Probe top-1 | Regression query |
|---|---|---|
| `all-MiniLM-L6-v2`, body only | 3/5 | incorrect |
| `all-MiniLM-L6-v2` + title | 3/5 | incorrect |
| `multi-qa-mpnet-base-dot-v1`, body only | 4/5 | correct |
| `multi-qa-mpnet-base-dot-v1` + title | 4/5 | correct |

**The improvement came from the model switch, not from title augmentation.**
Both MPNet variants scored 4/5, so title augmentation is not claimed as a
contribution. It is retained because it costs nothing, but its benefit is
unmeasured.

## 3. Why MPNet-QA

`all-MiniLM-L6-v2` is trained for *symmetric* semantic similarity — how alike
two sentences are. Clause search is *asymmetric*: a short question against a
longer passage. `multi-qa-mpnet-base-dot-v1` is trained on that objective.

Phase 1's lexical diagnostic explained the failure concretely: the word "own"
never appears in the Intellectual Property clause (which says "sole and
exclusive property" and "assign all rights"), but does appear in Independent
Contractor Status in an unrelated sense ("their own work schedule"). Both
clauses contain "contractor" and "work", so body-only symmetric embeddings had
almost nothing to separate them.

## 4. Retrieval evaluation

`scripts/evaluate_retrieval.py` scores both retrievers on the **existing 29
queries** in `data/eval/retrieval_queries.json`. No new dataset was created.

**Definition of correct:** the retrieved clause's title matches the query's
`expected` title exactly. A Phase 2 test already asserts every `expected` value
corresponds to a clause the segmenter actually emits, so the ground truth
cannot silently drift when segmentation changes.

Metrics: **Hit@1, Hit@3, Hit@5, MRR**, reported for the baseline and improved
retrievers, broken down by the query types recorded in Phase 1 (direct,
paraphrase, lay, inference, cross, distractor-pair, regression). Results are
written to `evaluation/results/retrieval_metrics.json`.

These labels are hand-written by the project authors. They encode which clause
answers each question; they are not expert annotations and not a benchmark.

## 5. RAG architecture

```
question → ClauseRetriever (MPNet-QA) → candidate pool
         → deduplicate → top-k → [optional flag rule]
         → budget packing → RAGContext → (Phase 7: FLAN-T5)
```

**`src/retrieval.py` is not modified.** `RetrievalResult` already carries
`clause_id` and `ClauseAnalysis` is keyed by the same field, so Phase 4/5
metadata joins on that key. `src/rag.py` declares a `Retriever` protocol that
the existing `ClauseRetriever` already satisfies — which also makes the whole
context layer testable without downloading embeddings.

## 6. Context construction

`RetrievedClause` carries rank, score, clause id, title, text, page, and —
when analyses are supplied — category, salience, calibrated confidence, flags
and extractions. Nothing is copied that could be referenced: `flags` and
`extractions` hold the same objects Phases 4 and 5 produced.

`RAGContext` carries the query, the structured records, the formatted string,
character and token counts, and the budget accounting.

## 7. Provenance and grounding

Every clause in a context has a citation of the form
`Clause 6: Intellectual Property (p2)`, emitted both in the structured records
and as a header in the formatted text, so a Phase 7 answer can be traced to
its sources. `RAGContext.sources` lists them in context order.

Clause titles are repeated in the context headers deliberately: titles carry
topical signal the body sometimes lacks — the Intellectual Property clause
never uses the word "own".

## 8. Interaction with salience, flags and extractions

**Ranking stays purely semantic.** Salience and classifier confidence are
attached as metadata and are never multiplied into the similarity score.
Reordering retrieval by an importance prior would confound two different
signals and make the retrieval evaluation uninterpretable. Salience answers
"how much should a reader care about this clause"; similarity answers "does
this clause address this question". They are not interchangeable, and no
combined score is produced.

### The flag-support rule

Phase 4 found that a low-salience *Boilerplate & Administrative* clause can
still carry Choice-of-law and Arbitration flags — in the sample contract, the
"Miscellaneous" section holds exactly those provisions. Important evidence can
therefore sit far down a salience ordering.

But appending every flagged clause to every query would flood the context. The
rule implemented here is deliberately conservative and **off by default**. A
flagged clause is admitted only if it:

1. carries a Phase 4 flag, **and**
2. appears in the retriever's candidate pool, **and**
3. scores at least `flag_score_ratio` (default 0.6) of the top hit.

Without the score floor this would attach arbitration boilerplate to unrelated
questions. Because this is a secondary rule, `scripts/evaluate_retrieval.py`
**measures** what it does — how often it fires, how many clauses it adds, and
how often it recovers a gold clause that semantic top-3 missed — rather than
assuming it helps. It never reorders retrieval, so it cannot change Hit@k and
is reported separately.

## 9. Context-budget strategy

Default budget is 1600 characters (≈400 tokens at ~4 chars/token), leaving
headroom inside FLAN-T5's 512-token input for the instruction and question.
The estimator is a heuristic; Phase 7 should inject the real FLAN-T5 tokenizer
via `RAGContextBuilder(token_counter=...)`.

Packing is deterministic and prefers complete clauses:

- A clause that does not fit is **dropped whole**, and its id is recorded in
  `dropped_clause_ids`.
- **Only the top hit is ever abridged**, so the context never contains a
  mid-clause fragment presented as if it were a complete provision.
- Abridgement cuts at a **sentence boundary** where at least one sentence
  fits, appends ` [...]`, sets `included_fully=False`, and records the id in
  `abridged_clause_ids`. Provenance is fully preserved.
- If even the first sentence exceeds the remaining room, it falls back to a
  hard character cut. Observed on the Intellectual Property clause, whose
  first sentence is 231 characters: at a 300-character budget it cuts at the
  sentence boundary; at 200 it hard-cuts.

## 10. Test results

**48 Phase 6 tests, all passing. Full suite: 233 passed, 1 skipped.** The skip
is the saved-model round-trip, pending the fine-tuned model download.

Tests use a deterministic stub retriever satisfying the `Retriever` protocol,
so selection, budgeting, provenance and metric arithmetic are verified without
model downloads or random seeds. Two tests assert that Phase 4/5 outputs are
not mutated by context building, and that abridgement does not mutate the
source clause.

## 11. Limitations

1. **Retrieval metrics are produced by the user's run, not asserted here.**
   The evaluation requires downloading MPNet-QA.
2. **29 queries over 2 documents is a small evaluation set**, one of which is
   a synthetic fixture. Metrics should be read as indicative, not benchmark
   quality.
3. **Ground truth is hand-written by the project authors**, not expert
   annotation.
4. **Exact title matching** counts a near-miss as a full miss. Where two
   clauses could both reasonably answer a question, this understates
   performance.
5. **The token estimate is a heuristic** until Phase 7 injects a tokenizer.
6. **No generation quality claim is possible.** Retrieval quality, grounding
   and budget behaviour are demonstrable now; answer quality is not.

## 12. Phase 7 integration

Phase 7 consumes `RAGContext.formatted_context` as the grounded input and
`RAGContext.sources` for citations, and should pass the FLAN-T5 tokenizer's
length function as `token_counter` for exact budgeting. No change to Phase 6
is anticipated.
