# Phase 7 — Grounded Generation

## 1. Objective

Turn the Phase 6 `RAGContext` into readable, grounded text for the four
user-facing outputs: Quick Summary, Key Clauses, Watch Out For, and Simple
Explanation.

Generation is grounded **structurally**, not by instruction. The pipeline
consumes a `RAGContext` and never touches the retriever, the raw PDF, or
arbitrary document text. There is no code path from a document to a generated
answer that bypasses retrieval. An empty context returns a fixed message
without invoking the model.

## 2. Model selection

**`google/flan-t5-base`**, used zero-shot via `AutoTokenizer` and
`AutoModelForSeq2SeqLM`.

Chosen because it follows instructions without fine-tuning, runs on CPU for
inference, and fits a free-tier T4 comfortably. `flan-t5-large` is excluded as
disproportionate. `flan-t5-small` remains available via
`--model google/flan-t5-small` if profiling shows base is too slow on CPU;
that decision should be made from a measured run, not assumed.

**No fine-tuning.** The project's deep-learning contribution is the fine-tuned
Legal-BERT classifier. A second training pipeline would cost time without
strengthening the argument, and the plain-English corpus available is far too
small to fine-tune on responsibly.

## 3. Architecture

```
query → ClauseRetriever (Phase 6) → RAGContextBuilder → RAGContext
      → GenerationPipeline → GeneratedResponse
```

`GenerationPipeline` accepts an injected model and tokenizer, which keeps the
unit tests free of a model download. A test asserts that `src/generation.py`
contains no reference to `build_improved`, `ClauseRetriever` or
`SentenceTransformer` — generation cannot retrieve on its own.

`GenerationPipeline.token_count()` exposes the FLAN-T5 tokenizer so Phase 6
can replace its character heuristic with exact counts via
`RAGContextBuilder(token_counter=...)`.

## 4. Prompt design

Three tasks — `quick_summary`, `simple_explanation`, `clause_explanation` —
share **identical guardrails**, so any difference in output between tasks is
attributable to the task instruction rather than to differing constraints:

> Use only the information in the context below. Do not add facts that are not
> stated. Do not give legal advice or say whether anything is legal, fair or
> valid. Write plainly and neutrally.

`clause_explanation` additionally receives Phase 4/5 metadata: the clause
category, the labels of any evidence-linked flags, and extracted values.
**Placeholder extractions are deliberately excluded** — feeding `[Insert Date]`
to the model invites it to invent a date.

Prompts are versioned (`PROMPT_VERSION`) so stored results can be matched to
the prompts that produced them.

## 5. Grounding strategy

- Context comes only from Phase 6.
- Empty context → `"No relevant clause context was retrieved for this
  question."` The model is never called.
- Every response carries `source_clause_ids`, `source_titles` and
  `source_citations` (`Clause 6: Intellectual Property (p2)`).
- `is_grounded` is false unless at least one source clause backs the response.

**Context text is never modified.** Phases 4–6 guarantee that evidence spans
index the original clause text; sanitising context here would break that
contract. The consequence is stated plainly in §11.

## 6. Output structures

`GeneratedResponse` — text, task, source clause ids/titles/citations, model
name, prompt version, the prompt itself, context size, empty-context flag,
grounding report, and the disclaimer.

`GroundingReport` — supported and unsupported numeric tokens, key-point
coverage, covered and missing key points.

## 7. Evaluation dataset

`data/eval/generation_queries.json` — **23 hand-written examples**, no new
dataset downloaded.

| Task | n | Document | n |
|---|---|---|---|
| clause_explanation | 15 | contractor agreement | 15 |
| simple_explanation | 5 | privacy policy / ToS | 8 |
| quick_summary | 3 | | |

Coverage: intellectual property, termination, payment, confidentiality,
indemnity, compliance, scope, governing law and disputes, data collection,
data sharing, retention, user rights, automatic renewal, arbitration.

**Every reference answer is derived strictly from the source documents.** A
test asserts that each `expected_sources` entry is a real clause title and
that every `key_point` actually occurs in that clause's text — a key point
absent from the source would be an unwinnable metric. One (`"belongs"`, absent
from an IP clause that says "property of the Client") was caught by that check
and corrected.

These are author-written references, not expert annotations.

## 8. Evaluation metrics

`scripts/evaluate_generation.py` reports ROUGE-1/2/L, BERTScore, key-point
coverage, count of examples containing unsupported numeric tokens, and how
often all expected sources were cited — overall and per task.

**Context comparison.** Each example is generated twice: once from the full
Phase 6 context (top-k clauses with titles, categories and citations) and once
from a reduced baseline (top-1 clause text only, no metadata). Same model,
same decoding, same template — only the context differs, isolating whether
richer grounding changes output quality.

## 9. Results

Executed locally on CPU with `google/flan-t5-base`, 23 examples, `top_k=3`,
prompt version `v1`. Decoding: `num_beams=4`, `do_sample=false`,
`early_stopping=true`, `no_repeat_ngram_size=3`, `max_input_tokens=512`.

Runtime: **137.1 s total, 5.96 s per example** — and each example is generated
twice (RAG context and top-1 baseline), so that figure covers two generations.

### Overall

| Metric | RAG context (top-3) | Top-1 baseline |
|---|---|---|
| ROUGE-1 | 0.4380 | **0.4499** |
| ROUGE-2 | 0.2643 | **0.2758** |
| ROUGE-L | 0.3874 | **0.4087** |
| BERTScore F1 | 0.8968 | **0.8997** |
| BERTScore precision | 0.9085 | 0.9137 |
| BERTScore recall | 0.8859 | 0.8863 |
| Key-point coverage | 0.4703 | **0.4739** |
| Unsupported numeric outputs | 0/23 | 0/23 |
| All expected sources cited | **21/23** | 19/23 |
| Mean output length (chars) | 140.8 | 115.7 |

### Per task, RAG context

| Task | n | ROUGE-1 | ROUGE-L | Key-point coverage |
|---|---|---|---|---|
| clause_explanation | 15 | 0.5026 | 0.4673 | 0.4600 |
| simple_explanation | 5 | 0.3561 | 0.2842 | 0.5833 |
| quick_summary | 3 | 0.2513 | 0.1599 | 0.3333 |

### Interpretation

**The richer RAG context did not improve the automatic similarity metrics.**
The top-1 baseline scored marginally higher on ROUGE-1, ROUGE-2, ROUGE-L,
BERTScore F1 and key-point coverage. That result is reported as measured; no
claim of a generation-quality improvement from richer context is made, and the
Phase 6 context builder is retained on its own merits rather than on the basis
of this comparison.

A plausible reading, offered as a hypothesis rather than a finding: the
references are short and clause-focused, and the top-1 variant produces shorter
output (115.7 vs 140.8 chars). ROUGE F-measure penalises the extra material
that additional clauses invite. That would mean the metric rewards brevity here
rather than the baseline being better grounded. Testing it would require
length-controlled decoding, which was not run.

**The one measured advantage of the richer context is provenance coverage:
21/23 versus 19/23 examples citing all expected sources.** With top-k=3 the
context can span the several clauses a multi-part question actually needs,
whereas top-1 structurally cannot. That matters for the application, where the
interface shows which clauses support an explanation, but it is a
context-coverage result, not a text-quality result.

**Task-level spread is large and the sample is tiny.** `clause_explanation`
(n=15) scores roughly twice `quick_summary` (n=3) on ROUGE-L. Summarising
several clauses into a short paragraph is a harder task than restating one, but
with n=3 the summary figure should be treated as anecdotal.

**Generation quality is bounded by retrieval quality.** Several weak outputs
trace directly to what was retrieved:

- *"Who else can see my personal data?"* retrieved **Your Rights and Choices**
  at rank 1, with the expected **Sharing With Third Parties** at rank 2. The
  model faithfully summarised the rank-1 clause — access, correction and
  deletion rights — and scored 0.00 key-point coverage. The generation was
  correct with respect to its context; the context was wrong.
- *"Summarize how my data is collected, shared and retained"* retrieved
  **Preamble** into the top-3, spending budget on the document title instead of
  the retention clause. Coverage 0.50.

Two other weak cases were **not** retrieval failures, and it would be wrong to
attribute them to retrieval:

- *"How and when is the contractor paid?"* retrieved **Payment Terms** at rank
  1 correctly, but the output selected only the tax sentence, missing rate,
  milestones and invoicing. Coverage 0.00.
- *"Summarize the payment and termination terms"* retrieved both correct
  clauses and returned the same tax sentence.

These are extraction-and-selection failures inside the generator: FLAN-T5-base,
zero-shot, picking a peripheral sentence from a correctly retrieved clause.

## 10. Hallucination and grounding observations

**The numeric grounding check flagged nothing: 0/23 examples contained
unsupported numeric tokens, under both context variants.** Given that the
sample contract is an unfilled template whose amounts and dates are all
placeholders (Phase 5), there was little numeric material to fabricate. The
clean result should not be read as evidence that the model does not hallucinate.

**It plainly does.** The most instructive output in the run:

> Query: *"Which law governs the agreement and how are disputes handled?"*
> Retrieved (rank 1): **Miscellaneous** — the correct clause, containing the
> Governing Law and Dispute Resolution provisions.
> Output: *"Payoneer is a software company that specializes in contract
> management."*

Retrieval succeeded. The word "Payoneer" does appear in the context, inside the
Payment Terms clause's list of payment methods. But **"software company that
specializes in contract management" appears nowhere in the document.** The model
supplied it from its own parametric knowledge, latched onto a brand name, and
ignored the governing-law text it was given.

**The grounding check scored this output as clean**, because it contains no
numeric tokens. This is precisely the blind spot documented in §11 and pinned by
`test_grounding_check_is_surface_level_only`. It is recorded here as a measured
instance rather than a hypothetical: a fluent, entirely ungrounded, brand-name
hallucination passed a diagnostic that reports 0/23 unsupported outputs.

Key-point coverage is the more informative signal at 0.4703 — under half the
terms an adequate answer should contain were present. Read together, the two
numbers say the model rarely invents *numbers* but frequently *omits* the
substance of the clause, and occasionally substitutes external knowledge for it.

**No claim is made that hallucination has been addressed.** The diagnostic is a
lightweight academic instrument, and this run demonstrates its limits as clearly
as its usefulness.

## 11. Limitations

1. **23 examples over 2 documents**, one of them a synthetic fixture. Per-task
   figures rest on as few as 3 examples. Indicative only.
2. **References are author-written**, not expert annotations.
3. **ROUGE and BERTScore measure similarity to those references.** They say
   nothing about legal correctness, and a legally accurate answer phrased
   differently scores poorly.
4. **The grounding diagnostic is not a hallucination detector.** Demonstrated
   above: an entirely ungrounded output scored clean because it contained no
   numbers.
5. **No defence against instructions embedded in a source document.** Context
   passes through verbatim to preserve the evidence-span contract of Phases
   4–6. Task and provenance are computed from structure, so they cannot be
   forged, but the model's output can be influenced. No injection resistance is
   claimed.
6. **Zero-shot FLAN-T5-base is not adapted to legal text.** The Payoneer case
   shows parametric knowledge overriding supplied context.
7. **512-token encoder limit.** Long contexts are truncated by the Phase 6
   budget before the model sees them, which contributes to the weak
   `quick_summary` scores.
8. **CPU beam search is slow** at 5.96 s per example for two generations. Use
   `--limit` before a full run.
9. **The RAG-vs-top-1 comparison is confounded by output length.** The
   length-controlled experiment that would separate the two was not run.

## 12. Conclusion

Phase 7 implements grounded generation over Phase 6 contexts: provenance
retained end to end, deterministic decoding, a safe empty-context path that
never invokes the model, and a lightweight grounding diagnostic. All 23
examples were evaluated with real metrics.

The honest summary of what was measured:

- Automatic similarity is **moderate** — ROUGE-1 0.4380, ROUGE-L 0.3874,
  BERTScore F1 0.8968 against the RAG context. BERTScore is high in absolute
  terms mostly because both texts are short English prose about the same
  subject; it is not evidence of accuracy.
- **The richer context did not beat the top-1 baseline on similarity metrics.**
  Its measured advantage is provenance coverage, 21/23 versus 19/23.
- **Generation quality is bounded above by retrieval quality**, and separately
  by the zero-shot model's tendency to select a peripheral sentence from an
  otherwise correct clause.
- **Hallucination occurs and the diagnostic missed it.** One output invented a
  company description and passed the numeric check cleanly.

For the viva, the defensible claim is that the system produces grounded,
traceable explanations with measured quality and honestly characterised
failure modes — not that it produces reliable legal explanations. The
components a stronger result would need are identified: better retrieval for
multi-clause questions, and either a legal-domain generator or a fine-tuned one.
