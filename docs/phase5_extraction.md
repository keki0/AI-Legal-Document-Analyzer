# Phase 5 — Information Extraction

## Purpose

Phase 4 produces, for each clause, a category, a calibrated confidence, a
salience score and evidence-linked flags. Phase 5 adds the structured facts a
reader actually wants to look up: who the parties are, what is owed, by when,
and on what notice.

The layer is deterministic and rule-based. It makes no legal judgements: it
reports what a document says, never whether a term is valid, enforceable,
fair or advisable.

## Why rules rather than a model

All seven extraction types are surface phenomena with reliable lexical
signals. A transformer NER model would add a dependency and an unexplainable
failure mode in exchange for no measurable gain, and we have no labelled
extraction data with which to justify or tune one. spaCy was also considered
and not adopted: on templated contracts its entity recogniser fires largely on
bracketed placeholders, so it would cost a model download for no observed
benefit. The hook remains available if a later document set justifies it.

Every rule is inspectable, and every extracted item names the rule that
produced it.

## Output schema

```
ExtractedItem
    type              party | date | duration | money |
                      notice_period | renewal_term | obligation
    value             the matched text, verbatim
    normalized_value  ISO-8601 where applicable (P30D, 2026-01-15)
    evidence          the sentence containing the match
    span_start/end    offsets into the clause text
    method            the rule that fired, e.g. regex:duration_near_notice
    confidence        rule specificity (see caveat below)
    clause_id/title/category   provenance from Phases 1 and 3
    is_placeholder    True for unfilled template fields
```

`text[span_start:span_end] == value` holds for every non-obligation item and
is asserted by tests, so the interface can highlight the exact evidence.

### What `confidence` means

It is a **hand-assigned indicator of how specific the matching rule is**, on a
0–1 scale. It is not a probability, it is not learned, and it is **not
comparable to the calibrated classifier confidence from Phase 4**. A tightly
anchored pattern such as a notice period scores above a bare capitalised role
word. Only the ordering is defended. This mirrors the honesty rule applied to
the Phase 4 category priors.

## Integration

`ClauseAnalysis` gains one additive field, `extractions`. No Phase 4 field is
duplicated and no calibration behaviour changes; a test asserts that salience,
calibrated confidence and flags are byte-identical before and after
extraction is attached. The clause category is copied from Phase 4 onto each
item rather than re-derived.

The sentence splitter is imported from `src.importance` rather than
reimplemented, so flags and extractions segment text identically.

Governing law is deliberately **not** an extraction type: Phase 4's
`choice_of_law` flag already covers it, and a second detector would duplicate
an abstraction.

A separate `DocumentExtraction` aggregate exists because some facts are
document-level rather than clause-level. Parties are defined once in the
preamble and referred to throughout, so a purely per-clause view would report
the same party a dozen times.

## Finding: the sample contract is an unfilled template

Extraction over `sample-independent-contractor-agreement.pdf` produced 41
items, of which **every date, monetary amount and notice period is a bracketed
placeholder**: `[Insert Date]`, `$[Rate]`, `[7/14] days`.

| Type | Value | Clause | Placeholder |
|---|---|---|---|
| date | Insert Date | Preamble | yes |
| date | Start Date | Term and Termination | yes |
| notice_period | [7/14] days | Term and Termination | yes |
| money | $[Rate], $X, $XX, $XY | Payment Terms | yes |
| date | Effective Date | Miscellaneous | yes |

This is why placeholders are first-class rather than filtered out. Extracting
`[Insert Date]` as a date would be wrong; discarding it silently would lose
real information. Reporting *"an effective date is referenced but not filled
in"* is both accurate and useful.

`placeholder_ratio` measures this over **fillable types only** (date, money,
duration, notice period). An earlier version measured it across all items and
scored the contract below 30% — obligations and role-word parties, which can
never be placeholders, diluted the signal. Corrected, the contract scores
1.00 and the filled policy document scores 0.33.

## Limitations

1. **No precision, recall or F1 is claimed.** There is no expert-labelled
   extraction dataset for this project. Testing is qualitative: each rule is
   verified to fire on representative drafting and to stay silent on
   constructions designed to trip it. **Recall against real-world drafting
   variety is unmeasured.**
2. **English only**, and conventional legal drafting is assumed.
3. **Obligation extraction is shallow by design.** It identifies sentences
   carrying deontic language and a rough subject. It performs no semantic role
   labelling and does not reliably resolve who bears an obligation.
4. **Party detection depends on role vocabulary.** Documents written in
   first and second person — most privacy policies, which say "we" and "you" —
   yield no parties. Observed on our policy fixture.
5. **Date parsing assumes US ordering** for the ambiguous `01/02/2026` form.
6. **Durations are normalised, amounts are not.** No currency conversion, no
   arithmetic, no totals — deliberately, since those would be financial
   inferences rather than extraction.

## False positives found and fixed during development

Three, all caught by tests rather than by inspection:

- **`"Draft Month"` matched as a duration.** The contract's milestone table
  flattens into prose as "Rough Draft Month [Date], [Year]". A pattern of the
  form `\w+ months?` matched it. Fixed by requiring an explicit cardinal.
- **Percentages never extracted.** A trailing `\b` after `%` requires a word
  character to follow, so `"1.5% per month"` failed to match.
- **Role words inside placeholders extracted twice.** `[Company Name]`
  produced both the placeholder and a spurious non-placeholder `"Company"`
  matched from inside the brackets. Fixed by skipping role matches contained
  within a placeholder span.
