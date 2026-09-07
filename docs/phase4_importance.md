# Phase 4 — Importance & Attention Flags

## Purpose

The baseline (`notebooks/01_baseline.ipynb`) rated clause significance with a
keyword table: if a clause contained a listed phrase, it was labelled HIGH.
That approach had three defects. It was circular — one of its HIGH keywords,
"sole and exclusive property", was lifted verbatim from the single document it
was evaluated on. It was unevaluable, since there was no held-out set. And it
issued a verdict with no evidence, so a reader could neither check it nor
disagree with it.

Phase 4 replaces it with three separate outputs.

| Output | Source | Epistemic status |
|---|---|---|
| **Category** | Legal-BERT argmax | Learned from 20,000 labelled provisions |
| **Salience** | calibrated confidence × category prior | Learned signal × documented design choice |
| **Attention flags** | regex detectors, span-linked | Rules, with evidence attached |

There is deliberately **no single risk score**, and no output field contains
the word "risk" (asserted by a test). The system describes documents; it does
not judge them.

## Confidence calibration

A fine-tuned Transformer's softmax output is systematically overconfident, so
raw confidence is a poor input to a salience score. We apply **temperature
scaling** (Guo et al., 2017): a single scalar `T` divides the logits before
the softmax, fitted by minimising negative log-likelihood.

`T` is fitted on the **validation** split and evaluated on **test**. Fitting
on test would leak and make the reported ECE meaningless.

We report Expected Calibration Error, Maximum Calibration Error, NLL,
confidence-binned accuracy, and a before/after reliability diagram.

### A property that must not be misread

**Temperature scaling cannot change which class is predicted.** Dividing every
logit by the same positive constant is monotonic, so the argmax is invariant
and accuracy is mathematically unchanged. Any accuracy difference appearing in
a report is a floating-point tie-break, not an effect.

This is enforced by `test_temperature_scaling_never_changes_a_prediction`. The
purpose of calibration here is a trustworthy **confidence** signal for
downstream salience — not better classification, and it must never be
presented as such.

## Salience

```
salience = calibrated_confidence × category_prior
```

Category priors live in `data/eval/salience_priors.yaml`. **They are
hand-assigned design choices, not learned values.** They were not fitted or
derived from any labelled importance dataset. The scale is ordinal: a prior of
0.90 means "ranks above 0.80 and below 0.95", not "90% likely to matter". Only
the ordering is defended.

The ordering answers one question: if someone signs this without reading it,
which clauses are most likely to surprise them later? Intellectual Property
and Liability & Indemnity sit at 0.95; Boilerplate & Administrative sits at
0.15. Down-weighting drafting machinery is the main job the priors do, and it
is what replaces the baseline's keyword rules.

The output is called a **salience score** or **attention priority**, never a
legal risk score.

## Evidence-linked attention flags

These are **rule-based flags, not Transformer attention weights** — the name
refers to what a reader should attend to.

Eight flag concepts are drawn from the UNFAIR-ToS label set of LexGLUE (Lippi
et al., *CLAUDETTE*): limitation of liability, unilateral termination,
unilateral change, content removal, contract by using, choice of law,
jurisdiction, arbitration. **We use the published concepts for provenance; we
do not use the dataset, and the patterns are ours, not expert annotations.**
Three further flags cover the contract domain our documents contain: broad IP
assignment, broad indemnity, automatic renewal.

What distinguishes these from the baseline's keywords is not that they stopped
being rules. It is that **every flag carries the exact span that triggered it
and the sentence containing that span**, so the reasoning is auditable and a
reader can disagree with it. A test verifies that
`text[span_start:span_end] == trigger_span` for every hit.

All wording follows the legal-safety rule. Every explanation begins *"This
clause may be significant because…"* and describes what the text does. A test
scans every explanation for the words *illegal, unfair, invalid, unlawful,
void, definitely* and fails if any appears.

## Finding: salience and flags disagree, by design

An end-to-end run over the sample contract produced this ranking:

| Clause | Category | Salience | Flags |
|---|---|---|---|
| Intellectual Property | Intellectual Property | 0.950 | Broad IP assignment |
| Indemnification | Liability & Indemnity | 0.950 | Broad indemnity |
| Term and Termination | Term & Termination | 0.900 | Unilateral termination |
| Payment Terms | Payment & Fees | 0.900 | — |
| … | | | |
| **Miscellaneous** | **Boilerplate & Administrative** | **0.150** | **Choice of law, Arbitration** |

The lowest-salience clause carries two flags. That is not a bug — it is the
limitation the two-signal design exists to cover.

The contract's "Miscellaneous" section contains its governing-law and
dispute-resolution provisions, which materially affect what remedies remain
available. A category-prior model ranks it as boilerplate because its heading
and much of its content genuinely are. The flags catch what the priors miss.

**Design consequence for the UI (Phase 9): flagged clauses must be surfaced
regardless of salience rank.** Ranking by salience alone would bury the
arbitration clause.

## Limitations

1. **No ground-truth evaluation of importance.** We considered grounding
   importance in CUAD, whose 41 categories are expert-defined. It was dropped
   to keep the project in scope. Consequently this project reports **no
   precision, recall or F1 for importance detection**. It is evaluated through
   calibration metrics and qualitative inspection only.
2. **Priors are unvalidated.** They are one team's judgement. A different team
   would produce a different ordering, and we have no data to adjudicate.
3. **Low confidence suppresses salience.** An important clause the classifier
   finds ambiguous is ranked down. Flags run independently precisely so such
   clauses can still surface.
4. **Regex flags have unmeasured recall.** They fire on the patterns we wrote.
   Unusual drafting will evade them. We have verified precision qualitatively
   on two documents; recall is unknown.
5. **Flag detection is English-only** and assumes reasonably conventional
   legal drafting.

## Architectural role

The salience layer is load-bearing, not decorative. FLAN-T5 accepts at most
512 input tokens, so a full contract cannot be summarised directly. The
salience ranking selects the top-K clauses that are passed to the generator in
Phase 6.

```
PDF → segmentation → Legal-BERT → calibrated confidence
                                → salience + flags
                                → top-K clauses → MPNet-QA retrieval
                                → FLAN-T5 → grounded explanation
```

The classifier therefore feeds the generator through this layer. That
coupling is what makes the project one system rather than several
demonstrations.
