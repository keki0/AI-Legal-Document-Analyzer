# Phase 9 — Streamlit Application

## 1. Objective

Make the Phase 1–8 pipeline demonstrable through a single-file Streamlit
interface: upload a PDF, see what the system found, and ask questions answered
from the document's own clauses.

The application is a **presentation layer**. It performs no PDF extraction, no
segmentation, no classification, no calibration, no salience scoring, no flag
detection, no information extraction, no retrieval and no generation. Every one
of those is a call into `src.pipeline`.

## 2. UI architecture

`app.py` is split in two, and the split is what makes it testable:

- **Data preparation** — `document_stats`, `key_clauses`, `attention_flags`,
  `grouped_extractions`, `source_clauses`, `answer_payload`, `build_report`,
  `analyze_upload`. These contain **no Streamlit calls**, so `tests/test_app.py`
  runs them headlessly.
- **Rendering** — `render_overview`, `render_key_clauses`, `render_flags`,
  `render_extractions`, `render_ask`, and `main`. Not unit tested; screenshot
  tests would be fragile and would verify Streamlit rather than this project.

## 3. How Streamlit connects to `pipeline.py`

```
upload  -> analyze_upload()      -> LegalDocumentPipeline.analyze_document()
summary -> render_overview()     -> LegalDocumentPipeline.document_summary()
question-> render_ask()          -> LegalDocumentPipeline.ask()
```

Streamlit hands over an in-memory buffer; PyMuPDF needs a path, so
`analyze_upload` writes the bytes to a temporary file. That is the only
file handling in the application.

### The one compatibility change

`src/pipeline.py` gained two methods: `summary_context()` and
`document_summary()`. Nothing else in `src/` was modified.

**Why it was necessary.** Both existing entry points — `ask()` and
`context_for()` — are query-driven through retrieval. There was no path to a
*document-level* summary, because a summary has no query to retrieve against.
The Phase 4 design stated that salience selects the top-K clauses for the
generator, but that wiring had never been built.

**Why it belongs in `pipeline.py`, not `app.py`.** The presentation layer must
not assemble contexts. `summary_context()` implements the selection as a small
adapter satisfying the Phase 6 `Retriever` protocol, returning clauses in
salience order. `RAGContextBuilder` then does the deduplication, budgeting and
formatting exactly as it does for retrieval, so no Phase 6 logic is duplicated
and no Phase 6 behaviour changed.

This is also the point where the deep-learning component visibly feeds the
generator: on the sample contract the summary is built from Intellectual
Property, Indemnification, Term and Termination, and Payment Terms — selected
by calibrated classifier confidence weighted by category prior, not by keyword.

## 4. Main UI sections

| Section | Source | Notes |
|---|---|---|
| Sidebar | `document_stats` | Upload, metrics, degraded-mode warnings, report download |
| **Overview** | `document_summary` | Generated on demand, not on upload |
| **Key clauses** | `top_k_salient` | Salience-ranked, category filter, expandable full text |
| **Watch out for** | Phase 4 flags | Each flag shows the sentence that triggered it |
| **Extracted information** | Phase 5 | Grouped by type, placeholder status shown |
| **Ask the document** | `pipeline.ask` | Example buttons, answer style toggle, expandable sources |

Two deliberate choices. The summary is **generated on a button press, not on
upload**, because FLAN-T5 on CPU takes seconds and a slow upload feels broken.
And the *Watch out for* tab is **independent of salience ranking** — Phase 4
found that a low-salience "Boilerplate & Administrative" clause can carry
Choice-of-law and Arbitration flags, so ranking by salience alone would bury
them. A test asserts that at least one flagged clause falls outside the top 3.

## 5. Session-state handling

`st.session_state` holds `analysis`, `history` and `summary`. Streamlit
re-executes the whole script on every interaction, so without this the document
would be re-analysed on every keystroke.

The user can ask any number of questions against one analysis; only `ask()`
runs per question. Uploading a new document clears the history and summary so
answers from a previous document cannot linger.

## 6. Model caching

`load_pipeline()` is decorated with `@st.cache_resource`, so Legal-BERT,
MPNet-QA and FLAN-T5 load once per session rather than on every rerun. Nothing
is downloaded at start-up beyond what the pipeline already requires, and no
model is ever retrained.

## 7. Error handling

No Python traceback reaches the interface. Handled cases:

| Situation | Behaviour |
|---|---|
| No upload | Getting-started panel |
| Non-PDF | "Only PDF files are supported." |
| Empty file | "The uploaded file is empty." |
| Corrupt / encrypted PDF | Named message suggesting scanned, encrypted or corrupted |
| PDF with no extractable text | Explains there is no OCR |
| **Classifier missing** | Red error naming `models/legal-bert-clause-classifier` and the notebook that produces it; app continues with clauses marked *Unclassified* |
| `temperature.json` missing | Warning that confidence is uncalibrated, names `scripts/calibrate.py` |
| Retrieval failure | Recorded in `stage_errors` and shown in the sidebar |
| Generation failure | "This question could not be answered." |
| Empty retrieval context | The Phase 7 safe message; the model is never called |

The missing-classifier case is an explicit error rather than a silent fallback,
so the user is never shown blank categories without explanation.

## 8. Legal disclaimer

Shown at the top of every page, repeated in the footer, and included in the
downloadable report:

> **Academic prototype only.** This tool provides document analysis and
> simplified explanations for informational purposes. It is not legal advice
> and should not be used as a substitute for advice from a qualified legal
> professional.

A second caveat accompanies every generated answer:

> Answers are generated from the clauses shown below them. A cited answer is
> traceable, not necessarily correct — always read the source clause.

That wording is deliberate. Phase 8 recorded 11/11 structurally grounded
responses, and Phase 7 measured a case where retrieval succeeded, the model
asserted something absent from the document, and the numeric grounding check
scored it clean. **Grounded means traceable, not correct**, and the interface
must not blur the two. A test asserts the caveat contains "not necessarily
correct".

Flag wording is inherited unchanged from Phase 4 — every explanation begins
"This clause may be significant because…" — and a test scans for *illegal*,
*unfair*, *invalid*, *you should* and *definitely risky*.

## 9. Testing

`tests/test_app.py` — **31 tests**, all models stubbed, no browser and no
downloads. Coverage: document statistics, salience ordering, preview
truncation, flag evidence and neutral wording, extraction grouping and
placeholder exposure, source-clause mapping, empty and failed queries, upload
validation (non-PDF, empty, corrupt, valid), degraded mode, report generation,
and disclaimer wording.

Full suite after Phase 9: **343 passed, 1 skipped, 0 failed** — up from 312
passed / 1 skipped at Phase 8, with no regressions.

## 10. How to run

```bat
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens on `http://localhost:8501`. For the full experience the fine-tuned
classifier must be present at `models/legal-bert-clause-classifier`; without it
the app runs and says so.

## 11. Limitations

1. **Not a legal tool.** It describes documents and gives no legal advice.
2. **Generation quality is moderate and measured.** Phase 7: ROUGE-1 0.4380,
   key-point coverage 0.4703, with a documented hallucination the grounding
   diagnostic missed. The UI presents answers accordingly.
3. **No OCR.** Scanned or image-only PDFs produce no text.
4. **Segmentation assumes conventional structure.** Heavily tabular or unusual
   layouts may segment poorly.
5. **Classification is out of distribution.** Legal-BERT was fine-tuned on
   LEDGAR (SEC filings); its 0.9126 macro-F1 does not transfer to arbitrary
   uploaded contracts.
6. **CPU generation is slow** — seconds per answer. This is why the summary is
   on demand.
7. **Single user, no persistence.** No database, authentication or deployment
   configuration; analysis is lost when the session ends.
8. **Validated on two documents only** (Phase 8). Behaviour on other documents
   is unverified.
