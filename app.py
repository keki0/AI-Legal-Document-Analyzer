"""Streamlit interface for the AI-Powered Legal Document Analyzer.

A presentation layer and nothing more. Every piece of analysis is a call into
``src.pipeline``; this file contains no PDF handling, no segmentation, no
classification, no retrieval and no generation logic.

    streamlit run app.py

The data-shaping functions at the top are deliberately free of Streamlit calls
so they can be unit tested without launching a browser. Everything below
``render_*`` is presentation.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from src.config import PATHS  # noqa: E402
from src.generation import DISCLAIMER, GenerationTask  # noqa: E402
from src.pipeline import UNCLASSIFIED, LegalDocumentPipeline  # noqa: E402

APP_TITLE = "AI-Powered Legal Document Analyzer"

DISCLAIMER_TEXT = (
    "**Academic prototype only.** This tool provides document analysis and "
    "simplified explanations for informational purposes. It is not legal "
    "advice and should not be used as a substitute for advice from a "
    "qualified legal professional."
)

# Shown next to generated answers. Phase 7 measured a case where retrieval
# succeeded and the model still asserted something absent from the document,
# and the numeric grounding check scored it clean. The interface must not
# imply that a cited answer is a correct one.
GROUNDING_CAVEAT = (
    "Answers are generated from the clauses shown below them. A cited answer "
    "is traceable, not necessarily correct — always read the source clause."
)

EXAMPLE_QUESTIONS = [
    "Who owns the work produced by the contractor?",
    "How can the contract be terminated?",
    "How much does the contractor get paid?",
    "Is there an automatic renewal?",
    "How are disputes resolved?",
]

EXTRACTION_LABELS = {
    "party": "Parties",
    "date": "Dates",
    "duration": "Durations",
    "money": "Money & Fees",
    "notice_period": "Notice periods",
    "renewal_term": "Renewal & term",
    "obligation": "Obligations",
}


# ===========================================================================
# Data preparation -- no Streamlit calls, unit tested in tests/test_app.py
# ===========================================================================

def document_stats(analysis) -> dict:
    """Headline numbers for the sidebar."""
    return {
        "Filename": analysis.path.name,
        "Pages": analysis.document.n_pages,
        "Words": analysis.document.word_count,
        "Clauses": analysis.n_clauses,
        "Classified": analysis.n_classified,
        "Flagged": analysis.n_with_flags,
        "Extracted items": len(analysis.extraction.items) if analysis.extraction else 0,
    }


def key_clauses(analysis, limit: int = 6) -> list[dict]:
    """Most salient clauses, highest first.

    Salience is the Phase 4 signal: calibrated classifier confidence weighted
    by a documented category prior. It is not a risk score.
    """
    rows = []
    for item in analysis.top_salient(limit):
        text = item.text
        rows.append({
            "clause_id": item.clause_id,
            "title": item.title,
            "category": item.category,
            "salience": round(item.salience, 3),
            "confidence": round(item.calibrated_confidence, 3),
            "page": item.page,
            "preview": (text[:180] + "...") if len(text) > 180 else text,
            "text": text,
            "flags": item.flag_labels,
        })
    return rows


def attention_flags(analysis) -> list[dict]:
    """Evidence-linked flags, grouped per clause.

    Deliberately independent of salience. Phase 4 found that a low-salience
    "Boilerplate & Administrative" clause can still carry Choice-of-law and
    Arbitration flags, so ranking by salience alone would bury them.
    """
    rows = []
    for item in analysis.analyses:
        for hit in item.flags:
            rows.append({
                "label": hit.label,
                "clause_id": item.clause_id,
                "clause_title": item.title,
                "category": item.category,
                "explanation": hit.explanation,
                "evidence": hit.sentence,
                "trigger": hit.trigger_span,
                "page": item.page,
            })
    return rows


def grouped_extractions(analysis) -> dict[str, list[dict]]:
    """Extracted items grouped by type, in a stable display order."""
    if analysis.extraction is None:
        return {}

    grouped: dict[str, list[dict]] = {}
    for item in analysis.extraction.items:
        grouped.setdefault(item.type, []).append({
            "Value": item.value,
            "Normalized": item.normalized_value or "—",
            "Clause": item.clause_title or "—",
            "Category": item.clause_category or "—",
            "Unfilled placeholder": "yes" if item.is_placeholder else "no",
            "Evidence": item.evidence,
        })

    ordered = {}
    for key in EXTRACTION_LABELS:
        if key in grouped:
            ordered[key] = grouped[key]
    for key, rows in grouped.items():   # any type not in the label map
        ordered.setdefault(key, rows)
    return ordered


def source_clauses(result) -> list[dict]:
    """Source clauses supporting a generated answer."""
    if result is None or result.context is None:
        return []
    return [
        {
            "clause_id": clause.clause_id,
            "citation": clause.citation,
            "title": clause.title,
            "category": clause.category or UNCLASSIFIED,
            "page": clause.page,
            "score": round(clause.score, 4),
            "text": clause.text,
            "flags": clause.flag_labels,
            "abridged": not clause.included_fully,
        }
        for clause in result.context.retrieved
    ]


def answer_payload(result) -> dict:
    """Normalise a QueryResult into what the UI renders."""
    if result is None:
        return {"status": "none", "text": "", "sources": []}
    if result.error:
        return {"status": "error", "text": result.error, "sources": []}
    if result.empty_context:
        text = result.response.text if result.response else ""
        return {"status": "empty", "text": text, "sources": []}
    return {
        "status": "ok",
        "text": result.response.text if result.response else "",
        "sources": source_clauses(result),
        "seconds": round(result.seconds, 1),
    }


def build_report(analysis, history: list) -> str:
    """Plain-text analysis report for download."""
    lines = [APP_TITLE, "=" * len(APP_TITLE), ""]
    lines.append("ACADEMIC PROTOTYPE - NOT LEGAL ADVICE")
    lines.append("")
    for key, value in document_stats(analysis).items():
        lines.append(f"{key}: {value}")

    lines += ["", "KEY CLAUSES", "-" * 40]
    for row in key_clauses(analysis):
        lines.append(f"[{row['salience']:.3f}] {row['title']} ({row['category']})")
        lines.append(f"  page {row['page']}: {row['preview']}")

    flags = attention_flags(analysis)
    lines += ["", "WATCH OUT FOR", "-" * 40]
    if not flags:
        lines.append("No attention flags detected.")
    for flag in flags:
        lines.append(f"{flag['label']} - {flag['clause_title']} (p{flag['page']})")
        lines.append(f"  {flag['explanation']}")
        lines.append(f"  Evidence: {flag['evidence']}")

    lines += ["", "EXTRACTED INFORMATION", "-" * 40]
    for type_, rows in grouped_extractions(analysis).items():
        lines.append(f"{EXTRACTION_LABELS.get(type_, type_)} ({len(rows)})")
        for row in rows[:20]:
            marker = " [placeholder]" if row["Unfilled placeholder"] == "yes" else ""
            lines.append(f"  - {row['Value']}{marker}  ({row['Clause']})")

    if history:
        lines += ["", "QUESTIONS", "-" * 40]
        for question, result in history:
            payload = answer_payload(result)
            lines.append(f"Q: {question}")
            lines.append(f"A: {payload['text']}")
            for source in payload["sources"]:
                lines.append(f"   source: {source['citation']}")
            lines.append("")

    lines += ["", DISCLAIMER]
    return "\n".join(lines)


# ===========================================================================
# Pipeline loading
# ===========================================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load models once per session. Cached across reruns.

    Streamlit re-executes this script on every interaction, so without
    caching Legal-BERT, MPNet-QA and FLAN-T5 would reload on every click.
    """
    pipeline = LegalDocumentPipeline.from_defaults(top_k=3)
    return pipeline


def analyze_upload(pipeline, uploaded) -> tuple[object | None, str | None]:
    """Write the upload to a temp file and analyse it.

    Returns ``(analysis, error_message)``. Streamlit gives an in-memory
    buffer; PyMuPDF wants a path.
    """
    if uploaded is None:
        return None, "No document uploaded."
    if not uploaded.name.lower().endswith(".pdf"):
        return None, "Only PDF files are supported."

    data = uploaded.getvalue()
    if not data:
        return None, "The uploaded file is empty."

    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / uploaded.name
    temp_path.write_bytes(data)

    try:
        analysis = pipeline.analyze_document(temp_path)
    except Exception as error:  # noqa: BLE001 - surfaced as a message, not a trace
        return None, f"Could not read this PDF ({type(error).__name__}). It may be scanned, encrypted or corrupted."

    if analysis.n_clauses == 0:
        return None, ("No text could be extracted. Scanned or image-only PDFs "
                      "are not supported — this prototype has no OCR.")
    return analysis, None


# ===========================================================================
# Rendering
# ===========================================================================

CSS = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1100px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .doc-hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
    padding: 1.4rem 1.6rem; border-radius: 10px; color: #fff; margin-bottom: 1.2rem;
  }
  .doc-hero h1 {color: #fff; margin: 0 0 .3rem 0; font-size: 1.55rem;}
  .doc-hero p {margin: 0; opacity: .85; font-size: .92rem;}
  .flag-card {
    border-left: 4px solid #d97706; background: #fffbeb;
    padding: .8rem 1rem; border-radius: 6px; margin-bottom: .7rem;
  }
  .flag-card .flag-title {font-weight: 600; color: #92400e;}
  .flag-card .flag-meta {font-size: .82rem; color: #78716c;}
  .evidence {
    border-left: 3px solid #cbd5e1; background: #f8fafc;
    padding: .55rem .8rem; font-size: .87rem; color: #334155;
    margin-top: .5rem; border-radius: 4px;
  }
  .answer-card {
    background: #f0f7ff; border: 1px solid #bfdbfe;
    padding: 1rem 1.2rem; border-radius: 8px; font-size: 1rem; line-height: 1.6;
  }
  .pill {
    display: inline-block; background: #e2e8f0; color: #334155;
    padding: .12rem .55rem; border-radius: 999px;
    font-size: .74rem; margin-right: .35rem;
  }
</style>
"""


def render_overview(pipeline, analysis) -> None:
    st.subheader("Quick summary")
    st.caption(
        "Generated from the highest-salience clauses, not the whole document — "
        "the generator accepts a limited amount of text."
    )

    if st.session_state.get("summary") is None:
        if st.button("Generate summary", type="primary"):
            with st.spinner("Summarising the most significant clauses..."):
                st.session_state["summary"] = pipeline.document_summary(analysis)
            st.rerun()
        st.info("Select **Generate summary** to run the generator.")
        return

    payload = answer_payload(st.session_state["summary"])
    if payload["status"] == "error":
        st.error(f"The summary could not be generated. {payload['text']}")
        return
    if payload["status"] == "empty" or not payload["text"]:
        st.warning("No clause content was available to summarise.")
        return

    st.markdown(f"<div class='answer-card'>{payload['text']}</div>",
                unsafe_allow_html=True)
    st.caption(GROUNDING_CAVEAT)

    if payload["sources"]:
        st.markdown("**Clauses used**")
        for source in payload["sources"]:
            with st.expander(f"{source['citation']} — {source['category']}"):
                st.write(source["text"])


def render_key_clauses(analysis) -> None:
    st.subheader("Key clauses")
    st.caption(
        "Ranked by salience: calibrated classifier confidence weighted by a "
        "documented category prior. This is an attention ranking, not a risk score."
    )

    rows = key_clauses(analysis, limit=8)
    if not rows:
        st.info("No clauses were identified.")
        return

    categories = sorted({r["category"] for r in rows})
    chosen = st.multiselect("Filter by category", categories, default=categories)

    for row in [r for r in rows if r["category"] in chosen]:
        with st.container(border=True):
            left, right = st.columns([5, 1])
            with left:
                st.markdown(f"**{row['title']}**")
                pills = f"<span class='pill'>{row['category']}</span>"
                for flag in row["flags"]:
                    pills += f"<span class='pill'>⚠ {flag}</span>"
                st.markdown(pills, unsafe_allow_html=True)
            with right:
                st.metric("Salience", f"{row['salience']:.2f}")
            st.write(row["preview"])
            with st.expander(f"Full clause text — page {row['page']}"):
                st.write(row["text"])


def render_flags(analysis) -> None:
    st.subheader("Watch out for")
    st.caption(
        "Pattern-based flags, each linked to the sentence that triggered it. "
        "These are observations about the text, not legal conclusions."
    )

    flags = attention_flags(analysis)
    if not flags:
        st.success("No attention flags were detected in this document.")
        return

    for flag in flags:
        st.markdown(
            f"<div class='flag-card'>"
            f"<div class='flag-title'>⚠ {flag['label']}</div>"
            f"<div class='flag-meta'>{flag['clause_title']} · page {flag['page']}"
            f" · {flag['category']}</div>"
            f"<div style='margin-top:.5rem'>{flag['explanation']}</div>"
            f"<div class='evidence'><em>Evidence:</em> {flag['evidence']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_extractions(analysis) -> None:
    st.subheader("Extracted information")
    grouped = grouped_extractions(analysis)
    if not grouped:
        st.info("No structured information was extracted.")
        return

    if analysis.extraction and analysis.extraction.is_unfilled_template:
        st.warning(
            "Most fillable fields in this document are unfilled template "
            "placeholders such as `[Insert Date]`. Values below marked as "
            "placeholders are blanks in the source, not real values."
        )

    for type_, rows in grouped.items():
        label = EXTRACTION_LABELS.get(type_, type_.replace("_", " ").title())
        with st.expander(f"{label} ({len(rows)})", expanded=type_ == "party"):
            st.dataframe(rows, use_container_width=True, hide_index=True)


def render_ask(pipeline, analysis) -> None:
    st.subheader("Ask about this document")
    st.caption(
        "Questions are answered from clauses retrieved out of this document. "
        "The system does not answer from general legal knowledge."
    )

    cols = st.columns(len(EXAMPLE_QUESTIONS[:3]))
    for col, example in zip(cols, EXAMPLE_QUESTIONS[:3]):
        if col.button(example, use_container_width=True):
            st.session_state["pending_question"] = example

    question = st.text_input(
        "Your question",
        value=st.session_state.pop("pending_question", ""),
        placeholder="Ask a question about this document...",
    )
    task = st.radio(
        "Answer style",
        [GenerationTask.SIMPLE_EXPLANATION, GenerationTask.CLAUSE_EXPLANATION],
        format_func=lambda t: {
            GenerationTask.SIMPLE_EXPLANATION: "Plain-language explanation",
            GenerationTask.CLAUSE_EXPLANATION: "Explain the single best-matching clause",
        }[t],
        horizontal=True,
    )

    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving clauses and generating an answer..."):
            result = pipeline.ask(analysis, question, task=task)
        st.session_state["history"].insert(0, (question, result))

    for question_text, result in st.session_state["history"]:
        payload = answer_payload(result)
        with st.container(border=True):
            st.markdown(f"**Q: {question_text}**")

            if payload["status"] == "error":
                st.error("This question could not be answered. "
                         "The retrieval or generation step failed.")
                continue
            if payload["status"] == "empty":
                st.warning(payload["text"] or
                           "No relevant clause was found for this question.")
                continue

            st.markdown(f"<div class='answer-card'>{payload['text']}</div>",
                        unsafe_allow_html=True)
            st.caption(GROUNDING_CAVEAT)
            st.markdown("**Source clauses**")
            for source in payload["sources"]:
                suffix = " (abridged)" if source["abridged"] else ""
                with st.expander(
                    f"{source['citation']} — {source['category']}"
                    f" · similarity {source['score']:.3f}{suffix}"
                ):
                    st.write(source["text"])
                    if source["flags"]:
                        st.caption("Flags: " + ", ".join(source["flags"]))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("summary", None)

    st.markdown(
        f"<div class='doc-hero'><h1>⚖️ {APP_TITLE}</h1>"
        f"<p>Clause segmentation · Legal-BERT classification · calibrated "
        f"salience · evidence-linked flags · information extraction · "
        f"grounded question answering</p></div>",
        unsafe_allow_html=True,
    )
    st.info(DISCLAIMER_TEXT, icon="ℹ️")

    with st.spinner("Loading models (first run only)..."):
        pipeline = load_pipeline()

    if pipeline.classifier is None:
        st.error(
            "**Clause classifier not found.** Classification and salience are "
            f"unavailable.\n\nExpected the fine-tuned Legal-BERT model at "
            f"`{PATHS.classifier_dir}`. Train it with "
            "`notebooks/03_classification_colab.ipynb` and unzip the result "
            "there.\n\nSegmentation, flags, extraction and retrieval still work, "
            "and clauses will be shown as *Unclassified*.",
            icon="⚠️",
        )
    if pipeline.scaler is None:
        st.warning(
            "No `evaluation/results/temperature.json` found — confidence values "
            "are uncalibrated. Run `python scripts/calibrate.py`.",
            icon="⚠️",
        )

    with st.sidebar:
        st.header("Document")
        uploaded = st.file_uploader("Upload a legal PDF", type=["pdf"])

        if uploaded is not None and st.button("Analyze document", type="primary"):
            with st.spinner("Extracting, segmenting, classifying, extracting..."):
                analysis, error = analyze_upload(pipeline, uploaded)
            if error:
                st.session_state["analysis"] = None
                st.error(error)
            else:
                st.session_state["analysis"] = analysis
                st.session_state["history"] = []
                st.session_state["summary"] = None
                st.success(f"Analysed {analysis.n_clauses} clauses.")

        analysis = st.session_state.get("analysis")
        if analysis is not None:
            st.divider()
            stats = document_stats(analysis)
            st.caption(stats["Filename"])
            row1 = st.columns(2)
            row1[0].metric("Pages", stats["Pages"])
            row1[1].metric("Clauses", stats["Clauses"])
            row2 = st.columns(2)
            row2[0].metric("Classified", stats["Classified"])
            row2[1].metric("Flagged", stats["Flagged"])
            st.metric("Extracted items", stats["Extracted items"])

            if analysis.degraded:
                st.warning("Running without the classifier.", icon="⚠️")
            if analysis.stage_errors:
                st.warning("Some stages did not complete: "
                           + "; ".join(analysis.stage_errors))

            st.divider()
            st.download_button(
                "Download report (.txt)",
                data=build_report(analysis, st.session_state["history"]),
                file_name=f"analysis-{Path(stats['Filename']).stem}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.markdown("### Getting started")
        st.write(
            "Upload a PDF in the sidebar and select **Analyze document**. "
            "The pipeline segments the document into clauses, classifies each "
            "one, scores how much attention it warrants, flags notable "
            "patterns, extracts structured values, and lets you ask questions "
            "answered only from the document's own text."
        )
        st.caption(
            "Works best on text-based PDFs with clear clause structure. "
            "Scanned documents are not supported — there is no OCR."
        )
        return

    overview, clauses, flags, extracted, ask = st.tabs(
        ["Overview", "Key clauses", "Watch out for", "Extracted information",
         "Ask the document"]
    )
    with overview:
        render_overview(pipeline, analysis)
    with clauses:
        render_key_clauses(analysis)
    with flags:
        render_flags(analysis)
    with extracted:
        render_extractions(analysis)
    with ask:
        render_ask(pipeline, analysis)

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
