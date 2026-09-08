"""Tests for the Streamlit application's data layer.

``app.py`` separates data preparation from rendering: the functions tested
here contain no Streamlit calls, so they run headlessly. The ``render_*``
functions are not tested — screenshot tests would be fragile and would verify
Streamlit rather than this project.

All models are stubbed, so this suite requires no downloads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("streamlit", reason="streamlit not installed")

import app as application  # noqa: E402

from src.config import PATHS  # noqa: E402
from src.generation import EMPTY_CONTEXT_MESSAGE  # noqa: E402
from src.pipeline import UNCLASSIFIED, LegalDocumentPipeline, QueryResult  # noqa: E402
from tests.test_pipeline import (  # noqa: E402
    TITLE_TO_CATEGORY_BY_TEXT,
    StubClassifier,
    StubGenerator,
    StubRetriever,
)
from src.importance import TemperatureScaler  # noqa: E402

CONTRACT = PATHS.data_raw / "sample-independent-contractor-agreement.pdf"


@pytest.fixture(scope="module")
def pipeline():
    return LegalDocumentPipeline(
        classifier=StubClassifier(TITLE_TO_CATEGORY_BY_TEXT),
        scaler=TemperatureScaler(1.1198),
        generator=StubGenerator(),
        retriever_factory=StubRetriever,
        top_k=3,
    )


@pytest.fixture(scope="module")
def analysis(pipeline):
    if not CONTRACT.exists():
        pytest.skip("contract fixture not present")
    return pipeline.analyze_document(CONTRACT)


class FakeUpload:
    """Mimics Streamlit's UploadedFile."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


# ===========================================================================
# Document statistics
# ===========================================================================

def test_document_stats_reports_real_numbers(analysis):
    stats = application.document_stats(analysis)
    assert stats["Filename"] == CONTRACT.name
    assert stats["Pages"] == 3
    assert stats["Clauses"] == analysis.n_clauses
    assert stats["Classified"] == analysis.n_classified
    assert stats["Extracted items"] > 0


def test_document_stats_survives_missing_extraction(analysis):
    analysis.extraction, saved = None, analysis.extraction
    try:
        assert application.document_stats(analysis)["Extracted items"] == 0
    finally:
        analysis.extraction = saved


# ===========================================================================
# Key clauses
# ===========================================================================

def test_key_clauses_are_sorted_by_salience(analysis):
    rows = application.key_clauses(analysis)
    scores = [r["salience"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_key_clauses_respect_the_limit(analysis):
    assert len(application.key_clauses(analysis, limit=3)) == 3


def test_key_clause_rows_carry_everything_the_ui_renders(analysis):
    row = application.key_clauses(analysis, limit=1)[0]
    for field in ("clause_id", "title", "category", "salience", "page",
                  "preview", "text", "flags"):
        assert field in row
    assert row["clause_id"] in analysis.clause_ids


def test_preview_is_truncated_but_full_text_is_kept(analysis):
    for row in application.key_clauses(analysis):
        assert len(row["preview"]) <= 183
        if len(row["text"]) > 180:
            assert row["preview"].endswith("...")
            assert len(row["text"]) > len(row["preview"])


# ===========================================================================
# Attention flags
# ===========================================================================

def test_flags_include_evidence_and_neutral_wording(analysis):
    flags = application.attention_flags(analysis)
    assert flags
    for flag in flags:
        assert flag["explanation"].startswith("This clause may be significant")
        assert flag["evidence"]
        assert flag["clause_id"] in analysis.clause_ids


def test_flag_wording_makes_no_legal_judgement(analysis):
    """The UI must not restate flags as verdicts."""
    forbidden = ("illegal", "unfair", "invalid", "you should", "definitely risky")
    for flag in application.attention_flags(analysis):
        lowered = flag["explanation"].lower()
        for word in forbidden:
            assert word not in lowered


def test_expected_contract_flags_surface(analysis):
    labels = {f["label"] for f in application.attention_flags(analysis)}
    assert "Broad IP assignment" in labels
    assert "Broad indemnity" in labels


def test_flags_are_independent_of_salience(analysis):
    """Phase 4: a low-salience clause can still carry important flags."""
    flagged_titles = {f["clause_title"] for f in application.attention_flags(analysis)}
    top_titles = {r["title"] for r in application.key_clauses(analysis, limit=3)}
    assert flagged_titles - top_titles, (
        "every flagged clause is already in the top 3; the independence of "
        "flags from salience is no longer being exercised"
    )


# ===========================================================================
# Extracted information
# ===========================================================================

def test_extractions_are_grouped_by_type(analysis):
    grouped = application.grouped_extractions(analysis)
    assert grouped
    assert "party" in grouped
    for rows in grouped.values():
        for row in rows:
            assert "Value" in row and "Evidence" in row


def test_placeholder_status_is_exposed(analysis):
    """The sample contract is an unfilled template; the UI must show that."""
    grouped = application.grouped_extractions(analysis)
    flat = [row for rows in grouped.values() for row in rows]
    assert any(row["Unfilled placeholder"] == "yes" for row in flat)


def test_party_group_is_ordered_first(analysis):
    keys = list(application.grouped_extractions(analysis))
    assert keys[0] == "party"


def test_grouped_extractions_handles_missing_extraction(analysis):
    analysis.extraction, saved = None, analysis.extraction
    try:
        assert application.grouped_extractions(analysis) == {}
    finally:
        analysis.extraction = saved


# ===========================================================================
# Answers and source mapping
# ===========================================================================

def test_answer_payload_maps_a_successful_query(pipeline, analysis):
    result = pipeline.ask(analysis, "Who owns the work produced by the contractor?")
    payload = application.answer_payload(result)
    assert payload["status"] == "ok"
    assert payload["text"]
    assert payload["sources"]


def test_source_clauses_map_back_to_real_clauses(pipeline, analysis):
    result = pipeline.ask(analysis, "how is the contractor paid?")
    for source in application.source_clauses(result):
        assert source["clause_id"] in analysis.clause_ids
        assert source["citation"].startswith(f"Clause {source['clause_id']}:")
        assert source["text"]


def test_answer_payload_handles_an_empty_context(pipeline, analysis):
    class _Empty:
        def search(self, query, top_k=3):
            return []

    saved, analysis.retriever = analysis.retriever, _Empty()
    try:
        payload = application.answer_payload(pipeline.ask(analysis, "anything"))
        assert payload["status"] == "empty"
        assert payload["text"] == EMPTY_CONTEXT_MESSAGE
        assert payload["sources"] == []
    finally:
        analysis.retriever = saved


def test_answer_payload_handles_a_failed_query(pipeline, analysis):
    saved, analysis.retriever = analysis.retriever, None
    try:
        payload = application.answer_payload(pipeline.ask(analysis, "anything"))
        assert payload["status"] == "error"
        assert payload["sources"] == []
    finally:
        analysis.retriever = saved


def test_answer_payload_handles_none():
    payload = application.answer_payload(None)
    assert payload["status"] == "none"
    assert payload["sources"] == []


def test_source_clauses_empty_for_none():
    assert application.source_clauses(None) == []
    assert application.source_clauses(QueryResult("q", "t", "d")) == []


# ===========================================================================
# Upload handling
# ===========================================================================

def test_non_pdf_upload_is_rejected(pipeline):
    analysis, error = application.analyze_upload(
        pipeline, FakeUpload("contract.docx", b"data")
    )
    assert analysis is None
    assert "PDF" in error


def test_empty_upload_is_rejected(pipeline):
    analysis, error = application.analyze_upload(pipeline, FakeUpload("x.pdf", b""))
    assert analysis is None
    assert "empty" in error.lower()


def test_missing_upload_is_rejected(pipeline):
    analysis, error = application.analyze_upload(pipeline, None)
    assert analysis is None
    assert error


def test_corrupt_pdf_gives_a_message_not_a_traceback(pipeline):
    analysis, error = application.analyze_upload(
        pipeline, FakeUpload("broken.pdf", b"this is not a PDF at all")
    )
    assert analysis is None
    assert "Traceback" not in error
    assert "scanned, encrypted or corrupted" in error


def test_valid_pdf_upload_is_analysed(pipeline):
    if not CONTRACT.exists():
        pytest.skip("contract fixture not present")
    analysis, error = application.analyze_upload(
        pipeline, FakeUpload(CONTRACT.name, CONTRACT.read_bytes())
    )
    assert error is None
    assert analysis is not None and analysis.n_clauses >= 9


# ===========================================================================
# Missing model artefacts
# ===========================================================================

def test_degraded_pipeline_marks_clauses_unclassified():
    if not CONTRACT.exists():
        pytest.skip("contract fixture not present")

    pipeline = LegalDocumentPipeline(
        classifier=None, generator=StubGenerator(), retriever_factory=StubRetriever
    )
    analysis = pipeline.analyze_document(CONTRACT)
    assert analysis.degraded is True
    assert application.document_stats(analysis)["Classified"] == 0
    # Rule-based sections still populate.
    assert application.attention_flags(analysis)
    assert application.grouped_extractions(analysis)
    assert all(r["category"] == UNCLASSIFIED
               for r in application.key_clauses(analysis))


# ===========================================================================
# Report generation
# ===========================================================================

def test_report_contains_every_section(analysis):
    report = application.build_report(analysis, [])
    for heading in ("KEY CLAUSES", "WATCH OUT FOR", "EXTRACTED INFORMATION"):
        assert heading in report
    assert "NOT LEGAL ADVICE" in report


def test_report_includes_question_history(pipeline, analysis):
    result = pipeline.ask(analysis, "who owns the work?")
    report = application.build_report(analysis, [("who owns the work?", result)])
    assert "QUESTIONS" in report
    assert "who owns the work?" in report
    assert "source:" in report


def test_report_is_plain_text_and_non_empty(analysis):
    report = application.build_report(analysis, [])
    assert isinstance(report, str)
    assert len(report) > 400


# ===========================================================================
# Disclaimer
# ===========================================================================

def test_disclaimer_is_present_and_appropriately_hedged():
    text = application.DISCLAIMER_TEXT.lower()
    assert "not legal advice" in text
    assert "academic prototype" in text
    for overclaim in ("accurate", "guarantee", "certified", "reliable"):
        assert overclaim not in text


def test_grounding_caveat_does_not_equate_citation_with_correctness():
    """Phase 7 measured a cited answer that was still wrong."""
    caveat = application.GROUNDING_CAVEAT.lower()
    assert "not necessarily correct" in caveat
