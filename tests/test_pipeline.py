"""Integration tests for the Phase 8 pipeline.

Legal-BERT, MPNet-QA and FLAN-T5 are all stubbed. The pipeline accepts every
component by injection, so the full orchestration path is exercised without a
single model download. Real model behaviour is measured by the phase-specific
evaluation scripts, not here.

What these tests verify is **wiring**: that a real PDF flows through every
stage, that the clause identifier survives each hop, and that the pipeline
degrades and fails safely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS  # noqa: E402
from src.generation import (  # noqa: E402
    EMPTY_CONTEXT_MESSAGE,
    GenerationPipeline,
    GenerationTask,
)
from src.importance import TemperatureScaler  # noqa: E402
from src.pipeline import (  # noqa: E402
    UNCLASSIFIED,
    DocumentAnalysis,
    LegalDocumentPipeline,
    QueryResult,
)
from src.retrieval import RetrievalResult  # noqa: E402

CONTRACT = PATHS.data_raw / "sample-independent-contractor-agreement.pdf"
POLICY = PATHS.data_raw / "synthetic-unnumbered-policy.pdf"

# Maps the contract's clause titles onto taxonomy categories, so the stub
# classifier behaves plausibly without loading Legal-BERT.
TITLE_TO_CATEGORY = {
    "Preamble": "Boilerplate & Administrative",
    "Scope of Work": "Boilerplate & Administrative",
    "Term and Termination": "Term & Termination",
    "Payment Terms": "Payment & Fees",
    "Independent Contractor Status": "Employment & Compensation",
    "Intellectual Property": "Intellectual Property",
    "Confidentiality": "Confidentiality & Publicity",
    "Compliance with Laws": "Compliance & Approvals",
    "Indemnification": "Liability & Indemnity",
    "Miscellaneous": "Boilerplate & Administrative",
}


class StubClassifier:
    """Deterministic stand-in for ClauseClassifier.

    Matches the real interface exactly: a probability vector per clause, not
    just an argmax, because Phase 4 consumes the distribution.
    """

    def __init__(self, texts_to_category: dict[str, str] | None = None) -> None:
        self.mapping = texts_to_category or {}
        self.calls = 0

    def predict(self, texts, batch_size: int = 16):
        self.calls += 1
        out = []
        for text in texts:
            category = next(
                (c for key, c in self.mapping.items() if key in text),
                "Boilerplate & Administrative",
            )
            out.append({
                "category": category,
                "confidence": 0.87,
                "probabilities": {category: 0.87, "Tax": 0.13},
            })
        return out


class StubRetriever:
    """Ranks clauses by naive token overlap. No embeddings involved."""

    def __init__(self, clauses) -> None:
        self.clauses = list(clauses)

    def search(self, query: str, top_k: int = 3):
        if not query.strip():
            return []
        terms = {t for t in query.lower().split() if len(t) > 3}
        scored = []
        for clause in self.clauses:
            haystack = f"{clause.title} {clause.text}".lower()
            overlap = sum(1 for t in terms if t in haystack)
            scored.append((overlap / max(len(terms), 1), clause))
        scored.sort(key=lambda pair: (-pair[0], pair[1].clause_id))
        return [
            RetrievalResult(
                rank=rank, clause_id=clause.clause_id, title=clause.title,
                score=float(score), text=clause.text, page=clause.page_start,
            )
            for rank, (score, clause) in enumerate(scored[:top_k], start=1)
        ]


class StubGenerator(GenerationPipeline):
    """GenerationPipeline with the model replaced, empty-path logic intact."""

    def __init__(self) -> None:
        super().__init__()
        self.generate_calls = 0

    def _generate(self, prompt: str, task: str) -> str:
        self.generate_calls += 1
        return f"[stub:{task}] generated from {len(prompt)} chars of prompt."


@pytest.fixture(scope="module")
def contract_available():
    if not CONTRACT.exists():
        pytest.skip("contract fixture not present")


@pytest.fixture
def generator():
    return StubGenerator()


@pytest.fixture
def pipeline(generator):
    return LegalDocumentPipeline(
        classifier=StubClassifier(TITLE_TO_CATEGORY_BY_TEXT),
        scaler=TemperatureScaler(1.1198),   # the measured Phase 4 temperature
        generator=generator,
        retriever_factory=StubRetriever,
        top_k=3,
    )


# Match on distinctive clause text rather than titles, since the stub sees text.
TITLE_TO_CATEGORY_BY_TEXT = {
    "sole and exclusive property": "Intellectual Property",
    "indemnify and hold harmless": "Liability & Indemnity",
    "written notice": "Term & Termination",
    "agrees to pay the Contractor": "Payment & Fees",
    "not to disclose": "Confidentiality & Publicity",
    "independent contractor and not an employee": "Employment & Compensation",
    "comply with all applicable": "Compliance & Approvals",
}


@pytest.fixture
def analysis(pipeline, contract_available):
    return pipeline.analyze_document(CONTRACT)


# ===========================================================================
# 1-2. A real PDF produces clauses
# ===========================================================================

def test_real_pdf_is_processed_end_to_end(analysis):
    assert isinstance(analysis, DocumentAnalysis)
    assert analysis.document.n_pages == 3
    assert analysis.n_clauses >= 9
    assert analysis.stage_errors == []
    assert analysis.seconds > 0


def test_expected_contract_clauses_are_present(analysis):
    titles = [c.title for c in analysis.clauses]
    for expected in ("Intellectual Property", "Term and Termination",
                     "Payment Terms", "Confidentiality", "Indemnification"):
        assert expected in titles


def test_unnumbered_policy_also_processes(pipeline):
    if not POLICY.exists():
        pytest.skip("policy fixture not present")
    result = pipeline.analyze_document(POLICY)
    assert result.n_clauses >= 6
    assert result.stage_errors == []


# ===========================================================================
# 3. Clause identifiers are stable through every stage
# ===========================================================================

def test_clause_ids_are_unique_and_stable(analysis):
    ids = analysis.clause_ids
    assert len(ids) == len(set(ids))
    assert [a.clause_id for a in analysis.analyses] == ids


def test_clause_id_survives_from_pdf_to_generated_response(pipeline, analysis):
    """The single property that makes an answer traceable to a page."""
    result = pipeline.ask(analysis, "Who owns the work produced by the contractor?")
    assert result.response is not None

    retrieved = result.retrieved_clause_ids
    assert retrieved, "nothing retrieved"

    # segmentation -> analysis -> extraction -> retrieval -> generation
    for clause_id in retrieved:
        assert clause_id in analysis.clause_ids
        matching = [a for a in analysis.analyses if a.clause_id == clause_id]
        assert len(matching) == 1
        for item in matching[0].extractions:
            assert item.clause_id == clause_id

    assert result.response.source_clause_ids == retrieved


def test_extraction_items_carry_clause_provenance(analysis):
    assert analysis.extraction is not None
    for item in analysis.extraction.items:
        assert item.clause_id in analysis.clause_ids
        assert item.clause_title
        assert item.clause_category is not None


# ===========================================================================
# 4-5. Categories, confidence and salience are attached
# ===========================================================================

def test_categories_are_attached_to_every_clause(analysis):
    assert not analysis.degraded
    assert analysis.n_classified == analysis.n_clauses
    for item in analysis.analyses:
        assert item.category and item.category != UNCLASSIFIED


def test_calibrated_confidence_and_salience_attached(analysis):
    for item in analysis.analyses:
        assert 0.0 <= item.calibrated_confidence <= 1.0
        assert 0.0 <= item.salience <= 1.0
        assert item.category_prior > 0


def test_calibration_softens_confidence_when_temperature_exceeds_one(analysis):
    """T = 1.1198 was measured in Phase 4; T > 1 must reduce confidence."""
    for item in analysis.analyses:
        assert item.calibrated_confidence <= item.raw_confidence + 1e-9


def test_boilerplate_ranks_below_substantive_clauses(analysis):
    by_title = {a.title: a for a in analysis.analyses}
    assert by_title["Intellectual Property"].salience > by_title["Miscellaneous"].salience


# ===========================================================================
# 6. Flags are attached where applicable
# ===========================================================================

def test_flags_attached_to_the_expected_clauses(analysis):
    flags = {a.title: {f.flag_id for f in a.flags} for a in analysis.analyses}
    assert "broad_ip_assignment" in flags["Intellectual Property"]
    assert "broad_indemnity" in flags["Indemnification"]
    assert "unilateral_termination" in flags["Term and Termination"]
    assert not flags["Scope of Work"]


def test_flag_evidence_spans_index_the_clause_text(analysis):
    for item in analysis.analyses:
        for hit in item.flags:
            assert item.text[hit.span_start:hit.span_end].strip() == hit.trigger_span


def test_low_salience_clause_can_still_carry_flags(analysis):
    """The Phase 4 finding, re-asserted at integration level."""
    misc = next(a for a in analysis.analyses if a.title == "Miscellaneous")
    assert {"choice_of_law", "arbitration"} & {f.flag_id for f in misc.flags}


# ===========================================================================
# 7. Extraction is attached
# ===========================================================================

def test_extractions_attached_and_aggregated(analysis):
    assert analysis.n_with_extractions > 0
    assert analysis.extraction is not None
    assert analysis.extraction.items
    names = {p["name"] for p in analysis.extraction.parties}
    assert {"Client", "Contractor"} <= names


def test_template_document_is_identified(analysis):
    """The sample contract is an unfilled template (Phase 5 finding)."""
    assert analysis.extraction.is_unfilled_template is True


# ===========================================================================
# 8-9. Retrieval and RAG context
# ===========================================================================

def test_retrieval_returns_valid_clause_ids(pipeline, analysis):
    context = pipeline.context_for(analysis, "how is the contractor paid?")
    assert context.retrieved
    for clause in context.retrieved:
        assert clause.clause_id in analysis.clause_ids
        assert clause.title
        assert clause.text


def test_rag_context_carries_phase_4_and_5_metadata(pipeline, analysis):
    context = pipeline.context_for(analysis, "who owns the intellectual property?")
    first = context.retrieved[0]
    assert first.category is not None
    assert first.salience is not None
    assert first.citation.startswith(f"Clause {first.clause_id}:")


def test_no_duplicate_clauses_in_context(pipeline, analysis):
    for query in ("payment terms", "termination notice", "confidential information"):
        context = pipeline.context_for(analysis, query)
        ids = [c.clause_id for c in context.retrieved]
        assert len(ids) == len(set(ids)), f"duplicate clause for {query!r}"


def test_context_respects_the_budget(pipeline, analysis):
    context = pipeline.context_for(analysis, "what are the payment terms?")
    assert context.within_budget
    assert context.token_estimate > 0


# ===========================================================================
# 10-11. Generation consumes only the context, and preserves provenance
# ===========================================================================

def test_generation_receives_only_the_rag_context(pipeline, analysis, generator):
    result = pipeline.ask(analysis, "what must be kept confidential?")
    assert generator.generate_calls == 1
    assert result.response is not None
    assert str(len(result.response.prompt)) in result.response.text
    # Every retrieved clause's text must appear in the prompt.
    for clause in result.context.retrieved:
        assert clause.text[:60] in result.response.prompt


def test_generated_response_preserves_sources(pipeline, analysis):
    result = pipeline.ask(analysis, "who owns the work?")
    response = result.response
    assert response.source_clause_ids == result.retrieved_clause_ids
    assert response.source_titles == result.retrieved_titles
    assert len(response.source_citations) == len(response.source_clause_ids)
    assert response.is_grounded


def test_clause_explanation_cites_a_single_clause(pipeline, analysis):
    result = pipeline.ask(
        analysis, "explain the indemnification clause",
        task=GenerationTask.CLAUSE_EXPLANATION,
    )
    assert len(result.response.source_clause_ids) == 1


def test_disclaimer_present_on_generated_output(pipeline, analysis):
    result = pipeline.ask(analysis, "what are the payment terms?")
    assert "not legal advice" in result.response.disclaimer


# ===========================================================================
# 12. Empty context must not invoke the model
# ===========================================================================

def test_empty_context_does_not_invoke_generation(pipeline, analysis, generator):
    class _Empty:
        def search(self, query, top_k=3):
            return []

    analysis.retriever = _Empty()
    result = pipeline.ask(analysis, "anything at all")

    assert generator.generate_calls == 0
    assert result.empty_context
    assert result.response.text == EMPTY_CONTEXT_MESSAGE
    assert result.is_grounded is False
    assert result.response.source_clause_ids == []


def test_blank_query_yields_empty_context(pipeline, analysis, generator):
    result = pipeline.ask(analysis, "   ")
    assert result.empty_context
    assert generator.generate_calls == 0


# ===========================================================================
# Failure isolation and degraded mode
# ===========================================================================

def test_missing_classifier_degrades_rather_than_crashing(generator):
    pipeline = LegalDocumentPipeline(
        classifier=None, generator=generator, retriever_factory=StubRetriever
    )
    if not CONTRACT.exists():
        pytest.skip("contract fixture not present")

    result = pipeline.analyze_document(CONTRACT)
    assert result.degraded is True
    assert result.n_clauses >= 9
    assert all(a.category == UNCLASSIFIED for a in result.analyses)
    # Rule-based stages are unaffected by the missing model.
    assert result.n_with_flags > 0
    assert result.n_with_extractions > 0


def test_degradation_is_recorded_not_silent(generator):
    if not CONTRACT.exists():
        pytest.skip("contract fixture not present")
    pipeline = LegalDocumentPipeline(
        classifier=None, generator=generator, retriever_factory=StubRetriever
    )
    assert pipeline.analyze_document(CONTRACT).summary()["degraded"] is True


def test_retriever_failure_is_captured_not_raised(generator):
    if not CONTRACT.exists():
        pytest.skip("contract fixture not present")

    def broken(clauses):
        raise RuntimeError("no embeddings available")

    pipeline = LegalDocumentPipeline(
        classifier=StubClassifier(), generator=generator, retriever_factory=broken
    )
    result = pipeline.analyze_document(CONTRACT)
    assert result.retriever is None
    assert any("retrieval" in e for e in result.stage_errors)
    # Analysis up to that point survives.
    assert result.n_clauses >= 9


def test_query_failure_is_captured_in_the_result(pipeline, analysis):
    analysis.retriever = None
    result = pipeline.ask(analysis, "who owns the work?")
    assert result.error is not None
    assert "No retriever" in result.error
    assert result.response is None


def test_missing_pdf_raises_clearly(pipeline):
    with pytest.raises(FileNotFoundError):
        pipeline.analyze_document(PATHS.data_raw / "nope.pdf")


# ===========================================================================
# Batch run and serialisation
# ===========================================================================

def test_run_answers_a_batch_of_queries(pipeline, contract_available):
    queries = [
        {"query": "who owns the work?"},
        {"query": "how much notice is required?"},
        {"query": "explain confidentiality",
         "task": GenerationTask.CLAUSE_EXPLANATION},
    ]
    analysis, results = pipeline.run(CONTRACT, queries)
    assert len(results) == 3
    assert all(isinstance(r, QueryResult) for r in results)
    assert all(r.error is None for r in results)


def test_results_serialise_to_json(pipeline, analysis):
    result = pipeline.ask(analysis, "who owns the work?")
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["retrieved_clauses"]
    assert payload["source_clause_ids"]
    assert payload["is_grounded"] is True
    assert "flags" in payload["retrieved_clauses"][0]


def test_document_summary_serialises(analysis):
    payload = json.loads(json.dumps(analysis.summary()))
    assert payload["clauses"] == analysis.n_clauses
    assert payload["degraded"] is False


# ===========================================================================
# Earlier phases must not have regressed
# ===========================================================================

def test_baseline_notebook_is_unchanged():
    """Phase 1's documented "before" system stays frozen."""
    import hashlib

    notebook = PATHS.notebooks / "01_baseline.ipynb"
    if not notebook.exists():
        pytest.skip("baseline notebook not present")
    digest = hashlib.md5(notebook.read_bytes()).hexdigest()
    assert digest == "7ba8d2cc297b9f61f65c7534d4ef7820"


def test_pipeline_reimplements_no_phase_logic():
    """The integration layer orchestrates; it must not duplicate analysis.

    If these appear in pipeline.py, logic has been copied out of the module
    that owns it and the two will drift.
    """
    import inspect

    import src.pipeline as pipeline_module

    source = inspect.getsource(pipeline_module)
    for forbidden in ("re.compile", "SentenceTransformer", "AutoModel",
                      "softmax", "TfidfVectorizer", "_FLAG_DEFINITIONS"):
        assert forbidden not in source, f"pipeline.py contains {forbidden}"
