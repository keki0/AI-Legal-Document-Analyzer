"""Tests for the Phase 6 retrieval + RAG layer.

A deterministic stub retriever is used throughout. ``RAGContextBuilder``
depends on the :class:`~src.rag.Retriever` protocol rather than on
``ClauseRetriever`` directly, so the selection, budgeting and provenance logic
is testable without downloading embeddings and without a fixed random seed.

Real retrieval quality is measured separately by
``scripts/evaluate_retrieval.py``, which needs the MPNet-QA model.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS  # noqa: E402
from src.document_processing import Clause  # noqa: E402
from src.extraction import attach_extractions  # noqa: E402
from src.importance import TemperatureScaler, analyze_clauses  # noqa: E402
from src.rag import (  # noqa: E402
    DEFAULT_BUDGET_CHARS,
    RAGContext,
    RAGContextBuilder,
    RetrievedClause,
    estimate_tokens,
)
from src.retrieval import RetrievalResult  # noqa: E402


# ===========================================================================
# Fixtures
# ===========================================================================

CLAUSES = [
    Clause(1, "1", "Intellectual Property",
           "All work product shall be the sole and exclusive property of the "
           "Client, and the Contractor agrees to assign all rights, title, and "
           "interest to the Client.", 2, 2),
    Clause(2, "2", "Payment Terms",
           "The Client agrees to pay the Contractor $5,000 within 30 days of "
           "invoice.", 1, 1),
    Clause(3, "3", "Miscellaneous",
           "This Agreement shall be governed by the laws of the State of "
           "Delaware. Any disputes shall be resolved through binding "
           "arbitration.", 3, 3),
    Clause(4, "4", "Headings",
           "Headings are for convenience only and do not affect "
           "interpretation.", 3, 3),
]

PREDICTIONS = [
    {"category": "Intellectual Property", "confidence": 0.93,
     "probabilities": {"Intellectual Property": 0.93}},
    {"category": "Payment & Fees", "confidence": 0.90,
     "probabilities": {"Payment & Fees": 0.90}},
    {"category": "Boilerplate & Administrative", "confidence": 0.88,
     "probabilities": {"Boilerplate & Administrative": 0.88}},
    {"category": "Boilerplate & Administrative", "confidence": 0.95,
     "probabilities": {"Boilerplate & Administrative": 0.95}},
]


@dataclass
class StubRetriever:
    """Returns a fixed ranking. Satisfies the Retriever protocol."""

    order: list[int]
    scores: list[float] | None = None
    clauses: list[Clause] = None

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if not query.strip():
            return []
        source = {c.clause_id: c for c in (self.clauses or CLAUSES)}
        scores = self.scores or [0.9 - 0.1 * i for i in range(len(self.order))]
        out = []
        for rank, (clause_id, score) in enumerate(zip(self.order, scores), start=1):
            if rank > top_k:
                break
            clause = source[clause_id]
            out.append(RetrievalResult(
                rank=rank, clause_id=clause.clause_id, title=clause.title,
                score=score, text=clause.text, page=clause.page_start,
            ))
        return out


@pytest.fixture
def analyses():
    result = analyze_clauses(CLAUSES, PREDICTIONS, scaler=TemperatureScaler(1.4))
    attach_extractions(result)
    return result


@pytest.fixture
def retriever():
    return StubRetriever(order=[1, 2, 3, 4])


# ===========================================================================
# Token estimation
# ===========================================================================

def test_token_estimate_scales_with_length():
    assert estimate_tokens("") == 1
    short, long = estimate_tokens("a" * 40), estimate_tokens("a" * 400)
    assert long > short
    assert 8 <= short <= 12


def test_custom_token_counter_is_used(retriever):
    builder = RAGContextBuilder(retriever, token_counter=lambda t: 12345)
    assert builder.build("who owns the work?").token_estimate == 12345


# ===========================================================================
# Result structure and provenance
# ===========================================================================

def test_retrieved_clause_carries_full_provenance(retriever, analyses):
    context = RAGContextBuilder(retriever, analyses=analyses, top_k=2).build("q")
    first = context.retrieved[0]
    assert isinstance(first, RetrievedClause)
    assert first.clause_id == 1
    assert first.title == "Intellectual Property"
    assert first.page == 2
    assert first.rank == 1
    assert first.original_char_count == len(CLAUSES[0].text)
    assert first.citation == "Clause 1: Intellectual Property (p2)"


def test_metadata_joins_from_phase_4_and_5(retriever, analyses):
    context = RAGContextBuilder(retriever, analyses=analyses, top_k=3).build("q")
    ip = context.retrieved[0]
    assert ip.category == "Intellectual Property"
    assert ip.salience is not None and ip.salience > 0
    assert ip.calibrated_confidence is not None
    assert "Broad IP assignment" in ip.flag_labels          # Phase 4
    assert any(e.type == "party" for e in ip.extractions)   # Phase 5


def test_builder_works_without_any_analyses(retriever):
    """Metadata is optional; retrieval alone must still produce a context."""
    context = RAGContextBuilder(retriever, top_k=2).build("q")
    assert len(context.retrieved) == 2
    for clause in context.retrieved:
        assert clause.category is None
        assert clause.salience is None
        assert clause.flags == []


def test_context_serialises_to_json(retriever, analyses):
    import json

    context = RAGContextBuilder(retriever, analyses=analyses, top_k=2).build("q")
    payload = json.loads(json.dumps(context.to_dict()))
    assert payload["sources"]
    assert payload["retrieved"][0]["flag_labels"]


# ===========================================================================
# Ranking and selection
# ===========================================================================

def test_ranking_order_is_preserved_exactly(analyses):
    builder = RAGContextBuilder(StubRetriever(order=[3, 1, 2, 4]),
                                analyses=analyses, top_k=3)
    assert [c.clause_id for c in builder.build("q").retrieved] == [3, 1, 2]


def test_salience_never_reorders_semantic_ranking(analyses):
    """Headings has the highest classifier confidence and lowest salience.

    Neither may move it: retrieval order is semantic, full stop.
    """
    builder = RAGContextBuilder(StubRetriever(order=[4, 1]), analyses=analyses,
                                top_k=2)
    retrieved = builder.build("q").retrieved
    assert retrieved[0].clause_id == 4
    assert retrieved[0].salience < retrieved[1].salience


def test_scores_are_monotonically_non_increasing(retriever, analyses):
    context = RAGContextBuilder(retriever, analyses=analyses, top_k=4).build("q")
    scores = [c.score for c in context.retrieved]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_configurable(analyses):
    for k in (1, 2, 3, 4):
        builder = RAGContextBuilder(StubRetriever(order=[1, 2, 3, 4]),
                                    analyses=analyses, top_k=k,
                                    budget_chars=100_000)
        assert len(builder.build("q").retrieved) == k


def test_duplicate_clauses_are_removed(analyses):
    builder = RAGContextBuilder(StubRetriever(order=[1, 1, 2, 2]),
                                analyses=analyses, top_k=3)
    ids = [c.clause_id for c in builder.build("q").retrieved]
    assert ids == sorted(set(ids), key=ids.index)
    assert len(ids) == len(set(ids))


def test_selection_is_deterministic(retriever, analyses):
    builder = RAGContextBuilder(retriever, analyses=analyses, top_k=3)
    first = builder.build("who owns the work?")
    second = builder.build("who owns the work?")
    assert first.formatted_context == second.formatted_context
    assert [c.clause_id for c in first.retrieved] == [c.clause_id for c in second.retrieved]


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_top_k_rejected(retriever, bad):
    with pytest.raises(ValueError, match="top_k"):
        RAGContextBuilder(retriever, top_k=bad)


def test_invalid_budget_rejected(retriever):
    with pytest.raises(ValueError, match="budget_chars"):
        RAGContextBuilder(retriever, budget_chars=0)


def test_invalid_flag_ratio_rejected(retriever):
    with pytest.raises(ValueError, match="flag_score_ratio"):
        RAGContextBuilder(retriever, flag_score_ratio=1.5)


# ===========================================================================
# Flag-support rule
# ===========================================================================

def test_flag_rule_is_off_by_default(analyses):
    """It is a secondary rule and must be opted into."""
    builder = RAGContextBuilder(StubRetriever(order=[1, 2, 3]),
                                analyses=analyses, top_k=2)
    assert builder.include_flagged is False
    reasons = {c.selection_reason for c in builder.build("q").retrieved}
    assert reasons == {"semantic"}


def test_flag_rule_admits_a_relevant_flagged_clause(analyses):
    """Miscellaneous is low-salience but carries Choice of law + Arbitration.

    This is the Phase 4 finding the rule exists to cover.
    """
    builder = RAGContextBuilder(
        StubRetriever(order=[1, 2, 3, 4], scores=[0.90, 0.85, 0.80, 0.10]),
        analyses=analyses, top_k=2, include_flagged=True, flag_score_ratio=0.6,
    )
    retrieved = builder.build("how are disputes resolved?").retrieved
    misc = [c for c in retrieved if c.clause_id == 3]
    assert misc, "flagged Miscellaneous clause was not admitted"
    assert misc[0].selection_reason == "flag_supported"
    assert "Arbitration" in misc[0].flag_labels


def test_flag_rule_respects_the_relevance_floor(analyses):
    """Without the floor this would append arbitration to every question."""
    builder = RAGContextBuilder(
        StubRetriever(order=[1, 2, 3], scores=[0.90, 0.85, 0.05]),
        analyses=analyses, top_k=2, include_flagged=True, flag_score_ratio=0.6,
    )
    ids = [c.clause_id for c in builder.build("q").retrieved]
    assert 3 not in ids


def test_flag_rule_never_admits_unflagged_clauses(analyses):
    builder = RAGContextBuilder(
        StubRetriever(order=[1, 2, 4], scores=[0.90, 0.88, 0.86]),
        analyses=analyses, top_k=2, include_flagged=True,
    )
    # Clause 4 (Headings) scores above the floor but carries no flags.
    assert 4 not in [c.clause_id for c in builder.build("q").retrieved]


def test_flag_rule_does_not_duplicate_a_clause_already_selected(analyses):
    builder = RAGContextBuilder(
        StubRetriever(order=[3, 1, 2], scores=[0.90, 0.88, 0.86]),
        analyses=analyses, top_k=3, include_flagged=True,
    )
    ids = [c.clause_id for c in builder.build("q").retrieved]
    assert ids.count(3) == 1


# ===========================================================================
# Context budget
# ===========================================================================

def test_budget_drops_clauses_rather_than_cutting_them(analyses):
    builder = RAGContextBuilder(StubRetriever(order=[1, 2, 3]),
                                analyses=analyses, top_k=3, budget_chars=200)
    context = builder.build("q")
    assert context.dropped_clause_ids
    for clause in context.retrieved:
        if clause.included_fully:
            original = next(c for c in CLAUSES if c.clause_id == clause.clause_id)
            assert clause.text == original.text


def test_complete_clause_text_preserved_when_budget_allows(analyses):
    builder = RAGContextBuilder(StubRetriever(order=[1, 2, 3, 4]),
                                analyses=analyses, top_k=4, budget_chars=100_000)
    context = builder.build("q")
    assert context.dropped_clause_ids == []
    assert context.abridged_clause_ids == []
    for clause in context.retrieved:
        original = next(c for c in CLAUSES if c.clause_id == clause.clause_id)
        assert clause.text == original.text
        assert clause.included_fully


def test_top_hit_is_abridged_at_a_sentence_boundary(analyses):
    """Only the top hit may be abridged, and never mid-sentence."""
    builder = RAGContextBuilder(StubRetriever(order=[3, 1]),
                                analyses=analyses, top_k=2, budget_chars=80)
    context = builder.build("q")
    first = context.retrieved[0]
    assert first.included_fully is False
    assert first.clause_id in context.abridged_clause_ids
    assert first.text.endswith("[...]")
    assert first.original_char_count == len(CLAUSES[2].text)


def test_abridged_clause_keeps_its_provenance(analyses):
    builder = RAGContextBuilder(StubRetriever(order=[3]), analyses=analyses,
                                top_k=1, budget_chars=80)
    first = builder.build("q").retrieved[0]
    assert first.clause_id == 3
    assert first.title == "Miscellaneous"
    assert first.citation == "Clause 3: Miscellaneous (p3)"
    assert first.category == "Boilerplate & Administrative"


def test_abridgement_is_marked_in_the_formatted_context(analyses):
    builder = RAGContextBuilder(StubRetriever(order=[3]), analyses=analyses,
                                top_k=1, budget_chars=80)
    assert "(abridged)" in builder.build("q").formatted_context


def test_only_the_top_hit_is_ever_abridged(analyses):
    builder = RAGContextBuilder(StubRetriever(order=[2, 1, 3]),
                                analyses=analyses, top_k=3, budget_chars=120)
    context = builder.build("q")
    for clause in context.retrieved[1:]:
        assert clause.included_fully


def test_budget_accounting_is_reported(analyses):
    builder = RAGContextBuilder(StubRetriever(order=[1, 2, 3, 4]),
                                analyses=analyses, top_k=4, budget_chars=100_000)
    context = builder.build("q")
    assert context.budget_chars == 100_000
    assert context.within_budget
    assert context.char_count == len(context.formatted_context)
    assert context.token_estimate > 0


def test_default_budget_targets_flan_t5_input_limit():
    """1600 chars ~ 400 tokens, leaving headroom in FLAN-T5's 512."""
    assert DEFAULT_BUDGET_CHARS == 1600
    assert estimate_tokens("x" * DEFAULT_BUDGET_CHARS) < 512


# ===========================================================================
# Formatting and grounding
# ===========================================================================

def test_formatted_context_labels_every_clause_with_its_citation(retriever, analyses):
    context = RAGContextBuilder(retriever, analyses=analyses, top_k=3).build("q")
    for clause in context.retrieved:
        assert clause.citation in context.formatted_context
        assert clause.text in context.formatted_context


def test_formatted_context_includes_titles_and_categories(retriever, analyses):
    context = RAGContextBuilder(retriever, analyses=analyses, top_k=2).build("q")
    assert "Intellectual Property" in context.formatted_context
    assert "Category:" in context.formatted_context


def test_sources_match_retrieved_clauses_in_order(retriever, analyses):
    context = RAGContextBuilder(retriever, analyses=analyses, top_k=3).build("q")
    assert context.sources == [c.citation for c in context.retrieved]


# ===========================================================================
# Edge cases
# ===========================================================================

@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_empty_query_yields_an_empty_context(retriever, query):
    context = RAGContextBuilder(retriever).build(query)
    assert isinstance(context, RAGContext)
    assert context.is_empty
    assert context.formatted_context == ""
    assert context.within_budget


def test_empty_document_yields_an_empty_context():
    builder = RAGContextBuilder(StubRetriever(order=[], clauses=[]))
    context = builder.build("who owns the work?")
    assert context.is_empty
    assert context.sources == []


def test_single_clause_document():
    only = [CLAUSES[0]]
    builder = RAGContextBuilder(StubRetriever(order=[1], clauses=only), top_k=3)
    context = builder.build("who owns the work?")
    assert len(context.retrieved) == 1
    assert context.retrieved[0].clause_id == 1


def test_top_k_larger_than_available_clauses(analyses):
    builder = RAGContextBuilder(StubRetriever(order=[1, 2]), analyses=analyses,
                                top_k=10, budget_chars=100_000)
    assert len(builder.build("q").retrieved) == 2


def test_analyses_for_clauses_never_retrieved_are_ignored(analyses):
    builder = RAGContextBuilder(StubRetriever(order=[1]), analyses=analyses, top_k=1)
    assert len(builder.build("q").retrieved) == 1


# ===========================================================================
# Compatibility with Phase 1
# ===========================================================================

def test_clause_retriever_satisfies_the_retriever_protocol():
    """Phase 1 needs no modification to work with Phase 6."""
    import inspect

    from src.retrieval import ClauseRetriever

    signature = inspect.signature(ClauseRetriever.search)
    assert "query" in signature.parameters
    assert "top_k" in signature.parameters


def test_retrieval_result_still_carries_the_grounding_fields():
    """Guards the interface Phase 7 will depend on."""
    for field_name in ("rank", "clause_id", "title", "score", "text", "page"):
        assert field_name in RetrievalResult.__annotations__


def test_phase_4_and_5_outputs_are_not_mutated_by_context_building(analyses):
    before = [(a.salience, a.calibrated_confidence, len(a.flags), len(a.extractions))
              for a in analyses]
    RAGContextBuilder(StubRetriever(order=[1, 2, 3, 4]), analyses=analyses,
                      top_k=4, budget_chars=50).build("q")
    after = [(a.salience, a.calibrated_confidence, len(a.flags), len(a.extractions))
             for a in analyses]
    assert before == after


def test_abridgement_does_not_mutate_the_source_clause(analyses):
    original = CLAUSES[2].text
    RAGContextBuilder(StubRetriever(order=[3]), analyses=analyses,
                      top_k=1, budget_chars=80).build("q")
    assert CLAUSES[2].text == original


# ===========================================================================
# Evaluation metric arithmetic
# ===========================================================================

def _metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "hit_at_1": sum(r == 1 for r in ranks) / n,
        "hit_at_3": sum(r is not None and r <= 3 for r in ranks) / n,
        "hit_at_5": sum(r is not None and r <= 5 for r in ranks) / n,
        "mrr": sum(1.0 / r if r else 0.0 for r in ranks) / n,
    }


def test_hit_at_k_is_cumulative():
    m = _metrics([1, 2, 3, 4, None])
    assert m["hit_at_1"] == pytest.approx(0.2)
    assert m["hit_at_3"] == pytest.approx(0.6)
    assert m["hit_at_5"] == pytest.approx(0.8)
    assert m["hit_at_1"] <= m["hit_at_3"] <= m["hit_at_5"]


def test_mrr_matches_hand_computed_value():
    # 1/1 + 1/2 + 1/4 + 0 = 1.75, over 4 queries.
    assert _metrics([1, 2, 4, None])["mrr"] == pytest.approx(1.75 / 4)


def test_all_correct_and_all_missed_are_the_extremes():
    assert _metrics([1, 1, 1])["mrr"] == pytest.approx(1.0)
    assert _metrics([None, None])["mrr"] == 0.0
    assert _metrics([None, None])["hit_at_5"] == 0.0


# ===========================================================================
# The documented regression case (structure only)
# ===========================================================================

def test_regression_query_is_still_in_the_evaluation_set():
    """Phase 6 does not re-measure this; scripts/evaluate_retrieval.py does.

    No claim is made here about whether the query succeeds -- that requires
    the real MPNet-QA model. This only guards the query's continued presence.
    """
    import json

    path = PATHS.data_eval / "retrieval_queries.json"
    if not path.exists():
        pytest.skip("query file not present")
    queries = json.loads(path.read_text(encoding="utf-8"))["queries"]
    regression = [q for q in queries if q["type"] == "regression"]
    assert len(regression) == 1
    assert regression[0]["query"] == "Who owns the work produced by the contractor?"
    assert regression[0]["expected"] == "Intellectual Property"


def test_context_for_the_regression_query_grounds_on_the_ip_clause(analyses):
    """Given correct retrieval, the context must cite the IP clause.

    Uses a stub that returns the correct ranking, so this tests context
    construction, not retrieval accuracy.
    """
    builder = RAGContextBuilder(StubRetriever(order=[1, 2]), analyses=analyses,
                                top_k=2)
    context = builder.build("Who owns the work produced by the contractor?")
    assert context.retrieved[0].title == "Intellectual Property"
    assert "Clause 1: Intellectual Property (p2)" in context.sources
    assert "sole and exclusive property" in context.formatted_context
