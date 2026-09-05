"""Tests for Phase 1 document processing.

Model-free: nothing here downloads weights, so the suite runs offline and
fast. Retrieval quality is verified separately by scripts/retrieval_spike.py,
which needs Sentence-Transformers models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS  # noqa: E402
from src.document_processing import (  # noqa: E402
    clauses_to_chunks,
    extract_document,
    load_and_segment,
    segment_clauses,
)

CONTRACT = PATHS.data_raw / "sample-independent-contractor-agreement.pdf"
POLICY = PATHS.data_raw / "synthetic-unnumbered-policy.pdf"

EXPECTED_CONTRACT_TITLES = [
    "Scope of Work",
    "Term and Termination",
    "Payment Terms",
    "Independent Contractor Status",
    "Intellectual Property",
    "Confidentiality",
    "Compliance with Laws",
    "Indemnification",
    "Miscellaneous",
]


@pytest.fixture(scope="module")
def contract():
    return load_and_segment(CONTRACT)


@pytest.fixture(scope="module")
def policy():
    if not POLICY.exists():
        pytest.skip("unnumbered policy fixture not present")
    return load_and_segment(POLICY)


# --- extraction ------------------------------------------------------------

def test_extraction_reports_expected_page_count(contract):
    document, _ = contract
    assert document.n_pages == 3


def test_body_font_size_is_detected(contract):
    document, _ = contract
    assert document.body_size == pytest.approx(11.0)


def test_repeated_footer_is_removed(contract):
    """The page footer must not leak into clause text."""
    document, clauses = contract
    assert any("informational purposes" in line for line in document.removed_boilerplate)
    for clause in clauses:
        assert "Seek professional advice" not in clause.text


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_document(PATHS.data_raw / "does-not-exist.pdf")


# --- segmentation ----------------------------------------------------------

def test_all_nine_numbered_clauses_found(contract):
    _, clauses = contract
    titles = [clause.title for clause in clauses]
    for expected in EXPECTED_CONTRACT_TITLES:
        assert expected in titles, f"missing clause: {expected}"


def test_clause_numbers_are_sequential(contract):
    _, clauses = contract
    numbers = [clause.number for clause in clauses if clause.number]
    assert numbers == [str(n) for n in range(1, 10)]


def test_front_matter_merged_into_single_preamble(contract):
    _, clauses = contract
    assert sum(1 for c in clauses if c.title == "Preamble") == 1
    assert clauses[0].title == "Preamble"


def test_heading_is_not_duplicated_in_body(contract):
    """The baseline included the heading line inside clause text; this does not."""
    _, clauses = contract
    ip = next(c for c in clauses if c.title == "Intellectual Property")
    assert not ip.text.startswith("5.")
    assert not ip.text.startswith("Intellectual Property")


def test_every_clause_has_substance(contract):
    _, clauses = contract
    assert all(clause.word_count >= 3 for clause in clauses)


def test_unnumbered_policy_is_segmented(policy):
    """The case the baseline regex cannot handle at all."""
    _, clauses = policy
    titles = [clause.title for clause in clauses]
    assert len(clauses) >= 6
    assert "Automatic Renewal of Your Subscription" in titles
    assert "Sharing With Third Parties" in titles
    # No numbering anywhere in this document.
    assert all(clause.number is None for clause in clauses)


def test_segmentation_never_returns_empty(contract):
    document, _ = contract
    assert len(segment_clauses(document, min_clauses=99)) > 0


# --- retrieval representation ---------------------------------------------

def test_title_augmentation_changes_the_chunk(contract):
    _, clauses = contract
    plain = clauses_to_chunks(clauses, title_augmented=False)
    titled = clauses_to_chunks(clauses, title_augmented=True)
    assert len(plain) == len(titled) == len(clauses)
    assert all(t.startswith(c.title) for t, c in zip(titled, clauses))


def test_ownership_vocabulary_trap_is_present(contract):
    """Pin the documented cause of the baseline retrieval failure.

    The query asks who *owns* the work. The word "own" appears nowhere in the
    Intellectual Property clause (which says "sole and exclusive property" and
    "assign all rights"), but it does appear in Independent Contractor Status
    ("their own work schedule") in an unrelated sense. If this ever stops
    being true, the retrieval spike is no longer testing what it claims to.
    """
    import re

    _, clauses = contract
    ip = next(c for c in clauses if c.title == "Intellectual Property")
    distractor = next(c for c in clauses if c.title == "Independent Contractor Status")

    assert not re.search(r"\bown\w*", ip.text.lower())
    assert re.search(r"\bown\b", distractor.text.lower())
    # Both share the query's other content words, so body-only embeddings have
    # little to separate them.
    for token in ("contractor", "work"):
        assert token in ip.text.lower()
        assert token in distractor.text.lower()


def test_title_augmentation_supplies_the_missing_topic(contract):
    """Title augmentation is what puts a topical signal on the IP clause."""
    _, clauses = contract
    ip = next(c for c in clauses if c.title == "Intellectual Property")
    assert ip.as_chunk().lower().startswith("intellectual property")
    assert "intellectual" not in ip.text.lower()


# --- taxonomy and evaluation set (Phase 2) --------------------------------

def test_taxonomy_covers_all_100_ledgar_labels():
    """Every LEDGAR label maps exactly once; no invented names."""
    from src.dataset import load_taxonomy

    taxonomy = load_taxonomy()
    assert len(taxonomy.label_to_category) == 100
    assert 10 <= taxonomy.n_categories <= 15
    # Sentinel labels that must land where a viva examiner would expect.
    assert taxonomy.label_to_category["Intellectual Property"] == "Intellectual Property"
    assert taxonomy.label_to_category["Counterparts"] == "Boilerplate & Administrative"
    assert taxonomy.label_to_category["Arbitration"] == "Governing Law & Dispute Resolution"


def test_retrieval_queries_reference_real_clauses():
    """Guards the eval set against silent drift in the segmenter."""
    import json

    path = PATHS.data_eval / "retrieval_queries.json"
    if not path.exists():
        pytest.skip("retrieval_queries.json not present")

    queries = json.loads(path.read_text(encoding="utf-8"))["queries"]
    assert len(queries) >= 25

    cache: dict[str, set[str]] = {}
    for item in queries:
        name = item["document"]
        if name not in cache:
            doc_path = PATHS.data_raw / name
            if not doc_path.exists():
                pytest.skip(f"fixture {name} not present")
            _, clauses = load_and_segment(doc_path)
            cache[name] = {clause.title for clause in clauses}
        assert item["expected"] in cache[name], (
            f"query {item['query']!r} expects clause {item['expected']!r} "
            f"which the segmenter does not produce"
        )


def test_regression_query_is_present_in_eval_set():
    import json

    path = PATHS.data_eval / "retrieval_queries.json"
    if not path.exists():
        pytest.skip("retrieval_queries.json not present")
    queries = json.loads(path.read_text(encoding="utf-8"))["queries"]
    regression = [q for q in queries if q["type"] == "regression"]
    assert len(regression) == 1
    assert regression[0]["expected"] == "Intellectual Property"
