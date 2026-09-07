"""Tests for the Phase 5 information extraction layer.

Rule-based and deterministic, so these are exact-behaviour tests rather than
statistical ones.

**These are qualitative tests.** There is no expert-labelled extraction
dataset for this project, so no precision, recall or F1 is claimed anywhere.
What is verified is that each rule fires on representative drafting, stays
silent on constructions designed to trip it, and preserves evidence exactly.
Recall against real-world drafting variety is unmeasured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS  # noqa: E402
from src.document_processing import Clause, load_and_segment  # noqa: E402
from src.extraction import (  # noqa: E402
    DocumentExtraction,
    ExtractedItem,
    ExtractionType,
    InformationExtractor,
    _normalise_date,
    _normalise_duration,
    _word_to_number,
    attach_extractions,
    extract_from_clauses,
)
from src.importance import TemperatureScaler, analyze_clauses  # noqa: E402

CONTRACT = PATHS.data_raw / "sample-independent-contractor-agreement.pdf"


@pytest.fixture(scope="module")
def extractor():
    return InformationExtractor()


def types_in(items) -> set[str]:
    return {item.type for item in items}


def values_of(items, type_) -> list[str]:
    return [i.value for i in items if i.type == type_]


# ===========================================================================
# Normalisation primitives
# ===========================================================================

@pytest.mark.parametrize("word,expected", [
    ("one", 1), ("twelve", 12), ("thirty", 30), ("ninety", 90),
    ("twenty-four", 24), ("sixty-five", 65), ("forty-two", 42),
])
def test_number_words_convert(word, expected):
    assert _word_to_number(word) == expected


@pytest.mark.parametrize("word", ["banana", "", "twenty-banana", "hundred-two"])
def test_unknown_number_words_return_none(word):
    assert _word_to_number(word) is None


@pytest.mark.parametrize("amount,unit,expected", [
    ("30", "days", "P30D"), ("12", "months", "P12M"), ("1", "year", "P1Y"),
    ("thirty", "days", "P30D"), ("twenty-four", "months", "P24M"),
    ("2", "weeks", "P2W"), ("1,000", "days", "P1000D"),
])
def test_duration_normalises_to_iso8601(amount, unit, expected):
    assert _normalise_duration(amount, unit) == expected


def test_duration_normalisation_rejects_unknown_units():
    assert _normalise_duration("30", "fortnights") is None
    assert _normalise_duration("banana", "days") is None


@pytest.mark.parametrize("raw,expected", [
    ("January 15, 2026", "2026-01-15"),
    ("Jan 15, 2026", "2026-01-15"),
    ("15 January 2026", "2026-01-15"),
    ("1st March 2026", "2026-03-01"),
    ("2026-01-15", "2026-01-15"),
    ("01/15/2026", "2026-01-15"),
])
def test_date_normalises_to_iso(raw, expected):
    assert _normalise_date(raw) == expected


@pytest.mark.parametrize("raw", [
    "February 31, 2026",   # impossible calendar date
    "Bananuary 5, 2026",   # not a month
    "sometime next year",
    "",
])
def test_unparseable_dates_return_none_rather_than_guessing(raw):
    assert _normalise_date(raw) is None


# ===========================================================================
# Dates
# ===========================================================================

def test_real_date_extracted_and_normalised(extractor):
    items = extractor.extract("This Agreement is effective as of January 15, 2026.")
    dates = [i for i in items if i.type == ExtractionType.DATE]
    assert len(dates) == 1
    assert dates[0].value == "January 15, 2026"
    assert dates[0].normalized_value == "2026-01-15"
    assert dates[0].is_placeholder is False


def test_date_placeholder_marked_not_valued(extractor):
    items = extractor.extract("This Agreement is entered into as of [Insert Date].")
    dates = [i for i in items if i.type == ExtractionType.DATE]
    assert dates
    assert dates[0].is_placeholder is True
    assert dates[0].normalized_value is None


def test_impossible_date_is_not_extracted(extractor):
    """A calendar-invalid date must be dropped, not normalised to a guess."""
    items = extractor.extract("The term begins on February 31, 2026.")
    assert not [i for i in items if i.type == ExtractionType.DATE]


# ===========================================================================
# Durations
# ===========================================================================

@pytest.mark.parametrize("text,expected_norm", [
    ("either party may cancel within 30 days", "P30D"),
    ("a period of twelve months", "P12M"),
    ("retained for twenty-four months", "P24M"),
    ("a term of one year", "P1Y"),
    ("within 10 business days", "P10D"),
])
def test_durations_extracted_with_normalisation(extractor, text, expected_norm):
    items = [i for i in extractor.extract(text) if i.type == ExtractionType.DURATION]
    assert items
    assert expected_norm in {i.normalized_value for i in items}


def test_duration_requires_a_cardinal_not_just_a_unit(extractor):
    """Guards a real false positive found on the sample contract.

    Its milestone table flattens to "Rough Draft Month [Date], [Year]", which
    a pattern of the form ``\\w+ months?`` matched as a duration.
    """
    items = extractor.extract("Rough Draft Month and Client Edits Month follow.")
    assert not [i for i in items if i.type == ExtractionType.DURATION]


def test_bracketed_duration_alternatives_marked_placeholder(extractor):
    items = [i for i in extractor.extract("terminated with [7/14] days' written notice")
             if i.type == ExtractionType.DURATION]
    assert items
    assert items[0].is_placeholder is True


# ===========================================================================
# Money
# ===========================================================================

@pytest.mark.parametrize("text,expected", [
    ("a fee of $5,000 per month", "$5,000"),
    ("payment of USD 10,000", "USD 10,000"),
    ("the sum of INR 50,000", "INR 50,000"),
    ("a charge of £250.50", "£250.50"),
    ("a deposit of Rs. 20,000", "Rs. 20,000"),
])
def test_currency_amounts_extracted(extractor, text, expected):
    assert expected in values_of(extractor.extract(text), ExtractionType.MONEY)


def test_percentage_extracted_as_financial_term(extractor):
    items = [i for i in extractor.extract("a late fee of 1.5% per month")
             if i.type == ExtractionType.MONEY]
    assert items
    assert items[0].value == "1.5%"


def test_money_placeholders_marked(extractor):
    items = [i for i in extractor.extract("Rate: $[Rate] per hour, then $X and $XX")
             if i.type == ExtractionType.MONEY]
    assert items
    assert all(i.is_placeholder for i in items)


def test_no_money_extracted_from_plain_numbers(extractor):
    """Bare integers must not be read as amounts."""
    items = extractor.extract("Section 5 refers to paragraph 12 and clause 300.")
    assert not [i for i in items if i.type == ExtractionType.MONEY]


# ===========================================================================
# Notice periods
# ===========================================================================

@pytest.mark.parametrize("text", [
    "terminated by either party with 30 days' written notice",
    "upon at least 60 days prior written notice to the other party",
    "the Client shall give notice of not less than fourteen days",
])
def test_notice_periods_detected(extractor, text):
    items = [i for i in extractor.extract(text)
             if i.type == ExtractionType.NOTICE_PERIOD]
    assert items, f"no notice period found in {text!r}"


def test_duration_without_notice_context_is_not_a_notice_period(extractor):
    """Proximity to 'notice' is the whole signal; without it, nothing fires."""
    items = extractor.extract("The Contractor shall complete the work within 30 days.")
    assert ExtractionType.DURATION in types_in(items)
    assert ExtractionType.NOTICE_PERIOD not in types_in(items)


def test_notice_period_normalised(extractor):
    items = [i for i in extractor.extract("with 30 days' written notice")
             if i.type == ExtractionType.NOTICE_PERIOD]
    assert items[0].normalized_value == "P30D"


# ===========================================================================
# Renewal / term
# ===========================================================================

@pytest.mark.parametrize("text,subtype", [
    ("This subscription will automatically renew each year.", "automatic_renewal"),
    ("The initial term shall be twelve months from the Effective Date.", "initial_term"),
    ("followed by successive renewal terms of one year each", "renewal_period"),
    ("This Agreement expires on the third anniversary.", "expiration"),
])
def test_renewal_subtypes_detected(extractor, text, subtype):
    items = [i for i in extractor.extract(text)
             if i.type == ExtractionType.RENEWAL_TERM]
    assert subtype in {i.normalized_value for i in items}


def test_no_renewal_language_yields_no_renewal_items(extractor):
    items = extractor.extract("The Contractor shall provide website design services.")
    assert ExtractionType.RENEWAL_TERM not in types_in(items)


# ===========================================================================
# Obligations
# ===========================================================================

@pytest.mark.parametrize("text,kind", [
    ("The Contractor shall deliver all services in writing.", "requirement"),
    ("The Contractor must comply with all applicable laws.", "requirement"),
    ("The Contractor agrees to assign all rights to the Client.", "requirement"),
    ("The Contractor may not disclose confidential information.", "prohibition"),
    ("The Client shall not be liable for indirect damages.", "prohibition"),
])
def test_obligations_detected_with_modality(extractor, text, kind):
    items = [i for i in extractor.extract(text)
             if i.type == ExtractionType.OBLIGATION]
    assert items
    assert any(kind in (i.normalized_value or "") for i in items)


def test_prohibition_ranked_above_plain_requirement(extractor):
    prohibition = [i for i in extractor.extract("The Contractor may not sublicense.")
                   if i.type == ExtractionType.OBLIGATION][0]
    requirement = [i for i in extractor.extract("The Contractor will deliver files.")
                   if i.type == ExtractionType.OBLIGATION][0]
    assert prohibition.confidence > requirement.confidence


def test_purely_descriptive_sentence_has_no_obligation(extractor):
    items = extractor.extract("This document represents the entire agreement.")
    assert ExtractionType.OBLIGATION not in types_in(items)


def test_obligation_evidence_is_the_whole_sentence(extractor):
    text = ("The parties met. The Contractor shall deliver the work. "
            "Nothing else applies.")
    item = [i for i in extractor.extract(text)
            if i.type == ExtractionType.OBLIGATION][0]
    assert "Contractor shall deliver" in item.evidence
    assert "Nothing else applies" not in item.evidence


# ===========================================================================
# Parties
# ===========================================================================

def test_quoted_definition_is_the_strongest_party_signal(extractor):
    items = [i for i in extractor.extract(
        'between Acme Corp (the "Client") and Jane Doe (the "Contractor").')
        if i.type == ExtractionType.PARTY]
    names = {i.normalized_value for i in items}
    assert {"Client", "Contractor"} <= names
    quoted = [i for i in items if i.method == "regex:quoted_definition"]
    assert quoted and all(i.confidence >= 0.9 for i in quoted)


def test_role_lexicon_finds_unquoted_parties(extractor):
    items = [i for i in extractor.extract(
        "The Employer shall pay the Employee monthly.")
        if i.type == ExtractionType.PARTY]
    assert {"Employer", "Employee"} <= {i.normalized_value for i in items}


def test_party_placeholders_marked(extractor):
    items = [i for i in extractor.extract("between [Company Name] and [Contractor Name]")
             if i.type == ExtractionType.PARTY]
    assert items
    assert all(i.is_placeholder for i in items)


def test_parties_deduplicated_within_a_clause(extractor):
    items = [i for i in extractor.extract(
        "The Client shall pay. The Client shall also review. The Client agrees.")
        if i.type == ExtractionType.PARTY]
    assert len(items) == 1


def test_agreement_is_not_treated_as_a_party(extractor):
    items = extractor.extract('This Independent Contractor Agreement ("Agreement") is made.')
    assert "Agreement" not in {i.normalized_value
                               for i in items if i.type == ExtractionType.PARTY}


# ===========================================================================
# Evidence and provenance
# ===========================================================================

def test_spans_reproduce_the_extracted_value_exactly(extractor):
    text = ("This Agreement dated January 15, 2026 between Acme Corp "
            '(the "Client") and Jane Doe. The Client shall pay $5,000 within '
            "30 days' written notice.")
    for item in extractor.extract(text):
        if item.type == ExtractionType.OBLIGATION:
            continue  # obligations span a whole sentence
        assert text[item.span_start:item.span_end] == item.value, (
            f"span mismatch for {item.type}: {item.value!r}"
        )


def test_every_item_carries_evidence_and_method(extractor):
    text = "The Client shall pay $5,000 within 30 days' written notice."
    items = extractor.extract(text)
    assert items
    for item in items:
        assert item.evidence, f"{item.type} has no evidence"
        assert item.evidence in text
        assert item.method
        assert 0.0 <= item.confidence <= 1.0


def test_clause_provenance_attached(extractor):
    clause = Clause(7, "7", "Payment Terms", "The Client shall pay $5,000.", 2, 2)
    items = extractor.extract_from_clause(clause)
    assert items
    for item in items:
        assert item.clause_id == 7
        assert item.clause_title == "Payment Terms"


def test_items_serialise_to_json():
    import json

    clause = Clause(1, "1", "Payment", "Pay $5,000 within 30 days.", 1, 1)
    doc = extract_from_clauses([clause])
    json.dumps(doc.to_dict())  # must be safe for the API and UI


# ===========================================================================
# Edge cases
# ===========================================================================

@pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
def test_empty_text_yields_nothing(extractor, text):
    assert extractor.extract(text) == []


def test_clause_with_no_extractable_content(extractor):
    assert extractor.extract("Headings are for convenience only.") == []


def test_multiple_entity_types_in_one_clause(extractor):
    text = ("The Client shall pay the Contractor $10,000 on January 15, 2026, "
            "and may terminate with 30 days' written notice.")
    found = types_in(extractor.extract(text))
    assert {ExtractionType.PARTY, ExtractionType.MONEY, ExtractionType.DATE,
            ExtractionType.NOTICE_PERIOD, ExtractionType.OBLIGATION} <= found


def test_extractor_can_be_restricted_to_selected_types():
    limited = InformationExtractor(types=[ExtractionType.MONEY])
    items = limited.extract("The Client shall pay $5,000 within 30 days.")
    assert types_in(items) == {ExtractionType.MONEY}


def test_results_are_ordered_by_position(extractor):
    items = extractor.extract("Pay $5,000 by January 15, 2026 within 30 days.")
    starts = [i.span_start for i in items]
    assert starts == sorted(starts)


# ===========================================================================
# Document-level aggregation
# ===========================================================================

def test_parties_aggregated_and_ranked_by_mentions():
    clauses = [
        Clause(1, "1", "Preamble", 'Acme Corp (the "Client") and Jane Doe.', 1, 1),
        Clause(2, "2", "Payment", "The Client shall pay the Contractor.", 1, 1),
        Clause(3, "3", "Scope", "The Client reviews the work.", 1, 1),
    ]
    parties = extract_from_clauses(clauses).parties
    assert parties[0]["name"] == "Client"
    assert parties[0]["mentions"] == 3


def test_placeholder_ratio_uses_only_fillable_types():
    """Regression: obligations and role words must not dilute the ratio.

    Measured over all items, the sample contract scored under 30% despite
    having placeholders in 100% of its dates, amounts and periods.
    """
    clauses = [Clause(1, "1", "Terms",
                      "The Client shall pay $[Rate] on [Insert Date] with "
                      "[7/14] days' written notice, and the Contractor "
                      "agrees to comply and must deliver on time.", 1, 1)]
    doc = extract_from_clauses(clauses)
    assert doc.placeholder_ratio == 1.0
    assert doc.is_unfilled_template is True


def test_filled_document_is_not_flagged_as_a_template():
    clauses = [Clause(1, "1", "Terms",
                      "The Client shall pay $5,000 on January 15, 2026 with "
                      "30 days' written notice.", 1, 1)]
    doc = extract_from_clauses(clauses)
    assert doc.placeholder_ratio == 0.0
    assert doc.is_unfilled_template is False


def test_empty_document_extraction_is_safe():
    doc = DocumentExtraction([])
    assert doc.parties == []
    assert doc.placeholders == []
    assert doc.placeholder_ratio == 0.0
    assert doc.is_unfilled_template is False
    assert doc.summary()["total_items"] == 0


# ===========================================================================
# Integration with Phase 4
# ===========================================================================

def test_attach_extractions_populates_clause_analysis():
    clauses = [
        Clause(1, "1", "Payment Terms",
               "The Client shall pay $5,000 within 30 days' written notice.", 1, 1),
        Clause(2, "2", "Headings", "Headings are for convenience only.", 1, 1),
    ]
    predictions = [
        {"category": "Payment & Fees", "confidence": 0.9,
         "probabilities": {"Payment & Fees": 0.9}},
        {"category": "Boilerplate & Administrative", "confidence": 0.9,
         "probabilities": {"Boilerplate & Administrative": 0.9}},
    ]
    analyses = analyze_clauses(clauses, predictions, scaler=TemperatureScaler(1.5))
    doc = attach_extractions(analyses)

    assert analyses[0].extractions
    assert analyses[1].extractions == []
    assert doc.summary()["total_items"] == len(analyses[0].extractions)


def test_clause_category_propagates_from_phase_4():
    """Items must carry the Legal-BERT category, not re-derive it."""
    clauses = [Clause(1, "1", "Payment Terms", "The Client shall pay $5,000.", 1, 1)]
    predictions = [{"category": "Payment & Fees", "confidence": 0.9,
                    "probabilities": {"Payment & Fees": 0.9}}]
    analyses = analyze_clauses(clauses, predictions)
    attach_extractions(analyses)
    assert all(i.clause_category == "Payment & Fees" for i in analyses[0].extractions)


def test_attaching_extractions_does_not_disturb_phase_4_outputs():
    """Salience, flags and calibration must be untouched by Phase 5."""
    clauses = [Clause(1, "1", "Intellectual Property",
                      "The Contractor agrees to assign all rights, title, and "
                      "interest to the Client within 30 days.", 1, 1)]
    predictions = [{"category": "Intellectual Property", "confidence": 0.9,
                    "probabilities": {"Intellectual Property": 0.9}}]
    analyses = analyze_clauses(clauses, predictions, scaler=TemperatureScaler(1.5))

    before = (analyses[0].salience, analyses[0].calibrated_confidence,
              [f.flag_id for f in analyses[0].flags])
    attach_extractions(analyses)
    after = (analyses[0].salience, analyses[0].calibrated_confidence,
             [f.flag_id for f in analyses[0].flags])
    assert before == after
    assert "broad_ip_assignment" in after[2]


# ===========================================================================
# End-to-end on the real document (qualitative)
# ===========================================================================

def test_sample_contract_is_identified_as_an_unfilled_template():
    """Qualitative check against the real smoke-test document."""
    if not CONTRACT.exists():
        pytest.skip("contract fixture not present")

    _, clauses = load_and_segment(CONTRACT)
    doc = extract_from_clauses(clauses)

    assert doc.is_unfilled_template is True
    assert doc.placeholder_ratio == 1.0
    names = {p["name"] for p in doc.parties}
    assert {"Client", "Contractor"} <= names
    # The 7/14-day termination notice must be found.
    notice = doc.by_type(ExtractionType.NOTICE_PERIOD)
    assert notice and notice[0].clause_title == "Term and Termination"
