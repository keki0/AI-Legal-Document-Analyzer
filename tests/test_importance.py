"""Tests for the Phase 4 importance layer.

Model-free and network-free: temperature scaling is a one-dimensional
optimisation in NumPy/SciPy, so the whole suite runs on CPU in seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS  # noqa: E402
from src.document_processing import Clause, load_and_segment  # noqa: E402
from src.importance import (  # noqa: E402
    ClauseAnalysis,
    FlagDetector,
    FlagHit,
    SalienceScorer,
    TemperatureScaler,
    analyze_clauses,
    compute_ece,
    confidence_bins,
    evaluate_calibration,
    top_k_salient,
)

CONTRACT = PATHS.data_raw / "sample-independent-contractor-agreement.pdf"


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def overconfident():
    """Synthetic logits from a deliberately overconfident 15-class model.

    Mirrors the usual fine-tuned-Transformer failure mode: the softmax is far
    sharper than the accuracy justifies, so a correct implementation should
    recover T > 1.
    """
    rng = np.random.default_rng(0)
    n, k = 4000, 15
    labels = rng.integers(0, k, n)
    base = rng.normal(0, 1, (n, k))
    base[np.arange(n), labels] += 1.6
    return base * 2.5, labels


@pytest.fixture
def clauses():
    return [
        Clause(1, "1", "Intellectual Property",
               "All work product shall be the sole and exclusive property of the "
               "Client, and the Contractor agrees to assign all rights, title, and "
               "interest to the Client.", 1, 1),
        Clause(2, "2", "Counterparts",
               "This Agreement may be executed in counterparts, each of which shall "
               "be deemed an original.", 2, 2),
    ]


@pytest.fixture
def predictions():
    return [
        {"category": "Intellectual Property", "confidence": 0.92,
         "probabilities": {"Intellectual Property": 0.92,
                           "Boilerplate & Administrative": 0.05,
                           "Payment & Fees": 0.03}},
        {"category": "Boilerplate & Administrative", "confidence": 0.88,
         "probabilities": {"Boilerplate & Administrative": 0.88,
                           "Intellectual Property": 0.07,
                           "Payment & Fees": 0.05}},
    ]


# ===========================================================================
# Temperature scaling
# ===========================================================================

def test_recovers_temperature_above_one_for_overconfident_model(overconfident):
    logits, labels = overconfident
    scaler = TemperatureScaler().fit(logits, labels)
    assert scaler.fitted
    assert scaler.temperature > 1.0


def test_recovers_temperature_below_one_for_underconfident_model():
    rng = np.random.default_rng(1)
    n, k = 3000, 10
    labels = rng.integers(0, k, n)
    base = rng.normal(0, 1, (n, k))
    base[np.arange(n), labels] += 3.0
    scaler = TemperatureScaler().fit(base * 0.3, labels)
    assert scaler.temperature < 1.0


def test_temperature_scaling_never_changes_a_prediction(overconfident):
    """The invariant that keeps the report honest.

    Dividing every logit by the same positive constant is monotonic, so the
    argmax cannot move and accuracy is mathematically unchanged. If this ever
    fails, any claimed accuracy gain from calibration is a bug.
    """
    logits, labels = overconfident
    scaler = TemperatureScaler().fit(logits, labels)
    report = evaluate_calibration(scaler, logits, labels)
    assert report.accuracy_before == report.accuracy_after

    before = logits.argmax(axis=1)
    after = scaler.transform(logits).argmax(axis=1)
    assert np.array_equal(before, after)


def test_calibration_reduces_ece_on_overconfident_logits(overconfident):
    logits, labels = overconfident
    split = len(labels) // 2
    scaler = TemperatureScaler().fit(logits[:split], labels[:split])
    report = evaluate_calibration(scaler, logits[split:], labels[split:])
    assert report.ece_after < report.ece_before
    assert report.ece_improved
    # Softening should pull mean confidence toward accuracy.
    assert abs(report.mean_confidence_after - report.accuracy_after) < abs(
        report.mean_confidence_before - report.accuracy_before
    )


def test_transform_returns_valid_probability_distributions(overconfident):
    logits, labels = overconfident
    scaler = TemperatureScaler().fit(logits, labels)
    probabilities = scaler.transform(logits)
    assert probabilities.shape == logits.shape
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_calibrate_probabilities_matches_transform_on_logits(overconfident):
    """Recovering logits via log(p) is exact, not an approximation.

    Softmax is invariant to an additive constant, which is the only thing
    log(p) loses.
    """
    logits, labels = overconfident
    scaler = TemperatureScaler(1.7)
    from src.importance import _softmax

    raw = _softmax(logits[:20])
    via_probabilities = scaler.calibrate_probabilities(raw)
    via_logits = scaler.transform(logits[:20])
    assert np.allclose(via_probabilities, via_logits, atol=1e-9)


def test_scaler_roundtrips_through_disk(tmp_path, overconfident):
    logits, labels = overconfident
    scaler = TemperatureScaler().fit(logits, labels)
    path = tmp_path / "t.json"
    scaler.save(path)
    reloaded = TemperatureScaler.load(path)
    assert reloaded.temperature == pytest.approx(scaler.temperature)
    assert np.allclose(reloaded.transform(logits), scaler.transform(logits))


def test_extreme_logits_do_not_overflow():
    """Large logits divided by a small temperature must not produce NaN."""
    logits = np.array([[500.0, -500.0, 0.0], [-800.0, 800.0, 100.0]])
    probabilities = TemperatureScaler(0.05).transform(logits)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_temperature_rejected(bad):
    with pytest.raises(ValueError, match="positive"):
        TemperatureScaler(bad)


def test_fit_rejects_malformed_input():
    with pytest.raises(ValueError, match="2-D"):
        TemperatureScaler().fit(np.array([1.0, 2.0]), np.array([0]))
    with pytest.raises(ValueError, match="length mismatch"):
        TemperatureScaler().fit(np.zeros((5, 3)), np.zeros(4, dtype=int))
    with pytest.raises(ValueError, match="out of range"):
        TemperatureScaler().fit(np.zeros((5, 3)), np.array([0, 1, 2, 3, 9]))


# ===========================================================================
# Calibration metrics
# ===========================================================================

def test_perfectly_calibrated_predictions_have_near_zero_ece():
    rng = np.random.default_rng(3)
    confidence = rng.uniform(0.5, 1.0, 20000)
    correct = rng.random(20000) < confidence
    probabilities = np.stack([confidence, 1 - confidence], axis=1)
    labels = np.where(correct, 0, 1)
    ece, _ = compute_ece(probabilities, labels, n_bins=15)
    assert ece < 0.02


def test_maximally_overconfident_predictions_have_large_ece():
    n = 1000
    probabilities = np.zeros((n, 2))
    probabilities[:, 0] = 1.0          # always fully confident in class 0
    labels = np.array([0, 1] * (n // 2))  # right only half the time
    ece, mce = compute_ece(probabilities, labels, n_bins=15)
    assert ece == pytest.approx(0.5, abs=0.01)
    assert mce == pytest.approx(0.5, abs=0.01)


def test_every_prediction_lands_in_exactly_one_bin():
    rng = np.random.default_rng(4)
    logits = rng.normal(0, 2, (500, 5))
    from src.importance import _softmax

    bins = confidence_bins(_softmax(logits), rng.integers(0, 5, 500), n_bins=15)
    assert sum(b["count"] for b in bins) == 500


def test_ece_handles_empty_input():
    assert compute_ece(np.zeros((0, 3)), np.zeros(0, dtype=int)) == (0.0, 0.0)


# ===========================================================================
# Salience
# ===========================================================================

def test_priors_yaml_loads_and_covers_the_taxonomy():
    from src.dataset import load_taxonomy

    scorer = SalienceScorer.from_yaml()
    taxonomy = load_taxonomy()
    for category in taxonomy.categories:
        assert category in scorer.priors, f"no prior for {category!r}"


def test_priors_are_declared_unlearned():
    """Provenance must survive into the object, not just the YAML comments."""
    scorer = SalienceScorer.from_yaml()
    assert scorer.learned is False


def test_yaml_claiming_learned_priors_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("learned: true\npriors:\n  A: 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hand-assigned"):
        SalienceScorer.from_yaml(path)


def test_boilerplate_ranks_below_high_stakes_categories():
    """The main job the priors do: down-weighting drafting machinery."""
    scorer = SalienceScorer.from_yaml()
    floor = scorer.prior_for("Boilerplate & Administrative")
    for category in ("Intellectual Property", "Liability & Indemnity",
                     "Term & Termination", "Payment & Fees"):
        assert scorer.prior_for(category) > floor


def test_confident_boilerplate_scores_below_unsure_ip():
    """The behaviour that replaces the baseline's keyword rules."""
    scorer = SalienceScorer.from_yaml()
    assert scorer.score("Boilerplate & Administrative", 0.99) < scorer.score(
        "Intellectual Property", 0.40
    )


def test_salience_is_bounded_and_monotonic_in_confidence():
    scorer = SalienceScorer.from_yaml()
    scores = [scorer.score("Payment & Fees", c) for c in (0.1, 0.5, 0.9)]
    assert scores == sorted(scores)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_unknown_category_falls_back_to_default_prior():
    scorer = SalienceScorer({"A": 0.9}, default_prior=0.42)
    assert scorer.prior_for("Nonexistent") == pytest.approx(0.42)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_out_of_range_confidence_rejected(bad):
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        SalienceScorer.from_yaml().score("Payment & Fees", bad)


# ===========================================================================
# Flags
# ===========================================================================

def test_all_eleven_flags_are_defined():
    detector = FlagDetector()
    expected = {
        "limitation_of_liability", "unilateral_termination", "unilateral_change",
        "content_removal", "contract_by_using", "choice_of_law", "jurisdiction",
        "arbitration", "broad_ip_assignment", "broad_indemnity",
        "automatic_renewal",
    }
    assert set(detector.flag_ids) == expected


@pytest.mark.parametrize("text,expected", [
    ("The Client may terminate immediately for breach.", "unilateral_termination"),
    ("The Contractor agrees to assign all rights, title, and interest to the Client.",
     "broad_ip_assignment"),
    ("The Contractor shall indemnify and hold harmless the Client.", "broad_indemnity"),
    ("This Agreement shall be governed by the laws of the State of Delaware.",
     "choice_of_law"),
    ("Disputes shall be resolved by binding arbitration.", "arbitration"),
    ("Your subscription will automatically renew each period.", "automatic_renewal"),
    ("We reserve the right to modify these terms at any time.", "unilateral_change"),
    ("In no event shall the Company be liable for indirect damages.",
     "limitation_of_liability"),
    ("By using the service, you agree to these terms.", "contract_by_using"),
    ("We may remove any content at our sole discretion.", "content_removal"),
    ("The parties submit to the jurisdiction of the courts of New York.",
     "jurisdiction"),
])
def test_each_flag_fires_on_representative_text(text, expected):
    hits = FlagDetector().detect(text)
    assert expected in {hit.flag_id for hit in hits}, (
        f"{expected} did not fire on {text!r}; got {[h.flag_id for h in hits]}"
    )


@pytest.mark.parametrize("text,must_not_fire", [
    # Bilateral amendment is not a unilateral change.
    ("Any changes must be made in writing and signed by both parties.",
     "unilateral_change"),
    # Termination requiring mutual consent is not unilateral.
    ("This Agreement may be terminated by mutual written agreement of the parties.",
     "unilateral_termination"),
    # Describing services is not an IP assignment.
    ("The Contractor agrees to provide website design and SEO optimization.",
     "broad_ip_assignment"),
    ("The Contractor will deliver all services in writing.", "broad_indemnity"),
])
def test_flags_do_not_fire_on_benign_text(text, must_not_fire):
    hits = FlagDetector().detect(text)
    assert must_not_fire not in {hit.flag_id for hit in hits}


def test_flag_wording_is_legally_cautious():
    """Every explanation must hedge and must never assert a legal conclusion."""
    detector = FlagDetector()
    forbidden = ("illegal", "unfair", "invalid", "unlawful", "void",
                 "definitely", "you should", "must not sign")
    for flag_id, label, source, reason, _ in detector._definitions:
        explanation = f"This clause may be significant because {reason}"
        assert explanation.startswith("This clause may be significant because")
        lowered = explanation.lower()
        for word in forbidden:
            assert word not in lowered, f"{flag_id} says {word!r}: {explanation}"


def test_trigger_span_offsets_point_at_the_real_text():
    """Evidence must be verifiable: offsets have to index the source text."""
    text = ("The Contractor shall indemnify and hold harmless the Client from "
            "any claims. This Agreement shall be governed by the laws of Ohio.")
    hits = FlagDetector().detect(text)
    assert hits
    for hit in hits:
        assert text[hit.span_start:hit.span_end].strip() == hit.trigger_span
        assert hit.trigger_span in hit.sentence
        assert hit.sentence in text


def test_sentence_evidence_isolates_the_right_sentence():
    text = ("The Contractor shall provide services. "
            "Your subscription will automatically renew each year. "
            "Notices must be in writing.")
    hit = next(h for h in FlagDetector().detect(text) if h.flag_id == "automatic_renewal")
    assert "automatically renew" in hit.sentence
    assert "Notices must be in writing" not in hit.sentence


def test_one_hit_per_flag_even_with_repeated_matches():
    text = ("Party A shall indemnify and hold harmless Party B. "
            "Party C shall indemnify and hold harmless Party D.")
    hits = [h for h in FlagDetector().detect(text) if h.flag_id == "broad_indemnity"]
    assert len(hits) == 1


def test_multiple_distinct_flags_coexist():
    text = ("This Agreement shall be governed by the laws of Delaware, and any "
            "dispute shall be resolved by binding arbitration.")
    ids = {hit.flag_id for hit in FlagDetector().detect(text)}
    assert {"choice_of_law", "arbitration"} <= ids


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_text_yields_no_flags(text):
    assert FlagDetector().detect(text) == []


def test_flags_fire_on_the_real_contract():
    """End-to-end against the actual smoke-test document."""
    if not CONTRACT.exists():
        pytest.skip("contract fixture not present")

    _, clauses = load_and_segment(CONTRACT)
    detector = FlagDetector()
    found = {
        clause.title: {hit.flag_id for hit in detector.detect(clause.text)}
        for clause in clauses
    }
    assert "broad_ip_assignment" in found["Intellectual Property"]
    assert "broad_indemnity" in found["Indemnification"]
    assert "unilateral_termination" in found["Term and Termination"]
    assert "choice_of_law" in found["Miscellaneous"]
    # Scope of Work describes services and should stay clean.
    assert not found["Scope of Work"]


# ===========================================================================
# ClauseAnalysis and analyze_clauses
# ===========================================================================

def test_analyze_clauses_builds_complete_records(clauses, predictions):
    analyses = analyze_clauses(clauses, predictions)
    assert len(analyses) == 2

    ip = analyses[0]
    assert isinstance(ip, ClauseAnalysis)
    assert ip.category == "Intellectual Property"
    assert ip.clause_id == 1
    assert ip.page == 1
    assert 0.0 <= ip.salience <= 1.0
    assert ip.category_prior > 0
    assert ip.probabilities


def test_evidence_spans_survive_into_clause_analysis(clauses, predictions):
    """The requirement that makes flags auditable in the UI."""
    analyses = analyze_clauses(clauses, predictions)
    ip = analyses[0]
    assert ip.has_flags
    hit = next(h for h in ip.flags if h.flag_id == "broad_ip_assignment")
    assert isinstance(hit, FlagHit)
    assert hit.trigger_span
    assert ip.text[hit.span_start:hit.span_end].strip() == hit.trigger_span
    assert hit.trigger_span in hit.sentence


def test_uncalibrated_path_records_confidence_honestly(clauses, predictions):
    """With no scaler, calibrated must equal raw -- never silently differ."""
    analyses = analyze_clauses(clauses, predictions, scaler=None)
    for analysis in analyses:
        assert analysis.calibrated_confidence == analysis.raw_confidence


def test_scaler_changes_calibrated_but_not_raw_confidence(clauses, predictions):
    analyses = analyze_clauses(clauses, predictions, scaler=TemperatureScaler(2.0))
    ip = analyses[0]
    assert ip.raw_confidence == pytest.approx(0.92)
    assert ip.calibrated_confidence < ip.raw_confidence  # T>1 softens
    assert ip.salience == pytest.approx(
        ip.calibrated_confidence * ip.category_prior
    )


def test_salience_ranks_ip_above_boilerplate(clauses, predictions):
    analyses = analyze_clauses(clauses, predictions)
    ranked = top_k_salient(analyses, k=2)
    assert ranked[0].category == "Intellectual Property"
    assert ranked[0].salience > ranked[1].salience


def test_top_k_respects_k_and_handles_oversized_k(clauses, predictions):
    analyses = analyze_clauses(clauses, predictions)
    assert len(top_k_salient(analyses, k=1)) == 1
    assert len(top_k_salient(analyses, k=99)) == 2
    assert top_k_salient([], k=3) == []


def test_length_mismatch_is_rejected(clauses, predictions):
    with pytest.raises(ValueError, match="mismatch"):
        analyze_clauses(clauses, predictions[:1])


def test_analysis_serialises_with_flag_labels(clauses, predictions):
    payload = analyze_clauses(clauses, predictions)[0].to_dict()
    import json

    json.dumps(payload)  # must be JSON-safe for the API and UI
    assert "flag_labels" in payload
    assert "Broad IP assignment" in payload["flag_labels"]
    assert payload["flags"][0]["trigger_span"]


def test_no_output_field_is_named_risk(clauses, predictions):
    """Legal-safety rule: no 'risk score' anywhere in the public surface."""
    payload = analyze_clauses(clauses, predictions)[0].to_dict()
    for key in payload:
        assert "risk" not in key.lower()
