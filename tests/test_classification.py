"""Tests for Phase 3 classification.

Model-free by design: nothing here downloads weights or touches a GPU, so the
suite stays fast and runnable on the Windows laptop. Legal-BERT itself is
verified in Colab and by the round-trip check in
``test_saved_model_loads_if_present``, which skips when no model has been
downloaded yet.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification import (  # noqa: E402
    build_tfidf_svc,
    compute_class_weights,
    evaluate_predictions,
    save_confusion_matrix,
)
from src.config import PATHS  # noqa: E402
from src.dataset import Taxonomy, load_taxonomy  # noqa: E402

CATEGORIES = [f"Cat{i:02d}" for i in range(15)]
# Mirrors the real LEDGAR imbalance (~13x largest to smallest).
SIZES = [1300, 1100, 1000, 900, 800, 700, 600, 500, 450, 400, 350, 300, 250, 150, 100]


@pytest.fixture(scope="module")
def synthetic():
    """Separable synthetic data with LEDGAR-like class imbalance.

    Deliberately easy: this fixture tests that the plumbing works, not that
    the task is hard. Real difficulty is measured on LEDGAR.
    """
    rng = random.Random(0)
    vocab = {i: [f"term{i}_{j}" for j in range(8)] for i in range(15)}
    shared = [token for tokens in vocab.values() for token in tokens]

    texts, labels = [], []
    for class_id, size in enumerate(SIZES):
        for _ in range(size):
            tokens = rng.choices(vocab[class_id], k=12) + rng.choices(shared, k=6)
            texts.append(" ".join(tokens))
            labels.append(class_id)

    order = list(range(len(texts)))
    rng.shuffle(order)
    texts = [texts[i] for i in order]
    labels = [labels[i] for i in order]
    split = int(0.8 * len(texts))
    return texts[:split], labels[:split], texts[split:], labels[split:]


# --- metrics ---------------------------------------------------------------

def test_metrics_shape_and_range():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 1]
    metrics = evaluate_predictions(
        y_true, y_pred, ["A", "B", "C"], model_name="unit", split="test"
    )
    assert metrics.n_examples == 6
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert len(metrics.confusion_matrix) == 3
    assert len(metrics.per_class) == 3


def test_macro_and_weighted_f1_differ_under_imbalance():
    """The reason macro-F1 is the headline metric.

    A model that nails the majority class and fails the minority scores well
    on accuracy and weighted-F1 while macro-F1 correctly stays low.
    """
    y_true = [0] * 95 + [1] * 5
    y_pred = [0] * 100  # never predicts the rare class
    metrics = evaluate_predictions(
        y_true, y_pred, ["Major", "Rare"], model_name="unit", split="test"
    )
    assert metrics.accuracy == pytest.approx(0.95)
    assert metrics.macro_f1 < 0.55
    assert metrics.macro_f1 < metrics.weighted_f1


def test_metrics_roundtrip_json(tmp_path):
    metrics = evaluate_predictions(
        [0, 1], [0, 1], ["A", "B"], model_name="unit", split="test"
    )
    path = tmp_path / "m.json"
    metrics.save(path)
    import json

    loaded = json.loads(path.read_text())
    assert loaded["macro_f1"] == metrics.macro_f1
    assert loaded["categories"] == ["A", "B"]


# --- class weights ---------------------------------------------------------

def test_balanced_weights_favour_rare_classes():
    labels = [0] * 900 + [1] * 100
    weights = compute_class_weights(labels, 2, scheme="balanced")
    assert weights[1] > weights[0]
    # sklearn's formula: n / (k * count_c)
    assert weights[0] == pytest.approx(1000 / (2 * 900), rel=1e-4)
    assert weights[1] == pytest.approx(1000 / (2 * 100), rel=1e-4)


def test_sqrt_scheme_is_gentler_than_balanced():
    labels = [0] * 1300 + [1] * 100
    balanced = compute_class_weights(labels, 2, scheme="balanced")
    softened = compute_class_weights(labels, 2, scheme="sqrt_balanced")
    assert (softened.max() / softened.min()) < (balanced.max() / balanced.min())


def test_weights_handle_absent_class_without_dividing_by_zero():
    weights = compute_class_weights([0, 0, 1], 4, scheme="balanced")
    assert len(weights) == 4
    assert all(w > 0 for w in weights)


def test_unknown_weight_scheme_rejected():
    with pytest.raises(ValueError):
        compute_class_weights([0, 1], 2, scheme="nonsense")


# --- TF-IDF baseline -------------------------------------------------------

def test_tfidf_svc_trains_and_predicts(synthetic):
    x_train, y_train, x_test, y_test = synthetic
    pipeline = build_tfidf_svc(balanced=True)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    assert len(predictions) == len(y_test)
    assert set(predictions) <= set(range(15))
    metrics = evaluate_predictions(
        y_test, predictions, CATEGORIES, model_name="tfidf", split="test"
    )
    # Separable fixture, so this should be near-perfect. A low score here
    # means the pipeline is broken, not that the task is hard.
    assert metrics.macro_f1 > 0.8


def test_balanced_and_unweighted_variants_both_build():
    for balanced in (True, False):
        pipeline = build_tfidf_svc(balanced=balanced)
        weight = pipeline.named_steps["clf"].class_weight
        assert weight == ("balanced" if balanced else None)


def test_confusion_matrix_figure_written(tmp_path, synthetic):
    _, _, x_test, y_test = synthetic
    metrics = evaluate_predictions(
        y_test, y_test, CATEGORIES, model_name="perfect", split="test"
    )
    path = tmp_path / "cm.png"
    save_confusion_matrix(metrics, path)
    assert path.exists() and path.stat().st_size > 1000


# --- taxonomy --------------------------------------------------------------

def test_taxonomy_category_ids_are_stable_and_sorted():
    """id2label in the saved model is built from this ordering.

    If the ordering ever changed between training and inference, every
    prediction would silently map to the wrong category name.
    """
    taxonomy = load_taxonomy()
    assert taxonomy.categories == sorted(taxonomy.categories)
    for index, category in enumerate(taxonomy.categories):
        assert taxonomy.category_id(category) == index


def test_taxonomy_rejects_a_label_mapped_twice(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "categories:\n"
        "  A:\n    ledgar_labels: [Payments, Fees]\n"
        "  B:\n    ledgar_labels: [Fees]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mapped twice"):
        load_taxonomy(path)


def test_validate_against_dataset_catches_unmapped_labels():
    from src.dataset import validate_against_dataset

    taxonomy = Taxonomy({"Payments": "Payment & Fees"}, ["Payment & Fees"], {})
    with pytest.raises(ValueError, match="unmapped"):
        validate_against_dataset(taxonomy, ["Payments", "Arbitration"])


# --- interface contracts ---------------------------------------------------

def test_retrieval_result_carries_what_generation_will_need():
    """Pins the retrieval interface for the later FLAN-T5 stage.

    Phase 6 feeds top-k clauses to FLAN-T5 for grounded answers, which needs
    the clause text and a page reference for citation. This test fails loudly
    if either is ever dropped from RetrievalResult.
    """
    from src.retrieval import RetrievalResult

    annotations = RetrievalResult.__annotations__
    for field in ("clause_id", "title", "text", "page", "score", "rank"):
        assert field in annotations, f"RetrievalResult lost the {field!r} field"


def test_saved_model_loads_if_present():
    """Round-trip check, skipped until the Colab model is downloaded."""
    model_dir = PATHS.classifier_dir
    if not model_dir.exists():
        pytest.skip("no fine-tuned model downloaded yet")

    from src.classification import ClauseClassifier

    classifier = ClauseClassifier(model_dir)
    taxonomy = load_taxonomy()
    assert classifier.categories == taxonomy.categories

    result = classifier.predict(
        ["All work product created under this Agreement shall be the sole "
         "and exclusive property of the Client."]
    )[0]
    assert result["category"] in taxonomy.categories
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == taxonomy.n_categories


def test_classifier_raises_clear_error_when_model_missing(tmp_path):
    from src.classification import ClauseClassifier

    with pytest.raises(FileNotFoundError, match="Download it from Colab"):
        ClauseClassifier(tmp_path / "nope")
