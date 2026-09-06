"""Clause classification: classical baseline, metrics, and inference.

Both the TF-IDF baseline and the fine-tuned Legal-BERT model are scored by
:func:`evaluate_predictions` in this module. Sharing one metric
implementation is deliberate: if the two models were scored by separately
written code, any difference between them could be an artefact of the
scoring rather than the model.

**Fair-comparison note.** The classical baseline uses
``class_weight="balanced"`` and the Transformer uses a class-weighted loss.
Giving imbalance handling to only one of the two would rig the comparison in
its favour, so both get it, and the TF-IDF script reports the unweighted
variant as well to show what the weighting costs.

Nothing here imports torch. The Transformer inference wrapper imports it
lazily so the module stays usable on a machine with only scikit-learn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ClassificationMetrics",
    "evaluate_predictions",
    "build_tfidf_svc",
    "save_confusion_matrix",
    "compute_class_weights",
    "ClauseClassifier",
]


@dataclass
class ClassificationMetrics:
    """Metrics for one model on one split."""

    model_name: str
    split: str
    n_examples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.model_name} on {self.split} (n={self.n_examples:,}): "
            f"accuracy {self.accuracy:.4f} | macro-F1 {self.macro_f1:.4f} | "
            f"weighted-F1 {self.weighted_f1:.4f}"
        )

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "split": self.split,
            "n_examples": self.n_examples,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "categories": self.categories,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Saved metrics to %s", path)


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    categories: Sequence[str],
    *,
    model_name: str,
    split: str = "test",
) -> ClassificationMetrics:
    """Score predictions against gold labels.

    ``macro_f1`` is the headline number, not accuracy. With a 13x imbalance
    between the largest and smallest category, accuracy is dominated by the
    frequent classes and would look flattering while the model failed on
    Intellectual Property.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    labels = list(range(len(categories)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(categories),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": int(report[name]["support"]),
        }
        for name in categories
        if name in report
    }

    return ClassificationMetrics(
        model_name=model_name,
        split=split,
        n_examples=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        macro_precision=float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        macro_recall=float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        per_class=per_class,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        categories=list(categories),
    )


def build_tfidf_svc(*, balanced: bool = True, seed: int = 42):
    """TF-IDF + LinearSVC pipeline.

    Word unigrams and bigrams plus character n-grams. The character features
    matter for legal text: they capture recurring formulaic fragments
    ("hold harmless", "hereunder") robustly against inflection, and they cost
    almost nothing on CPU.

    Args:
        balanced: Apply ``class_weight="balanced"``. Should stay True for the
            headline comparison so the classical model gets the same
            imbalance handling as the Transformer.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import FeatureUnion

    features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_features=200_000,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=100_000,
            sublinear_tf=True,
            lowercase=True,
        )),
    ])

    return Pipeline([
        ("features", features),
        ("clf", LinearSVC(
            C=1.0,
            class_weight="balanced" if balanced else None,
            random_state=seed,
            max_iter=3000,
        )),
    ])


def compute_class_weights(
    labels: Sequence[int], n_classes: int, *, scheme: str = "balanced"
) -> np.ndarray:
    """Per-class loss weights for the Transformer.

    ``balanced`` uses ``n / (k * count_c)`` -- scikit-learn's formula, so the
    Transformer and the SVC are weighted identically.

    ``sqrt_balanced`` softens it to ``sqrt(n / count_c)``, normalised. Full
    balancing on a 13x-imbalanced set can over-correct: the model starts
    over-predicting rare classes and weighted-F1 falls further than macro-F1
    rises. If that happens in validation, switch schemes rather than
    abandoning weighting.
    """
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    total = counts.sum()

    if scheme == "balanced":
        weights = total / (n_classes * counts)
    elif scheme == "sqrt_balanced":
        weights = np.sqrt(total / counts)
        weights = weights / weights.mean()
    else:
        raise ValueError(f"unknown scheme: {scheme!r}")

    logger.info(
        "Class weights (%s): min %.3f, max %.3f, ratio %.1fx",
        scheme, weights.min(), weights.max(), weights.max() / weights.min(),
    )
    return weights.astype(np.float32)


def save_confusion_matrix(
    metrics: ClassificationMetrics,
    path: Path,
    *,
    normalise: bool = True,
) -> None:
    """Write a confusion-matrix figure.

    Row-normalised by default: with a 13x imbalance a raw-count matrix is a
    single dark row and communicates nothing about the rare classes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(metrics.confusion_matrix, dtype=float)
    if normalise:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, np.clip(row_sums, 1, None))

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalise else None)
    ax.set_xticks(range(len(metrics.categories)))
    ax.set_yticks(range(len(metrics.categories)))
    ax.set_xticklabels(metrics.categories, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(metrics.categories, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"{metrics.model_name} — {'row-normalised ' if normalise else ''}"
        f"confusion matrix\nmacro-F1 {metrics.macro_f1:.4f}"
    )
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", path)


class ClauseClassifier:
    """Inference wrapper for the fine-tuned Legal-BERT model.

    Used by the pipeline and the application. Loads from a local directory so
    the app never needs network access at runtime.
    """

    def __init__(self, model_dir: Path | str, *, device: str | None = None) -> None:
        # Validate the path BEFORE importing torch. A missing model is the
        # common mistake ("I forgot to download it from Colab"), and a heavy
        # import failing first would mask it behind an unrelated error.
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"No fine-tuned model at {model_dir}. Download it from Colab first."
            )

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

        config = self.model.config
        self.categories = [
            config.id2label[i] for i in sorted(int(k) for k in config.id2label)
        ]
        self.max_length = int(getattr(config, "max_position_embeddings", 512))
        self.max_length = min(self.max_length, 384)

    def predict(self, texts: Sequence[str], *, batch_size: int = 16) -> list[dict]:
        """Classify clauses, returning category and full probability vector.

        Probabilities are kept, not just the argmax: the importance layer in
        Phase 4 uses classifier confidence, so discarding it here would mean
        re-running the model later.
        """
        import torch

        results: list[dict] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            encoded = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probabilities:
                index = int(row.argmax())
                results.append({
                    "category": self.categories[index],
                    "confidence": float(row[index]),
                    "probabilities": {
                        name: float(p) for name, p in zip(self.categories, row)
                    },
                })
        return results
