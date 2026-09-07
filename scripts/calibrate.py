"""Phase 4: fit and evaluate temperature scaling on the saved Legal-BERT logits.

Uses the four arrays exported from the Colab training run. Does NOT retrain
anything and does not touch the saved model.

    python scripts/calibrate.py

Expects in evaluation/results/:
    val_logits.npy   (n, 15)      test_logits.npy  (n, 15)
    val_labels.npy   (n,)         test_labels.npy  (n,)

Temperature is fitted on VALIDATION and evaluated on TEST. Fitting on test
would leak and would make the reported ECE meaningless.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS, set_seed  # noqa: E402
from src.dataset import load_taxonomy  # noqa: E402
from src.importance import (  # noqa: E402
    TemperatureScaler,
    compute_ece,
    confidence_bins,
    evaluate_calibration,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

RESULTS = PATHS.root / "evaluation" / "results"
REQUIRED = ("val_logits.npy", "val_labels.npy", "test_logits.npy", "test_labels.npy")


def load_arrays() -> tuple[np.ndarray, ...]:
    missing = [name for name in REQUIRED if not (RESULTS / name).exists()]
    if missing:
        raise SystemExit(
            f"Missing from {RESULTS}: {', '.join(missing)}\n\n"
            "These are exported by notebooks/03_classification_colab.ipynb "
            "(cell 13). Download them from Drive into evaluation/results/."
        )
    return tuple(np.load(RESULTS / name) for name in REQUIRED)


def reliability_diagram(report, path: Path) -> None:
    """Reliability diagram before and after, with the identity line.

    A perfectly calibrated model sits on the diagonal. Bars below it mean
    overconfidence -- the usual failure mode for fine-tuned Transformers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    panels = [
        ("Before (raw softmax)", report.bins_before, report.ece_before),
        (f"After (T = {report.temperature:.3f})", report.bins_after, report.ece_after),
    ]

    for ax, (title, bins, ece) in zip(axes, panels):
        centres = [(b["lower"] + b["upper"]) / 2 for b in bins]
        accuracies = [b["accuracy"] for b in bins]
        counts = [b["count"] for b in bins]
        total = sum(counts) or 1

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        ax.bar(centres, accuracies, width=1 / len(bins) * 0.9,
               edgecolor="black", alpha=0.75, label="accuracy")
        for centre, accuracy, count in zip(centres, accuracies, counts):
            if count:
                ax.annotate(f"{100 * count / total:.0f}%", (centre, accuracy),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=6)
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("accuracy")
    fig.suptitle("Legal-BERT reliability (bar labels = % of test set in bin)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"reliability diagram -> {path}")


def print_bins(bins: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"{'confidence bin':<18}{'n':>8}{'mean conf':>12}{'accuracy':>11}{'gap':>9}")
    for b in bins:
        if not b["count"]:
            continue
        span = f"({b['lower']:.2f}, {b['upper']:.2f}]"
        print(
            f"{span:<18}{b['count']:>8,}{b['confidence']:>12.4f}"
            f"{b['accuracy']:>11.4f}{b['gap']:>+9.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()
    set_seed()

    taxonomy = load_taxonomy()
    val_logits, val_labels, test_logits, test_labels = load_arrays()

    print("=" * 74)
    print("TEMPERATURE SCALING — Legal-BERT confidence calibration")
    print("=" * 74)
    print(f"validation : {val_logits.shape} logits, {val_labels.shape} labels")
    print(f"test       : {test_logits.shape} logits, {test_labels.shape} labels")
    print(f"categories : {taxonomy.n_categories}")

    if val_logits.shape[1] != taxonomy.n_categories:
        raise SystemExit(
            f"Logits have {val_logits.shape[1]} classes but the taxonomy has "
            f"{taxonomy.n_categories}. These logits came from a different model."
        )

    print("\n--- fitting on VALIDATION (test is never used for fitting) ---")
    scaler = TemperatureScaler().fit(val_logits, val_labels)
    print(f"fitted temperature: T = {scaler.temperature:.4f}")
    if scaler.temperature > 1.0:
        print("T > 1: the model was OVERCONFIDENT; probabilities are softened.")
    elif scaler.temperature < 1.0:
        print("T < 1: the model was UNDERCONFIDENT; probabilities are sharpened.")
    else:
        print("T = 1: already calibrated; scaling is a no-op.")

    val_report = evaluate_calibration(scaler, val_logits, val_labels, n_bins=args.bins)
    print(f"\nvalidation (in-sample): {val_report.summary()}")

    print("\n--- evaluating on TEST (held out) ---")
    report = evaluate_calibration(scaler, test_logits, test_labels, n_bins=args.bins)
    print(report.summary())

    print(f"\n{'metric':<28}{'before':>12}{'after':>12}{'change':>12}")
    for name, before, after in [
        ("accuracy", report.accuracy_before, report.accuracy_after),
        ("ECE", report.ece_before, report.ece_after),
        ("MCE", report.mce_before, report.mce_after),
        ("NLL", report.nll_before, report.nll_after),
        ("mean confidence", report.mean_confidence_before, report.mean_confidence_after),
    ]:
        print(f"{name:<28}{before:>12.4f}{after:>12.4f}{after - before:>+12.4f}")

    print(
        "\nNote: accuracy is expected to be UNCHANGED. Dividing every logit by "
        "the same positive constant is monotonic, so the argmax cannot move. "
        "Temperature scaling calibrates confidence; it does not improve "
        "classification."
    )
    if abs(report.accuracy_after - report.accuracy_before) > 1e-9:
        print(
            "  (A tiny difference appeared -- this is a float tie-break, not "
            "an effect. Do not report it as an improvement.)"
        )

    print_bins(report.bins_before, "CONFIDENCE-BINNED ACCURACY — before (raw softmax)")
    print_bins(report.bins_after, f"CONFIDENCE-BINNED ACCURACY — after (T={scaler.temperature:.3f})")

    verdict = (
        "ECE improved: calibrated confidence is a better salience signal "
        "than raw softmax."
        if report.ece_improved else
        "ECE did NOT improve. Report this honestly and use raw confidence, "
        "or investigate before relying on the calibrated values."
    )
    print(f"\nVERDICT: {verdict}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    scaler.save(RESULTS / "temperature.json")
    report.save(RESULTS / "calibration_test.json")
    (RESULTS / "calibration_validation.json").write_text(
        json.dumps(val_report.to_dict(), indent=2), encoding="utf-8"
    )
    reliability_diagram(report, RESULTS / "reliability_diagram.png")

    print(f"\ntemperature -> {RESULTS / 'temperature.json'}")
    print(f"test report -> {RESULTS / 'calibration_test.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
