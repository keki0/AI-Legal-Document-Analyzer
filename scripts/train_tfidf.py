"""Phase 3, part 1: the classical baseline. Runs locally on CPU.

Trains TF-IDF + LinearSVC on the prepared 15-category LEDGAR data and writes
metrics and a confusion matrix. This is the number Legal-BERT has to beat; if
the Transformer cannot clear it, that is a finding worth reporting, not a
failure to hide.

Two variants are trained because they answer different questions:

  balanced    class_weight="balanced" -- the headline result, matched to the
              Transformer's class-weighted loss so the comparison is fair.
  unweighted  no class weighting -- shows what the weighting costs in
              accuracy and buys in macro-F1.

    python scripts/train_tfidf.py
    python scripts/train_tfidf.py --quick    # 3k training rows, for a fast check

Requires scripts/prepare_dataset.py --save to have been run first.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification import (  # noqa: E402
    build_tfidf_svc,
    evaluate_predictions,
    save_confusion_matrix,
)
from src.config import PATHS, set_seed  # noqa: E402
from src.dataset import load_taxonomy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PREPARED = PATHS.data_processed / "ledgar_mapped"
RESULTS = PATHS.root / "evaluation" / "results"


def load_prepared():
    from datasets import load_from_disk

    if not PREPARED.exists():
        raise SystemExit(
            f"No prepared dataset at {PREPARED}.\n"
            "Run:  python scripts/prepare_dataset.py --save"
        )
    return load_from_disk(str(PREPARED))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use 3k training rows")
    args = parser.parse_args()
    set_seed()

    taxonomy = load_taxonomy()
    categories = taxonomy.categories
    dataset = load_prepared()

    train = dataset["train"]
    if args.quick:
        train = train.select(range(min(3000, len(train))))

    x_train, y_train = train["text"], train["category_id"]
    x_test, y_test = dataset["test"]["text"], dataset["test"]["category_id"]

    print("=" * 70)
    print("TF-IDF + LinearSVC — classical baseline")
    print("=" * 70)
    print(f"train      : {len(x_train):,}")
    print(f"test       : {len(x_test):,}")
    print(f"categories : {len(categories)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    results = {}

    for variant, balanced in (("balanced", True), ("unweighted", False)):
        name = f"TF-IDF+LinearSVC ({variant})"
        print(f"\n--- {name} ---")

        pipeline = build_tfidf_svc(balanced=balanced)
        start = time.perf_counter()
        pipeline.fit(x_train, y_train)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = pipeline.predict(x_test)
        infer_seconds = time.perf_counter() - start

        metrics = evaluate_predictions(
            y_test, predictions, categories, model_name=name, split="test"
        )
        print(metrics.summary())
        print(f"train time     : {train_seconds:.1f}s")
        print(
            f"inference      : {infer_seconds:.1f}s for {len(x_test):,} "
            f"({1000 * infer_seconds / len(x_test):.2f} ms/clause)"
        )

        payload = metrics.to_dict()
        payload["train_seconds"] = round(train_seconds, 2)
        payload["inference_seconds"] = round(infer_seconds, 2)
        payload["n_train"] = len(x_train)

        path = RESULTS / f"tfidf_svc_{variant}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        save_confusion_matrix(metrics, RESULTS / f"confusion_tfidf_{variant}.png")
        results[variant] = metrics

    # Per-class table for the headline variant only.
    headline = results["balanced"]
    print("\n" + "=" * 70)
    print("PER-CLASS — TF-IDF+LinearSVC (balanced)")
    print("=" * 70)
    print(f"{'category':<38}{'prec':>7}{'rec':>7}{'F1':>7}{'n':>8}")
    for category in sorted(
        headline.per_class, key=lambda c: headline.per_class[c]["f1"]
    ):
        row = headline.per_class[category]
        print(
            f"{category:<38}{row['precision']:>7.3f}{row['recall']:>7.3f}"
            f"{row['f1']:>7.3f}{row['support']:>8,}"
        )

    print("\n" + "=" * 70)
    print("WHAT CLASS WEIGHTING COST AND BOUGHT")
    print("=" * 70)
    print(f"{'variant':<14}{'accuracy':>10}{'macro-F1':>10}{'weighted-F1':>13}")
    for variant, metrics in results.items():
        print(
            f"{variant:<14}{metrics.accuracy:>10.4f}"
            f"{metrics.macro_f1:>10.4f}{metrics.weighted_f1:>13.4f}"
        )
    print(
        "\nExpect weighting to raise macro-F1 and lower accuracy slightly. "
        "Macro-F1 is the headline metric: with a 13x imbalance, accuracy is "
        "dominated by the frequent categories."
    )
    print(f"\nResults written to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
