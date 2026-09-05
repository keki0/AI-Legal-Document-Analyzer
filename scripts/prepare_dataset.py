"""Phase 2 dataset preparation and analysis.

Produces the evidence needed to fix ``max_length`` and to report class
balance honestly, then writes the prepared splits to disk.

    python scripts/prepare_dataset.py               # analysis only
    python scripts/prepare_dataset.py --save        # also save prepared splits

Downloads LEDGAR (~16 MB) and the Legal-BERT tokenizer on first run.
Nothing here trains anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODELS, PATHS, TRAIN, set_seed  # noqa: E402
from src.dataset import (  # noqa: E402
    apply_taxonomy,
    class_distribution,
    describe_split_provenance,
    load_ledgar,
    load_taxonomy,
    subsample_train,
    token_length_report,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="write prepared splits")
    parser.add_argument(
        "--subsample",
        type=int,
        default=TRAIN.train_subsample,
        help="target training-set size (0 = keep all 60k)",
    )
    args = parser.parse_args()
    set_seed()

    rule("1. TAXONOMY")
    taxonomy = load_taxonomy()
    print(f"categories       : {taxonomy.n_categories}")
    print(f"labels mapped    : {len(taxonomy.label_to_category)}")

    rule("2. LOADING LEDGAR")
    dataset = load_ledgar()

    rule("3. SPLIT PROVENANCE")
    provenance = describe_split_provenance(dataset)
    print(f"features             : {provenance['features']}")
    print(f"temporal fields      : {provenance['temporal_fields_found'] or 'NONE FOUND'}")
    print(f"chronology verifiable: {provenance['can_verify_chronology']}")
    for split, size in provenance["split_sizes"].items():
        print(f"  {split:<12} {size:,}")
    print(f"\n{provenance['description']}")

    rule("4. APPLYING TAXONOMY")
    dataset = apply_taxonomy(dataset, taxonomy)
    example = dataset["train"][0]
    print(f"example LEDGAR label : {example['ledgar_label']}")
    print(f"example category     : {example['category']}")

    rule("5. CLASS DISTRIBUTION (before subsampling)")
    for split in ("train", "validation", "test"):
        counts = class_distribution(dataset, split)
        total = sum(counts.values())
        print(f"\n--- {split} (n={total:,}) ---")
        print(f"{'category':<38}{'count':>8}{'%':>8}")
        for category, count in counts.most_common():
            print(f"{category:<38}{count:>8,}{100 * count / total:>7.2f}%")
        smallest = counts.most_common()[-1]
        print(
            f"imbalance ratio (largest/smallest): "
            f"{counts.most_common()[0][1] / smallest[1]:.1f}x  "
            f"(smallest: {smallest[0]} = {smallest[1]})"
        )

    rule("6. TOKEN LENGTH (evidence for max_length)")
    report = token_length_report(dataset, MODELS.classifier)
    print(f"tokenizer : {report['model']}")
    print(f"sampled   : {report['sampled']:,} provisions")
    print(
        f"mean {report['mean']} | median {report['median']} | "
        f"p90 {report['p90']} | p95 {report['p95']} | "
        f"p99 {report['p99']} | max {report['max']}"
    )
    print(f"\n{'max_length':>12}{'% kept whole':>16}")
    for length, coverage in report["coverage"].items():
        print(f"{length:>12}{coverage:>15.2f}%")
    print(
        "\nDecision rule: choose the smallest max_length covering >=95% of "
        "provisions. Shorter sequences train quadratically faster in "
        "attention, which matters on a free-tier T4."
    )

    if args.subsample:
        rule(f"7. SUBSAMPLING TRAIN -> ~{args.subsample:,}")
        dataset = subsample_train(dataset, args.subsample)
        counts = class_distribution(dataset, "train")
        total = sum(counts.values())
        print(f"{'category':<38}{'count':>8}{'%':>8}")
        for category, count in counts.most_common():
            print(f"{category:<38}{count:>8,}{100 * count / total:>7.2f}%")

    if args.save:
        rule("8. SAVING")
        out = PATHS.data_processed / "ledgar_mapped"
        dataset.save_to_disk(str(out))
        summary = {
            "provenance": provenance,
            "token_lengths": report,
            "n_categories": taxonomy.n_categories,
            "categories": taxonomy.categories,
            "split_sizes": {s: len(dataset[s]) for s in dataset},
        }
        path = PATHS.data_processed / "dataset_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"splits  -> {out}")
        print(f"summary -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
