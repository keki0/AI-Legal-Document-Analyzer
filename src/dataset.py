"""LEDGAR loading, taxonomy mapping and dataset preparation.

Design decisions that matter, and why:

**Mapping is keyed by label name, never by integer.** The HF dataset card
shows an example taxes provision with ``label: 32``, but index 32 in the
ClassLabel array is "Duties" ("Taxes" is 87). The card's worked example is
illustrative and its integer is unreliable. Names are resolved to ids at
runtime from ``features["label"].names``.

**The published splits are used unmodified and are NOT described as
chronological.** See :func:`describe_split_provenance` for the evidence.

**Nothing is discarded.** All 100 labels map into 15 categories. Subsampling
touches only the training split; validation and test stay whole.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.config import DATASETS, PATHS, SEED

if TYPE_CHECKING:  # pragma: no cover
    from datasets import DatasetDict

logger = logging.getLogger(__name__)

__all__ = [
    "Taxonomy",
    "load_taxonomy",
    "load_ledgar",
    "apply_taxonomy",
    "subsample_train",
    "class_distribution",
    "token_length_report",
    "describe_split_provenance",
]


@dataclass(frozen=True)
class Taxonomy:
    """A validated LEDGAR-label to project-category mapping."""

    label_to_category: dict[str, str]
    categories: list[str]
    descriptions: dict[str, str]

    @property
    def n_categories(self) -> int:
        return len(self.categories)

    def category_id(self, category: str) -> int:
        return self.categories.index(category)


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    """Load and validate ``taxonomy.yaml``.

    Raises:
        ValueError: if a label is mapped twice or a name is not a real LEDGAR
            label. Validation happens here rather than at training time so a
            typo surfaces in seconds instead of after a GPU run.
    """
    path = path or PATHS.taxonomy
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    mapping: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    for category, body in raw["categories"].items():
        descriptions[category] = body.get("description", "")
        for label in body["ledgar_labels"]:
            if label in mapping:
                raise ValueError(
                    f"LEDGAR label {label!r} mapped twice: "
                    f"{mapping[label]!r} and {category!r}"
                )
            mapping[label] = category

    categories = sorted(raw["categories"].keys())
    logger.info(
        "Loaded taxonomy: %d labels -> %d categories", len(mapping), len(categories)
    )
    return Taxonomy(mapping, categories, descriptions)


def validate_against_dataset(taxonomy: Taxonomy, label_names: list[str]) -> None:
    """Check the taxonomy covers the dataset's actual labels exactly.

    Guards against the dataset changing under us and against typos in the
    YAML that would otherwise silently route provisions to a wrong category.
    """
    mapped = set(taxonomy.label_to_category)
    actual = set(label_names)

    missing = sorted(actual - mapped)
    invented = sorted(mapped - actual)
    if missing:
        raise ValueError(f"{len(missing)} dataset labels are unmapped: {missing}")
    if invented:
        raise ValueError(f"{len(invented)} mapped names are not real labels: {invented}")
    logger.info("Taxonomy validated against %d dataset labels.", len(actual))


def load_ledgar() -> "DatasetDict":
    """Load the LEDGAR config of LexGLUE with its published splits."""
    from datasets import load_dataset

    dataset = load_dataset(
        DATASETS.ledgar_repo,
        DATASETS.ledgar_config,
        trust_remote_code=DATASETS.ledgar_trust_remote_code,
    )
    logger.info(
        "Loaded LEDGAR: %s",
        {split: len(dataset[split]) for split in dataset},
    )
    return dataset


def describe_split_provenance(dataset: "DatasetDict") -> dict[str, object]:
    """Report what can and cannot be verified about the split.

    We were asked whether LEDGAR carries a reliable temporal field. It does
    not. The distributed features are ``text`` and ``label`` only; there is no
    date, filing year, or contract identifier, so the ordering of the split
    cannot be independently checked from the data.

    Note also that the LexGLUE dataset card explicitly describes the ECtHR
    split as chronological, and does *not* say this for LEDGAR. Some secondary
    papers do describe LEDGAR's split as chronological, but that is a claim we
    cannot verify. We therefore use the published splits unmodified and
    describe them as such.
    """
    features = list(dataset["train"].features)
    temporal = [f for f in features if any(
        token in f.lower() for token in ("date", "year", "time", "filing")
    )]
    return {
        "features": features,
        "temporal_fields_found": temporal,
        "can_verify_chronology": bool(temporal),
        "split_sizes": {split: len(dataset[split]) for split in dataset},
        "description": (
            "Published LexGLUE train/validation/test splits, used unmodified. "
            "No temporal field is distributed with the data, so the split "
            "ordering cannot be independently verified; it is not described "
            "as chronological in this project."
        ),
    }


def apply_taxonomy(dataset: "DatasetDict", taxonomy: Taxonomy) -> "DatasetDict":
    """Add ``category`` and ``category_id`` columns derived from the label name.

    Resolution goes id -> name -> category. Never id -> category.
    """
    label_names = dataset["train"].features["label"].names
    validate_against_dataset(taxonomy, label_names)

    def add_category(batch: dict) -> dict:
        names = [label_names[i] for i in batch["label"]]
        categories = [taxonomy.label_to_category[name] for name in names]
        return {
            "ledgar_label": names,
            "category": categories,
            "category_id": [taxonomy.category_id(c) for c in categories],
        }

    return dataset.map(add_category, batched=True, desc="Applying taxonomy")


def subsample_train(
    dataset: "DatasetDict",
    n: int | None,
    *,
    seed: int = SEED,
) -> "DatasetDict":
    """Stratified subsample of the training split only.

    Validation and test are left whole so metrics stay comparable to the full
    benchmark. Stratifying preserves the class balance; a plain random sample
    would starve small categories such as Intellectual Property.
    """
    if n is None or n >= len(dataset["train"]):
        logger.info("No subsampling applied (requested %s).", n)
        return dataset

    train = dataset["train"]
    counts = Counter(train["category_id"])
    total = sum(counts.values())

    keep: list[int] = []
    import random

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for index, cid in enumerate(train["category_id"]):
        by_class.setdefault(cid, []).append(index)

    for cid, indices in by_class.items():
        # Proportional allocation, but never fewer than 50 examples for a
        # class that has them, so rare categories remain learnable.
        target = max(min(len(indices), 50), round(n * counts[cid] / total))
        target = min(target, len(indices))
        keep.extend(rng.sample(indices, target))

    rng.shuffle(keep)
    dataset["train"] = train.select(keep)
    logger.info("Subsampled train to %d examples.", len(keep))
    return dataset


def class_distribution(dataset: "DatasetDict", split: str = "train") -> Counter:
    """Category counts for one split."""
    return Counter(dataset[split]["category"])


def token_length_report(
    dataset: "DatasetDict",
    model_name: str,
    *,
    split: str = "train",
    sample: int = 5000,
    candidates: tuple[int, ...] = (64, 128, 192, 256, 384, 512),
) -> dict[str, object]:
    """Measure provision token lengths and report coverage per max_length.

    ``max_length`` should be chosen from this table, not assumed. Truncation
    silently discards text, so the report states what fraction of provisions
    each candidate keeps whole.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = dataset[split]["text"][:sample]
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in texts]
    lengths.sort()

    def percentile(p: float) -> int:
        return lengths[min(len(lengths) - 1, int(len(lengths) * p))]

    return {
        "model": model_name,
        "sampled": len(lengths),
        "mean": round(sum(lengths) / len(lengths), 1),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": lengths[-1],
        "coverage": {
            n: round(100.0 * sum(1 for x in lengths if x <= n) / len(lengths), 2)
            for n in candidates
        },
    }
