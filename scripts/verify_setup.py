"""Phase 0 verification.

Run this after installing dependencies, on the local machine and again in
Colab. It checks the environment, confirms the baseline is intact, and tries
to load LEDGAR.

    python scripts/verify_setup.py            # skip the dataset download
    python scripts/verify_setup.py --data     # also download LEDGAR (~a few hundred MB)

Nothing here writes to the baseline notebook or to data/processed.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Make `src` importable no matter where this is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    DATASETS,
    MODELS,
    PATHS,
    describe_environment,
    ensure_dirs,
    set_seed,
)

PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"

# (import name, name shown to the user)
CORE_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("sentence_transformers", "sentence-transformers"),
    ("datasets", "datasets"),
    ("huggingface_hub", "huggingface_hub"),
    ("pymupdf", "pymupdf"),
    ("spacy", "spacy"),
    ("yaml", "pyyaml"),
]

# Not needed until later phases; missing ones are warnings, not failures.
OPTIONAL_PACKAGES = [
    ("streamlit", "streamlit"),
    ("evaluate", "evaluate"),
    ("rouge_score", "rouge-score"),
    ("bert_score", "bert-score"),
    ("rank_bm25", "rank-bm25"),
    ("accelerate", "accelerate"),
    ("sentencepiece", "sentencepiece"),
]

failures: list[str] = []
warnings: list[str] = []


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def check_packages(packages: list[tuple[str, str]], *, required: bool) -> None:
    for module_name, display_name in packages:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "?")
            print(f"{PASS} {display_name:<24} {version}")
        except ImportError as exc:
            if required:
                print(f"{FAIL} {display_name:<24} {exc}")
                failures.append(f"{display_name} is not installed")
            else:
                print(f"{WARN} {display_name:<24} not installed (needed later)")
                warnings.append(f"{display_name} not installed")


def check_numpy_abi() -> None:
    """Catch the numpy 1.x / 2.x ABI mismatch early.

    The symptom otherwise is an opaque crash deep inside another import.
    """
    try:
        import numpy as np
    except ImportError:
        return
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        print(f"{WARN} numpy {np.__version__} — requirements.txt pins <2.0")
        warnings.append(
            "numpy 2.x detected; if imports crash, run: pip install 'numpy<2.0'"
        )
    else:
        print(f"{PASS} numpy ABI       {np.__version__} (below 2.0, as pinned)")


def check_paths() -> None:
    ensure_dirs()
    for label, path in [
        ("repo root", PATHS.root),
        ("data/raw", PATHS.data_raw),
        ("data/processed", PATHS.data_processed),
        ("data/eval", PATHS.data_eval),
        ("models", PATHS.models),
        ("notebooks", PATHS.notebooks),
    ]:
        state = PASS if path.exists() else FAIL
        print(f"{state} {label:<16} {path}")
        if not path.exists():
            failures.append(f"missing directory: {path}")


def check_baseline_intact() -> None:
    """Confirm the baseline notebook is present and unmodified in shape.

    Phase 0 must not touch it. This reports its cell count so any later
    accidental edit is visible in a diff of the verification output.
    """
    notebook = PATHS.notebooks / "01_baseline.ipynb"
    if not notebook.exists():
        print(f"{FAIL} 01_baseline.ipynb not found at {notebook}")
        failures.append("baseline notebook missing")
        return

    import json

    with notebook.open(encoding="utf-8") as handle:
        content = json.load(handle)
    cells = len(content.get("cells", []))
    with_output = sum(1 for cell in content["cells"] if cell.get("outputs"))
    print(f"{PASS} 01_baseline.ipynb present — {cells} cells, {with_output} with outputs")
    print("       (expected 21 cells / 18 with outputs — unchanged since Phase 0)")

    if PATHS.sample_pdf.exists():
        size_kb = PATHS.sample_pdf.stat().st_size // 1024
        print(f"{PASS} sample PDF present ({size_kb} KB)")
    else:
        print(f"{WARN} sample PDF not at {PATHS.sample_pdf}")
        warnings.append("sample contractor PDF missing from data/raw/")


def check_spacy_model() -> None:
    try:
        import spacy
    except ImportError:
        return
    try:
        spacy.load(MODELS.spacy_model)
        print(f"{PASS} spaCy model      {MODELS.spacy_model}")
    except OSError:
        print(f"{WARN} spaCy model {MODELS.spacy_model} not downloaded")
        warnings.append(
            f"run: python -m spacy download {MODELS.spacy_model}"
        )


def check_ledgar(download: bool) -> None:
    """Attempt to load LEDGAR, trying each known access path in turn."""
    if not download:
        print(f"{WARN} skipped (pass --data to actually download)")
        warnings.append("LEDGAR load not verified — rerun with --data")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print(f"{FAIL} datasets not installed")
        failures.append("cannot verify LEDGAR: datasets missing")
        return

    attempts = [
        (
            "script loader + trust_remote_code",
            lambda: load_dataset(
                DATASETS.ledgar_repo,
                DATASETS.ledgar_config,
                trust_remote_code=True,
            ),
        ),
        (
            "script loader without trust_remote_code",
            lambda: load_dataset(DATASETS.ledgar_repo, DATASETS.ledgar_config),
        ),
        (
            "legacy canonical name 'lex_glue'",
            lambda: load_dataset("lex_glue", DATASETS.ledgar_config, trust_remote_code=True),
        ),
    ]

    dataset = None
    for description, loader in attempts:
        try:
            print(f"       trying: {description} ...")
            dataset = loader()
            print(f"{PASS} loaded via {description}")
            break
        except Exception as exc:  # noqa: BLE001 - we want the reason, whatever it is
            print(f"{WARN} failed: {type(exc).__name__}: {str(exc)[:160]}")

    if dataset is None:
        print(f"{FAIL} LEDGAR could not be loaded by any method")
        failures.append("LEDGAR load failed — see messages above")
        return

    for split in ("train", "validation", "test"):
        if split in dataset:
            print(f"{PASS} split {split:<11} {len(dataset[split]):,} rows")

    labels = dataset["train"].features["label"].names
    print(f"{PASS} label names      {len(labels)} classes")
    print(f"       first 8: {labels[:8]}")
    print(f"       last 4:  {labels[-4:]}")

    if len(labels) != 100:
        warnings.append(f"expected 100 LEDGAR labels, found {len(labels)}")

    sample = dataset["train"][0]
    print(f"{PASS} sample record    label={labels[sample['label']]!r}, "
          f"{len(sample['text'].split())} words")

    # Written out now so Phase 1 can build taxonomy.yaml without re-downloading.
    out = PATHS.data_eval / "ledgar_label_names.txt"
    out.write_text("\n".join(labels), encoding="utf-8")
    print(f"{PASS} label names written to {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        action="store_true",
        help="also download and verify LEDGAR (several hundred MB)",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("AI LEGAL DOCUMENT ANALYZER — PHASE 0 VERIFICATION")
    print("=" * 62)

    section("Environment")
    for key, value in describe_environment().items():
        print(f"{key:>12}: {value}")
    set_seed()

    section("Required packages")
    check_packages(CORE_PACKAGES, required=True)
    check_numpy_abi()

    section("Optional packages (needed in later phases)")
    check_packages(OPTIONAL_PACKAGES, required=False)

    section("Paths")
    check_paths()

    section("Baseline integrity")
    check_baseline_intact()

    section("spaCy model")
    check_spacy_model()

    section("LEDGAR dataset")
    check_ledgar(args.data)

    section("Summary")
    if failures:
        print(f"{FAIL} {len(failures)} blocking problem(s):")
        for item in failures:
            print(f"       - {item}")
    else:
        print(f"{PASS} no blocking problems")

    if warnings:
        print(f"{WARN} {len(warnings)} warning(s):")
        for item in warnings:
            print(f"       - {item}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
