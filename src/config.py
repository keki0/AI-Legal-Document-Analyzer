"""Central configuration for the AI Legal Document Analyzer.

Single source of truth for paths, model identifiers and runtime settings, so
that the same code runs unchanged on a local Windows machine and in a Google
Colab notebook.

Two things this module deliberately gets right:

1. Paths are derived from this file's own location, never from the current
   working directory. The existing baseline notebook uses
   ``Path.cwd().parent``, which is correct only when the kernel starts in
   ``notebooks/``. That assumption breaks in Colab and breaks again when a
   module is imported from the repository root. Anchoring on ``__file__``
   removes the problem entirely.

2. ``torch`` is imported lazily, inside the functions that need it, so this
   module can be imported for its path constants alone in an environment
   where the deep-learning stack is not installed.

Usage::

    from src.config import PATHS, MODELS, get_device, set_seed

    set_seed()
    print(PATHS.data_raw)
    print(get_device())
"""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

__all__ = [
    "PATHS",
    "MODELS",
    "DATASETS",
    "TRAIN",
    "SEED",
    "IN_COLAB",
    "get_device",
    "set_seed",
    "ensure_dirs",
    "add_repo_to_syspath",
    "describe_environment",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Environment detection
# --------------------------------------------------------------------------

def _detect_colab() -> bool:
    """Return True when running inside Google Colab.

    Checks for the ``google.colab`` module rather than an environment
    variable, because the module is present only in a genuine Colab kernel.
    """
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


IN_COLAB: Final[bool] = _detect_colab()

# Repository root = parent of the directory holding this file (src/).
# Resolved from __file__, so it is independent of the current working
# directory and of whether the caller is a notebook, a test or the app.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    """Filesystem locations used across the project."""

    root: Path = REPO_ROOT
    app: Path = REPO_ROOT / "app"
    data: Path = REPO_ROOT / "data"
    data_raw: Path = REPO_ROOT / "data" / "raw"
    data_processed: Path = REPO_ROOT / "data" / "processed"
    data_eval: Path = REPO_ROOT / "data" / "eval"
    docs: Path = REPO_ROOT / "docs"
    models: Path = REPO_ROOT / "models"
    notebooks: Path = REPO_ROOT / "notebooks"
    tests: Path = REPO_ROOT / "tests"

    # Specific files referenced from more than one place.
    sample_pdf: Path = (
        REPO_ROOT / "data" / "raw" / "sample-independent-contractor-agreement.pdf"
    )
    taxonomy: Path = REPO_ROOT / "data" / "eval" / "taxonomy.yaml"
    retrieval_queries: Path = REPO_ROOT / "data" / "eval" / "retrieval_queries.json"

    # Destination for the fine-tuned classifier, downloaded from Colab.
    classifier_dir: Path = REPO_ROOT / "models" / "legal-bert-clause-classifier"

    def writable(self) -> tuple[Path, ...]:
        """Directories that should exist before anything writes to them."""
        return (
            self.data_raw,
            self.data_processed,
            self.data_eval,
            self.models,
            self.docs,
        )


PATHS: Final[Paths] = Paths()


def ensure_dirs() -> None:
    """Create any missing writable directories. Safe to call repeatedly."""
    for directory in PATHS.writable():
        directory.mkdir(parents=True, exist_ok=True)


def add_repo_to_syspath() -> Path:
    """Put the repository root on ``sys.path`` and return it.

    Needed in Colab, where the notebook's working directory is ``/content``
    and ``import src.config`` would otherwise fail.
    """
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelIDs:
    """Hugging Face model identifiers, fixed in one place.

    Changing a model means changing it here, not hunting through notebooks.
    """

    # Primary deep-learning component: fine-tuned for clause classification.
    classifier: str = "nlpaueb/legal-bert-base-uncased"

    # Smaller fallback. Switch to this only if the Day 1 timing gate shows the
    # base model will not finish inside the available Colab window.
    classifier_fallback: str = "nlpaueb/legal-bert-small-uncased"

    # Retrieval. `baseline` reproduces the existing notebook and must not
    # change — it is the documented "before" system.
    retrieval_baseline: str = "all-MiniLM-L6-v2"
    # Trained for asymmetric query->passage retrieval, unlike the symmetric
    # similarity model above. This mismatch is one identified cause of the
    # baseline retrieval failure.
    retrieval_improved: str = "multi-qa-mpnet-base-dot-v1"

    # Generation: clause simplification and document summarisation.
    generator: str = "google/flan-t5-base"

    # Named entity recognition.
    spacy_model: str = "en_core_web_sm"


MODELS: Final[ModelIDs] = ModelIDs()


@dataclass(frozen=True)
class DatasetIDs:
    """Dataset identifiers and their loading quirks."""

    ledgar_repo: str = "coastalcph/lex_glue"
    ledgar_config: str = "ledgar"
    # The lex_glue repository carries a loading script, so datasets 2.x
    # requires explicit opt-in. See requirements.txt for why the version is
    # held below 3.0.
    ledgar_trust_remote_code: bool = True

    # Manor & Li (2019) plain-English contract summaries, already parsed.
    # 446 pairs, a single "train" split. Used as a reference set for
    # evaluating clause simplification.
    simplification_repo: str = "joelniklaus/plain_english_contracts_summarization"


DATASETS: Final[DatasetIDs] = DatasetIDs()


# --------------------------------------------------------------------------
# Training defaults
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """Starting hyperparameters for classifier fine-tuning.

    ``max_length`` and ``train_subsample`` are provisional. Both are set from
    evidence in Phase 3: the token-length distribution of LEDGAR provisions,
    and a timed short run extrapolated to a full epoch.
    """

    max_length: int = 256
    train_batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    num_epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    # Stratified subsample of the 60k LEDGAR training split. Fourteen mapped
    # categories do not need the full set, and this keeps a free-tier T4
    # session inside a safe window.
    train_subsample: int | None = 20_000
    fp16: bool = True  # ignored automatically when no GPU is present


TRAIN: Final[TrainConfig] = TrainConfig()

SEED: Final[int] = 42


# --------------------------------------------------------------------------
# Runtime helpers
# --------------------------------------------------------------------------

def get_device() -> str:
    """Return ``"cuda"`` when a GPU is usable, otherwise ``"cpu"``.

    torch is imported here rather than at module scope so that path constants
    remain importable without the deep-learning stack installed.
    """
    try:
        import torch
    except ImportError:
        logger.warning("torch is not installed; falling back to CPU.")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED) -> int:
    """Seed Python, NumPy and torch. Returns the seed for logging.

    Full determinism is not guaranteed on GPU; this makes runs reproducible
    enough to compare, which is what the evaluation requires.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        logger.warning("numpy is not installed; skipping its seed.")

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.warning("torch is not installed; skipping its seed.")

    return seed


def describe_environment() -> dict[str, object]:
    """Collect a snapshot of the runtime for logging and reports."""
    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": sys.platform,
        "in_colab": IN_COLAB,
        "repo_root": str(REPO_ROOT),
        "device": get_device(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ensure_dirs()
    for key, value in describe_environment().items():
        print(f"{key:>12}: {value}")
