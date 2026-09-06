"""Regression tests for the generated Colab notebook.

These exist because of a real defect: the generator built ``source`` with
``text.split("\n")``, which strips the newline characters. nbformat joins
``source`` with ``"".join(...)``, so every statement in a cell was
concatenated onto one line and Colab raised SyntaxError on open.

The original validation missed it by joining with ``"\n".join(...)`` --
reconstructing the intended text rather than the artifact. Every check here
therefore uses ``"".join(...)``, exactly as a notebook reader does.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "03_classification_colab.ipynb"
BASELINE = Path(__file__).resolve().parents[1] / "notebooks" / "01_baseline.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip("notebook not generated; run scripts/build_colab_notebook.py")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _strip_shell(source: str) -> str:
    return "\n".join(
        line for line in source.split("\n")
        if not line.lstrip().startswith(("!", "%"))
    )


def test_every_source_line_keeps_its_newline(notebook):
    """The exact defect. Without trailing newlines, statements run together."""
    offenders = [
        (index, position, entry[:60])
        for index, cell in enumerate(notebook["cells"])
        for position, entry in enumerate(cell["source"][:-1])
        if not entry.endswith("\n")
    ]
    assert not offenders, f"source entries missing trailing newline: {offenders}"


def test_all_code_cells_parse_as_python(notebook):
    """Joined with "".join, as nbformat does -- never "\\n".join."""
    failures = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        try:
            ast.parse(_strip_shell("".join(cell["source"])))
        except SyntaxError as error:
            failures.append((index, error.lineno, error.msg))
    assert not failures, f"cells failed to parse: {failures}"


def test_multi_statement_cells_actually_span_multiple_lines(notebook):
    """Guards against a cell collapsing to one line while still parsing.

    Sequential imports concatenated onto one line are a SyntaxError, but some
    concatenations happen to remain valid Python. This checks structure, not
    just parseability.
    """
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if len(cell["source"]) > 3:
            assert source.count("\n") >= 2, (
                f"cell has {len(cell['source'])} source entries but "
                f"{source.count(chr(10))} newlines: {source[:80]!r}"
            )


def test_notebook_validates_against_nbformat(notebook):
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    nbformat.validate(nb)


def test_approved_methodology_is_unchanged(notebook):
    """Regeneration must not silently alter the agreed configuration."""
    source = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    for fragment in (
        "MAX_LENGTH = 384",            # measured p95 = 360
        "BATCH  = 16",                 # 384-token seqs at 32 risk OOM on T4
        "ACCUM  = 2",                  # effective batch 32
        "max_steps=100",               # the timing gate
        "/content/drive/MyDrive/legal-doc-analyzer",
        "weight=class_weights_t",      # class-weighted loss
        "DataCollatorWithPadding",     # dynamic padding
        "metric_for_best_model='macro_f1'",
    ):
        assert fragment in source, f"methodology changed: {fragment!r} missing"


def test_baseline_notebook_untouched():
    import hashlib

    if not BASELINE.exists():
        pytest.skip("baseline notebook not present")
    digest = hashlib.md5(BASELINE.read_bytes()).hexdigest()
    assert digest == "7ba8d2cc297b9f61f65c7534d4ef7820", (
        "01_baseline.ipynb has been modified; it must stay frozen as the "
        "documented 'before' system"
    )
