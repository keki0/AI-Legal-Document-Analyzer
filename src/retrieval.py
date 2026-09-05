"""Semantic retrieval over segmented clauses.

The baseline retriever reproduces ``01_baseline.ipynb`` exactly: body-only
clause text, ``all-MiniLM-L6-v2``, cosine similarity. It exists so the
"before" system can be re-run on demand rather than quoted from a notebook.

The improved retriever changes two things, kept separable so their
contributions can be measured independently:

1. **Representation.** Each clause is embedded with its heading prepended.
   A clause about IP ownership may never contain the word "own"; its heading
   says "Intellectual Property". Body-only embedding discards that.

2. **Model.** ``all-MiniLM-L6-v2`` is trained for *symmetric* semantic
   similarity — how alike are two sentences. Clause search is *asymmetric*:
   a short question against a longer passage. ``multi-qa-mpnet-base-dot-v1``
   is trained on exactly that objective.

An optional BM25 stage adds lexical matching via reciprocal rank fusion. It
is off by default and degrades to pure dense retrieval if ``rank_bm25`` is
absent, so it cannot destabilise the pipeline.

``sentence_transformers`` is imported lazily so this module can be imported
for its types alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from src.document_processing import Clause

logger = logging.getLogger(__name__)

__all__ = ["RetrievalResult", "ClauseRetriever", "build_baseline", "build_improved"]

BASELINE_MODEL = "all-MiniLM-L6-v2"
IMPROVED_MODEL = "multi-qa-mpnet-base-dot-v1"


@dataclass
class RetrievalResult:
    """One retrieved clause and its score."""

    rank: int
    clause_id: int
    title: str
    score: float
    text: str
    page: int

    def __str__(self) -> str:
        return f"#{self.rank} [{self.score:.4f}] {self.title} (p{self.page})"


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class ClauseRetriever:
    """Dense clause retriever, optionally fused with BM25.

    Args:
        clauses: Segmented clauses to index.
        model_name: Sentence-Transformers model identifier.
        title_augmented: Prepend each clause heading to its body before
            embedding. False reproduces the baseline representation.
        use_bm25: Fuse a BM25 ranking with the dense ranking. Silently
            disabled if ``rank_bm25`` is not installed.
    """

    def __init__(
        self,
        clauses: Sequence["Clause"],
        *,
        model_name: str = BASELINE_MODEL,
        title_augmented: bool = False,
        use_bm25: bool = False,
    ) -> None:
        if not clauses:
            raise ValueError("Cannot build a retriever over zero clauses.")

        self.clauses = list(clauses)
        self.model_name = model_name
        self.title_augmented = title_augmented

        self.chunks = [
            clause.as_chunk() if title_augmented else clause.text
            for clause in self.clauses
        ]

        from sentence_transformers import SentenceTransformer  # lazy

        logger.info(
            "Loading %s (title_augmented=%s) over %d clauses",
            model_name, title_augmented, len(self.chunks),
        )
        self.model = SentenceTransformer(model_name)
        self.embeddings = _normalise(
            self.model.encode(self.chunks, convert_to_numpy=True, show_progress_bar=False)
        )

        self.bm25 = None
        if use_bm25:
            try:
                from rank_bm25 import BM25Okapi

                self.bm25 = BM25Okapi([chunk.lower().split() for chunk in self.chunks])
                logger.info("BM25 stage enabled.")
            except ImportError:
                logger.warning("rank_bm25 not installed; continuing dense-only.")

    def _dense_scores(self, query: str) -> np.ndarray:
        vector = self.model.encode([query], convert_to_numpy=True)
        return (_normalise(vector) @ self.embeddings.T)[0]

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return the top-k clauses for a query, highest score first."""
        scores = self._dense_scores(query)

        if self.bm25 is not None:
            # Reciprocal rank fusion. Rank-based rather than score-based
            # because BM25 and cosine scores are on incomparable scales.
            lexical = np.asarray(self.bm25.get_scores(query.lower().split()))
            k = 60.0
            dense_rank = np.empty_like(scores)
            dense_rank[np.argsort(scores)[::-1]] = np.arange(len(scores))
            lex_rank = np.empty_like(lexical)
            lex_rank[np.argsort(lexical)[::-1]] = np.arange(len(lexical))
            scores = 1.0 / (k + dense_rank + 1) + 1.0 / (k + lex_rank + 1)

        order = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalResult(
                rank=position + 1,
                clause_id=self.clauses[index].clause_id,
                title=self.clauses[index].title,
                score=float(scores[index]),
                text=self.clauses[index].text,
                page=self.clauses[index].page_start,
            )
            for position, index in enumerate(order)
        ]


def build_baseline(clauses: Sequence["Clause"]) -> ClauseRetriever:
    """The documented "before" system. Do not change this configuration."""
    return ClauseRetriever(
        clauses, model_name=BASELINE_MODEL, title_augmented=False, use_bm25=False
    )


def build_improved(
    clauses: Sequence["Clause"], *, use_bm25: bool = False
) -> ClauseRetriever:
    """Title-augmented chunks with an asymmetric retrieval model."""
    return ClauseRetriever(
        clauses, model_name=IMPROVED_MODEL, title_augmented=True, use_bm25=use_bm25
    )
