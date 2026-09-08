"""End-to-end orchestration of Phases 1-7.

This module contains **no analysis logic of its own**. Every stage is a call
into an existing module; the only thing added here is sequencing, the joining
of stages on ``clause_id``, and failure isolation. If a behaviour is wrong,
the fix belongs in the phase that owns it, not here.

    PDF
     -> extract_document / segment_clauses        (Phase 1)
     -> ClauseClassifier.predict                  (Phase 3)
     -> analyze_clauses: calibration, salience, flags   (Phase 4)
     -> attach_extractions                        (Phase 5)
     -> build_improved -> RAGContextBuilder       (Phase 1 / 6)
     -> GenerationPipeline.run                    (Phase 7)

**The clause id is the join key throughout.** ``Clause.clause_id`` becomes
``ClauseAnalysis.clause_id``, ``ExtractedItem.clause_id``,
``RetrievalResult.clause_id``, ``RetrievedClause.clause_id`` and finally
``GeneratedResponse.source_clause_ids``. A test asserts the identifier is
unchanged at every hop, which is what makes a generated sentence traceable
back to a page of a PDF.

**Degraded mode.** The classifier is a downloaded artefact and may be absent.
Rather than crashing, the pipeline runs without it: clauses are marked
``UNCLASSIFIED``, salience collapses to zero, and ``degraded`` is set on the
result. Flags and extraction are rule-based and still work. Degradation is
always recorded, never silent -- an interface must be able to tell the user
that categories are missing rather than showing them blanks.

Every component is injectable, so the integration tests run without
downloading Legal-BERT, MPNet-QA or FLAN-T5.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from src.config import PATHS
from src.document_processing import extract_document, segment_clauses
from src.extraction import attach_extractions
from src.generation import GenerationPipeline, GenerationTask
from src.importance import TemperatureScaler, analyze_clauses, top_k_salient
from src.rag import RAGContextBuilder

if TYPE_CHECKING:  # pragma: no cover
    from src.document_processing import Clause, Document
    from src.extraction import DocumentExtraction
    from src.generation import GeneratedResponse
    from src.importance import ClauseAnalysis
    from src.rag import RAGContext

logger = logging.getLogger(__name__)

__all__ = [
    "UNCLASSIFIED",
    "DocumentAnalysis",
    "QueryResult",
    "LegalDocumentPipeline",
]

# Sentinel category used when no classifier is available.
UNCLASSIFIED = "Unclassified"


@dataclass
class DocumentAnalysis:
    """Everything the pipeline knows about one document."""

    path: Path
    document: "Document"
    clauses: list["Clause"]
    analyses: list["ClauseAnalysis"]
    extraction: "DocumentExtraction | None" = None
    retriever: object | None = None
    degraded: bool = False
    stage_errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)

    @property
    def clause_ids(self) -> list[int]:
        return [c.clause_id for c in self.clauses]

    @property
    def n_classified(self) -> int:
        return sum(1 for a in self.analyses if a.category != UNCLASSIFIED)

    @property
    def n_with_flags(self) -> int:
        return sum(1 for a in self.analyses if a.flags)

    @property
    def n_with_extractions(self) -> int:
        return sum(1 for a in self.analyses if a.extractions)

    def top_salient(self, k: int = 5) -> list["ClauseAnalysis"]:
        return top_k_salient(self.analyses, k=k)

    def summary(self) -> dict:
        return {
            "document": self.path.name,
            "pages": self.document.n_pages,
            "words": self.document.word_count,
            "clauses": self.n_clauses,
            "classified": self.n_classified,
            "with_flags": self.n_with_flags,
            "with_extractions": self.n_with_extractions,
            "extracted_items": len(self.extraction.items) if self.extraction else 0,
            "degraded": self.degraded,
            "stage_errors": self.stage_errors,
            "seconds": round(self.seconds, 2),
        }


@dataclass
class QueryResult:
    """One question answered against one analysed document."""

    query: str
    task: str
    document: str
    context: "RAGContext | None" = None
    response: "GeneratedResponse | None" = None
    error: str | None = None
    seconds: float = 0.0

    @property
    def retrieved_clause_ids(self) -> list[int]:
        return [c.clause_id for c in self.context.retrieved] if self.context else []

    @property
    def retrieved_titles(self) -> list[str]:
        return [c.title for c in self.context.retrieved] if self.context else []

    @property
    def empty_context(self) -> bool:
        return self.context is None or self.context.is_empty

    @property
    def is_grounded(self) -> bool:
        return bool(self.response and self.response.is_grounded)

    def to_dict(self) -> dict:
        """Machine-readable record, including per-clause metadata."""
        clauses = []
        if self.context:
            for clause in self.context.retrieved:
                clauses.append({
                    "clause_id": clause.clause_id,
                    "title": clause.title,
                    "page": clause.page,
                    "score": round(clause.score, 4),
                    "category": clause.category,
                    "salience": (
                        round(clause.salience, 4)
                        if clause.salience is not None else None
                    ),
                    "calibrated_confidence": (
                        round(clause.calibrated_confidence, 4)
                        if clause.calibrated_confidence is not None else None
                    ),
                    "flags": clause.flag_labels,
                    "extractions": [
                        {
                            "type": item.type,
                            "value": item.value,
                            "normalized_value": item.normalized_value,
                            "is_placeholder": item.is_placeholder,
                        }
                        for item in clause.extractions
                    ],
                    "selection_reason": clause.selection_reason,
                    "included_fully": clause.included_fully,
                })

        return {
            "query": self.query,
            "task": self.task,
            "document": self.document,
            "retrieved_clause_ids": self.retrieved_clause_ids,
            "retrieved_titles": self.retrieved_titles,
            "retrieved_clauses": clauses,
            "context_chars": self.context.char_count if self.context else 0,
            "context_token_estimate": (
                self.context.token_estimate if self.context else 0
            ),
            "generated_text": self.response.text if self.response else None,
            "source_clause_ids": (
                self.response.source_clause_ids if self.response else []
            ),
            "source_titles": self.response.source_titles if self.response else [],
            "source_citations": (
                self.response.source_citations if self.response else []
            ),
            "is_grounded": self.is_grounded,
            "empty_context": self.empty_context,
            "unsupported_numbers": (
                self.response.grounding.unsupported_numbers
                if self.response and self.response.grounding else []
            ),
            "error": self.error,
            "seconds": round(self.seconds, 2),
        }


class LegalDocumentPipeline:
    """Orchestrates the Phase 1-7 components.

    Args:
        classifier: A ``ClauseClassifier``. When None the pipeline attempts to
            load one from ``PATHS.classifier_dir`` and falls back to degraded
            mode if it is absent.
        scaler: Phase 4 temperature scaler. Loaded from
            ``evaluation/results/temperature.json`` when available; without it
            calibrated confidence equals raw confidence, which
            ``analyze_clauses`` already records honestly.
        generator: Phase 7 ``GenerationPipeline``. Constructed lazily so
            documents can be analysed without loading FLAN-T5.
        top_k: Clauses per RAG context.
        budget_chars: Phase 6 context budget.
    """

    def __init__(
        self,
        *,
        classifier=None,
        scaler: TemperatureScaler | None = None,
        generator: GenerationPipeline | None = None,
        retriever_factory=None,
        top_k: int = 3,
        budget_chars: int | None = None,
        include_flagged: bool = False,
    ) -> None:
        self.classifier = classifier
        self.scaler = scaler
        self._generator = generator
        self._retriever_factory = retriever_factory
        self.top_k = top_k
        self.budget_chars = budget_chars
        self.include_flagged = include_flagged

    # -- lazy component loading -----------------------------------------

    @classmethod
    def from_defaults(cls, **kwargs) -> "LegalDocumentPipeline":
        """Build a pipeline using on-disk artefacts where they exist."""
        pipeline = cls(**kwargs)
        pipeline._load_classifier()
        pipeline._load_scaler()
        return pipeline

    def _load_classifier(self) -> None:
        if self.classifier is not None:
            return
        try:
            from src.classification import ClauseClassifier

            self.classifier = ClauseClassifier(PATHS.classifier_dir)
            logger.info("Loaded classifier from %s", PATHS.classifier_dir)
        except Exception as error:  # noqa: BLE001 - degraded mode is the point
            logger.warning(
                "No classifier available (%s); running in degraded mode.", error
            )

    def _load_scaler(self) -> None:
        if self.scaler is not None:
            return
        path = PATHS.root / "evaluation" / "results" / "temperature.json"
        if path.exists():
            self.scaler = TemperatureScaler.load(path)
            logger.info("Loaded temperature T=%.4f", self.scaler.temperature)
        else:
            logger.warning(
                "No temperature.json; calibrated confidence will equal raw "
                "confidence."
            )

    @property
    def generator(self) -> GenerationPipeline:
        if self._generator is None:
            self._generator = GenerationPipeline()
        return self._generator

    def _build_retriever(self, clauses: Sequence["Clause"]):
        if self._retriever_factory is not None:
            return self._retriever_factory(clauses)
        from src.retrieval import build_improved

        return build_improved(clauses)

    # -- stage 1: document analysis --------------------------------------

    def _predict(self, clauses: Sequence["Clause"]) -> tuple[list[dict], bool]:
        """Classify clauses, or produce degraded placeholders."""
        if self.classifier is None:
            return (
                [
                    {"category": UNCLASSIFIED, "confidence": 0.0, "probabilities": {}}
                    for _ in clauses
                ],
                True,
            )
        return self.classifier.predict([c.text for c in clauses]), False

    def analyze_document(self, path: str | Path) -> DocumentAnalysis:
        """Run Phases 1-6 over one PDF and return the analysed document.

        Generation is not performed here: it is per-query, and analysing a
        document should not require FLAN-T5.
        """
        path = Path(path)
        started = time.perf_counter()
        errors: list[str] = []

        document = extract_document(path)
        clauses = segment_clauses(document)
        logger.info("%s: %d clauses", path.name, len(clauses))

        predictions, degraded = self._predict(clauses)
        analyses = analyze_clauses(clauses, predictions, scaler=self.scaler)

        # Extraction is an enrichment; a failure there must not lose the
        # classification work already done.
        extraction = None
        try:
            extraction = attach_extractions(analyses)
        except Exception as error:  # noqa: BLE001
            errors.append(f"extraction: {type(error).__name__}: {error}")
            logger.exception("Extraction failed for %s", path.name)

        retriever = None
        try:
            retriever = self._build_retriever(clauses)
        except Exception as error:  # noqa: BLE001
            errors.append(f"retrieval: {type(error).__name__}: {error}")
            logger.exception("Retriever construction failed for %s", path.name)

        return DocumentAnalysis(
            path=path,
            document=document,
            clauses=clauses,
            analyses=analyses,
            extraction=extraction,
            retriever=retriever,
            degraded=degraded,
            stage_errors=errors,
            seconds=time.perf_counter() - started,
        )

    # -- stage 2: question answering -------------------------------------

    def context_for(self, analysis: DocumentAnalysis, query: str) -> "RAGContext":
        """Build a Phase 6 context. Raises if no retriever was constructed."""
        if analysis.retriever is None:
            raise RuntimeError(
                f"No retriever for {analysis.path.name}; cannot build context."
            )
        kwargs = {
            "analyses": analysis.analyses,
            "top_k": self.top_k,
            "include_flagged": self.include_flagged,
        }
        if self.budget_chars is not None:
            kwargs["budget_chars"] = self.budget_chars
        return RAGContextBuilder(analysis.retriever, **kwargs).build(query)

    def ask(
        self,
        analysis: DocumentAnalysis,
        query: str,
        *,
        task: str = GenerationTask.SIMPLE_EXPLANATION,
    ) -> QueryResult:
        """Answer a question against an analysed document.

        Failures are captured in the result rather than raised, so one bad
        query cannot abort a batch evaluation.
        """
        started = time.perf_counter()
        result = QueryResult(query=query, task=task, document=analysis.path.name)

        try:
            context = self.context_for(analysis, query)
            result.context = context

            clause = None
            if task == GenerationTask.CLAUSE_EXPLANATION and context.retrieved:
                clause = context.retrieved[0]

            # Generation receives the context and nothing else.
            result.response = self.generator.run(task, context, clause=clause)
        except Exception as error:  # noqa: BLE001
            result.error = f"{type(error).__name__}: {error}"
            logger.exception("Query failed: %s", query)

        result.seconds = time.perf_counter() - started
        return result

    def run(
        self,
        path: str | Path,
        queries: Sequence[dict],
        *,
        analysis: DocumentAnalysis | None = None,
    ) -> tuple[DocumentAnalysis, list[QueryResult]]:
        """Analyse a document and answer a batch of queries against it.

        Args:
            path: The PDF.
            queries: Dicts with ``query`` and optionally ``task``.
            analysis: Reuse an existing analysis instead of recomputing.
        """
        analysis = analysis or self.analyze_document(path)
        results = [
            self.ask(
                analysis,
                item["query"],
                task=item.get("task", GenerationTask.SIMPLE_EXPLANATION),
            )
            for item in queries
        ]
        return analysis, results
