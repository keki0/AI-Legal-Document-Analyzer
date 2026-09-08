"""Lightweight RAG: retrieval, context selection, and grounded context building.

Phase 6 stops short of generation. It produces a structured, provenance-
carrying context that Phase 7 will hand to FLAN-T5. No text is generated here
and no claim is made about answer quality, because no answer is produced.

**Nothing in `src/retrieval.py` is modified.** ``ClauseRetriever`` already
returns ``RetrievalResult`` objects carrying ``clause_id``, and
``ClauseAnalysis`` is keyed by the same field, so Phase 4/5 metadata joins on
that key. This module declares a :class:`Retriever` protocol that the existing
retriever already satisfies, which also makes the context builder testable
without downloading a model.

Three design decisions worth stating plainly:

**Ranking stays purely semantic.** Salience and classifier confidence are
attached as metadata, never multiplied into the similarity score. Reordering
retrieval by an importance prior would confound two different signals and make
the retrieval evaluation uninterpretable. Salience answers "how much should a
reader care about this clause"; similarity answers "does this clause address
this question". They are not interchangeable.

**Flag support is a context-selection rule, not a ranking rule**, and it is
OFF by default. Phase 4 found that a low-salience "Boilerplate &
Administrative" clause can still carry Choice-of-law and Arbitration flags, so
important evidence can sit far down a salience ordering. But appending every
flagged clause to every query would flood the context. The rule implemented
here admits a flagged clause only if it is already in the retriever's
candidate pool *and* scores above a fraction of the top hit -- so it must
still be semantically relevant to the question. Because it is a secondary
rule, ``scripts/evaluate_retrieval.py`` measures what it actually does rather
than assuming it helps.

**Complete clauses are preferred to truncated ones.** A clause that does not
fit the budget is skipped rather than cut, unless it is the top hit and
nothing else would fit, in which case it is abridged at a sentence boundary
and marked ``included_fully=False``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable, Protocol, Sequence

from src.importance import split_sentences

if TYPE_CHECKING:  # pragma: no cover
    from src.importance import ClauseAnalysis
    from src.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

__all__ = [
    "Retriever",
    "RetrievedClause",
    "RAGContext",
    "RAGContextBuilder",
    "estimate_tokens",
]

# FLAN-T5 accepts 512 input tokens. At roughly four characters per token for
# English prose, ~1600 characters of clause text leaves headroom for the
# instruction prompt and the question itself. Configurable, and Phase 7 should
# override the estimator with the real tokenizer.
DEFAULT_BUDGET_CHARS = 1600
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Approximate token count from character length.

    A heuristic, deliberately. Phase 7 has the FLAN-T5 tokenizer and should
    inject it via ``RAGContextBuilder(token_counter=...)`` for exact counts;
    pulling a tokenizer in here would add a model download to a module that
    otherwise needs none.
    """
    return int(len(text) / CHARS_PER_TOKEN) + 1


class Retriever(Protocol):
    """The retrieval interface this module depends on.

    ``src.retrieval.ClauseRetriever`` already satisfies this, so no change to
    Phase 1 code is required. Declaring it as a protocol also allows the
    context builder to be tested without downloading embeddings.
    """

    def search(self, query: str, top_k: int = 3) -> "list[RetrievalResult]": ...


@dataclass
class RetrievedClause:
    """A retrieved clause with its provenance and any available metadata.

    Phase 4/5 fields are optional: the context builder works on raw clauses
    alone, and is enriched when analyses are supplied. Nothing is copied that
    could be looked up -- ``flags`` and ``extractions`` hold references to the
    same objects Phase 4 and 5 produced.
    """

    rank: int
    score: float
    clause_id: int
    title: str
    text: str
    page: int

    # Joined from ClauseAnalysis on clause_id when available.
    category: str | None = None
    salience: float | None = None
    calibrated_confidence: float | None = None
    flags: list = field(default_factory=list)
    extractions: list = field(default_factory=list)

    # Why this clause is in the context, and whether it survived intact.
    selection_reason: str = "semantic"
    included_fully: bool = True
    original_char_count: int = 0

    @property
    def flag_labels(self) -> list[str]:
        return [getattr(f, "label", str(f)) for f in self.flags]

    @property
    def citation(self) -> str:
        """Short human-readable source reference for a generated answer."""
        return f"Clause {self.clause_id}: {self.title} (p{self.page})"

    def to_dict(self) -> dict:
        payload = {
            "rank": self.rank,
            "score": self.score,
            "clause_id": self.clause_id,
            "title": self.title,
            "text": self.text,
            "page": self.page,
            "category": self.category,
            "salience": self.salience,
            "calibrated_confidence": self.calibrated_confidence,
            "selection_reason": self.selection_reason,
            "included_fully": self.included_fully,
            "original_char_count": self.original_char_count,
            "citation": self.citation,
            "flag_labels": self.flag_labels,
        }
        payload["flags"] = [
            f.to_dict() if hasattr(f, "to_dict") else str(f) for f in self.flags
        ]
        payload["extractions"] = [
            e.to_dict() if hasattr(e, "to_dict") else str(e) for e in self.extractions
        ]
        return payload


@dataclass
class RAGContext:
    """A grounded context ready for Phase 7.

    Carries both the formatted string a generator consumes and the structured
    records behind it, so an answer can be traced back to source clauses.
    """

    query: str
    retrieved: list[RetrievedClause]
    formatted_context: str
    char_count: int
    token_estimate: int
    budget_chars: int
    within_budget: bool
    dropped_clause_ids: list[int] = field(default_factory=list)
    abridged_clause_ids: list[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.retrieved

    @property
    def sources(self) -> list[str]:
        """Citations for every clause in the context, in context order."""
        return [clause.citation for clause in self.retrieved]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "formatted_context": self.formatted_context,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
            "budget_chars": self.budget_chars,
            "within_budget": self.within_budget,
            "dropped_clause_ids": self.dropped_clause_ids,
            "abridged_clause_ids": self.abridged_clause_ids,
            "sources": self.sources,
            "retrieved": [clause.to_dict() for clause in self.retrieved],
        }


def _abridge(text: str, budget: int) -> tuple[str, bool]:
    """Shorten text at a sentence boundary. Returns ``(text, was_abridged)``.

    Cutting mid-sentence would hand the generator a fragment that could be
    read as a complete provision, so whole sentences are kept and an explicit
    marker records the omission. Falls back to a hard character cut only if
    even the first sentence exceeds the budget.
    """
    if len(text) <= budget:
        return text, False

    marker = " [...]"
    room = max(budget - len(marker), 0)

    kept: list[str] = []
    length = 0
    for _, _, sentence in split_sentences(text):
        addition = len(sentence) + (1 if kept else 0)
        if length + addition > room:
            break
        kept.append(sentence)
        length += addition

    if kept:
        return " ".join(kept) + marker, True
    return text[:room].rstrip() + marker, True


class RAGContextBuilder:
    """Turns a question into a grounded, budgeted context.

    Args:
        retriever: Anything satisfying :class:`Retriever`. In production this
            is the Phase 1 ``ClauseRetriever`` built with MPNet-QA.
        analyses: Optional Phase 4 analyses, joined on ``clause_id`` to attach
            category, salience, flags and extractions.
        top_k: How many clauses to place in the context.
        candidate_pool: How many clauses to retrieve before selection. Larger
            than ``top_k`` so the flag rule has something to draw from.
        budget_chars: Character budget for the assembled clause text.
        include_flagged: Enable the secondary flag-support rule. OFF by
            default; see the module docstring.
        flag_score_ratio: A flagged clause is admitted only if its similarity
            is at least this fraction of the top hit's similarity.
        token_counter: Override the token estimator. Phase 7 should pass the
            FLAN-T5 tokenizer's length function.
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        analyses: Sequence["ClauseAnalysis"] | None = None,
        top_k: int = 3,
        candidate_pool: int = 8,
        budget_chars: int = DEFAULT_BUDGET_CHARS,
        include_flagged: bool = False,
        flag_score_ratio: float = 0.6,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if budget_chars < 1:
            raise ValueError(f"budget_chars must be >= 1, got {budget_chars}")
        if not 0.0 <= flag_score_ratio <= 1.0:
            raise ValueError(
                f"flag_score_ratio must be in [0,1], got {flag_score_ratio}"
            )

        self.retriever = retriever
        self.top_k = top_k
        self.candidate_pool = max(candidate_pool, top_k)
        self.budget_chars = budget_chars
        self.include_flagged = include_flagged
        self.flag_score_ratio = flag_score_ratio
        self.token_counter = token_counter or estimate_tokens
        self._analyses = {a.clause_id: a for a in (analyses or [])}

    # -- enrichment ------------------------------------------------------

    def _enrich(self, result: "RetrievalResult", reason: str) -> RetrievedClause:
        analysis = self._analyses.get(result.clause_id)
        return RetrievedClause(
            rank=result.rank,
            score=result.score,
            clause_id=result.clause_id,
            title=result.title,
            text=result.text,
            page=result.page,
            category=getattr(analysis, "category", None),
            salience=getattr(analysis, "salience", None),
            calibrated_confidence=getattr(analysis, "calibrated_confidence", None),
            flags=list(getattr(analysis, "flags", []) or []),
            extractions=list(getattr(analysis, "extractions", []) or []),
            selection_reason=reason,
            original_char_count=len(result.text),
        )

    # -- selection -------------------------------------------------------

    def select(self, query: str) -> list[RetrievedClause]:
        """Retrieve and select clauses. Ranking is never reordered."""
        if not query or not query.strip():
            return []

        results = self.retriever.search(query, top_k=self.candidate_pool)
        if not results:
            return []

        # Deduplicate by clause_id, keeping the highest-ranked occurrence.
        seen: set[int] = set()
        ordered: list["RetrievalResult"] = []
        for result in results:
            if result.clause_id in seen:
                continue
            seen.add(result.clause_id)
            ordered.append(result)

        selected = [self._enrich(r, "semantic") for r in ordered[: self.top_k]]

        if self.include_flagged:
            selected.extend(self._flag_supported(ordered, selected))
        return selected

    def _flag_supported(
        self,
        candidates: Sequence["RetrievalResult"],
        already: Sequence[RetrievedClause],
    ) -> list[RetrievedClause]:
        """Admit flagged clauses that are relevant but fell outside top-k.

        Deliberately conservative. A clause qualifies only if it carries a
        Phase 4 flag AND appears in the candidate pool AND scores at least
        ``flag_score_ratio`` of the top hit. Without the score floor this would
        append arbitration boilerplate to every unrelated question.
        """
        if not candidates or not self._analyses:
            return []

        chosen = {clause.clause_id for clause in already}
        floor = candidates[0].score * self.flag_score_ratio
        extra: list[RetrievedClause] = []

        for result in candidates[self.top_k:]:
            if result.clause_id in chosen:
                continue
            analysis = self._analyses.get(result.clause_id)
            if not analysis or not getattr(analysis, "flags", None):
                continue
            if result.score < floor:
                continue
            extra.append(self._enrich(result, "flag_supported"))
            chosen.add(result.clause_id)

        if extra:
            logger.info(
                "Flag rule added %d clause(s): %s",
                len(extra), [c.clause_id for c in extra],
            )
        return extra

    # -- assembly --------------------------------------------------------

    def build(self, query: str) -> RAGContext:
        """Retrieve, select, budget, and format a grounded context."""
        selected = self.select(query)
        if not selected:
            return RAGContext(
                query=query,
                retrieved=[],
                formatted_context="",
                char_count=0,
                token_estimate=0,
                budget_chars=self.budget_chars,
                within_budget=True,
            )

        kept: list[RetrievedClause] = []
        dropped: list[int] = []
        abridged: list[int] = []
        used = 0

        for position, clause in enumerate(selected):
            remaining = self.budget_chars - used
            if len(clause.text) <= remaining:
                kept.append(clause)
                used += len(clause.text)
                continue

            # Only the top hit is abridged. Anything further down is dropped
            # whole, so the context never contains a fragment presented as if
            # it were a complete provision.
            if position == 0 and remaining > 0:
                text, was_abridged = _abridge(clause.text, remaining)
                clause.text = text
                clause.included_fully = not was_abridged
                kept.append(clause)
                used += len(text)
                if was_abridged:
                    abridged.append(clause.clause_id)
            else:
                dropped.append(clause.clause_id)

        formatted = self._format(kept)
        char_count = len(formatted)
        return RAGContext(
            query=query,
            retrieved=kept,
            formatted_context=formatted,
            char_count=char_count,
            token_estimate=self.token_counter(formatted),
            budget_chars=self.budget_chars,
            # The budget governs clause text; headers add a small fixed
            # overhead, so the assembled string may exceed it slightly.
            within_budget=used <= self.budget_chars,
            dropped_clause_ids=dropped,
            abridged_clause_ids=abridged,
        )

    @staticmethod
    def _format(clauses: Iterable[RetrievedClause]) -> str:
        """Render clauses as a labelled context block.

        Each block is headed by its citation so a generator can quote the
        source, and the clause title is repeated in the header because titles
        carry topical signal the body sometimes lacks -- the Intellectual
        Property clause never uses the word "own".
        """
        blocks: list[str] = []
        for clause in clauses:
            header = f"[{clause.citation}]"
            if clause.category:
                header += f" Category: {clause.category}"
            if not clause.included_fully:
                header += " (abridged)"
            blocks.append(f"{header}\n{clause.text}")
        return "\n\n".join(blocks)
