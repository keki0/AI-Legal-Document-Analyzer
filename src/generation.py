"""Grounded generation over Phase 6 RAG contexts.

Phase 6 produces a :class:`~src.rag.RAGContext`: retrieved clauses, their
provenance, and a formatted context string within a token budget. This module
turns that context into readable text with a small FLAN-T5 model.

**Grounding is structural, not requested.** The pipeline consumes a
``RAGContext`` and never touches the retriever, the raw PDF, or arbitrary
document text. There is no code path from a document to a generated answer
that bypasses retrieval. An empty context returns a fixed safe message without
calling the model at all.

**Context text is never modified.** Phases 4-6 guarantee that evidence spans
index the original clause text, and rewriting or sanitising context here would
break that contract. The consequence is that this module offers no defence
against instructions embedded in a source document; see ``docs/phase7_
generation.md`` for the honest limitation.

**No fine-tuning.** FLAN-T5 is used zero-shot with deterministic beam search.
The project's deep-learning contribution is the fine-tuned Legal-BERT
classifier; adding a second training pipeline would cost time without
strengthening the argument.

Nothing here presents itself as a lawyer. Prompts instruct neutral,
descriptive phrasing, and :data:`DISCLAIMER` is attached to every response.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from src.rag import RAGContext, RetrievedClause

logger = logging.getLogger(__name__)

__all__ = [
    "DISCLAIMER",
    "EMPTY_CONTEXT_MESSAGE",
    "PROMPT_VERSION",
    "GenerationTask",
    "GenerationSettings",
    "GeneratedResponse",
    "GroundingReport",
    "build_prompt",
    "check_grounding",
    "GenerationPipeline",
]

# Bumped whenever a prompt template changes, so stored evaluation results can
# be matched to the prompts that produced them.
PROMPT_VERSION = "v1"

DEFAULT_MODEL = "google/flan-t5-base"

DISCLAIMER = (
    "This is an automated, informational summary of the uploaded document. "
    "It is not legal advice. Consider reviewing important clauses with a "
    "qualified legal professional."
)

EMPTY_CONTEXT_MESSAGE = "No relevant clause context was retrieved for this question."


class GenerationTask:
    """Task identifiers. Plain constants so they serialise cleanly."""

    QUICK_SUMMARY = "quick_summary"
    SIMPLE_EXPLANATION = "simple_explanation"
    CLAUSE_EXPLANATION = "clause_explanation"

    ALL = (QUICK_SUMMARY, SIMPLE_EXPLANATION, CLAUSE_EXPLANATION)


# ===========================================================================
# Prompts
# ===========================================================================

# Shared constraints. Kept identical across tasks so that a difference in
# output between tasks is attributable to the task instruction, not to
# differing guardrails.
_RULES = (
    "Use only the information in the context below. "
    "Do not add facts that are not stated. "
    "Do not give legal advice or say whether anything is legal, fair or valid. "
    "Write plainly and neutrally."
)

_TEMPLATES: dict[str, str] = {
    GenerationTask.QUICK_SUMMARY: (
        "Summarize the key points of the following legal document extract.\n"
        f"{_RULES}\n"
        "Mention obligations, payments, dates, termination, intellectual "
        "property, confidentiality or liability only if the context states "
        "them.\n\n"
        "Context:\n{context}\n\n"
        "Summary:"
    ),
    GenerationTask.SIMPLE_EXPLANATION: (
        "Explain the following legal text in plain language a non-lawyer can "
        "understand.\n"
        f"{_RULES}\n"
        "Preserve the original meaning exactly.\n\n"
        "Question: {query}\n\n"
        "Context:\n{context}\n\n"
        "Plain-language explanation:"
    ),
    GenerationTask.CLAUSE_EXPLANATION: (
        "Explain what this single contract clause says and what it means in "
        "practice.\n"
        f"{_RULES}\n\n"
        "Clause title: {title}\n"
        "{metadata}"
        "Clause text:\n{context}\n\n"
        "Explanation:"
    ),
}


def _metadata_block(clause: "RetrievedClause | None") -> str:
    """Render Phase 4/5 metadata as prompt lines.

    Only facts already computed upstream are included: the clause category,
    the labels of any evidence-linked flags, and non-placeholder extracted
    values. Placeholder extractions are skipped because feeding "[Insert
    Date]" to the model invites it to invent a date.
    """
    if clause is None:
        return ""

    lines: list[str] = []
    if clause.category:
        lines.append(f"Clause category: {clause.category}")

    labels = clause.flag_labels
    if labels:
        lines.append(f"Points to note: {', '.join(labels)}")

    values = [
        f"{item.type}: {item.value}"
        for item in clause.extractions
        if not getattr(item, "is_placeholder", False)
        and getattr(item, "type", "") in {"date", "money", "duration", "notice_period"}
    ]
    if values:
        lines.append(f"Values found in this clause: {'; '.join(values[:6])}")

    return "\n".join(lines) + "\n" if lines else ""


def build_prompt(
    task: str,
    context: "RAGContext",
    *,
    clause: "RetrievedClause | None" = None,
) -> str:
    """Assemble the prompt for a task from a Phase 6 context.

    Args:
        task: One of :class:`GenerationTask`.
        context: The Phase 6 context. Its ``formatted_context`` supplies the
            grounding text and already respects the token budget.
        clause: Required for ``CLAUSE_EXPLANATION``; the specific clause to
            explain. Ignored by the other tasks.

    Raises:
        ValueError: for an unknown task, or for a clause explanation with no
            clause supplied.
    """
    if task not in _TEMPLATES:
        raise ValueError(f"unknown task {task!r}; expected one of {GenerationTask.ALL}")

    if task == GenerationTask.CLAUSE_EXPLANATION:
        if clause is None:
            raise ValueError("clause_explanation requires a clause")
        return _TEMPLATES[task].format(
            title=clause.title,
            metadata=_metadata_block(clause),
            context=clause.text,
        )

    return _TEMPLATES[task].format(
        context=context.formatted_context, query=context.query
    )


# ===========================================================================
# Settings and outputs
# ===========================================================================

@dataclass(frozen=True)
class GenerationSettings:
    """Deterministic decoding settings.

    Sampling is off: an academic evaluation must be reproducible, and a
    grounded explanation has no reason to be creative. Beam search with
    ``num_beams=4`` is the standard conservative choice for FLAN-T5.

    ``max_new_tokens`` is per-task. A clause explanation should be a short
    paragraph; a summary can be a little longer. Over-long limits invite the
    model to pad, and padding is where unsupported content appears.
    """

    model_name: str = DEFAULT_MODEL
    num_beams: int = 4
    do_sample: bool = False
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    max_input_tokens: int = 512  # FLAN-T5's encoder limit
    max_new_tokens: dict[str, int] = field(
        default_factory=lambda: {
            GenerationTask.QUICK_SUMMARY: 160,
            GenerationTask.SIMPLE_EXPLANATION: 120,
            GenerationTask.CLAUSE_EXPLANATION: 120,
        }
    )

    def tokens_for(self, task: str) -> int:
        return self.max_new_tokens.get(task, 120)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroundingReport:
    """Lightweight diagnostic for unsupported content.

    **This is not a hallucination detector.** It performs surface checks:
    which numeric and currency tokens in the output do not appear in the
    context, and what fraction of expected key terms the output covers. A
    fluent, entirely fabricated sentence containing no numbers would pass it.
    Reported as a diagnostic, never as a guarantee.
    """

    unsupported_numbers: list[str] = field(default_factory=list)
    supported_numbers: list[str] = field(default_factory=list)
    key_point_coverage: float | None = None
    covered_key_points: list[str] = field(default_factory=list)
    missing_key_points: list[str] = field(default_factory=list)

    @property
    def has_unsupported_numbers(self) -> bool:
        return bool(self.unsupported_numbers)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeneratedResponse:
    """Generated text with the provenance needed to trace it."""

    text: str
    task: str
    source_clause_ids: list[int] = field(default_factory=list)
    source_titles: list[str] = field(default_factory=list)
    source_citations: list[str] = field(default_factory=list)
    model_name: str = DEFAULT_MODEL
    prompt_version: str = PROMPT_VERSION
    prompt: str = ""
    context_char_count: int = 0
    generated_from_empty_context: bool = False
    grounding: GroundingReport | None = None
    disclaimer: str = DISCLAIMER

    @property
    def is_grounded(self) -> bool:
        """True when the response is backed by at least one source clause."""
        return bool(self.source_clause_ids) and not self.generated_from_empty_context

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["is_grounded"] = self.is_grounded
        return payload


# ===========================================================================
# Grounding diagnostic
# ===========================================================================

# Numbers, money and percentages -- the tokens most likely to be fabricated
# and most consequential if they are.
_NUMERIC = re.compile(r"(?:[$£€]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?%?)")


def _numeric_tokens(text: str) -> list[str]:
    return [t.replace(" ", "") for t in _NUMERIC.findall(text)]


def check_grounding(
    output: str, context: str, *, key_points: Sequence[str] | None = None
) -> GroundingReport:
    """Compare generated output against its context.

    Args:
        output: Generated text.
        context: The context the model was given.
        key_points: Optional terms an ideal answer would mention. Used for
            coverage, matched case-insensitively as substrings.

    Returns:
        A :class:`GroundingReport`. Surface-level only -- see that class.
    """
    context_numbers = set(_numeric_tokens(context))
    supported, unsupported = [], []
    for token in _numeric_tokens(output):
        (supported if token in context_numbers else unsupported).append(token)

    report = GroundingReport(
        unsupported_numbers=sorted(set(unsupported)),
        supported_numbers=sorted(set(supported)),
    )

    if key_points:
        lowered = output.lower()
        covered = [p for p in key_points if p.lower() in lowered]
        report.covered_key_points = covered
        report.missing_key_points = [p for p in key_points if p not in covered]
        report.key_point_coverage = len(covered) / len(key_points)

    return report


# ===========================================================================
# Pipeline
# ===========================================================================

class GenerationPipeline:
    """Generates grounded text from Phase 6 contexts using FLAN-T5.

    The model and tokenizer can be injected, which keeps the unit tests free
    of a model download. When they are not supplied they are loaded lazily on
    first use, so importing this module costs nothing.

    Args:
        settings: Decoding configuration.
        model: Optional preloaded ``AutoModelForSeq2SeqLM``.
        tokenizer: Optional preloaded ``AutoTokenizer``.
        device: ``"cuda"`` or ``"cpu"``. Auto-detected when omitted.
    """

    def __init__(
        self,
        settings: GenerationSettings | None = None,
        *,
        model=None,
        tokenizer=None,
        device: str | None = None,
    ) -> None:
        self.settings = settings or GenerationSettings()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    # -- lazy loading ----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        name = self.settings.model_name
        logger.info("Loading %s", name)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(name)
        if self._model is None:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(name)
            self._model.to(self.device).eval()

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        return self._device

    def token_count(self, text: str) -> int:
        """Exact token count, for Phase 6 budgeting.

        Pass this as ``RAGContextBuilder(token_counter=...)`` to replace the
        character heuristic Phase 6 uses by default.
        """
        self._ensure_loaded()
        return len(self._tokenizer(text, add_special_tokens=True)["input_ids"])

    # -- generation ------------------------------------------------------

    def _generate(self, prompt: str, task: str) -> str:
        self._ensure_loaded()
        import torch

        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.max_input_tokens,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self._model.generate(
                **encoded,
                max_new_tokens=self.settings.tokens_for(task),
                num_beams=self.settings.num_beams,
                do_sample=self.settings.do_sample,
                early_stopping=self.settings.early_stopping,
                no_repeat_ngram_size=self.settings.no_repeat_ngram_size,
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def _empty_response(self, task: str) -> GeneratedResponse:
        """Fixed message for an empty context. The model is never called."""
        return GeneratedResponse(
            text=EMPTY_CONTEXT_MESSAGE,
            task=task,
            model_name=self.settings.model_name,
            generated_from_empty_context=True,
        )

    def run(
        self,
        task: str,
        context: "RAGContext",
        *,
        clause: "RetrievedClause | None" = None,
        key_points: Sequence[str] | None = None,
    ) -> GeneratedResponse:
        """Generate for a task from a Phase 6 context.

        Args:
            task: One of :class:`GenerationTask`.
            context: Phase 6 output. If empty, the safe message is returned
                without invoking the model.
            clause: The clause to explain, for ``CLAUSE_EXPLANATION``.
            key_points: Optional terms for the coverage diagnostic.
        """
        if task not in GenerationTask.ALL:
            raise ValueError(f"unknown task {task!r}")
        if context is None or context.is_empty:
            return self._empty_response(task)

        prompt = build_prompt(task, context, clause=clause)
        text = self._generate(prompt, task)

        sources = [clause] if clause is not None else list(context.retrieved)
        grounding_text = clause.text if clause is not None else context.formatted_context

        return GeneratedResponse(
            text=text,
            task=task,
            source_clause_ids=[c.clause_id for c in sources],
            source_titles=[c.title for c in sources],
            source_citations=[c.citation for c in sources],
            model_name=self.settings.model_name,
            prompt=prompt,
            context_char_count=context.char_count,
            grounding=check_grounding(text, grounding_text, key_points=key_points),
        )

    # -- convenience wrappers -------------------------------------------

    def quick_summary(self, context: "RAGContext", **kwargs) -> GeneratedResponse:
        return self.run(GenerationTask.QUICK_SUMMARY, context, **kwargs)

    def simple_explanation(self, context: "RAGContext", **kwargs) -> GeneratedResponse:
        return self.run(GenerationTask.SIMPLE_EXPLANATION, context, **kwargs)

    def clause_explanation(
        self, context: "RAGContext", clause: "RetrievedClause", **kwargs
    ) -> GeneratedResponse:
        return self.run(
            GenerationTask.CLAUSE_EXPLANATION, context, clause=clause, **kwargs
        )
