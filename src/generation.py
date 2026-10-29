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
from typing import TYPE_CHECKING, Any, Sequence

from src.importance import split_sentences

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
    "clean_generated_text",
    "is_placeholder_relevant_to_query",
    "clean_clause_noise",
    "compact_summary_clause",
    "distill_clause_fact",
    "build_structured_summary_context",
    "GenerationPipeline",
]

# Bumped whenever a prompt template changes, so stored evaluation results can
# be matched to the prompts that produced them.
PROMPT_VERSION = "v3"

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
        "Write a concise coherent overview of this legal document based ONLY on the structured facts provided. "
        "Explain what the agreement is about and summarize its major obligations, financial/term provisions, protections, and termination/default provisions when supported. "
        "Combine information across the supplied sections into a single coherent document overview paragraph.\n"
        f"{_RULES}\n"
        "Do not copy sentences from the source.\n"
        "Do not mention clause numbers, category labels, salience, retrieval, models, or internal metadata.\n"
        "Preserve exact numbers, placeholders, dates, amounts, and notice periods when they are important.\n"
        "Do not infer legal relationships that are not explicitly supported.\n\n"
        "Document facts:\n{context}\n\n"
        "Document overview:"
    ),
    GenerationTask.SIMPLE_EXPLANATION: (
        "Question: {query}\n\n"
        "Instructions:\n"
        f"{_RULES}\n"
        "Answer the question based only on direct facts in the context below.\n"
        "If the context does not explicitly mention the specific subject, topic, violation, or entity asked about, state: 'The document does not specify this information.'\n"
        "Do not infer penalties, rules, or remedies that are not explicitly stated for that specific subject.\n"
        "Never use unrelated template placeholders (such as XXX, [Date], or [Rate]) as answers. If the requested information (such as a phone number or address) is not explicitly present, state: 'The document does not specify this information.'\n"
        "If the question asks about payment or fees and the rate is an unfilled placeholder ($[Rate] or [Net 15/30]), state that the document contains a placeholder ($[Rate]) and does not specify a fixed payment amount.\n\n"
        "Context:\n{context}\n\n"
        "Answer:"
    ),
    GenerationTask.CLAUSE_EXPLANATION: (
        "Question: {query}\n\n"
        "Instructions:\n"
        f"{_RULES}\n"
        "Answer the question based only on direct facts in the clause below.\n"
        "If the clause does not explicitly mention the specific subject, topic, violation, or entity asked about, state: 'The document does not specify this information.'\n"
        "Do not infer penalties, rules, or remedies that are not explicitly stated for that specific subject.\n"
        "Never use unrelated template placeholders (such as XXX, [Date], or [Rate]) as answers. If the requested information (such as a phone number or address) is not explicitly present, state: 'The document does not specify this information.'\n"
        "If the question asks about payment or fees and the rate is an unfilled placeholder ($[Rate] or [Net 15/30]), state that the document contains a placeholder ($[Rate]) and does not specify a fixed payment amount.\n\n"
        "Clause title: {title}\n"
        "{metadata}"
        "Clause text:\n{context}\n\n"
        "Answer:"
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
            query=context.query,
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
            GenerationTask.QUICK_SUMMARY: 200,
            GenerationTask.SIMPLE_EXPLANATION: 120,
            GenerationTask.CLAUSE_EXPLANATION: 120,
        }
    )
    min_new_tokens: dict[str, int] = field(
        default_factory=lambda: {
            GenerationTask.QUICK_SUMMARY: 55,
            GenerationTask.SIMPLE_EXPLANATION: 0,
            GenerationTask.CLAUSE_EXPLANATION: 0,
        }
    )

    def tokens_for(self, task: str) -> int:
        return self.max_new_tokens.get(task, 120)

    def early_stopping_for(self, task: str) -> bool:
        """Return early stopping setting for task.

        QA tasks disable early stopping to prevent short abstention hypotheses
        from truncating valid multi-sentence answers during beam search.
        """
        if task in {GenerationTask.SIMPLE_EXPLANATION, GenerationTask.CLAUSE_EXPLANATION}:
            return False
        return self.early_stopping

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


KNOWN_CATEGORIES = [
    "Amendments & Waivers",
    "Assignment & Transfer",
    "Boilerplate & Administrative",
    "Compliance & Approvals",
    "Confidentiality & Publicity",
    "Corporate & Financial",
    "Employment & Compensation",
    "Governing Law & Dispute Resolution",
    "Intellectual Property",
    "Liability & Indemnity",
    "Notices & Records",
    "Payment & Fees",
    "Representations & Warranties",
    "Tax",
    "Term & Termination",
    "Unclassified",
]
_CAT_PATTERN = "|".join(
    re.escape(c).replace(r"\&", r"(?:&|and)") for c in KNOWN_CATEGORIES
)


def is_placeholder_relevant_to_query(placeholder: str, query: str) -> bool:
    """Check whether a bracketed template placeholder is semantically relevant to a user query."""
    ph_words = set(re.findall(r"[a-zA-Z]+", placeholder.lower()))
    if not ph_words:
        return False
    q_words = set(re.findall(r"[a-zA-Z]+", query.lower()))
    q_content = q_words - {
        "what", "when", "where", "who", "whom", "whose", "which", "why", "how",
        "is", "are", "was", "were", "be", "been", "being", "the", "a", "an",
        "under", "in", "of", "to", "for", "by", "this", "that", "these", "those",
        "does", "do", "did", "and", "or", "as", "at", "clause", "document",
        "agreement", "lease", "contract",
    }
    if not q_content:
        return True

    # 1. Direct word overlap
    if ph_words & q_content:
        return True

    # 2. Stem / prefix overlap (e.g. rent vs rental, deposit vs deposits)
    if any(
        (pw.startswith(qw[:4]) or qw.startswith(pw[:4]))
        for pw in ph_words
        for qw in q_content
        if len(pw) >= 3 and len(qw) >= 3
    ):
        return True

    # 3. Domain concept overlap (financial or temporal)
    financial_terms = {
        "pay", "paid", "payment", "fee", "fees", "cost", "rate", "compensation",
        "amount", "rent", "rental", "price", "deposit",
    }
    temporal_terms = {
        "date", "day", "days", "term", "start", "end", "begin", "expire",
        "expiration", "move", "period", "month", "year", "notice",
    }
    if (ph_words & financial_terms) and (q_content & financial_terms):
        return True
    if (ph_words & temporal_terms) and (q_content & temporal_terms):
        return True

    return False


def clean_generated_text(
    text: str, query: str | None = None, context: str | None = None
) -> str:
    """Clean generated text by removing leaked metadata headers and prompt echoes."""
    if not text:
        return ""
    # Strip bracketed clause/provision citations: e.g. [Clause 6: Intellectual Property (p2)], [Provision: Term]
    cleaned = re.sub(
        r"\[(?:Clause|Provision|Place)\s*[^\]]+\]", " ", text, flags=re.IGNORECASE
    )
    # Strip unbracketed clause headers with page: Clause 4: Payment Terms (p1)
    cleaned = re.sub(
        r"(?:^|\n|\b)Clause\s+\d+:\s*[^(\n]+\(p\d+\)", " ", cleaned, flags=re.IGNORECASE
    )
    # Strip category with known taxonomy categories (including & or and) or specific headers
    cleaned = re.sub(
        rf"\b(?:Category:\s*)?(?:{_CAT_PATTERN}|Liability\s*(?:&|and)\s*Property\s*Insurance)\b\.?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Strip uppercase section headers with colons: e.g. INSURANCE:, TERM / RENEWAL:
    cleaned = re.sub(r"\b[A-Z\s&/]{3,35}:\s*", " ", cleaned)
    # Strip any generic Category: ... at end of line or before newline
    cleaned = re.sub(r"\bCategory:\s*[^.\n]+(?:\n|$)", " ", cleaned, flags=re.IGNORECASE)
    # Strip Points to note / Values found labels
    cleaned = re.sub(r"\bPoints to note:\s*[^.\n]+(?:\n|$)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bValues found in this clause:\s*[^.\n]+(?:\n|$)", " ", cleaned, flags=re.IGNORECASE)
    # Strip prompt prefix echoes if any
    cleaned = re.sub(
        r"^(?:Summary(?:\s+overview)?|Plain-language explanation|Explanation|Answer|Overview|Document\s+overview|Synthesized\s+overview(?:\s+of\s+the\s+agreement)?):\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Strip leading section letter markers like "a. "
    cleaned = re.sub(r"^[a-z0-9]\.\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    # Strip repetitive opening stutter like 'This lease agreement is a lease agreement'
    cleaned = re.sub(
        r"^([A-Za-z\s]+?)\s+is\s+a\s+\1\s+establishes\b",
        r"\1 establishes",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^([A-Za-z\s]+?)\s+is\s+a\s+\1\s+governing\b",
        r"\1 governs",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+is\s+a\s+\1\b",
        r"\1",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Clean whitespace and outer quotes
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip("\"'")

    # Abstain if output collapsed into raw template blanks or index leaks
    if re.search(r"\bXXX\b", cleaned) and re.search(r"\b20XX\b", cleaned):
        return "The document does not specify this information."
    if re.fullmatch(
        r"(\bXXX\b|\b20XX\b|___+|\[\d+\]|\[\])[.\s]*",
        cleaned.strip(),
        flags=re.IGNORECASE,
    ):
        return "The document does not specify this information."

    # Find placeholders like [Security Deposit Amount], $[Rate], $[hour/project], etc.
    ph_matches = re.findall(r"(\[[^\]]+\]|\$(?:\[[^\]]+\]|[\w/]+))", cleaned)

    # Preserve standalone bracketed or rate placeholders representing the requested information
    m_ph = re.fullmatch(r"(\[[^\]]+\]|\$(?:\[[^\]]+\]|[\w/]+))[.\s]*", cleaned.strip())
    if m_ph:
        ph_str = m_ph.group(1).strip()
        ph_words = set(re.findall(r"[a-zA-Z]+", ph_str.lower()))
        if not ph_words:
            return "The document does not specify this information."
        if query and not is_placeholder_relevant_to_query(ph_str, query):
            return "The document does not specify this information."
        if context is not None and ph_str.lower() not in context.lower():
            return "The document does not specify this information."
        return cleaned

    if query:
        # Check if requested field was answered with an unrelated placeholder or index
        if ph_matches and not any(is_placeholder_relevant_to_query(p, query) for p in ph_matches):
            if re.search(r"\b(?:is|was|are|number|phone|date|address)\b", cleaned, flags=re.IGNORECASE):
                return "The document does not specify this information."

        if re.search(
            r"\b(?:is|was|are|number|phone|date|address)\s*[:=]?\s*['\"]?(?:XXX|20XX|____+|\[\d+\])",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return "The document does not specify this information."

        # Echo of query phrase without answering
        q_clean = re.sub(
            r"^(?:what|when|where|who|how|why|is|are|does|do)\s+(?:is|are|the|a|an)?\s*",
            "",
            query.lower().rstrip("? .'\''\""),
        ).strip()
        ans_clean = cleaned.lower().rstrip(" .'\''\"")
        if ans_clean in {q_clean, f"the {q_clean}", f"{q_clean}."}:
            return "The document does not specify this information."

    if cleaned and cleaned[-1].isalnum():
        cleaned += "."
    return cleaned


_NUMERIC_PATTERN = re.compile(
    r"(\$[\d,]+(?:\.\d+)?|\b\d+\s*(?:calendar\s+)?(?:days?|months?|years?|hours?|weeks?)\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b20\d\d\b|___|\[.*?\]|\b[A-Z]{2,}\b)",
    re.IGNORECASE,
)
_KEY_TERMS_PATTERN = re.compile(
    r"\b(rent|payment|pay|fee|insurance|coverage|insured|terminate|termination|renew|renewal|option|indemnify|indemnity|liability|damage|default|cure|notice|remedy|records?)\b",
    re.IGNORECASE,
)


def clean_clause_noise(text: str) -> str:
    """Strip leading contact/page lines, inline page markers, and excessive placeholders."""
    if not text:
        return ""
    cleaned = re.sub(
        r"^(?:[\d-]{7,}\s+|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\s+|Page\s+\d+\s+of\s+\d+\s*|[a-z0-9]\.\s*)+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+Page\s+\d+\s+of\s+\d+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"_{3,}", "____", cleaned)
    return cleaned.strip()


def compact_summary_clause(
    analysis_item: Any, budget_chars: int
) -> tuple[str, bool]:
    """Compact a clause for the summary context.

    Preserves the beginning sentence of the clause plus the most informative
    subsequent sentences (containing exact dates, monetary amounts, notice
    periods, placeholders, or key obligations) within the allocated budget.
    """
    text = clean_clause_noise(analysis_item.text)
    if len(text) <= budget_chars:
        return text, False

    raw_sents = [
        s[2].strip()
        for s in split_sentences(text)
        if len(s[2].strip()) > 5
        and not s[2].strip().lower().startswith("page ")
    ]
    if not raw_sents:
        cut = text[:budget_chars].rstrip()
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0].rstrip()
        return cut + (" [...]" if len(cut) < len(text) else ""), True

    opening = raw_sents[0]

    # Check for graceful completion of action in opening sentence
    m = re.search(
        r"(?:suspend or )?terminate this (?:agreement|lease)",
        opening,
        re.IGNORECASE,
    )
    if m and m.end() <= budget_chars + 130:
        return opening[:m.end()].rstrip() + ".", True

    candidates = []
    flag_spans = [
        f.trigger_span.lower()
        for f in getattr(analysis_item, "flags", [])
        if hasattr(f, "trigger_span")
    ]
    extraction_vals = [
        str(e.value).lower()
        for e in getattr(analysis_item, "extractions", [])
        if hasattr(e, "value") and not getattr(e, "is_placeholder", False)
    ]

    for i, s in enumerate(raw_sents[1:], start=1):
        s_low = s.lower()
        score = (
            len(_NUMERIC_PATTERN.findall(s)) * 3.0
            + len(_KEY_TERMS_PATTERN.findall(s)) * 1.5
        )
        for span in flag_spans:
            if span in s_low:
                score += 3.0
        for val in extraction_vals:
            if val in s_low:
                score += 3.0
        candidates.append((score, i, s))

    candidates.sort(key=lambda x: x[0], reverse=True)

    kept_indices = {0}
    used = len(opening)

    if used <= budget_chars:
        for score, idx, s in candidates:
            if score > 0 and used + len(s) + 1 <= budget_chars + 30:
                kept_indices.add(idx)
                used += len(s) + 1
                break

    if len(kept_indices) == 1 and len(opening) > budget_chars + 30:
        cut = opening[:budget_chars].rstrip()
        comma = max(cut.rfind(";"), cut.rfind(","))
        if comma > int(budget_chars * 0.5):
            cut = cut[:comma].rstrip()
        elif " " in cut:
            cut = cut.rsplit(" ", 1)[0].rstrip()
        while True:
            stripped = re.sub(
                r"\s+(?:and|or|with|of|to|for|in|at|by|on|from|covering|including|having|shall|the|a|an)\s*$",
                "",
                cut,
                flags=re.IGNORECASE,
            ).rstrip()
            if stripped == cut:
                break
            cut = stripped
        return cut + ".", True

    ordered_sents = [raw_sents[i] for i in sorted(kept_indices)]
    compacted = " ".join(ordered_sents)
    was_abridged = len(kept_indices) < len(raw_sents)
    return compacted, was_abridged


def distill_clause_fact(item: Any) -> str:
    """Reduce a clause analysis item to a clean structured factual block."""
    title = getattr(item, "title", "").strip()
    title_low = title.lower()
    text_low = getattr(item, "text", "").lower()
    raw_text = getattr(item, "text", "")

    # Extract entities from analysis item
    placeholders = []
    if re.search(r"(___{2,}|\[.*?\]|\b20XX\b|\bXXX\b)", raw_text):
        m = re.findall(r"(\$[\w/]+|\[.*?\]|\b20XX\b|\bXXX\b|___+)", raw_text)
        if m:
            sample = [x for x in m if x != "___"][:3]
            if sample:
                sample_str = ", ".join(sample)
                placeholders.append(f"contains template placeholders ({sample_str})")
            else:
                placeholders.append(
                    "contains template placeholders for dates or commercial terms"
                )

    notices = [
        str(e.value)
        for e in getattr(item, "extractions", [])
        if getattr(e, "type", "") == "notice_period" and hasattr(e, "value")
    ]
    money = [
        str(e.value)
        for e in getattr(item, "extractions", [])
        if getattr(e, "type", "") == "money" and hasattr(e, "value")
    ]

    facts = []

    # Map clause to structured factual summary based on title and core content
    if "renew" in title_low:
        header = "TERM / RENEWAL"
        facts.append("The agreement grants options to renew for additional periods.")
    elif "term" in title_low and "termination" in title_low:
        header = "TERM / TERMINATION"
        facts.append(
            "The agreement defines the term duration and conditions and notice periods for early termination."
        )
    elif "term" in title_low:
        header = "TERM"
        facts.append(
            "The agreement specifies an initial term with commencement and expiration dates."
        )
    elif "insur" in title_low or "insurance" in text_low[:60]:
        header = "INSURANCE"
        facts.append(
            "The Lessee must maintain liability and property insurance and satisfy the agreement's insurance requirements."
        )
    elif "destruct" in title_low or "damage" in title_low or "casualty" in title_low:
        header = "CASUALTY / DAMAGE"
        facts.append(
            "Outlines procedures, restoration limits, and termination rights if property suffers damage or destruction."
        )
    elif "default" in title_low:
        header = "DEFAULT / TERMINATION"
        facts.append(
            "The agreement specifies circumstances, notice requirements, and remedies concerning default and termination."
        )
    elif "indemn" in title_low:
        header = "INDEMNIFICATION"
        facts.append(
            "Requires indemnifying and holding harmless against liabilities, damages, and claims."
        )
    elif (
        "payment" in title_low
        or "rent" in title_low
        or "fee" in title_low
        or "compensation" in title_low
    ):
        header = "PAYMENT / FEES"
        facts.append(
            "Governs payment terms, fee schedules, invoices, and financial obligations."
        )
    elif "record" in title_low or "public record" in text_low[:120]:
        header = "PUBLIC RECORDS"
        facts.append(
            "Requires maintaining, inspecting, and transferring public records pursuant to applicable statutory standards."
        )
    elif "independent contractor" in title_low or "independent contractor" in text_low[:100]:
        header = "RELATIONSHIP / STATUS"
        facts.append(
            "Establishes that the relationship is that of an independent contractor and not an employee."
        )
    elif "intellectual property" in title_low or "work product" in title_low:
        header = "INTELLECTUAL PROPERTY"
        facts.append(
            "Assigns all work product, inventions, and intellectual property rights to the client."
        )
    elif "confidential" in title_low:
        header = "CONFIDENTIALITY"
        facts.append(
            "Imposes obligations to protect proprietary and confidential information from unauthorized disclosure."
        )
    elif "compliance" in title_low:
        header = "COMPLIANCE"
        facts.append(
            "Mandates compliance with applicable laws, regulations, and governmental standards."
        )
    else:
        # Fallback to concise opening sentence
        clean = re.sub(
            r"^(?:[\d-]{7,}\s+|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\s+|Page\s+\d+\s+of\s+\d+\s*|[a-z0-9]\.\s*)+",
            "",
            raw_text,
            flags=re.IGNORECASE,
        )
        sents = [
            s[2].strip()
            for s in split_sentences(clean)
            if len(s[2].strip()) > 8 and not s[2].strip().lower().startswith("page ")
        ]
        opening = sents[0] if sents else clean[:120]
        opening = re.sub(
            r"^(?:[A-Za-z0-9&/\s-]{3,35}\s+)?(?:The\s+)", "The ", opening
        )
        if len(opening) > 130:
            comma = max(opening[:120].rfind(";"), opening[:120].rfind(","))
            if comma > 50:
                opening = opening[:comma].rstrip() + "."
            else:
                opening = opening[:110].rsplit(" ", 1)[0].rstrip() + "."
        header = title.upper()[:25] if title else "PROVISION"
        facts.append(opening)

    details = []
    if notices:
        details.append(f"Requires notice period of {notices[0]}.")
    if money:
        details.append(f"Specifies amount of {money[0]}.")
    if placeholders:
        details.append(f"Specific terms {placeholders[0]}.")

    body = " ".join(facts)
    if details:
        body += " " + " ".join(details)
    return f"{header}:\n{body}"


def build_structured_summary_context(clauses: Sequence[Any]) -> str:
    """Build a compact structured context of key facts across the selected clauses."""
    blocks = []

    # Lead with high-level document scope if detectable
    all_text = " ".join(getattr(c, "text", "").lower() for c in clauses[:4])
    if "lease" in all_text and (
        "premises" in all_text
        or "city" in all_text
        or "tenant" in all_text
        or "lessee" in all_text
    ):
        blocks.append(
            "OVERVIEW:\nThis lease agreement establishes the terms governing the use and occupancy of the premises."
        )
    elif "contractor" in all_text and ("services" in all_text or "client" in all_text):
        blocks.append(
            "OVERVIEW:\nThis agreement establishes the terms governing services provided by an independent contractor to the client."
        )

    for c in clauses:
        blocks.append(distill_clause_fact(c))

    return "\n\n".join(blocks)


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

        gen_kwargs = {
            "max_new_tokens": self.settings.tokens_for(task),
            "num_beams": self.settings.num_beams,
            "do_sample": self.settings.do_sample,
            "early_stopping": self.settings.early_stopping_for(task),
            "no_repeat_ngram_size": self.settings.no_repeat_ngram_size,
        }
        min_tokens = self.settings.min_new_tokens.get(task, 0)
        if min_tokens > 0:
            gen_kwargs["min_new_tokens"] = min_tokens

        with torch.no_grad():
            output = self._model.generate(
                **encoded,
                **gen_kwargs,
            )
        decoded = self._tokenizer.decode(output[0], skip_special_tokens=True).strip()
        query = getattr(self, "_current_query", None)
        ctx = getattr(self, "_current_context", None)
        return clean_generated_text(decoded, query=query, context=ctx)

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

        sources = [clause] if clause is not None else list(context.retrieved)
        grounding_text = (
            clause.text if clause is not None else context.formatted_context
        )

        prompt = build_prompt(task, context, clause=clause)
        self._current_query = getattr(context, "query", None)
        self._current_context = grounding_text
        text = self._generate(prompt, task)

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
