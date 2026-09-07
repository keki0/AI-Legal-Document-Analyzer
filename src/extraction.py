"""Information extraction: deterministic, explainable, evidence-preserving.

Every extracted item carries the exact span it came from, so the interface can
show a reader where a fact originated and the reader can disagree with it.
This mirrors the evidence contract established for flags in Phase 4.

**Why rules rather than a model.** The eight required extraction types --
parties, dates, durations, money, notice periods, renewal terms, obligations
-- are all surface phenomena with reliable lexical signals. A transformer NER
model would add a dependency and an unexplainable failure mode in exchange for
no measurable gain, and we have no labelled extraction data to justify one.
Where regex is reliable, regex is the right answer.

**Placeholders are first-class.** Contract templates are full of unfilled
fields: ``[Insert Date]``, ``$[Rate]``, ``[7/14] days``. Extracting those as
values would be wrong, and ignoring them would lose real information. They are
extracted and marked ``is_placeholder=True``, so the interface can report
"an effective date is referenced but not filled in".

**What ``confidence`` means here.** It is a HAND-ASSIGNED indicator of how
specific the matching rule is, on a 0-1 scale. It is not a probability, it is
not learned, and it is not comparable to the calibrated classifier confidence
from Phase 4. A tightly anchored pattern such as a notice period scores higher
than a bare capitalised role word. Only the ordering is defended.

**No legal conclusions.** This module reports what a document says. It never
assesses whether a term is valid, enforceable, fair or advisable.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Iterable, Sequence

from src.importance import _split_sentences

if TYPE_CHECKING:  # pragma: no cover
    from src.document_processing import Clause
    from src.importance import ClauseAnalysis

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractionType",
    "ExtractedItem",
    "DocumentExtraction",
    "InformationExtractor",
    "extract_from_clauses",
    "attach_extractions",
]


class ExtractionType:
    """Extraction type constants. A plain namespace, not an Enum.

    These values are serialised into JSON for the interface, so they are kept
    as stable strings rather than enum members that would need conversion at
    every boundary.
    """

    PARTY = "party"
    DATE = "date"
    DURATION = "duration"
    MONEY = "money"
    NOTICE_PERIOD = "notice_period"
    RENEWAL_TERM = "renewal_term"
    OBLIGATION = "obligation"

    ALL = (PARTY, DATE, DURATION, MONEY, NOTICE_PERIOD, RENEWAL_TERM, OBLIGATION)


@dataclass
class ExtractedItem:
    """One extracted fact with its provenance.

    ``evidence`` is the sentence containing the match; ``value`` is the matched
    text itself. ``text[span_start:span_end]`` always reproduces ``value``
    exactly, which is asserted by tests.
    """

    type: str
    value: str
    evidence: str
    span_start: int
    span_end: int
    method: str
    confidence: float
    normalized_value: str | None = None
    clause_id: int | None = None
    clause_title: str | None = None
    clause_category: str | None = None
    is_placeholder: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ===========================================================================
# Normalisation helpers
# ===========================================================================

_NUMBER_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

# Alternation ordered longest-first so "twenty-four" is not matched as "two".
_NUMBER_WORD_PATTERN = "|".join(
    sorted((*_NUMBER_WORDS, *(f"{t}-{u}" for t in _TENS for u in
                              ("one", "two", "three", "four", "five",
                               "six", "seven", "eight", "nine"))),
           key=len, reverse=True)
)

_UNIT_TO_ISO = {"day": "D", "week": "W", "month": "M", "year": "Y"}


def _word_to_number(word: str) -> int | None:
    """Convert an English cardinal to an integer. Handles 'twenty-four'."""
    word = word.lower().strip()
    if word in _NUMBER_WORDS:
        return _NUMBER_WORDS[word]
    if "-" in word:
        tens, _, units = word.partition("-")
        if tens in _TENS and units in _NUMBER_WORDS:
            return _TENS[tens] + _NUMBER_WORDS[units]
    return None


def _parse_amount(token: str) -> int | None:
    """Parse a digit string or number word into an integer."""
    token = token.strip()
    digits = token.replace(",", "")
    if digits.isdigit():
        return int(digits)
    return _word_to_number(token)


def _normalise_duration(amount: str, unit: str) -> str | None:
    """Render a duration as an ISO-8601 period, e.g. 30 days -> P30D.

    ISO-8601 is used so durations sort and compare consistently regardless of
    whether the source wrote "30 days", "thirty days" or "30 calendar days".
    """
    number = _parse_amount(amount)
    if number is None:
        return None
    letter = _UNIT_TO_ISO.get(unit.lower().rstrip("s"))
    if letter is None:
        return None
    # ISO-8601 puts weeks and dates in the date component: P30D, P2W, P1Y.
    return f"P{number}{letter}"


_MONTHS = {
    m.lower(): i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"], start=1)
}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})


def _normalise_date(text: str) -> str | None:
    """Return an ISO date string, or None if the text is not a real date.

    Deliberately conservative: an unparseable fragment yields None rather than
    a guess. Written by hand rather than pulling in dateutil, since only a
    handful of formats occur in contracts.
    """
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text.strip(), flags=re.I)

    # 2026-01-15
    match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", cleaned)
    if match:
        year, month, day = (int(g) for g in match.groups())
    else:
        # January 15, 2026  /  Jan 15 2026
        match = re.match(r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$", cleaned)
        if match and match.group(1).lower() in _MONTHS:
            month = _MONTHS[match.group(1).lower()]
            day, year = int(match.group(2)), int(match.group(3))
        else:
            # 15 January 2026
            match = re.match(r"^(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})$", cleaned)
            if match and match.group(2).lower() in _MONTHS:
                day = int(match.group(1))
                month = _MONTHS[match.group(2).lower()]
                year = int(match.group(3))
            else:
                # 01/15/2026 -- assumed US order, which is ambiguous.
                match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", cleaned)
                if not match:
                    return None
                month, day, year = (int(g) for g in match.groups())

    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None  # e.g. 31 February


# ===========================================================================
# Patterns
# ===========================================================================

# A template placeholder: [Insert Date], [Company Name], [7/14], [Net 15/30].
_PLACEHOLDER = re.compile(r"\[[^\]\n]{1,40}\]")

_DURATION_UNIT = r"(?:calendar\s+|business\s+|working\s+)?(days?|weeks?|months?|years?)"
# A cardinal is REQUIRED. Without it, "Draft Month" in a flattened table row
# matches as a duration -- an actual false positive observed on the sample
# contract's milestone table.
_DURATION = re.compile(
    rf"\b(\d{{1,4}}|{_NUMBER_WORD_PATTERN})[\s-]+{_DURATION_UNIT}\b", re.I
)
# Bracketed alternatives in templates: "[7/14] days", "[30] days".
_DURATION_PLACEHOLDER = re.compile(
    rf"\[(\d{{1,4}}(?:\s*/\s*\d{{1,4}})*)\]\s*{_DURATION_UNIT}\b", re.I
)

_DATE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{1,2}-\d{1,2}"
    r"|[A-Z][a-z]{2,8}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]{2,8},?\s+\d{4}"
    r"|\d{1,2}/\d{1,2}/\d{4}"
    r")\b"
)
# "Effective Date", "Start Date", "termination date" followed by a placeholder.
_DATE_PLACEHOLDER = re.compile(
    r"\b((?:effective|commencement|start|termination|expiration|renewal|"
    r"execution|insert)\s+date)\b\s*[:\-]?\s*(\[[^\]\n]{1,40}\])?", re.I
)

_CURRENCY = r"(?:\$|US\$|USD|INR|Rs\.?|₹|£|€|EUR|GBP)"
_MONEY = re.compile(
    rf"{_CURRENCY}\s?\d[\d,]*(?:\.\d{{1,2}})?(?:\s*(?:million|billion|lakh|crore))?"
    rf"|\b\d[\d,]*(?:\.\d{{1,2}})?\s+(?:dollars|rupees|pounds|euros)\b",
    re.I,
)
_MONEY_PLACEHOLDER = re.compile(rf"{_CURRENCY}\s?\[[^\]\n]{{1,30}}\]|{_CURRENCY}\s?X+Y?\b")
# NOTE: no trailing \b after the alternation. '%' is a non-word character, so
# \b would require a word character immediately after it -- making "1.5% per
# month" fail to match. The word-boundary is applied inside the spelled-out
# alternative instead.
_PERCENT = re.compile(r"\b\d{1,3}(?:\.\d{1,2})?\s?(?:%|\bper\s?cent(?:um)?\b)", re.I)

_NOTICE_CUE = re.compile(r"\bnotice\b", re.I)

_RENEWAL = [
    ("automatic_renewal", re.compile(
        r"\b(?:automatically\s+(?:renew|extend)\w*|auto[\s-]?renew\w*"
        r"|renew\w*\s+automatically)\b", re.I)),
    ("initial_term", re.compile(
        r"\b(?:initial|original)\s+term\b[^.]{0,60}", re.I)),
    ("renewal_period", re.compile(
        r"\b(?:renewal|successive|additional)\s+(?:term|period)s?\b[^.]{0,60}", re.I)),
    ("expiration", re.compile(
        r"\b(?:expire|expires|expiration|terminates?\s+on)\b[^.]{0,60}", re.I)),
]

# Deontic cues. Ordered so the more specific multiword forms win.
_OBLIGATION_CUES = [
    ("prohibition", r"may\s+not|shall\s+not|must\s+not|will\s+not|is\s+prohibited\s+from"),
    ("requirement", r"is\s+required\s+to|are\s+required\s+to|agrees?\s+to|shall|must"),
    ("undertaking", r"\bwill\b"),
]
_OBLIGATION = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _OBLIGATION_CUES), re.I
)

# Defined-term pattern: [Company Name] ... ("Agreement"), (the "Client").
_QUOTED_PARTY = re.compile(r"\((?:the\s+)?[\"\u201c]([A-Z][A-Za-z ]{2,30})[\"\u201d]\)")
_ROLE_WORDS = (
    "Client", "Contractor", "Company", "Employer", "Employee", "Customer",
    "Supplier", "Vendor", "Licensor", "Licensee", "Landlord", "Tenant",
    "Service Provider", "Disclosing Party", "Receiving Party", "Consultant",
    "Purchaser", "Seller", "Lessor", "Lessee", "Subscriber", "User",
)
_ROLE_PATTERN = re.compile(
    r"\b(?:the\s+)?(" + "|".join(sorted(_ROLE_WORDS, key=len, reverse=True)) + r")\b"
)
_PARTY_PLACEHOLDER = re.compile(
    r"\[([A-Z][A-Za-z ]{0,30}(?:Name|Party|Company|Contractor|Client))\]"
)

# Words that are never a real monetary value even inside a currency match.
_NON_TERMS = {"agreement", "party", "parties", "section"}


# ===========================================================================
# Extractor
# ===========================================================================

class InformationExtractor:
    """Rule-based extraction over a single clause or a whole document.

    Each ``_extract_*`` method returns items for one type. They are kept
    separate so a type can be disabled or replaced without touching the
    others.
    """

    def __init__(self, *, types: Sequence[str] | None = None) -> None:
        self.types = tuple(types) if types else ExtractionType.ALL

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _sentence_for(text: str, start: int) -> str:
        for s_start, s_end, sentence in _split_sentences(text):
            if s_start <= start < s_end:
                return sentence
        return text.strip()

    def _item(
        self, text: str, match_text: str, start: int, end: int,
        *, type_: str, method: str, confidence: float,
        normalized: str | None = None, placeholder: bool = False,
    ) -> ExtractedItem:
        return ExtractedItem(
            type=type_,
            value=match_text,
            evidence=self._sentence_for(text, start),
            span_start=start,
            span_end=end,
            method=method,
            confidence=confidence,
            normalized_value=normalized,
            is_placeholder=placeholder,
        )

    # -- individual types ------------------------------------------------

    def _extract_dates(self, text: str) -> list[ExtractedItem]:
        items: list[ExtractedItem] = []
        for match in _DATE.finditer(text):
            raw = match.group(0)
            normalized = _normalise_date(raw)
            if normalized is None:
                continue  # not a real date; do not guess
            items.append(self._item(
                text, raw, match.start(), match.end(),
                type_=ExtractionType.DATE, method="regex:date",
                confidence=0.9, normalized=normalized,
            ))
        for match in _DATE_PLACEHOLDER.finditer(text):
            items.append(self._item(
                text, match.group(0).strip(), match.start(), match.end(),
                type_=ExtractionType.DATE, method="regex:date_placeholder",
                confidence=0.6, placeholder=True,
            ))
        return items

    def _extract_durations(self, text: str) -> list[ExtractedItem]:
        items: list[ExtractedItem] = []
        for match in _DURATION.finditer(text):
            amount, unit = match.group(1), match.group(2)
            items.append(self._item(
                text, match.group(0), match.start(), match.end(),
                type_=ExtractionType.DURATION, method="regex:duration",
                confidence=0.85, normalized=_normalise_duration(amount, unit),
            ))
        for match in _DURATION_PLACEHOLDER.finditer(text):
            items.append(self._item(
                text, match.group(0), match.start(), match.end(),
                type_=ExtractionType.DURATION, method="regex:duration_placeholder",
                confidence=0.6, placeholder=True,
            ))
        return items

    def _extract_money(self, text: str) -> list[ExtractedItem]:
        items: list[ExtractedItem] = []
        for match in _MONEY.finditer(text):
            items.append(self._item(
                text, match.group(0), match.start(), match.end(),
                type_=ExtractionType.MONEY, method="regex:money", confidence=0.9,
            ))
        for match in _MONEY_PLACEHOLDER.finditer(text):
            items.append(self._item(
                text, match.group(0), match.start(), match.end(),
                type_=ExtractionType.MONEY, method="regex:money_placeholder",
                confidence=0.6, placeholder=True,
            ))
        for match in _PERCENT.finditer(text):
            items.append(self._item(
                text, match.group(0), match.start(), match.end(),
                type_=ExtractionType.MONEY, method="regex:percentage",
                confidence=0.8, normalized=match.group(0).replace(" ", ""),
            ))
        return items

    def _extract_notice_periods(self, text: str, window: int = 90) -> list[ExtractedItem]:
        """A duration counts as a notice period when 'notice' is nearby.

        Proximity rather than a single fixed phrase, because drafting varies:
        "30 days' written notice", "notice of at least 30 days", "upon 30
        days prior written notice".
        """
        notice_spans = [m.span() for m in _NOTICE_CUE.finditer(text)]
        if not notice_spans:
            return []

        items: list[ExtractedItem] = []
        candidates = list(_DURATION.finditer(text)) + list(_DURATION_PLACEHOLDER.finditer(text))
        for match in candidates:
            start, end = match.span()
            near = any(
                abs(n_start - end) <= window or abs(start - n_end) <= window
                for n_start, n_end in notice_spans
            )
            if not near:
                continue
            placeholder = match.re is _DURATION_PLACEHOLDER
            normalized = (
                None if placeholder
                else _normalise_duration(match.group(1), match.group(2))
            )
            items.append(self._item(
                text, match.group(0), start, end,
                type_=ExtractionType.NOTICE_PERIOD,
                method="regex:duration_near_notice",
                confidence=0.75 if not placeholder else 0.55,
                normalized=normalized, placeholder=placeholder,
            ))
        return items

    def _extract_renewal_terms(self, text: str) -> list[ExtractedItem]:
        items: list[ExtractedItem] = []
        for subtype, pattern in _RENEWAL:
            match = pattern.search(text)
            if not match:
                continue
            items.append(self._item(
                text, match.group(0).strip(), match.start(), match.end(),
                type_=ExtractionType.RENEWAL_TERM,
                method=f"regex:{subtype}", confidence=0.8, normalized=subtype,
            ))
        return items

    def _extract_obligations(self, text: str) -> list[ExtractedItem]:
        """Sentences carrying deontic language.

        Deliberately shallow: the sentence is returned with the modal cue
        identified. No semantic role labelling, no attempt to resolve who
        bears the obligation beyond the leading phrase.
        """
        items: list[ExtractedItem] = []
        for s_start, s_end, sentence in _split_sentences(text):
            match = _OBLIGATION.search(sentence)
            if not match:
                continue
            kind = match.lastgroup or "requirement"
            # Rough subject: the text before the modal, if it looks like one.
            prefix = sentence[:match.start()].strip()
            subject = prefix if 0 < len(prefix) <= 80 else None
            items.append(ExtractedItem(
                type=ExtractionType.OBLIGATION,
                value=sentence,
                evidence=sentence,
                span_start=s_start,
                span_end=s_end,
                method=f"regex:modal_{kind}",
                confidence=0.7 if kind == "prohibition" else 0.6,
                normalized_value=f"{kind}:{subject}" if subject else kind,
            ))
        return items

    def _extract_parties(self, text: str) -> list[ExtractedItem]:
        items: list[ExtractedItem] = []
        seen: set[str] = set()

        for match in _QUOTED_PARTY.finditer(text):
            name = match.group(1).strip()
            if name.lower() in _NON_TERMS or name.lower() in seen:
                continue
            seen.add(name.lower())
            items.append(self._item(
                text, name, match.start(1), match.end(1),
                type_=ExtractionType.PARTY, method="regex:quoted_definition",
                confidence=0.9, normalized=name,
            ))

        for match in _PARTY_PLACEHOLDER.finditer(text):
            name = match.group(1).strip()
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            items.append(self._item(
                text, match.group(0), match.start(), match.end(),
                type_=ExtractionType.PARTY, method="regex:party_placeholder",
                confidence=0.6, normalized=name, placeholder=True,
            ))

        # Spans already consumed by a placeholder. Without this, "[Company
        # Name]" yields both the placeholder and a spurious non-placeholder
        # "Company" matched from inside the brackets.
        placeholder_spans = [m.span() for m in _PLACEHOLDER.finditer(text)]

        for match in _ROLE_PATTERN.finditer(text):
            name = match.group(1).strip()
            if name.lower() in seen:
                continue
            if any(start <= match.start(1) < end for start, end in placeholder_spans):
                continue
            seen.add(name.lower())
            items.append(self._item(
                text, name, match.start(1), match.end(1),
                type_=ExtractionType.PARTY, method="lexicon:role_word",
                confidence=0.5, normalized=name,
            ))
        return items

    # -- public API ------------------------------------------------------

    def extract(self, text: str) -> list[ExtractedItem]:
        """Extract every enabled type from a block of text."""
        if not text or not text.strip():
            return []

        dispatch = {
            ExtractionType.DATE: self._extract_dates,
            ExtractionType.DURATION: self._extract_durations,
            ExtractionType.MONEY: self._extract_money,
            ExtractionType.NOTICE_PERIOD: self._extract_notice_periods,
            ExtractionType.RENEWAL_TERM: self._extract_renewal_terms,
            ExtractionType.OBLIGATION: self._extract_obligations,
            ExtractionType.PARTY: self._extract_parties,
        }
        items: list[ExtractedItem] = []
        for type_ in self.types:
            handler = dispatch.get(type_)
            if handler is not None:
                items.extend(handler(text))
        return sorted(items, key=lambda item: (item.span_start, item.type))

    def extract_from_clause(self, clause: "Clause") -> list[ExtractedItem]:
        """Extract from a clause, tagging each item with its source."""
        items = self.extract(clause.text)
        for item in items:
            item.clause_id = clause.clause_id
            item.clause_title = clause.title
        return items


# ===========================================================================
# Document-level aggregation
# ===========================================================================

@dataclass
class DocumentExtraction:
    """Document-wide rollup of clause-level extractions.

    A separate structure is needed because some facts are document-level
    rather than clause-level: parties are defined once in the preamble but
    referred to throughout, so a per-clause view would report the same party
    a dozen times.
    """

    items: list[ExtractedItem] = field(default_factory=list)

    def by_type(self, type_: str) -> list[ExtractedItem]:
        return [item for item in self.items if item.type == type_]

    @property
    def parties(self) -> list[dict]:
        """Distinct parties, most frequently mentioned first."""
        counts: Counter[str] = Counter()
        representative: dict[str, ExtractedItem] = {}
        for item in self.by_type(ExtractionType.PARTY):
            key = (item.normalized_value or item.value).strip()
            counts[key] += 1
            # Keep the highest-confidence mention as the representative.
            if key not in representative or item.confidence > representative[key].confidence:
                representative[key] = item
        return [
            {
                "name": name,
                "mentions": count,
                "method": representative[name].method,
                "is_placeholder": representative[name].is_placeholder,
                "evidence": representative[name].evidence,
                "clause_title": representative[name].clause_title,
            }
            for name, count in counts.most_common()
        ]

    @property
    def placeholders(self) -> list[ExtractedItem]:
        """Unfilled template fields -- useful to surface to the reader."""
        return [item for item in self.items if item.is_placeholder]

    # Types where a template placeholder is possible. Obligations and
    # role-word parties are never placeholders, so including them in the
    # denominator dilutes the signal: the sample contract has placeholders in
    # 100% of its dates, amounts and periods, yet scored under 30% when
    # measured against all extracted items.
    FILLABLE_TYPES = (
        ExtractionType.DATE,
        ExtractionType.MONEY,
        ExtractionType.DURATION,
        ExtractionType.NOTICE_PERIOD,
    )

    @property
    def placeholder_ratio(self) -> float:
        """Fraction of fillable items that are unfilled placeholders."""
        fillable = [item for item in self.items if item.type in self.FILLABLE_TYPES]
        if not fillable:
            return 0.0
        return sum(item.is_placeholder for item in fillable) / len(fillable)

    @property
    def is_unfilled_template(self) -> bool:
        """True when most fillable fields are unfilled placeholders.

        Reported descriptively. It is a statement about the text, not a
        judgement about the document's validity or suitability.
        """
        return self.placeholder_ratio > 0.5

    def summary(self) -> dict:
        counts = Counter(item.type for item in self.items)
        return {
            "total_items": len(self.items),
            "by_type": dict(counts),
            "n_parties": len(self.parties),
            "n_placeholders": len(self.placeholders),
            "placeholder_ratio": round(self.placeholder_ratio, 3),
            "is_unfilled_template": self.is_unfilled_template,
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "parties": self.parties,
            "items": [item.to_dict() for item in self.items],
        }


def extract_from_clauses(
    clauses: Sequence["Clause"], *, extractor: InformationExtractor | None = None
) -> DocumentExtraction:
    """Run extraction across every clause and aggregate the results."""
    extractor = extractor or InformationExtractor()
    items: list[ExtractedItem] = []
    for clause in clauses:
        items.extend(extractor.extract_from_clause(clause))
    logger.info("Extracted %d items from %d clauses", len(items), len(clauses))
    return DocumentExtraction(items)


def attach_extractions(
    analyses: Sequence["ClauseAnalysis"],
    *,
    extractor: InformationExtractor | None = None,
) -> DocumentExtraction:
    """Attach extractions to existing Phase 4 analyses, in place.

    Populates ``ClauseAnalysis.extractions`` and returns the document-level
    rollup. The clause category from Phase 4 is copied onto each item so the
    interface can report "a notice period found in a Term & Termination
    clause" without a second lookup.
    """
    extractor = extractor or InformationExtractor()
    all_items: list[ExtractedItem] = []

    for analysis in analyses:
        items = extractor.extract(analysis.text)
        for item in items:
            item.clause_id = analysis.clause_id
            item.clause_title = analysis.title
            item.clause_category = analysis.category
        analysis.extractions = items
        all_items.extend(items)

    return DocumentExtraction(all_items)
