"""PDF extraction and clause segmentation.

The baseline (``notebooks/01_baseline.ipynb``) segments clauses with a single
regex over flattened text::

    ^\\s*(\\d+)\\.\\s+([A-Z][^\\n]+)

That works on a numbered contract and returns nothing at all on a privacy
policy or terms-of-service document, where headings are unnumbered prose. It
also keeps page furniture — repeated headers and footers — inside the clause
text, which pollutes downstream embeddings.

This module replaces it with three cooperating signals:

1. **Layout.** PyMuPDF exposes font size and weight per span. Body text is
   whichever size dominates by character count; anything meaningfully larger
   is a heading candidate, and anything meaningfully smaller is page
   furniture. This is what makes unnumbered documents work.
2. **Lexical patterns.** Numbered headings, ALL-CAPS headings, and a short
   list of heading-shaped cues. This catches headings that are not visually
   distinguished.
3. **Shape.** Headings are short, rarely end in a full stop, and are not
   themselves sentences.

The signals are scored rather than chained, so a document only needs to
satisfy some of them. If every signal fails, the segmenter falls back to the
baseline regex and then to paragraph blocks, so it always returns something.

Nothing here imports torch or transformers; it is pure text processing and
runs anywhere.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

__all__ = [
    "Span",
    "Line",
    "Clause",
    "Document",
    "extract_document",
    "segment_clauses",
    "clauses_to_chunks",
    "load_and_segment",
]

# PyMuPDF encodes bold in bit 4 of the span flags field.
_BOLD_FLAG = 1 << 4

# Heading shapes. Ordered from most to least specific.
_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]\s+(\S.*)$")
_ALLCAPS_HEADING = re.compile(r"^[A-Z][A-Z0-9 ,'&()/\-]{2,79}$")
_ARTICLE_HEADING = re.compile(
    r"^\s*(?:ARTICLE|SECTION|CLAUSE|Article|Section|Clause)\s+"
    r"([IVXLC]+|\d+(?:\.\d+)*)[.:)]?\s*(.*)$"
)

# A heading longer than this is almost certainly a sentence.
_MAX_HEADING_CHARS = 90
# Score at or above which a line is accepted as a heading.
_HEADING_SCORE_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Span:
    """A run of text sharing one font, size and weight."""

    text: str
    size: float
    bold: bool
    font: str
    page: int


@dataclass
class Line:
    """One visual line, with the layout attributes of its first span."""

    text: str
    size: float
    bold: bool
    page: int
    is_heading: bool = False
    heading_number: str | None = None
    heading_score: float = 0.0


@dataclass
class Clause:
    """A segmented clause: its heading and the body text beneath it."""

    clause_id: int
    number: str | None
    title: str
    text: str
    page_start: int
    page_end: int

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def as_chunk(self) -> str:
        """Return the clause with its title prepended.

        Used for retrieval. The heading carries most of the topical signal —
        a clause about IP ownership may never contain the word "own", but its
        heading says "Intellectual Property". Embedding the body alone throws
        that signal away.
        """
        return f"{self.title}. {self.text}" if self.title else self.text


@dataclass
class Document:
    """An extracted document: lines, page count, and diagnostics."""

    path: Path
    lines: list[Line]
    n_pages: int
    body_size: float
    removed_boilerplate: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Normalise whitespace and a few PDF artefacts.

    Zero-width spaces and bullet glyphs survive PyMuPDF extraction and would
    otherwise become tokens.
    """
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[•●▪◦]\s*", "", text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return text.strip()


def _extract_spans(doc: pymupdf.Document) -> list[list[Span]]:
    """Extract spans grouped into visual lines, preserving reading order."""
    lines: list[list[Span]] = []
    for page_number, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # skip images
                continue
            for line in block["lines"]:
                spans = [
                    Span(
                        text=span["text"],
                        size=round(span["size"], 1),
                        bold=bool(span["flags"] & _BOLD_FLAG),
                        font=span["font"],
                        page=page_number,
                    )
                    for span in line["spans"]
                    if span["text"].strip()
                ]
                if spans:
                    lines.append(spans)
    return lines


def _dominant_size(line_spans: list[list[Span]]) -> float:
    """Return the body-text font size.

    Weighted by character count, not line count: a document with many short
    headings and few long paragraphs would otherwise report the heading size
    as dominant.
    """
    weights: Counter[float] = Counter()
    for spans in line_spans:
        for span in spans:
            weights[span.size] += len(span.text.strip())
    if not weights:
        return 0.0
    return weights.most_common(1)[0][0]


def _find_boilerplate(lines: list[Line], n_pages: int, *, min_ratio: float = 0.6) -> set[str]:
    """Identify running headers and footers.

    A short line whose exact text recurs on most pages is page furniture, not
    content. On the sample contract this removes the footer that otherwise
    appends "Seek professional advice before using it" to the last clause of
    every page.
    """
    if n_pages < 2:
        return set()

    pages_by_text: dict[str, set[int]] = {}
    for line in lines:
        key = line.text.strip()
        if 0 < len(key) <= 150:
            pages_by_text.setdefault(key, set()).add(line.page)

    threshold = max(2, int(n_pages * min_ratio))
    return {text for text, pages in pages_by_text.items() if len(pages) >= threshold}


def extract_document(path: str | Path) -> Document:
    """Extract a PDF into cleaned lines with layout attributes.

    Args:
        path: Path to the PDF.

    Returns:
        A :class:`Document` with boilerplate removed.

    Raises:
        FileNotFoundError: if the PDF does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    with pymupdf.open(path) as doc:
        n_pages = doc.page_count
        line_spans = _extract_spans(doc)

    body_size = _dominant_size(line_spans)

    raw_lines: list[Line] = []
    for spans in line_spans:
        text = _clean("".join(span.text for span in spans))
        if not text:
            continue
        first = spans[0]
        raw_lines.append(
            Line(text=text, size=first.size, bold=first.bold, page=first.page)
        )

    boilerplate = _find_boilerplate(raw_lines, n_pages)

    # Drop repeated furniture and anything set noticeably smaller than body
    # text (footers, page numbers, fine print).
    small_cutoff = body_size * 0.85 if body_size else 0.0
    kept: list[Line] = []
    removed: list[str] = []
    for line in raw_lines:
        if line.text in boilerplate:
            removed.append(line.text)
            continue
        if small_cutoff and line.size < small_cutoff:
            removed.append(line.text)
            continue
        kept.append(line)

    logger.info(
        "Extracted %s: %d pages, %d lines kept, %d removed as furniture, body size %.1fpt",
        path.name, n_pages, len(kept), len(removed), body_size,
    )

    return Document(
        path=path,
        lines=kept,
        n_pages=n_pages,
        body_size=body_size,
        removed_boilerplate=sorted(set(removed)),
    )


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

def _score_heading(line: Line, body_size: float) -> tuple[float, str | None, str]:
    """Score a line's likelihood of being a clause heading.

    Returns ``(score, number, title)``. Signals are additive so that a
    document needs only some of them: a numbered contract scores on the regex,
    an unnumbered policy scores on font size and shape.
    """
    text = line.text.strip()
    if not text or len(text) > _MAX_HEADING_CHARS:
        return 0.0, None, text

    score = 0.0
    number: str | None = None
    title = text

    # --- lexical signals ---
    match = _ARTICLE_HEADING.match(text)
    if match:
        score += 2.0
        number = match.group(1)
        title = match.group(2).strip() or text
    else:
        match = _NUMBERED_HEADING.match(text)
        if match:
            score += 2.0
            number = match.group(1)
            title = match.group(2).strip()

    if _ALLCAPS_HEADING.match(title) and len(title.split()) <= 8:
        score += 1.0

    # --- layout signals ---
    if body_size:
        ratio = line.size / body_size
        if ratio >= 1.15:
            score += 2.0
        elif ratio >= 1.05:
            score += 0.5
    if line.bold and line.size >= body_size:
        score += 1.0

    # --- shape signals ---
    # A heading is short and is not a sentence.
    if len(title.split()) <= 10:
        score += 0.5
    if title.endswith((".", ";", ",", ":")) and not number:
        score -= 1.0
    # Lines ending in a colon usually introduce a list within a clause.
    if text.endswith(":") and not number:
        score -= 0.5

    return score, number, title


def _mark_headings(document: Document) -> int:
    """Annotate lines in place. Returns the number of headings found."""
    found = 0
    # The largest text on page 1 is the document title, not a clause heading.
    max_size = max((line.size for line in document.lines), default=0.0)
    title_zone_seen = False

    for line in document.lines:
        score, number, title = _score_heading(line, document.body_size)
        line.heading_score = score

        # Skip the document title block: outsized text at the very top.
        if (
            not title_zone_seen
            and line.page == 1
            and document.body_size
            and line.size >= max(max_size, document.body_size * 1.4)
        ):
            continue

        if score >= _HEADING_SCORE_THRESHOLD:
            line.is_heading = True
            line.heading_number = number
            line.text = line.text  # unchanged; title recomputed at assembly
            title_zone_seen = True
            found += 1

    return found


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _assemble(document: Document) -> list[Clause]:
    """Group lines into clauses at the marked heading boundaries."""
    clauses: list[Clause] = []
    current_title: str | None = None
    current_number: str | None = None
    current_body: list[str] = []
    page_start = 1

    def flush() -> None:
        nonlocal current_title, current_number, current_body, page_start
        if current_title is None and not current_body:
            return
        body = " ".join(current_body).strip()
        # Discard heading-only fragments with no substance beneath them.
        if current_title is not None and len(body.split()) < 3:
            current_title, current_number, current_body = None, None, []
            return
        clauses.append(
            Clause(
                clause_id=len(clauses) + 1,
                number=current_number,
                title=current_title or "Preamble",
                text=body,
                page_start=page_start,
                page_end=document.lines[-1].page if document.lines else page_start,
            )
        )
        current_title, current_number, current_body = None, None, []

    for line in document.lines:
        if line.is_heading:
            flush()
            _, number, title = _score_heading(line, document.body_size)
            current_title = title
            current_number = number
            page_start = line.page
        else:
            current_body.append(line.text)

    flush()

    # Fix page_end per clause now that boundaries are known.
    for clause in clauses:
        clause.page_end = max(clause.page_start, clause.page_end)
    return clauses


def _regex_fallback(document: Document) -> list[Clause]:
    """Baseline-equivalent segmentation, used when layout yields nothing."""
    logger.warning("Layout segmentation found too few headings; using regex fallback.")
    text = document.full_text
    matches = list(re.finditer(r"(?m)^\s*(\d+)\.\s+([A-Z][^\n]+)", text))
    clauses: list[Clause] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        clauses.append(
            Clause(
                clause_id=index + 1,
                number=match.group(1),
                title=match.group(2).strip(),
                text=text[match.end():end].strip(),
                page_start=1,
                page_end=document.n_pages,
            )
        )
    return clauses


def _paragraph_fallback(document: Document) -> list[Clause]:
    """Last resort: treat substantial paragraphs as clauses."""
    logger.warning("No headings detected at all; falling back to paragraph blocks.")
    clauses: list[Clause] = []
    for line in document.lines:
        if len(line.text.split()) >= 15:
            clauses.append(
                Clause(
                    clause_id=len(clauses) + 1,
                    number=None,
                    title=f"Paragraph {len(clauses) + 1}",
                    text=line.text,
                    page_start=line.page,
                    page_end=line.page,
                )
            )
    return clauses


def _merge_preamble(clauses: list[Clause], *, max_merge: int = 4) -> list[Clause]:
    """Fold front matter into a single "Preamble" clause.

    A contract's opening — title block, recitals, the paragraph naming the
    parties — is content worth keeping (it holds the parties and the effective
    date), but it is not a numbered clause and the title tends to be detected
    as a heading in its own right. Everything before the first numbered clause
    is merged so the front matter appears once.

    Skipped when the document has no numbered clauses at all, since in an
    unnumbered policy this would collapse the whole document into one clause.
    """
    first_numbered = next(
        (i for i, clause in enumerate(clauses) if clause.number is not None), None
    )
    if first_numbered is None or first_numbered == 0 or first_numbered > max_merge:
        return clauses

    head = clauses[:first_numbered]
    merged = Clause(
        clause_id=1,
        number=None,
        title="Preamble",
        text=" ".join(clause.text for clause in head).strip(),
        page_start=head[0].page_start,
        page_end=head[-1].page_end,
    )
    rest = clauses[first_numbered:]
    for offset, clause in enumerate(rest, start=2):
        clause.clause_id = offset
    return [merged, *rest]


def segment_clauses(document: Document, *, min_clauses: int = 2) -> list[Clause]:
    """Segment a document into clauses using layout, lexical and shape signals.

    Falls back to the baseline regex, then to paragraph blocks, so the return
    value is never empty for a document containing text.

    Args:
        document: Output of :func:`extract_document`.
        min_clauses: Below this count the layout result is considered a
            failure and the fallbacks are tried.

    Returns:
        Clauses in document order.
    """
    found = _mark_headings(document)
    logger.info("Heading detection marked %d candidate headings", found)

    clauses = _assemble(document) if found else []
    if len(clauses) >= min_clauses:
        return _merge_preamble(clauses)

    clauses = _regex_fallback(document)
    if len(clauses) >= min_clauses:
        return clauses

    return _paragraph_fallback(document)


def clauses_to_chunks(clauses: list[Clause], *, title_augmented: bool = True) -> list[str]:
    """Render clauses as retrieval units.

    Args:
        clauses: Segmented clauses.
        title_augmented: When True, prepend each clause's heading to its body.
            This is the improved representation; False reproduces the baseline.
    """
    if title_augmented:
        return [clause.as_chunk() for clause in clauses]
    return [clause.text for clause in clauses]


def load_and_segment(path: str | Path) -> tuple[Document, list[Clause]]:
    """Convenience wrapper: extract and segment in one call."""
    document = extract_document(path)
    return document, segment_clauses(document)
