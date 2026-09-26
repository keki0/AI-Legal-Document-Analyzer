"""Curated legal glossary for English-to-Marathi document translation.

LEGAL DISCLAIMER:
All glossary entries are curated for standard commercial and statutory
legal Marathi (e.g. Indian Contract Act, Specific Relief Act). They are
intended as recommendations to prevent known machine-translation distortions
and do not constitute certified legal advice.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class LegalTermEntry:
    """Documented legal term entry with source context and rationale."""

    source_term: str
    target_term: str
    context_example: str
    category: str  # 'title', 'substantive_distortion', 'statutory_terminology'
    legal_note: str


# Documented glossary entries strictly backed by sample documents and evaluation findings.
DOCUMENTED_LEGAL_GLOSSARY: list[LegalTermEntry] = [
    LegalTermEntry(
        source_term="NON-DISCLOSURE AGREEMENT",
        target_term="गोपनीयता करार",
        context_example="sample-nda.pdf Document Header",
        category="title",
        legal_note=(
            "Prevents NLLB phonetic confusion of 'Disclosure' with 'Description' "
            "('नॉन-डिस्क्रिप्शन एग्रीमेंट'). Uniformly rendered as 'गोपनीयता करार' "
            "in Indian commercial legal practice."
        ),
    ),
    LegalTermEntry(
        source_term="Non-Disclosure Agreement",
        target_term="गोपनीयता करार",
        context_example="sample-nda.pdf Preamble Recital",
        category="title",
        legal_note=("Standard commercial terminology for Non-Disclosure Agreements."),
    ),
    LegalTermEntry(
        source_term="liquidated damages",
        target_term="पूर्वनिर्धारित नुकसानभरपाई",
        context_example="sample-nda.pdf Remedies Clause",
        category="statutory_terminology",
        legal_note=(
            "Statutory Marathi equivalent under Section 74 of the Indian Contract Act, 1872."
        ),
    ),
    LegalTermEntry(
        source_term="injunctive relief",
        target_term="न्यायालयीन मनाईहुकूम",
        context_example="sample-nda.pdf Remedies Clause",
        category="statutory_terminology",
        legal_note=(
            "Statutory Marathi equivalent for equitable relief under the Specific Relief Act, 1963."
        ),
    ),
    LegalTermEntry(
        source_term="exclusive jurisdiction",
        target_term="अनन्य अधिकार क्षेत्र",
        context_example="sample-nda.pdf Governing Law Clause",
        category="statutory_terminology",
        legal_note=(
            "Standard judicial Marathi for exclusive jurisdiction (vs colloquial 'एकमेव अधिकार क्षेत्र')."
        ),
    ),
]


def get_documented_glossary_map() -> dict[str, str]:
    """Return dictionary mapping of source terms to Marathi translations."""
    return {entry.source_term: entry.target_term for entry in DOCUMENTED_LEGAL_GLOSSARY}


def apply_target_glossary(text: str) -> str:
    """Apply documented legal glossary terms to text using exact boundary matching.

    Only applies documented source examples to avoid speculative post-processing.
    """
    for entry in DOCUMENTED_LEGAL_GLOSSARY:
        pattern = rf"\b{re.escape(entry.source_term)}\b"
        text = re.sub(
            pattern,
            entry.target_term,
            text,
            flags=re.IGNORECASE if entry.category != "title" else 0,
        )
    return text
