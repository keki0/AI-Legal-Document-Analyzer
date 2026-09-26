"""Marathi legal translation package."""

from src.marathi.glossary import (
    DOCUMENTED_LEGAL_GLOSSARY,
    LegalTermEntry,
    apply_target_glossary,
    get_documented_glossary_map,
)
from src.marathi.translator import (
    LEGAL_ABBREVIATION_PATTERN,
    MASK_TOKEN_PATTERN,
    PLACEHOLDER_PATTERN,
    PlaceholderIntegrityError,
    PlaceholderValidationResult,
    TranslationClauseResult,
    TranslationReviewReport,
    check_translation_review_flags,
    clean_translation,
    normalize_translation_formatting,
    split_translation_chunks,
    translate_legal_clause,
    translate_preserving_placeholders,
    validate_placeholder_integrity,
)

__all__ = [
    "DOCUMENTED_LEGAL_GLOSSARY",
    "LegalTermEntry",
    "apply_target_glossary",
    "get_documented_glossary_map",
    "LEGAL_ABBREVIATION_PATTERN",
    "MASK_TOKEN_PATTERN",
    "PLACEHOLDER_PATTERN",
    "PlaceholderIntegrityError",
    "PlaceholderValidationResult",
    "TranslationClauseResult",
    "TranslationReviewReport",
    "check_translation_review_flags",
    "clean_translation",
    "normalize_translation_formatting",
    "split_translation_chunks",
    "translate_legal_clause",
    "translate_preserving_placeholders",
    "validate_placeholder_integrity",
]
