"""Marathi legal document translation orchestrator with strict placeholder integrity.

Provides sentence-aware chunking, deterministic atomic entity masking,
post-processing normalization, and programmatic placeholder validation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.translation import TranslationBackend


PLACEHOLDER_PATTERN = re.compile(r"\[[^\[\]\n]{1,80}\]")
MASK_TOKEN_PATTERN = re.compile(
    r"\b(?:__TAG_\d+__|_TAG_\d+_|TAG_\d+|ENTITY\s*_\s*\d+|<ENT_\d+>)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PlaceholderValidationResult:
    """Detailed programmatic report from placeholder integrity validation."""

    is_valid: bool
    source_placeholders: list[str]
    translated_placeholders: list[str]
    missing_placeholders: list[str]
    duplicated_placeholders: list[str]
    invented_placeholders: list[str]
    is_order_preserved: bool
    corrupted_tokens: list[str]
    error_message: str = ""


class PlaceholderIntegrityError(ValueError):
    """Raised when translated text fails strict placeholder integrity checks."""

    def __init__(
        self,
        message: str,
        *,
        validation_result: PlaceholderValidationResult,
    ):
        super().__init__(message)
        self.validation_result = validation_result


def clean_translation(text: str) -> str:
    """Clean common formatting and spacing issues in translated text."""
    replacements = {
        " .": ".",
        " ,": ",",
        " :": ":",
        "( ": "(",
        " )": ")",
        "[ ": "[",
        " ]": "]",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


# Common abbreviations and currency markers in legal text that must NOT trigger sentence splitting
LEGAL_ABBREVIATION_PATTERN = re.compile(
    r"\b(?:Rs|Re)\b\.?|"
    r"\b(?:No|Nos|Sec|Art|Para|Cl)\.\s*(?=[$€£₹\d\[])|"
    r"\b(?:vs?)\.\s*(?=[A-Za-z])|"
    r"\b(?:e\.g|i\.e)\.",
    re.IGNORECASE,
)


def split_translation_chunks(text: str, max_chars: int = 700) -> list[str]:
    """Split text while preserving line breaks, abbreviations, and sentence boundaries.

    Chunking by coherent sentences prevents NLLB decoder loop/tail-truncation
    on compound legal recitals, while protecting abbreviations like 'Rs.' from
    causing premature splits that drop verb predicates and remedies.
    """
    if not text.strip():
        return []

    chunks: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Temporarily protect decimal numbers and known legal abbreviations from splitting
        masked_line = re.sub(r"(\d+)\.(\d+)", r"\1___DOT___\2", line)
        masked_line = LEGAL_ABBREVIATION_PATTERN.sub(
            lambda m: m.group(0).replace(".", "___DOT___"), masked_line
        )

        # Split at sentence boundaries (period, exclamation, question followed by space and capital/bracket/quote)
        sentences = [
            s.replace("___DOT___", ".").strip()
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'\[])|(?<=[.!?])$", masked_line)
            if s.strip()
        ]
        if not sentences:
            continue

        for sentence in sentences:
            if len(sentence) <= max_chars:
                chunks.append(sentence)
                continue

            # Split unusually long sentences at word boundaries if exceeding max_chars
            words = sentence.split()
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if current and len(candidate) > max_chars:
                    chunks.append(current)
                    current = word
                else:
                    current = candidate

            if current:
                chunks.append(current)

    return chunks


def normalize_translation_formatting(text: str) -> str:
    """Normalize formatting and punctuation in machine-translated Marathi legal text."""

    # Normalize visarga mistakenly used as colon before placeholders or fields
    text = re.sub(r"ः(?=\s*(?:\[|\$|€|£|₹|\d|_))", ":", text)

    # Preserve currency symbols attached to placeholders: "$ [Amount]" -> "$[Amount]"
    text = re.sub(r"([$€£₹])\s+\[", r"\1[", text)

    # Avoid periods immediately before placeholders (e.g. "देय आहेत. [Number]")
    text = re.sub(r"\.\s*(?=[$€£₹]?\[)", " ", text)

    # Ensure space before opening placeholder or currency+placeholder if preceded by a normal character
    text = re.sub(r"(?<=[^\s(\[{/\$€£₹])(?=[$€£₹]?\[)", " ", text)

    # Ensure space between closing placeholder and following word
    text = re.sub(r"(?<=\])(?=[^\s.,;:!?।)\]}\'\"])", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:।])", r"\1", text)

    # Collapse duplicated punctuation
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"।{2,}", "।", text)
    text = re.sub(r",{2,}", ",", text)

    return " ".join(text.split()).strip()


def validate_placeholder_integrity(
    source_text: str,
    translated_text: str,
    *,
    placeholder_pattern: re.Pattern = PLACEHOLDER_PATTERN,
) -> PlaceholderValidationResult:
    """Validate that placeholders in translated text strictly match source placeholders.

    Verifies:
    1. Exact multiset equality (no missing, duplicated, or invented placeholders).
    2. Exact sequence ordering (no swapping of party/date/amount positions).
    3. Absence of corrupted or unrestored masking tokens (unless already present in source).
    """
    source_phs = placeholder_pattern.findall(source_text)
    trans_phs = placeholder_pattern.findall(translated_text)

    source_counts = Counter(source_phs)
    trans_counts = Counter(trans_phs)

    # Missing placeholders
    missing: list[str] = []
    for ph, count in source_counts.items():
        actual = trans_counts.get(ph, 0)
        if actual < count:
            missing.extend([ph] * (count - actual))

    # Duplicated placeholders
    duplicated: list[str] = []
    for ph, actual in trans_counts.items():
        expected = source_counts.get(ph, 0)
        if actual > expected and expected > 0:
            duplicated.extend([ph] * (actual - expected))

    # Invented placeholders
    invented: list[str] = []
    for ph in trans_counts:
        if ph not in source_counts:
            invented.extend([ph] * trans_counts[ph])

    # Sequence order check
    is_order_preserved = source_phs == trans_phs

    # Detect unrestored or corrupted masking tokens in translation that were NOT in source
    source_mask_tokens = set(MASK_TOKEN_PATTERN.findall(source_text))
    trans_mask_tokens = MASK_TOKEN_PATTERN.findall(translated_text)
    corrupted = [tok for tok in trans_mask_tokens if tok not in source_mask_tokens]

    errors: list[str] = []
    if missing:
        errors.append(f"Missing placeholders: {missing}")
    if duplicated:
        errors.append(f"Duplicated placeholders: {duplicated}")
    if invented:
        errors.append(f"Invented placeholders: {invented}")
    if not is_order_preserved and not missing and not duplicated and not invented:
        errors.append(
            f"Placeholder order swapped: expected {source_phs}, got {trans_phs}"
        )
    if corrupted:
        errors.append(f"Unrestored masking tokens: {corrupted}")

    is_valid = len(errors) == 0
    error_message = "; ".join(errors)

    return PlaceholderValidationResult(
        is_valid=is_valid,
        source_placeholders=source_phs,
        translated_placeholders=trans_phs,
        missing_placeholders=missing,
        duplicated_placeholders=duplicated,
        invented_placeholders=invented,
        is_order_preserved=is_order_preserved,
        corrupted_tokens=corrupted,
        error_message=error_message,
    )


def translate_preserving_placeholders(
    backend: TranslationBackend,
    text: str,
    *,
    on_error: str = "fallback",
    token_prefix: str = "__TAG_",
) -> str:
    """Translate legal prose while preserving placeholders and sentence cohesion using entity masking.

    Steps:
    1. Mask original placeholders with neutral atomic tokens (`__TAG_1__`, `__TAG_2__`, ...).
    2. Segment masked text into coherent sentence chunks.
    3. Translate chunks via backend.
    4. Restore original placeholders in reverse-numerical order to avoid prefix collisions.
    5. Clean and normalize punctuation/spacing.
    6. Strictly validate placeholder multiset, ordering, and absence of corrupted tokens.
    7. Flag or raise if validation fails; never silently append missing placeholders.
    """
    if not text.strip():
        return ""

    placeholders = PLACEHOLDER_PATTERN.findall(text)
    if not placeholders:
        chunks = split_translation_chunks(text)
        translated = " ".join(backend.translate(c).translated_text for c in chunks)
        return normalize_translation_formatting(translated)

    placeholder_map: dict[str, str] = {}
    counter = 0

    def _mask_match(match: re.Match) -> str:
        nonlocal counter
        while True:
            counter += 1
            token = f"{token_prefix}{counter}__"
            if token not in text:
                break
        placeholder_map[token] = match.group(0)
        return token

    masked_text = PLACEHOLDER_PATTERN.sub(_mask_match, text)

    chunks = split_translation_chunks(masked_text)
    translated_core = " ".join(
        backend.translate(chunk).translated_text for chunk in chunks
    )

    result = translated_core

    # Sort tokens in reverse numerical order (e.g. __TAG_10__ before __TAG_1__)
    # to avoid prefix replacement collisions.
    for token, original_placeholder in sorted(
        placeholder_map.items(),
        key=lambda item: int(re.search(r"\d+", item[0]).group(0)),
        reverse=True,
    ):
        num = re.search(r"\d+", token).group(0)
        # Match both __TAG_N__ and ENTITY_N in case a mock backend uses ENTITY_N
        pattern = re.compile(
            rf"(?:__{re.escape(token_prefix.strip('_'))}_{num}__|__TAG_{num}__|\bENTITY\s*_\s*{num}\b)",
            re.IGNORECASE,
        )
        if pattern.search(result):
            result = pattern.sub(original_placeholder, result)
        else:
            result = result.replace(token, original_placeholder)

    formatted_result = normalize_translation_formatting(result)

    # Programmatic multiset & order integrity validation (no silent appending)
    validation = validate_placeholder_integrity(text, formatted_result)
    if not validation.is_valid:
        if on_error == "raise":
            raise PlaceholderIntegrityError(
                f"Placeholder validation failed: {validation.error_message}",
                validation_result=validation,
            )
        return f"[Translation rejected: {validation.error_message}] {text}"

    return formatted_result


DEVANAGARI_DIGITS_MAP = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

DANGLING_CONJUNCTION_PATTERN = re.compile(
    r"(?:किंवा|आणि|तसेच|दरम्यान|च्या|वर|सह|ला|ने|कडून)\s*[.,!?।]?$",
    re.UNICODE,
)

HIGH_STAKES_TERMS_PATTERN = re.compile(
    r"\b(?:liquidated damages|injunctive relief|indemnif\w*|sole discretion|unilateral\w*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TranslationReviewReport:
    """Report evaluating translation safety, integrity, and human review flags."""

    requires_human_review: bool
    review_flags: list[str]
    reasons: list[str]
    placeholder_validation: PlaceholderValidationResult
    is_safe: bool = True


@dataclass(frozen=True)
class TranslationClauseResult:
    """Rich structured result for legal clause translation with review metadata."""

    source_text: str
    translated_text: str
    report: TranslationReviewReport


def check_translation_review_flags(
    source_text: str,
    translated_text: str,
    validation: PlaceholderValidationResult | None = None,
) -> TranslationReviewReport:
    """Check translation for critical legal defects, omissions, and advisory review triggers."""
    flags: list[str] = []
    reasons: list[str] = []

    # 1. Check placeholder integrity
    if validation is None:
        validation = validate_placeholder_integrity(source_text, translated_text)

    if not validation.is_valid:
        flags.append("PLACEHOLDER_INTEGRITY_FAILED")
        reasons.append(f"Placeholder validation failed: {validation.error_message}")

    # 2. Check for unrestored or corrupted internal masking tokens
    if validation.corrupted_tokens:
        flags.append("CORRUPTED_MASK_TOKENS")
        reasons.append(f"Unrestored mask tokens: {validation.corrupted_tokens}")

    # 3. Check for dropped numbers (monetary sums, notice deadlines, percentages)
    clean_src = PLACEHOLDER_PATTERN.sub("", source_text)
    clean_tgt = PLACEHOLDER_PATTERN.sub("", translated_text)
    clean_tgt_ascii = "".join(DEVANAGARI_DIGITS_MAP.get(c, c) for c in clean_tgt)

    src_numbers = set(re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", clean_src))
    for num in src_numbers:
        normalized_num = num.replace(",", "")
        tgt_has_num = (
            num in clean_tgt
            or num in clean_tgt_ascii
            or normalized_num in clean_tgt.replace(",", "")
            or normalized_num in clean_tgt_ascii.replace(",", "")
        )
        if not tgt_has_num:
            flags.append("NUMERICAL_DISCREPANCY")
            reasons.append(f"Source number '{num}' is missing in translated text.")
            break

    # 4. Check for abrupt termination on conjunctions or case markers
    stripped_tgt = translated_text.strip()
    if DANGLING_CONJUNCTION_PATTERN.search(stripped_tgt):
        flags.append("ABRUPT_TRUNCATION")
        reasons.append("Translation ends abruptly on a dangling conjunction or connector.")

    # 5. Check for residual untranslated Latin words outside placeholders
    latin_words = [
        w for w in re.findall(r"\b[A-Za-z]+\b", clean_tgt)
        if w.upper() not in {"TAG", "ENTITY", "USD", "INR", "EUR", "GBP"}
    ]
    if latin_words:
        flags.append("UNTRANSLATED_TOKENS")
        reasons.append(f"Untranslated Latin tokens in output: {latin_words}")

    # 6. High-consequence remedies advisory flag
    if HIGH_STAKES_TERMS_PATTERN.search(source_text):
        flags.append("HIGH_CONSEQUENCE_REMEDY")
        reasons.append(
            "Clause specifies high-consequence legal remedies (injunctive relief or liquidated damages)."
        )

    severe_flags = {
        "PLACEHOLDER_INTEGRITY_FAILED",
        "CORRUPTED_MASK_TOKENS",
        "NUMERICAL_DISCREPANCY",
        "ABRUPT_TRUNCATION",
        "UNTRANSLATED_TOKENS",
    }
    is_safe = not any(f in severe_flags for f in flags)
    requires_human_review = not is_safe or bool(flags)

    return TranslationReviewReport(
        requires_human_review=requires_human_review,
        review_flags=flags,
        reasons=reasons,
        placeholder_validation=validation,
        is_safe=is_safe,
    )


def translate_legal_clause(
    backend: TranslationBackend,
    text: str,
    *,
    on_error: str = "fallback",
    token_prefix: str = "__TAG_",
) -> TranslationClauseResult:
    """Translate a legal clause and produce an explainable human review assessment."""
    translated = translate_preserving_placeholders(
        backend,
        text,
        on_error=on_error,
        token_prefix=token_prefix,
    )
    validation = validate_placeholder_integrity(text, translated)
    report = check_translation_review_flags(text, translated, validation=validation)
    return TranslationClauseResult(
        source_text=text,
        translated_text=translated,
        report=report,
    )
