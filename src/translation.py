"""English-to-Marathi translation interfaces and local NLLB adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.marathi.translator import (
    PlaceholderIntegrityError,
    PlaceholderValidationResult,
    clean_translation,
)


@dataclass(frozen=True)
class TranslationResult:
    """Result returned by a translation backend."""

    source_text: str
    translated_text: str
    source_language: str = "English"
    target_language: str = "Marathi"
    backend: str = "unknown"


class TranslationBackend(ABC):
    """Common interface for translation backends."""

    @abstractmethod
    def translate(self, text: str) -> TranslationResult:
        """Translate English text into Marathi."""
        raise NotImplementedError


class LocalTranslationBackend(TranslationBackend):
    """Lazy-loading local NLLB English-to-Marathi translator."""

    MODEL_NAME = "facebook/nllb-200-distilled-600M"

    def __init__(
        self,
        translator: Callable[[str], str] | None = None,
        *,
        model_name: str = MODEL_NAME,
    ):
        self._translator = translator
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> None:
        """Load the model only when translation is first requested."""
        if self._tokenizer is not None and self._model is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()

    def _translate_with_model(self, text: str) -> str:
        self._load_model()

        self._tokenizer.src_lang = "eng_Latn"

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            output_tokens = self._model.generate(
                **inputs,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids("mar_Deva"),
                max_length=512,
                num_beams=4,
            )

        return self._tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True,
        )[0]

    def translate(self, text: str) -> TranslationResult:
        if not text.strip():
            return TranslationResult(
                source_text=text,
                translated_text="",
                backend="local",
            )

        if self._translator is not None:
            translated = clean_translation(self._translator(text))
        else:
            translated = clean_translation(self._translate_with_model(text))

        return TranslationResult(
            source_text=text,
            translated_text=translated,
            backend="local",
        )


class ExternalTranslationBackend(TranslationBackend):
    """Adapter for an external translation service."""

    def __init__(self, translator: Callable[[str], str] | None = None):
        self._translator = translator

    def translate(self, text: str) -> TranslationResult:
        if not text.strip():
            return TranslationResult(
                source_text=text,
                translated_text="",
                backend="external",
            )

        if self._translator is None:
            raise NotImplementedError(
                "The external translation service has not been configured."
            )

        return TranslationResult(
            source_text=text,
            translated_text=self._translator(text),
            backend="external",
        )


class HybridTranslationBackend(TranslationBackend):
    """Try the local backend and fall back to the external backend."""

    def __init__(
        self,
        primary: TranslationBackend,
        fallback: TranslationBackend,
    ):
        self.primary = primary
        self.fallback = fallback

    def translate(self, text: str) -> TranslationResult:
        try:
            return self.primary.translate(text)
        except (NotImplementedError, RuntimeError):
            return self.fallback.translate(text)


def create_translation_backend(
    backend: str = "hybrid",
    *,
    local_translator: Callable[[str], str] | None = None,
    external_translator: Callable[[str], str] | None = None,
) -> TranslationBackend:
    """Create a configured translation backend."""

    normalized_backend = backend.strip().lower()

    local = LocalTranslationBackend(local_translator)
    external = ExternalTranslationBackend(external_translator)

    if normalized_backend == "local":
        return local

    if normalized_backend in {"external", "api"}:
        return external

    if normalized_backend == "hybrid":
        return HybridTranslationBackend(
            primary=local,
            fallback=external,
        )

    raise ValueError(
        f"Unsupported translation backend: {backend!r}. "
        "Choose 'local', 'external', or 'hybrid'."
    )


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


def translate_quick_summary(
    summary_text: str,
    backend: TranslationBackend | None = None,
) -> TranslationClauseResult:
    """Translate an English quick summary into Marathi using the NLLB translation pipeline."""
    if backend is None:
        backend = create_translation_backend("local")
    return translate_legal_clause(backend, summary_text)


__all__ = [
    "TranslationResult",
    "TranslationBackend",
    "LocalTranslationBackend",
    "ExternalTranslationBackend",
    "HybridTranslationBackend",
    "create_translation_backend",
    "clean_translation",
    "split_translation_chunks",
    "normalize_translation_formatting",
    "validate_placeholder_integrity",
    "translate_preserving_placeholders",
    "translate_legal_clause",
    "translate_quick_summary",
    "check_translation_review_flags",
    "TranslationReviewReport",
    "TranslationClauseResult",
    "PlaceholderValidationResult",
    "PlaceholderIntegrityError",
    "PLACEHOLDER_PATTERN",
    "MASK_TOKEN_PATTERN",
    "LEGAL_ABBREVIATION_PATTERN",
    "DOCUMENTED_LEGAL_GLOSSARY",
    "LegalTermEntry",
    "apply_target_glossary",
    "get_documented_glossary_map",
]
