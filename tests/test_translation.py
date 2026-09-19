import pytest

from src.translation import (
    ExternalTranslationBackend,
    HybridTranslationBackend,
    LocalTranslationBackend,
    TranslationResult,
    create_translation_backend,
)
from src.translation import clean_translation


def test_clean_translation_fixes_spacing():
    result = clean_translation("हा करार आहे . हा [ करार ] आहे")

    assert result == "हा करार आहे. हा [करार] आहे"


def test_translation_result_defaults():
    result = TranslationResult(
        source_text="This is a contract.",
        translated_text="हा करार आहे.",
    )

    assert result.source_language == "English"
    assert result.target_language == "Marathi"
    assert result.backend == "unknown"


def test_local_backend_uses_injected_translator():
    backend = LocalTranslationBackend(translator=lambda text: f"मराठी: {text}")

    result = backend.translate("This is a contract.")

    assert result.translated_text == "मराठी: This is a contract."
    assert result.backend == "local"


def test_external_backend_uses_injected_translator():
    backend = ExternalTranslationBackend(translator=lambda text: f"API: {text}")

    result = backend.translate("This is a contract.")

    assert result.translated_text == "API: This is a contract."
    assert result.backend == "external"


def test_hybrid_uses_primary_backend():
    backend = HybridTranslationBackend(
        primary=LocalTranslationBackend(lambda _: "local result"),
        fallback=ExternalTranslationBackend(lambda _: "external result"),
    )

    result = backend.translate("test")

    assert result.translated_text == "local result"
    assert result.backend == "local"


def test_hybrid_falls_back_when_primary_fails():
    backend = HybridTranslationBackend(
        primary=LocalTranslationBackend(
            lambda _: (_ for _ in ()).throw(RuntimeError("local failed"))
        ),
        fallback=ExternalTranslationBackend(lambda _: "external result"),
    )

    result = backend.translate("test")

    assert result.translated_text == "external result"
    assert result.backend == "external"


def test_factory_creates_hybrid_backend():
    backend = create_translation_backend("hybrid")

    assert isinstance(backend, HybridTranslationBackend)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        create_translation_backend("unsupported")
