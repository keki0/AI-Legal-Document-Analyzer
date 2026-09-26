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


class MockBackend:
    """Mock translator recording input chunks and returning mapped or echoed translations."""

    def __init__(self, mapping: dict[str, str] | None = None):
        self.mapping = mapping or {}
        self.received_chunks: list[str] = []

    def translate(self, text: str) -> TranslationResult:
        self.received_chunks.append(text)
        translated = self.mapping.get(text, f"अनुवाद({text})")
        return TranslationResult(
            source_text=text,
            translated_text=translated,
            backend="mock",
        )


def test_invoices_payable_within_number_days():
    from src.translation import translate_preserving_placeholders

    mock = MockBackend(
        {
            "Invoices are payable within __TAG_1__ days after receipt.": (
                "प्राप्तीनंतर __TAG_1__ दिवसांच्या आत इनव्हॉइस देय आहेत."
            )
        }
    )
    clause = "Invoices are payable within [Number] days after receipt."

    result = translate_preserving_placeholders(mock, clause)

    assert result == "प्राप्तीनंतर [Number] दिवसांच्या आत इनव्हॉइस देय आहेत."
    assert mock.received_chunks == [
        "Invoices are payable within __TAG_1__ days after receipt."
    ]


def test_payments_will_be_made_by_payment_method():
    from src.translation import translate_preserving_placeholders

    mock = MockBackend(
        {"Payments will be made by __TAG_1__.": "__TAG_1__ द्वारे पैसे दिले जातील."}
    )
    clause = "Payments will be made by [Payment Method]."

    result = translate_preserving_placeholders(mock, clause)

    assert result == "[Payment Method] द्वारे पैसे दिले जातील."
    assert mock.received_chunks == ["Payments will be made by __TAG_1__."]


def test_liquidated_damages_and_currency_symbol_preservation():
    from src.translation import translate_preserving_placeholders

    mock = MockBackend(
        {
            (
                "The Disclosing Party shall be entitled to seek injunctive relief "
                "and liquidated damages in the amount of $__TAG_1__."
            ): (
                "प्रकटीकरण करणाऱ्या पक्षाला $__TAG_1__ च्या रकमेच्या आदेशात्मक भरपाई "
                "आणि विल्हेवाट नुकसान भरपाई मागण्याचा हक्क असेल."
            )
        }
    )
    clause = (
        "The Disclosing Party shall be entitled to seek injunctive relief and "
        "liquidated damages in the amount of $[Liquidated Damages Amount]."
    )

    result = translate_preserving_placeholders(mock, clause)

    assert "$[Liquidated Damages Amount]" in result
    assert "आदेशात्मक भरपाई" in result
    assert "विल्हेवाट नुकसान भरपाई" in result


def test_governing_law_clause():
    from src.translation import translate_preserving_placeholders

    mock = MockBackend(
        {
            "This Agreement is governed by the laws of __TAG_1__.": (
                "हा करार __TAG_1__ च्या कायद्यानुसार नियंत्रित केला जातो."
            )
        }
    )
    clause = "This Agreement is governed by the laws of [State or Country]."

    result = translate_preserving_placeholders(mock, clause)

    assert result == "हा करार [State or Country] च्या कायद्यानुसार नियंत्रित केला जातो."
    assert mock.received_chunks == [
        "This Agreement is governed by the laws of __TAG_1__."
    ]


def test_multiple_placeholders_in_one_sentence_and_ordering():
    from src.translation import translate_preserving_placeholders

    mock = MockBackend(
        {
            (
                "This Agreement is entered into on __TAG_1__, by and between "
                "__TAG_2__ and __TAG_3__."
            ): "हा करार __TAG_1__ रोजी __TAG_2__ आणि __TAG_3__ दरम्यान केला जातो."
        }
    )
    clause = (
        "This Agreement is entered into on [Date], by and between "
        "[Disclosing Party] and [Receiving Party]."
    )

    result = translate_preserving_placeholders(mock, clause)

    assert result == (
        "हा करार [Date] रोजी [Disclosing Party] आणि [Receiving Party] दरम्यान "
        "केला जातो."
    )
    idx_date = result.index("[Date]")
    idx_disclosing = result.index("[Disclosing Party]")
    idx_receiving = result.index("[Receiving Party]")
    assert 0 <= idx_date < idx_disclosing < idx_receiving


def test_exact_placeholder_preservation_and_content_unmodified():
    from src.translation import translate_preserving_placeholders

    mock = MockBackend()
    clause = (
        "SERVICE AGREEMENT Effective Date: [Date] This Service Agreement is made "
        "between [Client Name], located at [Client Address], and [Contractor Name], "
        "located at [Contractor Address]."
    )

    result = translate_preserving_placeholders(mock, clause)

    expected = [
        "[Date]",
        "[Client Name]",
        "[Client Address]",
        "[Contractor Name]",
        "[Contractor Address]",
    ]
    for placeholder in expected:
        assert placeholder in result


def test_currency_symbol_preservation_multiple_currencies():
    from src.translation import translate_preserving_placeholders

    mock = MockBackend(
        {
            "Fees: $__TAG_1__, €__TAG_2__, £__TAG_3__, ₹__TAG_4__.": (
                "शुल्क: $__TAG_1__, €__TAG_2__, £__TAG_3__, ₹__TAG_4__."
            )
        }
    )
    clause = "Fees: $[Amount_1], €[Amount_2], £[Amount_3], ₹[Amount_4]."

    result = translate_preserving_placeholders(mock, clause)

    assert "$[Amount_1]" in result
    assert "€[Amount_2]" in result
    assert "£[Amount_3]" in result
    assert "₹[Amount_4]" in result


def test_empty_and_whitespace_only_input():
    from src.translation import translate_preserving_placeholders

    mock = MockBackend()

    assert translate_preserving_placeholders(mock, "") == ""
    assert translate_preserving_placeholders(mock, "   ") == ""
    assert translate_preserving_placeholders(mock, "\n\t  ") == ""
    assert len(mock.received_chunks) == 0


def test_formatting_cleanup_avoids_periods_before_placeholders():
    from src.translation import normalize_translation_formatting

    raw_with_period = "आत देय आहेत. [Number] दिवसांच्या आत"
    cleaned = normalize_translation_formatting(raw_with_period)
    assert cleaned == "आत देय आहेत [Number] दिवसांच्या आत"

    raw_glued = "होऊ शकते.[Amount]"
    cleaned_glued = normalize_translation_formatting(raw_glued)
    assert cleaned_glued == "होऊ शकते [Amount]"


def test_formatting_cleanup_preserves_spaces_and_collapses_punctuation():
    from src.translation import normalize_translation_formatting

    raw = "[Client Address]आणि इतर तपशील.. तसेच।। काही मुद्दे ,,"
    cleaned = normalize_translation_formatting(raw)

    assert "[Client Address] आणि" in cleaned
    assert ".." not in cleaned
    assert "।। " not in cleaned
    assert ",," not in cleaned
    assert cleaned.endswith("काही मुद्दे,")


def test_formatting_cleanup_normalizes_visarga_colon():
    from src.translation import normalize_translation_formatting

    assert normalize_translation_formatting("तारीखः [Date]") == "तारीख: [Date]"


def test_entity_collision_and_similar_names():
    from src.translation import translate_preserving_placeholders

    # Test 1: More than 10 placeholders so ENTITY_1 and ENTITY_10 coexist
    placeholders = [f"[Field_{i}]" for i in range(1, 13)]
    clause = " ".join(placeholders)

    mock = MockBackend()
    result = translate_preserving_placeholders(mock, clause)

    for p in placeholders:
        assert p in result

    # Test 2: Source text already contains literal "ENTITY_1" and "ENTITY_10"
    clause_with_literal = "Contract regarding ENTITY_1 and ENTITY_10 for [Client Name]."
    mock2 = MockBackend()
    result2 = translate_preserving_placeholders(mock2, clause_with_literal)

    assert "ENTITY_1" in result2
    assert "ENTITY_10" in result2
    assert "[Client Name]" in result2


def test_validate_placeholder_integrity_multiset_matching():
    from src.marathi.translator import validate_placeholder_integrity

    source = "Between [Party A], located at [Address A], and [Party B]."

    # 1. Exact match
    valid_trans = "करार [Party A], [Address A] येथील, आणि [Party B] दरम्यान."
    res_valid = validate_placeholder_integrity(source, valid_trans)
    assert res_valid.is_valid is True
    assert res_valid.missing_placeholders == []
    assert res_valid.duplicated_placeholders == []
    assert res_valid.invented_placeholders == []

    # 2. Missing placeholder
    missing_trans = "करार [Party A] आणि [Party B] दरम्यान."
    res_missing = validate_placeholder_integrity(source, missing_trans)
    assert res_missing.is_valid is False
    assert res_missing.missing_placeholders == ["[Address A]"]
    assert "Missing placeholders" in res_missing.error_message

    # 3. Duplicated placeholder
    dup_trans = "करार [Party A], [Address A] येथील, [Party A] आणि [Party B] दरम्यान."
    res_dup = validate_placeholder_integrity(source, dup_trans)
    assert res_dup.is_valid is False
    assert res_dup.duplicated_placeholders == ["[Party A]"]
    assert "Duplicated placeholders" in res_dup.error_message

    # 4. Invented placeholder
    inv_trans = "करार [Party A], [Address A] येथील, आणि [Party B] [Extra] दरम्यान."
    res_inv = validate_placeholder_integrity(source, inv_trans)
    assert res_inv.is_valid is False
    assert res_inv.invented_placeholders == ["[Extra]"]
    assert "Invented placeholders" in res_inv.error_message


def test_validate_placeholder_integrity_detects_swapped_order():
    from src.marathi.translator import validate_placeholder_integrity

    source = "Made between [Client Name] and [Contractor Name]."
    swapped_trans = "केले [Contractor Name] आणि [Client Name] दरम्यान."

    res = validate_placeholder_integrity(source, swapped_trans)
    assert res.is_valid is False
    assert res.is_order_preserved is False
    assert "Placeholder order swapped" in res.error_message


def test_validate_placeholder_integrity_repeated_placeholder_names():
    from src.marathi.translator import validate_placeholder_integrity

    source = "[Company] shall notify [Partner]. If [Company] defaults, [Partner] may terminate."
    valid_trans = "[Company] ने [Partner] ला सूचित करावे. [Company] ने कसूर केल्यास, [Partner] रद्द करू शकतो."

    res_valid = validate_placeholder_integrity(source, valid_trans)
    assert res_valid.is_valid is True
    assert res_valid.source_placeholders == [
        "[Company]",
        "[Partner]",
        "[Company]",
        "[Partner]",
    ]
    assert res_valid.translated_placeholders == [
        "[Company]",
        "[Partner]",
        "[Company]",
        "[Partner]",
    ]

    # One instance of [Company] dropped
    invalid_trans = (
        "[Company] ने [Partner] ला सूचित करावे. कसूर केल्यास, [Partner] रद्द करू शकतो."
    )
    res_invalid = validate_placeholder_integrity(source, invalid_trans)
    assert res_invalid.is_valid is False
    assert res_invalid.missing_placeholders == ["[Company]"]


def test_validate_placeholder_integrity_detects_corrupted_tokens():
    from src.marathi.translator import validate_placeholder_integrity

    source = "Invoices are payable within [Number] days."

    # Unrestored __TAG_1__
    trans_unrestored = "प्राप्तीनंतर __TAG_1__ दिवसांच्या आत देय."
    res = validate_placeholder_integrity(source, trans_unrestored)
    assert res.is_valid is False
    assert "__TAG_1__" in res.corrupted_tokens

    # Residual ENTITY_1 when not in source
    trans_residual_entity = "प्राप्तीनंतर ENTITY_1 दिवसांच्या आत देय."
    res2 = validate_placeholder_integrity(source, trans_residual_entity)
    assert res2.is_valid is False
    assert "ENTITY_1" in res2.corrupted_tokens


def test_validate_placeholder_integrity_allows_source_literal_entities():
    from src.marathi.translator import validate_placeholder_integrity

    source = "Clause for ENTITY_1 and ENTITY_10 regarding [Client Name]."
    trans = "ENTITY_1 आणि ENTITY_10 बद्दल [Client Name] साठी कलम."

    res = validate_placeholder_integrity(source, trans)
    assert res.is_valid is True
    assert res.corrupted_tokens == []


def test_translate_preserving_placeholders_no_silent_repair_fallback():
    from src.translation import translate_preserving_placeholders

    # Mock backend drops __TAG_2__
    mock = MockBackend({"Pay __TAG_1__ on __TAG_2__.": "देय __TAG_1__."})
    source = "Pay [Amount] on [Date]."

    # In fallback mode, must NOT append "[Date]" to the end silently; must reject with explanation
    result = translate_preserving_placeholders(mock, source, on_error="fallback")
    assert result.startswith("[Translation rejected:")
    assert "Missing placeholders: ['[Date]']" in result
    assert result.endswith(source)
    # Ensure it did not silently output "देय [Amount]. [Date]"
    assert not result.startswith("देय [Amount]")


def test_translate_preserving_placeholders_raises_on_error():
    from src.translation import (
        PlaceholderIntegrityError,
        translate_preserving_placeholders,
    )

    # Mock backend swaps tags
    mock = MockBackend(
        {"From __TAG_1__ to __TAG_2__.": "__TAG_2__ कडून __TAG_1__ पर्यंत."}
    )
    source = "From [Sender] to [Receiver]."

    with pytest.raises(PlaceholderIntegrityError) as exc_info:
        translate_preserving_placeholders(mock, source, on_error="raise")

    err = exc_info.value
    assert not err.validation_result.is_valid
    assert err.validation_result.is_order_preserved is False
    assert "Placeholder order swapped" in str(err)


def test_documented_legal_glossary():
    from src.marathi.glossary import apply_target_glossary, DOCUMENTED_LEGAL_GLOSSARY

    assert len(DOCUMENTED_LEGAL_GLOSSARY) >= 4

    text = "NON-DISCLOSURE AGREEMENT"
    assert apply_target_glossary(text) == "गोपनीयता करार"

    recital = "This Non-Disclosure Agreement is subject to exclusive jurisdiction and liquidated damages."
    glossed = apply_target_glossary(recital)
    assert "गोपनीयता करार" in glossed
    assert "अनन्य अधिकार क्षेत्र" in glossed
    assert "पूर्वनिर्धारित नुकसानभरपाई" in glossed


def test_split_translation_chunks_sentence_boundaries():
    from src.translation import split_translation_chunks

    compound_recital = (
        "This Agreement is entered into on [Date]. "
        "The Client agrees to pay [Amount]. "
        "Invoices are payable within [Number] days."
    )
    chunks = split_translation_chunks(compound_recital)
    assert len(chunks) == 3
    assert chunks[0] == "This Agreement is entered into on [Date]."
    assert chunks[1] == "The Client agrees to pay [Amount]."
    assert chunks[2] == "Invoices are payable within [Number] days."


def test_split_translation_chunks_preserves_rs_and_legal_abbreviations():
    from src.translation import split_translation_chunks

    text = (
        "The penalty shall be Rs. 500,000 within 30 days. "
        "Refer to Sec. 4 and Art. 5 for details on Acme Inc. and Beta Ltd. "
        "The estimated value is 3.14 times baseline."
    )
    chunks = split_translation_chunks(text)
    assert len(chunks) == 3
    assert chunks[0] == "The penalty shall be Rs. 500,000 within 30 days."
    assert "Rs. 500,000" in chunks[0]
    assert chunks[1] == "Refer to Sec. 4 and Art. 5 for details on Acme Inc. and Beta Ltd."
    assert "Acme Inc." in chunks[1]
    assert "Beta Ltd." in chunks[1]
    assert chunks[2] == "The estimated value is 3.14 times baseline."
    assert "3.14" in chunks[2]


def test_remedies_clause_not_severed_at_rs():
    from src.translation import split_translation_chunks

    remedies = (
        "The Receiving Party acknowledges that any breach of this Agreement may cause irreparable harm. "
        "The Disclosing Party shall be entitled to seek injunctive relief and liquidated damages "
        "in the amount of $[Liquidated Damages Amount] or Rs. 500,000 within 30 days of written notice."
    )
    chunks = split_translation_chunks(remedies)
    assert len(chunks) == 2
    assert chunks[0] == (
        "The Receiving Party acknowledges that any breach of this Agreement may cause irreparable harm."
    )
    assert chunks[1] == (
        "The Disclosing Party shall be entitled to seek injunctive relief and liquidated damages "
        "in the amount of $[Liquidated Damages Amount] or Rs. 500,000 within 30 days of written notice."
    )
    assert "Rs. 500,000" in chunks[1]
    assert chunks[1].endswith("within 30 days of written notice.")


def test_remedies_clause_pipeline_chunk_delivery():
    from src.translation import translate_preserving_placeholders

    mock = MockBackend(
        {
            "The Receiving Party acknowledges that any breach of this Agreement may cause irreparable harm.": (
                "प्राप्ती करणारे पक्ष हे मान्य करतात की या कराराच्या कोणत्याही उल्लंघनामुळे अपूरणीय नुकसान होऊ शकते."
            ),
            (
                "The Disclosing Party shall be entitled to seek injunctive relief and liquidated damages "
                "in the amount of $__TAG_1__ or Rs. 500,000 within 30 days of written notice."
            ): (
                "प्रकटीकरण करणाऱ्या पक्षाला लिखित सूचना दिल्यानंतर 30 दिवसांच्या आत $__TAG_1__ किंवा 500,000 "
                "रुपयांच्या रकमेच्या निषेधात्मक नुकसान भरपाई आणि विल्हेवाट नुकसान भरपाईची मागणी करण्याचा हक्क असेल."
            ),
        }
    )
    remedies = (
        "The Receiving Party acknowledges that any breach of this Agreement may cause irreparable harm. "
        "The Disclosing Party shall be entitled to seek injunctive relief and liquidated damages "
        "in the amount of $[Liquidated Damages Amount] or Rs. 500,000 within 30 days of written notice."
    )

    result = translate_preserving_placeholders(mock, remedies)

    assert "$[Liquidated Damages Amount]" in result
    assert "500,000" in result
    assert "अपूरणीय नुकसान" in result
    assert "मागणी करण्याचा हक्क असेल" in result
    # Verify both full chunks were dispatched without premature splitting at Rs.
    assert len(mock.received_chunks) == 2
    assert "Rs. 500,000" in mock.received_chunks[1]


def test_translation_review_flags_on_defects():
    from src.translation import check_translation_review_flags

    source = (
        "The Disclosing Party shall be entitled to seek injunctive relief and liquidated damages "
        "in the amount of $[Liquidated Damages Amount] or Rs. 500,000 within 30 days of written notice."
    )
    # Defect 1: Dropped numbers and dropped verb
    broken_translation = (
        "प्रकटीकरण करणाऱ्या पक्षाला $[Liquidated Damages Amount] किंवा Rs. ३० दिवसांच्या आत."
    )
    report = check_translation_review_flags(source, broken_translation)
    assert report.requires_human_review is True
    assert report.is_safe is False
    # Detects dropped number (500,000) and untranslated Latin token ('Rs')
    assert "NUMERICAL_DISCREPANCY" in report.review_flags
    assert "UNTRANSLATED_TOKENS" in report.review_flags

    # Defect 2: Dangling conjunction
    dangling_translation = (
        "प्रकटीकरण करणाऱ्या पक्षाला $[Liquidated Damages Amount] किंवा 500,000 रुपयांच्या किंवा"
    )
    report2 = check_translation_review_flags(source, dangling_translation)
    assert report2.requires_human_review is True
    assert "ABRUPT_TRUNCATION" in report2.review_flags


def test_translation_review_flags_clean_on_valid_output():
    from src.translation import check_translation_review_flags

    source = "Invoices are payable within [Number] days."
    translation = "प्राप्तीनंतर [Number] दिवसांच्या आत इनव्हॉइस देय आहेत."

    report = check_translation_review_flags(source, translation)
    assert report.is_safe is True
    assert "PLACEHOLDER_INTEGRITY_FAILED" not in report.review_flags
    assert "NUMERICAL_DISCREPANCY" not in report.review_flags
    assert "UNTRANSLATED_TOKENS" not in report.review_flags


def test_translate_legal_clause_api():
    from src.translation import translate_legal_clause

    mock = MockBackend(
        {"Invoices are payable within __TAG_1__ days.": "देयके __TAG_1__ दिवसांच्या आत देय आहेत."}
    )
    clause = "Invoices are payable within [Number] days."

    res = translate_legal_clause(mock, clause)
    assert res.source_text == clause
    assert res.translated_text == "देयके [Number] दिवसांच्या आत देय आहेत."
    assert res.report.is_safe is True
    assert res.report.placeholder_validation.is_valid is True


def test_translate_quick_summary_api():
    from src.translation import translate_quick_summary

    mock = MockBackend(
        {
            "The agreement defines payment terms and termination conditions.": (
                "करार देयक अटी आणि समाप्तीच्या अटी परिभाषित करतो."
            )
        }
    )
    summary_text = "The agreement defines payment terms and termination conditions."
    res = translate_quick_summary(summary_text, backend=mock)
    assert res.source_text == summary_text
    assert "देयक अटी" in res.translated_text
    assert res.report.is_safe is True


def test_translate_quick_summary_empty():
    from src.translation import translate_quick_summary

    mock = MockBackend()
    res = translate_quick_summary("", backend=mock)
    assert res.translated_text == ""
