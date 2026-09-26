# Focused Translation Quality Review: English-to-Marathi Legal Translation

> [!CAUTION]
> **LEGAL NOTICE & DISCLAIMER**
> All translations evaluated and presented herein are **machine-generated** using NLLB-200 (`facebook/nllb-200-distilled-600M`). Any proposed Marathi reference translations are **draft recommendations for review** by qualified legal translators and do not constitute certified legal translations or formal legal advice.

---

## 1. Problematic Examples & Quality Analysis

| # | Clause / Section | Original English | Current Machine Marathi Output | Specific Defect | Legal Meaning Impact | Draft Human-Reviewed Reference (For Review Only) |
|---|---|---|---|---|---|---|
| **1** | **Agreement Title** (`sample-nda.pdf`) | `NON-DISCLOSURE AGREEMENT` | `नॉन-डिस्क्रिप्शन एग्रीमेंट` | Phonological confusion of "Disclosure" with "Description" (`डिस्क्रिप्शन`) and transliteration of "Agreement" (`एग्रीमेंट`). Output reads "Non-Description Agreement". | **Distorted** | `गोपनीयता करार` *(पर्यायी: माहिती उघड न करण्याचा करार)* |
| **2** | **Multi-Party Preamble** (`sample-nda.pdf`) | `This Non-Disclosure Agreement (the "Agreement") is entered into on [Date], by and between [Disclosing Party], having its principal place of business at [Disclosing Party Address] ("Disclosing Party"), and [Receiving Party], having its principal place of business at [Receiving Party Address] ("Receiving Party").` | `या प्रकटीकरण न करण्याच्या कराराचा (" कराराचा") [Date] वर [Disclosing Party] द्वारे आणि [Disclosing Party] द्वारे, ज्याचे मुख्य व्यवसाय ठिकाण [Disclosing Party Address] ("प्रकटीकरण करणारा पक्ष") येथे आहे आणि [Receiving Party] द्वारे, ज्याचे मुख्य व्यवसाय ठिकाण [Receiving Party Address] ("प्राप्त करणारा पक्ष) येथे आहे, दरम्यान केला जातो.` | Starts with uninflected genitive `या ... कराराचा`. Decoder repeats party token `[Disclosing Party] द्वारे आणि [Disclosing Party] द्वारे` in complex recitals. Missing closing quote. | **Awkwardly Expressed** | `हा गोपनीयता करार ("करार") [Date] रोजी [Disclosing Party], ज्यांचे मुख्य कार्यालय [Disclosing Party Address] येथे आहे ("प्रकटीकरण करणारा पक्ष"), आणि [Receiving Party], ज्यांचे मुख्य कार्यालय [Receiving Party Address] येथे आहे ("माहिती स्वीकारणारा पक्ष"), यांच्या दरम्यान करण्यात आला आहे.` |
| **3** | **Remedies & Damages** (`sample-nda.pdf`) | `The Receiving Party acknowledges that any breach of this Agreement may cause irreparable harm. The Disclosing Party shall be entitled to seek injunctive relief and liquidated damages in the amount of $[Liquidated Damages Amount] or Rs. 500,000 within 30 days of written notice.` | `प्राप्ती करणारे पक्ष हे मान्य करतात की या कराराच्या कोणत्याही उल्लंघनामुळे अपूरणीय नुकसान होऊ शकते. प्रकटीकरण करणाऱ्या पक्षाला $[Liquidated Damages Amount] किंवा Rs. ३० दिवसांच्या आत ५००,०००.` | First sentence on irreparable harm (`अपूरणीय नुकसान`) is accurate. Second sentence terminates abruptly at the number, dropping the operative verb predicate (`मागण्यास पात्र असेल`) and `injunctive relief`. | **Lost** | `माहिती स्वीकारणारा पक्ष हे मान्य करतो की या कराराच्या कोणत्याही उल्लंघनामुळे अपूरणीय नुकसान होऊ शकते. प्रकटीकरण करणारा पक्ष लेखी सूचना मिळाल्यापासून ३० दिवसांच्या आत न्यायालयीन मनाईहुकूम (injunctive relief) तसेच $[Liquidated Damages Amount] किंवा रु. ५,००,००० इतकी पूर्वनिर्धारित नुकसानभरपाई मागण्यास पात्र असेल.` |
| **4** | **Governing Law & Jurisdiction** (`sample-nda.pdf`) | `This Agreement shall be governed by and construed in accordance with the laws of [Jurisdiction]. Any legal dispute shall be subject to the exclusive jurisdiction of the courts in [City/State].` | `या कराराचे नियमन [Jurisdiction] च्या कायद्यानुसार केले जाईल आणि त्याची व्याख्या केली जाईल. कोणताही कायदेशीर वाद [City/State] च्या न्यायालयांच्या एकमेव अधिकार क्षेत्राच्या अधीन असेल.` | Coordination dropped second entity reference in "and construed in accordance with...". `एकमेव अधिकार क्षेत्र` conveys exclusive jurisdiction, though statutory legal Marathi uses `अनन्य अधिकार क्षेत्र`. | **Awkwardly Expressed** | `या कराराचे नियमन आणि अर्थनिर्वचन [Jurisdiction] च्या कायद्यानुसार केले जाईल. कोणताही कायदेशीर वाद [City/State] येथील न्यायालयांच्या अनन्य अधिकार क्षेत्राच्या अधीन राहील.` |
| **5** | **Payment Obligations** (`Service-Agreement-template.pdf`) | `The Client will pay the Contractor [Amount]. Invoices are payable within [Number] days after receipt. Payments will be made by [Payment Method].` | `ग्राहक कंत्राटदाराला [Amount] देईल. प्राप्तीनंतर [Number] दिवसांच्या आत इनव्हॉयर्स देय आहेत [Payment Method] द्वारे पैसे दिले जातील.` | Transliterates "Invoices" as `इनव्हॉयर्स` instead of `इनव्हॉइस`/`देयके`. Missing period between sentence 2 and sentence 3. `देईल` is informal compared to commercial `अदा करेल`. | **Awkwardly Expressed** | `ग्राहक कंत्राटदाराला [Amount] अदा करेल. इनव्हॉइस (देयके) मिळाल्यापासून [Number] दिवसांच्या आत देय राहतील. देयके [Payment Method] द्वारे अदा केली जातील.` |
| **6** | **Termination & Notice** (`Service-Agreement-template.pdf`) | `Either party may terminate this Agreement by providing [Notice Period] written notice. The Client will pay for services completed before termination.` | `दोन्ही पक्षांनी [Notice Period] ला लिखित सूचना देऊन हा करार रद्द करू शकतो. ग्राहक समाप्तीपूर्वी पूर्ण केलेल्या सेवांसाठी पैसे देईल.` | **Severe substantive distortion**: `Either party` (unilateral right) mistranslated as `दोन्ही पक्षांनी` (mutual/bilateral requirement). Distorts unilateral termination into a requirement for joint action. | **Distorted** | `कोणताही पक्ष [Notice Period] इतकी लेखी पूर्वसूचना देऊन हा करार समाप्त करू शकतो. समाप्तीपूर्वी पूर्ण झालेल्या सेवांसाठी ग्राहक रक्कम अदा करेल.` |

---

## 2. Categorization of Issues

### A. Fixed by Entity Masking
* **Preposition-to-Postposition Syntax**: English prepositions (`within`, `located at`, `governed by`, `by`) now properly attach as Marathi postpositions (`[Number] दिवसांच्या आत`, `[State] च्या कायद्यानुसार`, `[Payment Method] द्वारे`, `[Location] येथील`).
* **Payment Obligation Retention**: `[Payment Method] द्वारे पैसे दिले जातील` (Payments will be made by [Payment Method]) is no longer dropped.
* **Written Notice Obligation**: `लिखित सूचना देऊन` (by providing written notice) is no longer eliminated.
* **Placeholder Integrity**: 100% of bracketed tags (`[Client Name]`, `[Amount]`, `[Date]`) and currency symbols (`$[Liquidated Damages Amount]`) are preserved in order.

### B. Still Caused by the Underlying NLLB Model
* **Phonetic / Transliteration Hallucination**: `NON-DISCLOSURE` -> `नॉन-डिस्क्रिप्शन` ("Non-Description").
* **Distributive Pronoun Confusion**: Translating `Either party` as `दोन्ही पक्षांनी` ("both parties").
* **Decoder Looping / Stuttering on Long Multi-Entity Recitals**: Repeating party names (`[Disclosing Party] द्वारे आणि [Disclosing Party] द्वारे`) in complex preamble clauses.
* **Tail Truncation on Numerical Strings**: Omitting the verb predicate (`मागण्यास पात्र असेल`) at the tail end of long compound financial amounts.

### C. Caused by Formatting or Post-Processing
* **Whitespace before Bracketed Placeholders**: Addressed by `normalize_translation_formatting` ensuring spacing before `[` and around currency symbols `$`.
* **Visarga as Field Delimiter**: Addressed by normalizing `ः` back to `:` before placeholders and labels.
* **Sentence Glue**: When joining translated sentences, ensuring punctuation (periods) cleanly separates consecutive sentences.

### D. Not Safely Fixable Without Human-Reviewed Terminology
* **Legal Distinctions**:
  * `Either party` (`कोणताही पक्ष` / unilateral) vs `Both parties` (`दोन्ही पक्ष` / bilateral).
  * `Exclusive jurisdiction` (`अनन्य अधिकार क्षेत्र` vs `एकमेव अधिकार क्षेत्र`).
  * `Injunctive relief` (`न्यायालयीन मनाईहुकूम` / `मनाई आदेश`).
  * `Liquidated damages` (`पूर्वनिर्धारित नुकसानभरपाई`).
  * `Non-Disclosure Agreement` (`गोपनीयता करार`).
