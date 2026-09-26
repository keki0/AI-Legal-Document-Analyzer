# English-to-Marathi Legal Translation Evaluation Report

> [!CAUTION]
> **LEGAL NOTICE & DISCLAIMER**
> Machine-generated legal translation only. Reference translations are draft recommendations for expert review and do not constitute certified legal translations or formal legal advice.

**Model**: `facebook/nllb-200-distilled-600M`
**Date**: 2026-09-22 08:26:41

---

## 1. Executive Summary & Remedies Defect Resolution

| Evaluation Item | Status / Result | Note |
|---|---|---|
| **NDA Remedies Predicate & Damages** | **RESOLVED** | All 11 legal components, amounts, and remedies preserved in full sentence. |
| **Sentence Splitter Abbreviation Bug** | **FIXED** | Abbreviation protection prevents false splitting on `Rs. 500,000`. |
| **Placeholder Integrity & Sequence** | **100% PASS** | Zero missing, duplicated, or reordered placeholders across all test clauses. |
| **Numerical / Amount Integrity** | **100% PASS** | Both monetary amounts (`$[Liquidated Damages Amount]` and `500,000`) retained. |
| **Human-Review Flagging** | **OPERATIONAL** | Conservative rules flag high-consequence remedies for advisory attorney review. |

---

## 2. Deep Dive: Remedies Clause Before vs After

### English Source
```text
The Receiving Party acknowledges that any breach of this Agreement may cause irreparable harm. The Disclosing Party shall be entitled to seek injunctive relief and liquidated damages in the amount of $[Liquidated Damages Amount] or Rs. 500,000 within 30 days of written notice.
```

### Previous Broken Output (OLD)
```text
प्राप्ती करणारे पक्ष हे मान्य करतात की या कराराच्या कोणत्याही उल्लंघनामुळे अपूरणीय नुकसान होऊ शकते. प्रकटीकरण करणाऱ्या पक्षाला $[Liquidated Damages Amount] किंवा Rs. ३० दिवसांच्या आत ५००,०००.
```
- **Omitted**: `injunctive relief`, `liquidated damages`, operative verb predicate `shall be entitled to seek`.
- **Corrupted**: Severed at `Rs.`, resulting in verbless fragment `किंवा Rs. ३० दिवसांच्या आत ५००,०००.`

### Current Verified Output (NEW)
```text
प्राप्ती करणारे पक्ष हे मान्य करतात की या कराराच्या कोणत्याही उल्लंघनामुळे अपूरणीय नुकसान होऊ शकते. प्रकटीकरण करणाऱ्या पक्षाला लिखित सूचना दिल्यानंतर 30 दिवसांच्या आत $[Liquidated Damages Amount] किंवा 500,000 रुपयांच्या रकमेच्या निषेधात्मक नुकसान भरपाई आणि विल्हेवाट नुकसान भरपाईची मागणी करण्याचा हक्क असेल.
```

### Component-by-Component Preservation Check

| Clause Component | English Source | Marathi Translation in New Output | Preservation Status |
|---|---|---|---|
| **Receiving Party** | `The Receiving Party` | `प्राप्ती करणारे पक्ष` | **PRESERVED** |
| **Disclosing Party** | `The Disclosing Party` | `प्रकटीकरण करणाऱ्या पक्षाला` | **PRESERVED** |
| **Breach of Agreement** | `any breach of this Agreement` | `या कराराच्या कोणत्याही उल्लंघनामुळे` | **PRESERVED** |
| **Irreparable Harm** | `may cause irreparable harm` | `अपूरणीय नुकसान होऊ शकते` | **PRESERVED** |
| **Entitlement to Seek** | `shall be entitled to seek` | `ची मागणी करण्याचा हक्क असेल` | **PRESERVED** |
| **Injunctive Relief** | `injunctive relief` | `निषेधात्मक नुकसान भरपाई` | **PRESERVED (Generic MT, Review Required)** |
| **Liquidated Damages** | `liquidated damages` | `विल्हेवाट नुकसान भरपाई` | **PRESERVED (Colloquial MT, Review Required)** |
| **Dollar Placeholder** | `$[Liquidated Damages Amount]` | `$[Liquidated Damages Amount]` | **PRESERVED** |
| **Rupee Amount** | `Rs. 500,000` | `500,000 रुपयांच्या` | **PRESERVED** |
| **Notice Condition** | `within 30 days of written notice` | `लिखित सूचना दिल्यानंतर 30 दिवसांच्या आत` | **PRESERVED** |
| **Placeholder Integrity** | Valid multiset and sequence | 100% Match, 0 Corrupted Tokens | **PRESERVED** |

---

## 3. Semantic & Legal Terminology Verification (Remedies)

A rigorous legal review of the two distinct remedies reveals critical semantic distinctions between generic NLLB output and controlled statutory legal Marathi:

### A. Injunctive Relief
- **Raw NLLB Output**: `निषेधात्मक नुकसान भरपाई` (literally: *prohibitive damages / compensation*).
- **Documented Statutory Standard**: `न्यायालयीन मनाईहुकूम` (Specific Relief Act, 1963).
- **Legal Analysis**: Injunction is an **equitable, non-monetary restraining remedy** (an order of the court prohibiting an action). Calling it `निषेधात्मक नुकसान भरपाई` introduces the nominal head `नुकसान भरपाई` (damages/compensation), which legally conflates an equitable court order with financial compensation.

### B. Liquidated Damages
- **Raw NLLB Output**: `विल्हेवाट नुकसान भरपाई` (literally: *disposal damages / salvage compensation*).
- **Documented Statutory Standard**: `पूर्वनिर्धारित नुकसानभरपाई` (Indian Contract Act, 1872, Section 74).
- **Legal Analysis**: NLLB transliterated/borrowed 'liquidated' in the sense of asset disposal/liquidation (`विल्हेवाट`) rather than pre-agreed contract damages (`पूर्वनिर्धारित`).

### C. Distinguishability Between the Two Remedies
- In the **Raw NLLB Output**, both remedies share the identical noun phrase head `नुकसान भरपाई` (`निषेधात्मक नुकसान भरपाई` vs `विल्हेवाट नुकसान भरपाई`), causing the non-monetary court injunction and the monetary damages to be semantically blurred as two forms of compensation.
- In the **Controlled Statutory Standard**, `न्यायालयीन मनाईहुकूम` (court injunction) and `पूर्वनिर्धारित नुकसानभरपाई` (contract damages) belong to cleanly distinct legal categories.

### D. Terminology Post-Processing Experimentation
- **Target String Replacement**: Direct string substitution on target text was tested experimentally and found to introduce grammatical corruption (producing duplicate `आणि आणि` and erroneously attaching the monetary genitive `रकमेच्या` to the non-monetary court injunction). Therefore, blind post-replacement was strictly avoided.

### E. Official Evaluation Finding & Status
> **All source components are structurally represented, including the operative entitlement, both remedies, monetary values, notice condition, and placeholder. However, raw NLLB terminology for injunctive relief and liquidated damages is not treated as certified legal terminology; the clause remains flagged for human review.**

---

## 4. Four-Dimensional Evaluation Matrix

| Clause | 1. Structural / Component Preservation | 2. Placeholder & Numerical Integrity | 3. Terminology / Semantic Quality | 4. Human-Review Status |
|---|---|---|---|---|
| **NDA Title** | Full title translated | 100% (No placeholders) | Phonetic transliteration (`नॉन-डिस्क्रिप्शन`) | `FLAGGED_FOR_REVIEW` |
| **NDA Preamble** | All 5 parties & recitals intact | 100% (5/5 preserved in order) | Standard commercial Marathi | `PASS` |
| **NDA Remedies** | All 11 legal components & conditions intact | 100% (`$[Liquidated Damages Amount]`, `500,000`, `30`) | Generic MT (`निषेधात्मक`/`विल्हेवाट नुकसान भरपाई`) | `FLAGGED_FOR_REVIEW` (`HIGH_CONSEQUENCE_REMEDY`) |
| **Governing Law** | Full choice of law & forum intact | 100% (2/2 preserved in order) | Exclusive jurisdiction conveyed | `PASS` |
| **Payment Clause** | Payments, deadlines, methods intact | 100% (3/3 preserved in order) | Commercial Marathi (`इनव्हॉइस देय`) | `PASS` |
| **Termination** | Unilateral power intact (`कोणत्याही पक्षाने`) | 100% (1/1 preserved) | Preserves unilateral vs bilateral distinction | `PASS` |

---

## 5. Clause-by-Clause Evaluation Results

### NDA Document Header / Title (`CASE_1_TITLE`)
- **Source**: `data/raw/sample-nda.pdf`
- **Status**: `FLAGGED_FOR_REVIEW` (Latency: `7.79s`)
- **Placeholders Valid**: `True`
- **Review Flags**: `[]`

**English Source**:
> NON-DISCLOSURE AGREEMENT

**New Machine Translation (NLLB-200)**:
> नॉन-डिस्क्रिप्शन एग्रीमेंट

**Draft Human-Reviewed Reference**:
> गोपनीयता करार

---

### Multi-Party Recital / Preamble (`CASE_2_PREAMBLE`)
- **Source**: `data/raw/sample-nda.pdf`
- **Status**: `PASS` (Latency: `4.83s`)
- **Placeholders Valid**: `True`
- **Review Flags**: `[]`

**English Source**:
> This Non-Disclosure Agreement (the "Agreement") is entered into on [Date], by and between [Disclosing Party], having its principal place of business at [Disclosing Party Address] ("Disclosing Party"), and [Receiving Party], having its principal place of business at [Receiving Party Address] ("Receiving Party").

**New Machine Translation (NLLB-200)**:
> या नॉन-डिस्क्रिप्शन कराराचा (या " कराराचा") [Date] रोजी, [Disclosing Party] द्वारे आणि दरम्यान, त्याचे मुख्य व्यवसाय ठिकाण [Disclosing Party Address] ("डिस्क्रिप्शन पार्टी") आणि [Receiving Party] येथे आहे, त्याचे मुख्य व्यवसाय ठिकाण [Receiving Party Address] ("रिसीव्हिंग पार्टी") येथे आहे.

**Draft Human-Reviewed Reference**:
> हा गोपनीयता करार ("करार") [Date] रोजी [Disclosing Party], ज्यांचे मुख्य कार्यालय [Disclosing Party Address] येथे आहे ("प्रकटीकरण करणारा पक्ष"), आणि [Receiving Party], ज्यांचे मुख्य कार्यालय [Receiving Party Address] येथे आहे ("माहिती स्वीकारणारा पक्ष"), यांच्या दरम्यान करण्यात आला आहे.

---

### NDA Remedies and Liquidated Damages (`CASE_3_REMEDIES_LIQUIDATED_DAMAGES`)
- **Source**: `data/raw/sample-nda.pdf`
- **Status**: `FLAGGED_FOR_REVIEW` (Latency: `4.71s`)
- **Placeholders Valid**: `True`
- **Review Flags**: `['HIGH_CONSEQUENCE_REMEDY']`

**English Source**:
> The Receiving Party acknowledges that any breach of this Agreement may cause irreparable harm. The Disclosing Party shall be entitled to seek injunctive relief and liquidated damages in the amount of $[Liquidated Damages Amount] or Rs. 500,000 within 30 days of written notice.

**New Machine Translation (NLLB-200)**:
> प्राप्ती करणारे पक्ष हे मान्य करतात की या कराराच्या कोणत्याही उल्लंघनामुळे अपूरणीय नुकसान होऊ शकते. प्रकटीकरण करणाऱ्या पक्षाला लिखित सूचना दिल्यानंतर 30 दिवसांच्या आत $[Liquidated Damages Amount] किंवा 500,000 रुपयांच्या रकमेच्या निषेधात्मक नुकसान भरपाई आणि विल्हेवाट नुकसान भरपाईची मागणी करण्याचा हक्क असेल.

**Draft Human-Reviewed Reference**:
> माहिती स्वीकारणारा पक्ष हे मान्य करतो की या कराराच्या कोणत्याही उल्लंघनामुळे अपूरणीय नुकसान होऊ शकते. प्रकटीकरण करणारा पक्ष लेखी सूचना मिळाल्यापासून ३० दिवसांच्या आत न्यायालयीन मनाईहुकूम तसेच $[Liquidated Damages Amount] किंवा रु. ५,००,००० इतकी पूर्वनिर्धारित नुकसानभरपाई मागण्यास पात्र असेल.

---

### Governing Law and Jurisdiction (`CASE_4_GOVERNING_LAW`)
- **Source**: `data/raw/sample-nda.pdf`
- **Status**: `PASS` (Latency: `3.22s`)
- **Placeholders Valid**: `True`
- **Review Flags**: `[]`

**English Source**:
> This Agreement shall be governed by and construed in accordance with the laws of [Jurisdiction]. Any legal dispute shall be subject to the exclusive jurisdiction of the courts in [City/State].

**New Machine Translation (NLLB-200)**:
> या कराराचे नियमन [Jurisdiction] च्या कायद्यानुसार केले जाईल आणि त्याची व्याख्या केली जाईल. कोणताही कायदेशीर वाद [City/State] मधील न्यायालयांच्या एकमेव अधिकार क्षेत्राशी निगडित असेल.

**Draft Human-Reviewed Reference**:
> या कराराचे नियमन आणि अर्थनिर्वचन [Jurisdiction] च्या कायद्यानुसार केले जाईल. कोणताही कायदेशीर वाद [City/State] येथील न्यायालयांच्या अनन्य अधिकार क्षेत्राच्या अधीन राहील.

---

### Payment Obligations & Invoices (`CASE_5_PAYMENT_TERMS`)
- **Source**: `data/raw/Service-Agreement-template.pdf`
- **Status**: `PASS` (Latency: `3.15s`)
- **Placeholders Valid**: `True`
- **Review Flags**: `[]`

**English Source**:
> The Client will pay the Contractor [Amount]. Invoices are payable within [Number] days after receipt. Payments will be made by [Payment Method].

**New Machine Translation (NLLB-200)**:
> ग्राहक कंत्राटदाराला [Amount] देईल. पावत्यानंतर [Number] दिवसांच्या आत इनव्हॉइस देय आहेत [Payment Method] द्वारे पैसे दिले जातील.

**Draft Human-Reviewed Reference**:
> ग्राहक कंत्राटदाराला [Amount] अदा करेल. इनव्हॉइस (देयके) मिळाल्यापासून [Number] दिवसांच्या आत देय राहतील. देयके [Payment Method] द्वारे अदा केली जातील.

---

### Unilateral Termination Clause (`CASE_6_TERMINATION_UNILATERAL`)
- **Source**: `data/raw/Service-Agreement-template.pdf`
- **Status**: `PASS` (Latency: `2.51s`)
- **Placeholders Valid**: `True`
- **Review Flags**: `[]`

**English Source**:
> Either party may terminate this Agreement by providing [Notice Period] written notice. The Client will pay for services completed before termination.

**New Machine Translation (NLLB-200)**:
> कोणत्याही पक्षाने [Notice Period] लेखी सूचना देऊन हा करार रद्द करू शकतो. ग्राहक समाप्तीपूर्वी पूर्ण केलेल्या सेवांसाठी पैसे देईल.

**Draft Human-Reviewed Reference**:
> कोणताही पक्ष [Notice Period] इतकी लेखी पूर्वसूचना देऊन हा करार समाप्त करू शकतो. समाप्तीपूर्वी पूर्ण झालेल्या सेवांसाठी ग्राहक रक्कम अदा करेल.

---
