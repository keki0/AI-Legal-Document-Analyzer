# Baseline Implementation Report

## Environment

- OS: Windows
- IDE: VS Code
- Python: 3.10.11
- Environment: Python virtual environment (.venv)
- Heavy computation: Google Colab
- Repository:
  https://github.com/keki0/AI-Legal-Document-Analyzer

## Baseline Architecture

PDF
↓
PyMuPDF text extraction
↓
Text cleaning
↓
Regex-based section segmentation
↓
Sentence Transformer embeddings
↓
Cosine similarity retrieval
↓
Keyword-based risk analysis

## Baseline Test Document

sample-independent-contractor-agreement.pdf

## Baseline Results

- Pages: 3
- Words: 643
- Sections detected: 9
- Embedding model: all-MiniLM-L6-v2
- Risk detection: keyword-based

### Risk-relevant clauses detected

- Term and Termination: HIGH
- Intellectual Property: HIGH
- Confidentiality: MEDIUM
- Compliance with Laws: MEDIUM
- Indemnification: HIGH
- Miscellaneous: MEDIUM

### Semantic Retrieval Test

Query:

"Who owns the work produced by the contractor?"

Expected relevant clause:

"Intellectual Property"

Retrieved clause:

"Independent Contractor Status"

Similarity:

0.5407

This demonstrates a weakness in the baseline semantic retrieval system and should be addressed in the improved implementation.