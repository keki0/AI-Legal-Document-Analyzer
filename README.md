# AI-Powered Legal Document Simplifier & Clause Analyzer

An NLP + Deep Learning system that analyses legal documents and presents their
contents as structured, readable information: clause segmentation, Transformer-based
clause classification, importance and attention flags, information extraction,
plain-language simplification, summarisation, and semantic search.

> **Educational and research tool. Not legal advice.**
> The system explains documents. It does not make legal judgements, and it is
> not a substitute for a qualified legal professional.

## Status

| Phase | Component | State |
|---|---|---|
| 0 | Environment, config, dependencies | **Complete** |
| 1 | Document processing & clause segmentation | Not started |
| 2 | Dataset preparation & clause taxonomy | Not started |
| 3 | Clause classification (TF-IDF baseline + Legal-BERT) | Not started |
| 4 | Importance & attention flags | Not started |
| 5 | Information extraction | Not started |
| 6 | Simplification & summarisation (FLAN-T5) | Not started |
| 7 | Improved semantic retrieval | Not started |
| 8 | Streamlit application | Not started |
| 9 | Evaluation & documentation | Not started |

No model has been trained yet. **No metrics are reported anywhere in this
repository, because none have been produced.**

## Baseline

`notebooks/01_baseline.ipynb` is the completed, tested "before" system:
PyMuPDF extraction → regex section segmentation → MiniLM embeddings →
cosine retrieval → keyword risk rules. It is preserved unmodified and is the
comparison point for every improvement. See `docs/baseline.md`.

## Local setup (Windows, Python 3.10.11)

```bat
git clone https://github.com/keki0/AI-Legal-Document-Analyzer
cd AI-Legal-Document-Analyzer

py -3.10 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python scripts\verify_setup.py
```

Add `--data` to also download and verify LEDGAR (several hundred MB, one-off).

## Colab setup

```python
!git clone https://github.com/keki0/AI-Legal-Document-Analyzer
%cd AI-Legal-Document-Analyzer
!pip install -q -r requirements-colab.txt

import sys; sys.path.insert(0, '/content/AI-Legal-Document-Analyzer')
from src.config import describe_environment
print(describe_environment())
```

Do not install `requirements.txt` in Colab — it pins `torch` and would replace
Colab's CUDA build, costing GPU access and a runtime restart.

## Structure

```
app/          Streamlit application
data/raw/     source PDFs (sample contract tracked; everything else ignored)
data/eval/    taxonomy.yaml + retrieval queries — hand-authored, tracked
docs/         baseline report, methodology, results
models/       fine-tuned checkpoints (gitignored)
notebooks/    01_baseline (frozen) + experiment notebooks
scripts/      verify_setup.py
src/          reusable pipeline modules
tests/        pytest suite
```

## Datasets

| Dataset | Use | Licence |
|---|---|---|
| [LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) (LexGLUE), 60k/10k/10k, 100 labels | Classifier training | CC BY 4.0 |
| [plain_english_contracts_summarization](https://huggingface.co/datasets/joelniklaus/plain_english_contracts_summarization), 446 pairs | Simplification evaluation | Manor & Li (2019), NLLP Workshop |

The bundled contractor agreement is a smoke-test fixture only and is never
used as training data.
