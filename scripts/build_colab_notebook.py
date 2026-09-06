"""Generate notebooks/03_classification_colab.ipynb.

Kept as a generator rather than a hand-edited .ipynb so the notebook is
reviewable in diffs and cannot be committed with stale execution counts or
accidental output.

    python scripts/build_colab_notebook.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "03_classification_colab.ipynb"

GITHUB_REPO = "https://github.com/keki0/AI-Legal-Document-Analyzer"


def _source_lines(text: str) -> list[str]:
    """Split text into nbformat ``source`` entries.

    Every entry except the last must KEEP its trailing newline. nbformat
    joins ``source`` with ``"".join(...)``, not ``"\\n".join(...)``, so a
    plain ``.split("\\n")`` produces a list whose entries get concatenated
    into a single unbroken line -- valid JSON, broken Python.
    """
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(text.strip()),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(text.strip("\n")),
    }


CELLS = [
    md(f"""
# Legal-BERT clause classification — Colab training

Fine-tunes `nlpaueb/legal-bert-base-uncased` on the 15-category LEDGAR
taxonomy prepared in Phase 2.

**Runtime → Change runtime type → T4 GPU** before running anything.

Configuration is deliberately conservative for a free-tier T4:

| Setting | Value | Why |
|---|---|---|
| `max_length` | 384 | Measured: p95 = 360 tokens. Not chosen arbitrarily. |
| batch size | 16 | 384-token sequences at batch 32 risk OOM on a 16 GB T4. |
| grad accumulation | 2 | Restores an effective batch of 32 without the memory cost. |
| epochs | 3 | Cut to 2 if the timing gate says so. |
| fp16 | on | T4 is Turing and has tensor cores. |
| dynamic padding | on | See the note in the tokenisation cell — this is the big time saver. |
| class weights | balanced | Matches the `class_weight="balanced"` given to the SVC baseline. |

Repo: {GITHUB_REPO}
"""),

    md("## 1. Check the GPU\n\nStop here if this reports no GPU."),
    code("""
!nvidia-smi
import torch
assert torch.cuda.is_available(), "No GPU. Runtime -> Change runtime type -> T4 GPU."
print("GPU :", torch.cuda.get_device_name(0))
print("VRAM: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
"""),

    md("## 2. Mount Drive\n\nCheckpoints go to Drive. A free-tier session can "
       "disconnect mid-run, and a run that dies at 90% with nothing saved costs "
       "the whole slot."),
    code("""
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
DRIVE_OUT = Path('/content/drive/MyDrive/legal-doc-analyzer/legal-bert-clause-classifier')
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
print("checkpoints ->", DRIVE_OUT)
"""),

    md("## 3. Clone the repo and install\n\n`requirements-colab.txt` deliberately "
       "omits torch and numpy — replacing Colab's CUDA build of torch is the "
       "most common way to lose GPU access and costs a runtime restart."),
    code(f"""
!git clone {GITHUB_REPO} /content/AI-Legal-Document-Analyzer 2>/dev/null || (cd /content/AI-Legal-Document-Analyzer && git pull)
%cd /content/AI-Legal-Document-Analyzer
!pip install -q -r requirements-colab.txt

import sys
sys.path.insert(0, '/content/AI-Legal-Document-Analyzer')

from src.config import MODELS, TRAIN, PATHS, describe_environment, set_seed
set_seed()
print(describe_environment())
"""),

    md("## 4. Load data and apply the taxonomy\n\nRebuilt here rather than "
       "uploaded: LEDGAR is only ~16 MB and regenerating guarantees Colab and "
       "local use exactly the same mapping."),
    code("""
from src.dataset import (
    load_ledgar, load_taxonomy, apply_taxonomy,
    subsample_train, class_distribution, describe_split_provenance,
)

taxonomy = load_taxonomy()
dataset  = load_ledgar()

prov = describe_split_provenance(dataset)
print("features            :", prov['features'])
print("temporal fields     :", prov['temporal_fields_found'] or 'NONE')
print("chronology verifiable:", prov['can_verify_chronology'])

dataset = apply_taxonomy(dataset, taxonomy)
dataset = subsample_train(dataset, TRAIN.train_subsample)   # 20,000

CATEGORIES = taxonomy.categories
NUM_LABELS = len(CATEGORIES)
print(f"\\ncategories: {NUM_LABELS}")
for split in dataset:
    print(f"  {split:<12} {len(dataset[split]):,}")
"""),

    md("## 5. Class distribution\n\nDrives the loss weights in the next cell."),
    code("""
counts = class_distribution(dataset, 'train')
total  = sum(counts.values())
print(f"{'category':<38}{'count':>8}{'%':>8}")
for cat, n in counts.most_common():
    print(f"{cat:<38}{n:>8,}{100*n/total:>7.2f}%")

largest, smallest = counts.most_common()[0], counts.most_common()[-1]
print(f"\\nimbalance: {largest[1]/smallest[1]:.1f}x  "
      f"({largest[0]} = {largest[1]:,}  vs  {smallest[0]} = {smallest[1]:,})")
"""),

    md("""
## 6. Class weights

`balanced` uses `n / (k * count_c)` — scikit-learn's formula, so the
Transformer and the SVC baseline are weighted identically and the comparison
is fair.

If validation shows macro-F1 barely moving while weighted-F1 drops sharply,
the weighting has over-corrected and the model is over-predicting rare
classes. Switch `scheme` to `sqrt_balanced` (a gentler ~3.6x spread instead
of ~13x) rather than abandoning weighting altogether.
"""),
    code("""
import numpy as np, torch
from src.classification import compute_class_weights

WEIGHT_SCHEME = 'balanced'      # or 'sqrt_balanced'
class_weights = compute_class_weights(
    dataset['train']['category_id'], NUM_LABELS, scheme=WEIGHT_SCHEME
)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).cuda()

for cat, w in sorted(zip(CATEGORIES, class_weights), key=lambda x: -x[1]):
    print(f"{cat:<38}{w:>8.3f}")
"""),

    md("""
## 7. Tokenise

**`max_length=384` is a truncation ceiling, not a fixed padding width.**
We pad dynamically to the longest sequence in each batch (`DataCollatorWithPadding`)
and group similar-length examples together (`group_by_length=True`).

This matters a lot. p95 is 360 tokens but the median is far lower, so padding
every example to 384 would spend most of the GPU budget on padding tokens.
Dynamic padding changes nothing about the science — identical inputs, identical
truncation — and is the single largest time saving available here.
"""),
    code("""
from transformers import AutoTokenizer

MAX_LENGTH = 384        # measured: p95 = 360
tokenizer  = AutoTokenizer.from_pretrained(MODELS.classifier)

def tokenize(batch):
    # No padding here -- the collator pads per batch.
    return tokenizer(batch['text'], truncation=True, max_length=MAX_LENGTH)

tokenized = dataset.map(tokenize, batched=True, desc='Tokenizing')
tokenized = tokenized.rename_column('category_id', 'labels')
keep = {'input_ids', 'attention_mask', 'token_type_ids', 'labels'}
tokenized = tokenized.remove_columns(
    [c for c in tokenized['train'].column_names if c not in keep]
)
tokenized.set_format('torch')

lengths = [len(x) for x in tokenized['train']['input_ids'][:2000]]
print(f"actual token lengths -- mean {np.mean(lengths):.0f}, "
      f"median {np.median(lengths):.0f}, max {max(lengths)}")
print(f"padding waste if fixed-width 384: "
      f"{100*(1 - np.mean(lengths)/384):.0f}% of every batch")
"""),

    md("## 8. Model and weighted-loss Trainer\n\n`Trainer` uses unweighted "
       "cross-entropy by default, so `compute_loss` is overridden. Everything "
       "else stays stock."),
    code("""
from transformers import AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    MODELS.classifier,
    num_labels=NUM_LABELS,
    id2label={i: c for i, c in enumerate(CATEGORIES)},
    label2id={c: i for i, c in enumerate(CATEGORIES)},
)
print(f"parameters: {model.num_parameters()/1e6:.1f}M")

class WeightedTrainer(Trainer):
    \"\"\"Trainer with class-weighted cross-entropy.\"\"\"
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop('labels')
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, NUM_LABELS),
            labels.view(-1),
            weight=class_weights_t,
        )
        return (loss, outputs) if return_outputs else loss
"""),

    md("## 9. Metrics\n\nSame `evaluate_predictions` used by the SVC baseline, so "
       "any difference between the two models is the model, not the scoring."),
    code("""
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy':    accuracy_score(labels, preds),
        'macro_f1':    f1_score(labels, preds, average='macro',    zero_division=0),
        'weighted_f1': f1_score(labels, preds, average='weighted', zero_division=0),
    }
"""),

    md("""
## 10. TIMING GATE — run this before the real training

Trains for ~100 steps and extrapolates. **Do not skip this.** Free-tier T4
throughput varies with the machine you are allocated, so the only reliable
runtime estimate is a measured one.

Decision rule printed by the cell:

* under ~45 min → proceed with 3 epochs
* 45–75 min → drop to 2 epochs
* over ~75 min → switch to `MODELS.classifier_fallback`
  (`legal-bert-small-uncased`)
"""),
    code("""
import time
from transformers import TrainingArguments, DataCollatorWithPadding

collator = DataCollatorWithPadding(tokenizer)

BATCH  = 16     # 384-token sequences at 32 risk OOM on a 16 GB T4
ACCUM  = 2      # effective batch 32
EPOCHS = 3

probe_args = TrainingArguments(
    output_dir='/content/probe',
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,
    max_steps=100,
    learning_rate=TRAIN.learning_rate,
    fp16=True,
    group_by_length=True,
    logging_steps=50,
    report_to='none',
    save_strategy='no',
)

probe = WeightedTrainer(
    model=model, args=probe_args,
    train_dataset=tokenized['train'],
    data_collator=collator, compute_metrics=compute_metrics,
)

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
probe.train()
probe_seconds = time.perf_counter() - t0

peak_gb        = torch.cuda.max_memory_allocated() / 1e9
sec_per_step   = probe_seconds / 100
steps_epoch    = len(tokenized['train']) // (BATCH * ACCUM)
est_total_min  = sec_per_step * steps_epoch * EPOCHS / 60

print(f"\\n{'='*60}")
print(f"100 steps in       : {probe_seconds:.1f}s  ({sec_per_step:.2f}s/step)")
print(f"peak VRAM          : {peak_gb:.1f} GB")
print(f"optimizer steps/ep : {steps_epoch}")
print(f"ESTIMATED TRAINING : {est_total_min:.0f} min for {EPOCHS} epochs")
print(f"  (+ evaluation on {len(tokenized['validation']):,} validation rows per epoch)")
print('='*60)

if   est_total_min < 45: print("VERDICT: proceed with 3 epochs.")
elif est_total_min < 75: print("VERDICT: drop EPOCHS to 2 and re-run this cell.")
else:                    print("VERDICT: switch to MODELS.classifier_fallback (legal-bert-small).")
if peak_gb > 13:         print("WARNING: near the 16 GB limit -- drop BATCH to 8, ACCUM to 4.")
"""),

    md("## 11. Reload a clean model\n\nThe probe left 100 steps of updates on the "
       "model. Reload so the real run starts from the pretrained weights."),
    code("""
del probe, model
torch.cuda.empty_cache()

model = AutoModelForSequenceClassification.from_pretrained(
    MODELS.classifier,
    num_labels=NUM_LABELS,
    id2label={i: c for i, c in enumerate(CATEGORIES)},
    label2id={c: i for i, c in enumerate(CATEGORIES)},
)
print("model reloaded from pretrained weights")
"""),

    md("## 12. Train\n\nCheckpoints to Drive every epoch, best model selected on "
       "validation macro-F1 (not accuracy — the 13x imbalance makes accuracy "
       "the wrong selection signal)."),
    code("""
args = TrainingArguments(
    output_dir=str(DRIVE_OUT / 'checkpoints'),
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=TRAIN.learning_rate,
    weight_decay=TRAIN.weight_decay,
    warmup_ratio=TRAIN.warmup_ratio,
    fp16=True,
    group_by_length=True,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    greater_is_better=True,
    logging_steps=50,
    report_to='none',
    seed=42,
)

trainer = WeightedTrainer(
    model=model, args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
    data_collator=collator, compute_metrics=compute_metrics,
)

t0 = time.perf_counter()
train_result = trainer.train()
train_minutes = (time.perf_counter() - t0) / 60
print(f"\\nACTUAL TRAINING TIME: {train_minutes:.1f} min")
"""),

    md("## 13. Evaluate on the held-out test split\n\nTest is touched exactly "
       "once, here. Validation was used for model selection."),
    code("""
import ast
import json
from src.classification import evaluate_predictions, save_confusion_matrix

t0 = time.perf_counter()
output = trainer.predict(tokenized['test'])
infer_seconds = time.perf_counter() - t0

preds  = np.argmax(output.predictions, axis=-1)
labels = output.label_ids

metrics = evaluate_predictions(
    labels, preds, CATEGORIES,
    model_name='Legal-BERT (fine-tuned)', split='test',
)
print(metrics.summary())
print(f"inference: {infer_seconds:.1f}s for {len(labels):,} "
      f"({1000*infer_seconds/len(labels):.2f} ms/clause on GPU)")

print(f"\\n{'category':<38}{'prec':>7}{'rec':>7}{'F1':>7}{'n':>8}")
for cat in sorted(metrics.per_class, key=lambda c: metrics.per_class[c]['f1']):
    r = metrics.per_class[cat]
    print(f"{cat:<38}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}{r['support']:>8,}")
"""),

    md("## 14. Save model and results\n\nModel to Drive (never to git — it is "
       "hundreds of MB). Metrics JSON is small and *should* be committed."),
    code("""
FINAL = DRIVE_OUT / 'final'
trainer.save_model(str(FINAL))
tokenizer.save_pretrained(str(FINAL))

payload = metrics.to_dict()
payload.update({
    'max_length': MAX_LENGTH,
    'batch_size': BATCH,
    'gradient_accumulation_steps': ACCUM,
    'effective_batch_size': BATCH * ACCUM,
    'epochs': EPOCHS,
    'learning_rate': TRAIN.learning_rate,
    'weight_scheme': WEIGHT_SCHEME,
    'n_train': len(tokenized['train']),
    'train_minutes': round(train_minutes, 1),
    'inference_seconds_test': round(infer_seconds, 2),
    'peak_vram_gb': round(peak_gb, 2),
    'parameters_millions': round(model.num_parameters()/1e6, 1),
    'base_model': MODELS.classifier,
})

(DRIVE_OUT / 'legal_bert_metrics.json').write_text(json.dumps(payload, indent=2))
save_confusion_matrix(metrics, DRIVE_OUT / 'confusion_legal_bert.png')

import shutil
shutil.make_archive('/content/legal-bert-clause-classifier', 'zip', FINAL)
print("model  ->", FINAL)
print("zip    -> /content/legal-bert-clause-classifier.zip")
print("metrics->", DRIVE_OUT / 'legal_bert_metrics.json')
print(f"model size: {sum(f.stat().st_size for f in FINAL.rglob('*') if f.is_file())/1e6:.0f} MB")
"""),

    md("""
## 15. Download

Two things come back to the laptop:

1. **`legal_bert_metrics.json`** → `evaluation/results/` → **commit this**
2. **`legal-bert-clause-classifier.zip`** → unzip into
   `models/legal-bert-clause-classifier/` → **never commit** (gitignored)

Then verify locally:

```
python -c "from src.classification import ClauseClassifier; c=ClauseClassifier('models/legal-bert-clause-classifier'); print(c.predict(['All work product shall be the sole and exclusive property of the Client.']))"
```
"""),
    code("""
from google.colab import files
files.download('/content/legal-bert-clause-classifier.zip')
files.download(str(DRIVE_OUT / 'legal_bert_metrics.json'))
files.download(str(DRIVE_OUT / 'confusion_legal_bert.png'))
"""),

    md("""
## 16. Comparison table for the report

Fill the Legal-BERT row from cell 13 and the TF-IDF row from
`evaluation/results/tfidf_svc_balanced.json`.

| Model | Accuracy | Macro-F1 | Weighted-F1 | Params | Train time | ms/clause |
|---|---|---|---|---|---|---|
| TF-IDF + LinearSVC (balanced) | | | | n/a | | |
| Legal-BERT (fine-tuned) | | | | 110M | | |

Report macro-F1 as the headline. State plainly that 100 LEDGAR labels were
collapsed to 15 categories, which makes the task easier than the published
100-class benchmark and means these numbers are **not** comparable to the
LexGLUE leaderboard.
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


def _strip_shell(source: str) -> str:
    """Remove IPython shell/magic lines so the rest can be parsed as Python."""
    kept = []
    for line in source.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            continue
        kept.append(line)
    return "\n".join(kept)


def validate(cells: list[dict]) -> list[str]:
    """Check every code cell parses, joining source EXACTLY as nbformat does.

    The join here must be ``"".join(...)``. Using ``"\\n".join(...)`` would
    re-insert the separators this function exists to verify are present, and
    would pass even on a notebook whose statements are all concatenated onto
    one line.
    """
    problems: list[str] = []
    for index, cell in enumerate(cells):
        source = "".join(cell["source"])

        # Every entry but the last must end with a newline, or the reader
        # will run consecutive statements together.
        for position, entry in enumerate(cell["source"][:-1]):
            if not entry.endswith("\n"):
                problems.append(
                    f"cell {index}: source entry {position} has no trailing "
                    f"newline: {entry[:60]!r}"
                )
                break

        if cell["cell_type"] != "code":
            continue
        try:
            ast.parse(_strip_shell(source))
        except SyntaxError as error:
            problems.append(f"cell {index}: SyntaxError line {error.lineno}: {error.msg}")
    return problems


def main() -> int:
    problems = validate(CELLS)
    if problems:
        print("REFUSING TO WRITE -- notebook is malformed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")

    # Re-validate what actually landed on disk, not just what we held in
    # memory, so a serialisation fault cannot slip through either.
    written = json.loads(OUT.read_text(encoding="utf-8"))
    problems = validate(written["cells"])
    if problems:
        print("Written notebook failed validation:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    code_cells = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"wrote {OUT}")
    print(f"{len(CELLS)} cells ({code_cells} code, {len(CELLS)-code_cells} markdown)")
    print("validation: all code cells parse; all source lines keep newlines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
