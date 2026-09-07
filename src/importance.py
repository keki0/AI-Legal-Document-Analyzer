"""Importance layer: calibrated confidence, salience, and evidence-linked flags.

Three outputs are kept deliberately separate, because they are different
kinds of claim:

===========  ====================================  =========================
Output       Source                                Epistemic status
===========  ====================================  =========================
category     Legal-BERT argmax                     learned from 20k examples
salience     calibrated confidence x prior         learned x design choice
flags        regex detectors, span-linked          rules, evidence-attached
===========  ====================================  =========================

There is no single "risk score" and no legal judgement anywhere. Flag wording
follows the project's legal-safety rule: every explanation is phrased as
"This clause may be significant because...", never as an assertion that
something is unfair, invalid or illegal.

**What separates the flags from the baseline's keyword rules** is not that
they stopped being rules -- they are still rules. It is that every flag
carries the exact span that triggered it and the sentence containing that
span, so a reader can check the machine's reasoning and disagree with it. The
baseline emitted "Intellectual Property: HIGH" with no evidence and no
recourse.

**On the flag taxonomy.** Eight flag concepts are taken from the UNFAIR-ToS
label set of LexGLUE (Lippi et al., "CLAUDETTE: an Automated Detector of
Potentially Unfair Clauses in Online Terms of Service"). We use the published
concepts for provenance; we do NOT use the dataset, and these detectors are
regex rules written by us, not expert annotations. Three further flags cover
the contract domain our documents actually contain.

torch is never imported here. Temperature fitting is a one-dimensional
optimisation done in NumPy/SciPy, so the whole module runs on CPU without the
deep-learning stack.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import yaml

from src.config import PATHS

if TYPE_CHECKING:  # pragma: no cover
    from src.document_processing import Clause

logger = logging.getLogger(__name__)

__all__ = [
    "split_sentences",
    "TemperatureScaler",
    "CalibrationReport",
    "compute_ece",
    "confidence_bins",
    "SalienceScorer",
    "FlagHit",
    "FlagDetector",
    "ClauseAnalysis",
    "analyze_clauses",
    "top_k_salient",
]


# ===========================================================================
# Numerical helpers
# ===========================================================================

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted for numerical stability."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean negative log-likelihood at a given temperature.

    Computed via log-sum-exp rather than log(softmax(...)) so that confident
    logits divided by a small temperature do not underflow to log(0).
    """
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    log_partition = np.log(np.exp(shifted).sum(axis=-1)) + scaled.max(axis=-1)
    chosen = scaled[np.arange(len(labels)), labels]
    return float(-(chosen - log_partition).mean())


# ===========================================================================
# Temperature scaling
# ===========================================================================

@dataclass
class CalibrationReport:
    """Calibration measurements before and after temperature scaling."""

    temperature: float
    n_examples: int
    n_classes: int
    accuracy_before: float
    accuracy_after: float
    ece_before: float
    ece_after: float
    mce_before: float
    mce_after: float
    nll_before: float
    nll_after: float
    mean_confidence_before: float
    mean_confidence_after: float
    bins_before: list[dict] = field(default_factory=list)
    bins_after: list[dict] = field(default_factory=list)

    @property
    def ece_improved(self) -> bool:
        return self.ece_after < self.ece_before

    def summary(self) -> str:
        direction = "improved" if self.ece_improved else "did NOT improve"
        return (
            f"T = {self.temperature:.4f} | "
            f"ECE {self.ece_before:.4f} -> {self.ece_after:.4f} ({direction}) | "
            f"accuracy {self.accuracy_before:.4f} -> {self.accuracy_after:.4f}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Saved calibration report to %s", path)


class TemperatureScaler:
    """Single-parameter confidence calibration (Guo et al., 2017).

    Divides logits by a learned scalar T before the softmax. T is fitted by
    minimising negative log-likelihood on a held-out split -- here the
    validation split, never the test split, which would leak.

    **Temperature scaling cannot change which class is predicted.** Dividing
    every logit by the same positive constant is monotonic, so the argmax is
    invariant and accuracy is mathematically unchanged. Any accuracy
    difference in a report is floating-point noise or a tie-break, not an
    effect. This is asserted by a test.

    T > 1 means the model was overconfident and probabilities are softened.
    T < 1 means it was underconfident and they are sharpened.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)
        self.fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        bounds: tuple[float, float] = (0.05, 10.0),
    ) -> "TemperatureScaler":
        """Fit T by minimising NLL. Returns self."""
        from scipy.optimize import minimize_scalar

        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        if logits.ndim != 2:
            raise ValueError(f"logits must be 2-D, got shape {logits.shape}")
        if len(logits) != len(labels):
            raise ValueError(
                f"logits/labels length mismatch: {len(logits)} vs {len(labels)}"
            )
        if labels.min() < 0 or labels.max() >= logits.shape[1]:
            raise ValueError(
                f"labels out of range for {logits.shape[1]} classes: "
                f"[{labels.min()}, {labels.max()}]"
            )

        result = minimize_scalar(
            lambda t: _nll(logits, labels, t), bounds=bounds, method="bounded"
        )
        self.temperature = float(result.x)
        self.fitted = True

        logger.info(
            "Fitted temperature T=%.4f (NLL %.4f -> %.4f) on %d examples",
            self.temperature,
            _nll(logits, labels, 1.0),
            _nll(logits, labels, self.temperature),
            len(labels),
        )
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature and return probabilities."""
        return _softmax(np.asarray(logits, dtype=np.float64) / self.temperature)

    def calibrate_probabilities(self, probabilities: Sequence[float]) -> np.ndarray:
        """Re-calibrate an existing probability vector.

        ``ClauseClassifier.predict`` returns probabilities, not logits. Since
        softmax is invertible up to an additive constant, log(p) recovers the
        logits to that constant, and softmax is invariant to it -- so this is
        exact, not an approximation.
        """
        probabilities = np.asarray(probabilities, dtype=np.float64)
        recovered = np.log(np.clip(probabilities, 1e-12, None))
        return self.transform(recovered.reshape(1, -1) if recovered.ndim == 1 else recovered)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"temperature": self.temperature, "fitted": self.fitted}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        scaler = cls(payload["temperature"])
        scaler.fitted = payload.get("fitted", True)
        return scaler


# ===========================================================================
# Calibration metrics
# ===========================================================================

def confidence_bins(
    probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 15
) -> list[dict]:
    """Bin predictions by confidence and report accuracy within each bin.

    This is the table that answers "does confidence track correctness?".
    A well-calibrated model has accuracy approximately equal to mean
    confidence in every bin.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        # Lower-exclusive except for the first bin, so every point lands once.
        mask = (confidence > lower) & (confidence <= upper)
        if lower == 0.0:
            mask |= confidence == 0.0
        count = int(mask.sum())
        bins.append({
            "lower": float(lower),
            "upper": float(upper),
            "count": count,
            "accuracy": float(correct[mask].mean()) if count else 0.0,
            "confidence": float(confidence[mask].mean()) if count else 0.0,
            "gap": float(confidence[mask].mean() - correct[mask].mean()) if count else 0.0,
        })
    return bins


def compute_ece(
    probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 15
) -> tuple[float, float]:
    """Expected and Maximum Calibration Error.

    ECE is the count-weighted mean absolute gap between confidence and
    accuracy across bins; MCE is the largest such gap. Returns ``(ece, mce)``.
    """
    bins = confidence_bins(probabilities, labels, n_bins=n_bins)
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0, 0.0

    ece = sum(b["count"] / total * abs(b["gap"]) for b in bins)
    mce = max((abs(b["gap"]) for b in bins if b["count"] > 0), default=0.0)
    return float(ece), float(mce)


def evaluate_calibration(
    scaler: TemperatureScaler,
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> CalibrationReport:
    """Measure calibration before and after applying a fitted temperature."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)

    before = _softmax(logits)
    after = scaler.transform(logits)

    ece_before, mce_before = compute_ece(before, labels, n_bins=n_bins)
    ece_after, mce_after = compute_ece(after, labels, n_bins=n_bins)

    return CalibrationReport(
        temperature=scaler.temperature,
        n_examples=len(labels),
        n_classes=logits.shape[1],
        accuracy_before=float((before.argmax(1) == labels).mean()),
        accuracy_after=float((after.argmax(1) == labels).mean()),
        ece_before=ece_before,
        ece_after=ece_after,
        mce_before=mce_before,
        mce_after=mce_after,
        nll_before=_nll(logits, labels, 1.0),
        nll_after=_nll(logits, labels, scaler.temperature),
        mean_confidence_before=float(before.max(axis=1).mean()),
        mean_confidence_after=float(after.max(axis=1).mean()),
        bins_before=confidence_bins(before, labels, n_bins=n_bins),
        bins_after=confidence_bins(after, labels, n_bins=n_bins),
    )


# ===========================================================================
# Salience
# ===========================================================================

class SalienceScorer:
    """Salience = calibrated confidence x hand-assigned category prior.

    The priors live in ``data/eval/salience_priors.yaml`` and are DESIGN
    CHOICES, not learned values. See that file's header before quoting any
    number derived from them.

    The output is called a salience score or attention priority. It is not a
    risk score and carries no legal judgement.
    """

    def __init__(
        self,
        priors: dict[str, float],
        *,
        default_prior: float = 0.5,
        learned: bool = False,
    ) -> None:
        self.priors = dict(priors)
        self.default_prior = float(default_prior)
        # Carried through to any report so provenance cannot be lost.
        self.learned = learned

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "SalienceScorer":
        path = path or (PATHS.data_eval / "salience_priors.yaml")
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if payload.get("learned", False):
            raise ValueError(
                "salience_priors.yaml declares learned: true, but these priors "
                "are hand-assigned. Refusing to misrepresent their provenance."
            )
        return cls(
            payload["priors"],
            default_prior=payload.get("default_prior", 0.5),
            learned=False,
        )

    def prior_for(self, category: str) -> float:
        if category not in self.priors:
            logger.warning(
                "No prior for category %r; using default %.2f",
                category, self.default_prior,
            )
        return self.priors.get(category, self.default_prior)

    def score(self, category: str, calibrated_confidence: float) -> float:
        """Return a salience score in [0, 1]."""
        if not 0.0 <= calibrated_confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0,1], got {calibrated_confidence}"
            )
        return float(calibrated_confidence * self.prior_for(category))


# ===========================================================================
# Evidence-linked flags
# ===========================================================================

@dataclass
class FlagHit:
    """One triggered flag, with the evidence that caused it.

    ``trigger_span`` is the literal matched text and ``sentence`` is the
    sentence containing it. Both are preserved so the UI can highlight the
    evidence and the reader can disagree with the machine.
    """

    flag_id: str
    label: str
    explanation: str
    trigger_span: str
    span_start: int
    span_end: int
    sentence: str
    source: str  # "unfair-tos-inspired" or "contract-domain"

    def to_dict(self) -> dict:
        return asdict(self)


# Each entry: (flag_id, label, source, why-it-may-matter, [patterns])
#
# Wording rule: every explanation begins "This clause may be significant
# because" and describes what the text DOES. None asserts unfairness,
# illegality or invalidity.
_FLAG_DEFINITIONS: list[tuple[str, str, str, str, list[str]]] = [
    (
        "limitation_of_liability", "Limitation of liability", "unfair-tos-inspired",
        "it may limit or exclude what one party can be held responsible for.",
        [r"\b(?:in\s+no\s+event|under\s+no\s+circumstances)\b[^.]{0,120}\bliab",
         r"\blimitation\s+of\s+liability\b",
         r"\b(?:shall\s+not|will\s+not)\s+be\s+liable\b",
         r"\bdisclaims?\s+(?:any|all)\s+(?:liability|warrant)",
         r"\b(?:exclude|excludes|excluding)\s+(?:any|all)\s+liability\b"],
    ),
    (
        "unilateral_termination", "Unilateral termination", "unfair-tos-inspired",
        "it may allow one party to end the agreement on its own initiative.",
        [r"\b(?:may|can|shall\s+have\s+the\s+right\s+to)\s+terminate\b"
         r"[^.]{0,80}\b(?:at\s+any\s+time|immediately|without\s+(?:notice|cause)|"
         r"in\s+its\s+sole\s+discretion)\b",
         r"\bterminate\s+immediately\b",
         r"\b(?:we|the\s+\w+)\s+may\s+(?:suspend|terminate)\s+your\s+account\b"],
    ),
    (
        "unilateral_change", "Unilateral change", "unfair-tos-inspired",
        "it may allow one party to change the terms without the other's agreement.",
        [r"\b(?:may|reserve[sd]?\s+the\s+right\s+to)\s+(?:modify|change|amend|update|revise)\b"
         r"[^.]{0,100}\b(?:at\s+any\s+time|without\s+notice|in\s+(?:our|its)\s+sole\s+discretion)\b",
         r"\breserve[sd]?\s+the\s+right\s+to\s+(?:modify|change|amend|update)\b"],
    ),
    (
        "content_removal", "Content removal", "unfair-tos-inspired",
        "it may allow content to be removed or access withdrawn.",
        [r"\b(?:may|reserve[sd]?\s+the\s+right\s+to)\s+(?:remove|delete|disable|take\s+down)\b"
         r"[^.]{0,100}\b(?:content|material|post|submission)",
         r"\bremove\s+(?:any|all)\s+content\b[^.]{0,60}\b(?:without\s+notice|sole\s+discretion)\b"],
    ),
    (
        "contract_by_using", "Contract by using", "unfair-tos-inspired",
        "it may treat use of the service as acceptance of the terms.",
        [r"\bby\s+(?:using|accessing|continuing\s+to\s+use|visiting)\b[^.]{0,120}"
         r"\b(?:you\s+agree|constitutes?\s+(?:your\s+)?acceptance|you\s+accept)\b",
         r"\bcontinued\s+use\b[^.]{0,80}\bconstitutes?\s+acceptance\b"],
    ),
    (
        "choice_of_law", "Choice of law", "unfair-tos-inspired",
        "it specifies which jurisdiction's law governs the agreement.",
        [r"\bgoverned\s+(?:and\s+construed\s+)?by\s+the\s+laws?\s+of\b",
         r"\bchoice\s+of\s+law\b",
         r"\bgoverning\s+law\b"],
    ),
    (
        "jurisdiction", "Jurisdiction", "unfair-tos-inspired",
        "it specifies where a dispute must be brought, which may require "
        "travelling to another location.",
        [r"\b(?:exclusive|sole)\s+jurisdiction\b",
         r"\bsubmit\s+to\s+the\s+jurisdiction\b",
         r"\b(?:courts?|venue)\s+(?:located\s+)?in\s+(?:the\s+)?(?:State|County|City|District)\s+of\b",
         r"\bconsent\s+to\s+(?:the\s+)?(?:personal\s+)?jurisdiction\b"],
    ),
    (
        "arbitration", "Arbitration", "unfair-tos-inspired",
        "it may require disputes to be resolved outside the courts, which can "
        "affect the remedies available.",
        [r"\b(?:binding\s+)?arbitration\b",
         r"\barbitrat(?:ed|or|ion)\b",
         r"\b(?:waive|waiver\s+of)\b[^.]{0,80}\bclass\s+action\b",
         r"\bclass\s+action\s+waiver\b"],
    ),
    (
        "broad_ip_assignment", "Broad IP assignment", "contract-domain",
        "it may transfer ownership of work or intellectual property to the "
        "other party.",
        [r"\bassigns?\s+(?:all|any)\s+(?:right|rights)\b[^.]{0,80}\b(?:title|interest)\b",
         r"\bsole\s+and\s+exclusive\s+property\b",
         r"\ball\s+(?:work\s+product|deliverables|materials)\b[^.]{0,100}"
         r"\b(?:shall\s+be|are|is)\s+the\s+(?:sole\s+)?(?:and\s+exclusive\s+)?property\b",
         r"\bwork\s+made\s+for\s+hire\b",
         r"\bretains?\s+no\s+rights\b"],
    ),
    (
        "broad_indemnity", "Broad indemnity", "contract-domain",
        "it may require one party to cover the other's losses or legal costs.",
        [r"\bindemnify\b[^.]{0,60}\b(?:and\s+)?hold\s+harmless\b",
         r"\bhold\s+harmless\b",
         r"\bindemnif(?:y|ies|ication)\b[^.]{0,100}\b(?:any|all)\s+"
         r"(?:claims?|losses|liabilit)",
         r"\bdefend,?\s+indemnify\b"],
    ),
    (
        "automatic_renewal", "Automatic renewal", "contract-domain",
        "it may renew the agreement automatically unless cancelled in time.",
        [r"\bautomatically\s+(?:renew|extend)",
         r"\bauto(?:matic)?[\s-]?renew(?:al|s|ed|ing)?\b",
         r"\brenew(?:s|ed)?\s+automatically\b",
         r"\bunless\s+(?:you\s+)?cancel(?:led)?\b[^.]{0,80}\b(?:before|prior\s+to)\b"
         r"[^.]{0,60}\brenewal\b",
         r"\bsuccessive\s+(?:renewal\s+)?(?:terms?|periods?)\b"],
    ),
]

# Sentence splitter. Deliberately simple regex rather than spaCy: this keeps
# the importance layer free of a model download, and legal clauses are
# already short. Common legal abbreviations are protected from false splits.
_ABBREVIATIONS = r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bNo)(?<!\bInc)(?<!\bLtd)(?<!\bCo)"
_SENTENCE_SPLIT = re.compile(rf"{_ABBREVIATIONS}(?<=[.;])\s+(?=[A-Z(\[])")


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, sentence)`` triples with offsets into ``text``."""
    if not text.strip():
        return []
    spans: list[tuple[int, int, str]] = []
    position = 0
    for piece in _SENTENCE_SPLIT.split(text):
        if not piece:
            continue
        start = text.find(piece, position)
        if start == -1:
            start = position
        end = start + len(piece)
        spans.append((start, end, piece.strip()))
        position = end
    return spans or [(0, len(text), text.strip())]


class FlagDetector:
    """Regex detectors that attach their triggering evidence.

    These are RULES, written by the project authors. They are not expert
    annotations and carry no ground-truth status. The UNFAIR-ToS-derived
    flags borrow published *concepts* only; the patterns are ours.
    """

    def __init__(self, definitions=None) -> None:
        self._definitions = definitions or _FLAG_DEFINITIONS
        self._compiled = [
            (
                flag_id,
                label,
                source,
                reason,
                [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns],
            )
            for flag_id, label, source, reason, patterns in self._definitions
        ]

    @property
    def flag_ids(self) -> list[str]:
        return [definition[0] for definition in self._definitions]

    def detect(self, text: str) -> list[FlagHit]:
        """Return at most one hit per flag, using the first match found.

        One hit per flag keeps the UI readable: three phrasings of the same
        indemnity in one clause is still one thing to look at.
        """
        if not text or not text.strip():
            return []

        sentences = split_sentences(text)
        hits: list[FlagHit] = []

        for flag_id, label, source, reason, patterns in self._compiled:
            for pattern in patterns:
                match = pattern.search(text)
                if not match:
                    continue
                start, end = match.span()
                sentence = next(
                    (s for s_start, s_end, s in sentences if s_start <= start < s_end),
                    text.strip(),
                )
                hits.append(FlagHit(
                    flag_id=flag_id,
                    label=label,
                    explanation=f"This clause may be significant because {reason}",
                    trigger_span=text[start:end].strip(),
                    span_start=start,
                    span_end=end,
                    sentence=sentence,
                    source=source,
                ))
                break  # first pattern that fires wins
        return hits


# ===========================================================================
# Combined analysis
# ===========================================================================

@dataclass
class ClauseAnalysis:
    """A clause with its category, confidence, salience and flags.

    This is the object Phases 5-9 consume. ``probabilities`` is retained
    because downstream components may need the full distribution, and
    ``raw_confidence`` is kept alongside ``calibrated_confidence`` so the two
    are never confused in a report.
    """

    clause_id: int
    title: str
    text: str
    page: int
    category: str
    raw_confidence: float
    calibrated_confidence: float
    salience: float
    category_prior: float
    flags: list[FlagHit] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)
    # Populated by src.extraction.attach_extractions (Phase 5). Kept here
    # rather than in a parallel structure so the interface has one object per
    # clause. Typed loosely to avoid an import cycle: extraction imports the
    # sentence splitter from this module.
    extractions: list = field(default_factory=list)

    @property
    def flag_labels(self) -> list[str]:
        return [hit.label for hit in self.flags]

    @property
    def has_flags(self) -> bool:
        return bool(self.flags)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["flag_labels"] = self.flag_labels
        return payload


def analyze_clauses(
    clauses: Sequence["Clause"],
    predictions: Sequence[dict],
    *,
    scaler: TemperatureScaler | None = None,
    scorer: SalienceScorer | None = None,
    detector: FlagDetector | None = None,
) -> list[ClauseAnalysis]:
    """Combine classification, calibration, salience and flags.

    Args:
        clauses: Output of ``src.document_processing.segment_clauses``.
        predictions: Output of ``ClauseClassifier.predict`` -- one dict per
            clause with ``category``, ``confidence`` and ``probabilities``.
        scaler: Fitted temperature scaler. When None, calibrated confidence
            equals raw confidence and this is recorded honestly rather than
            silently.
        scorer: Salience scorer. Defaults to the YAML priors.
        detector: Flag detector. Defaults to the built-in rules.

    Returns:
        One :class:`ClauseAnalysis` per clause, in document order.
    """
    if len(clauses) != len(predictions):
        raise ValueError(
            f"clause/prediction count mismatch: {len(clauses)} vs {len(predictions)}"
        )

    scorer = scorer or SalienceScorer.from_yaml()
    detector = detector or FlagDetector()
    if scaler is None:
        logger.warning(
            "No temperature scaler supplied; calibrated_confidence will equal "
            "raw_confidence. Report it as uncalibrated."
        )

    analyses: list[ClauseAnalysis] = []
    for clause, prediction in zip(clauses, predictions):
        category = prediction["category"]
        raw_confidence = float(prediction["confidence"])
        probabilities = prediction.get("probabilities", {})

        if scaler is not None and probabilities:
            names = list(probabilities)
            vector = np.array([probabilities[n] for n in names], dtype=np.float64)
            calibrated_vector = scaler.calibrate_probabilities(vector)[0]
            calibrated = dict(zip(names, calibrated_vector.tolist()))
            calibrated_confidence = float(calibrated[category])
        else:
            calibrated = dict(probabilities)
            calibrated_confidence = raw_confidence

        analyses.append(ClauseAnalysis(
            clause_id=clause.clause_id,
            title=clause.title,
            text=clause.text,
            page=clause.page_start,
            category=category,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            salience=scorer.score(category, calibrated_confidence),
            category_prior=scorer.prior_for(category),
            flags=detector.detect(clause.text),
            probabilities=calibrated,
        ))
    return analyses


def top_k_salient(
    analyses: Iterable[ClauseAnalysis], k: int = 5
) -> list[ClauseAnalysis]:
    """Highest-salience clauses first.

    Used later to choose which clauses are fed to the generator, since
    FLAN-T5's 512-token input cannot take a whole contract.
    """
    return sorted(analyses, key=lambda a: a.salience, reverse=True)[:k]


# Backwards-compatible private alias. src.extraction imports the public name.
_split_sentences = split_sentences
