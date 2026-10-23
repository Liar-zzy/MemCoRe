"""CPU-only primary QA metrics, with explicit v1 F1 tokenization versions.

No model-based metrics or resource downloads are needed. NLTK data must already
be installed locally for BLEU-1.
"""

import re
from typing import Dict

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


HISTORICAL_V1 = "historical_v1"
NORMALIZED_V1 = "normalized_v1"
F1_VERSIONS = (HISTORICAL_V1, NORMALIZED_V1)


def _historical_tokens(text: str) -> set[str]:
    # Historical archive behavior: quotes and articles remain tokens.
    return set(text.lower().replace(".", " ").replace(",", " ")
               .replace("!", " ").replace("?", " ").split())


def _normalized_tokens(text: str) -> set[str]:
    # Preserve the alternative normalizer's punctuation/article removal.
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return set(text.split()) if text else set()


def _set_f1(prediction: str, reference: str, version: str) -> float:
    tokenize = _historical_tokens if version == HISTORICAL_V1 else _normalized_tokens
    predicted, expected = tokenize(prediction), tokenize(reference)
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _bleu1(prediction: str, reference: str) -> float:
    try:
        predicted = word_tokenize(prediction.lower())
        expected = word_tokenize(reference.lower())
    except LookupError as exc:
        raise RuntimeError(
            "BLEU-1 requires local NLTK punkt tokenization data "
            "(punkt and, on newer NLTK versions, punkt_tab). "
            "Install those resources in the chosen environment before scoring; "
            "core_metrics never downloads them."
        ) from exc
    if not predicted or not expected:
        return 0.0
    return float(sentence_bleu(
        [expected], predicted, weights=(1, 0, 0, 0),
        smoothing_function=SmoothingFunction().method1,
    ))


def calculate_core_metrics(
    prediction: str, reference: str, f1_version: str = HISTORICAL_V1
) -> Dict[str, float]:
    """Return exact match, set F1, and historical NLTK BLEU-1.

    Empty answers score zero, including empty/empty, matching legacy v1.
    ``historical_v1`` reproduces archived F1. ``normalized_v1`` uses the
    current v1 normalizer's punctuation/article removal. BLEU-1 is identical
    across F1 versions and never uses a model or network request.
    """
    if f1_version not in F1_VERSIONS:
        raise ValueError(f"Unknown F1 version {f1_version!r}; choose one of {F1_VERSIONS}")
    if not prediction or not reference:
        return {"exact_match": 0, "f1": 0.0, "bleu1": 0.0}
    prediction, reference = str(prediction).strip(), str(reference).strip()
    if not prediction or not reference:
        return {"exact_match": 0, "f1": 0.0, "bleu1": 0.0}
    return {
        "exact_match": int(prediction.lower() == reference.lower()),
        "f1": _set_f1(prediction, reference, f1_version),
        "bleu1": _bleu1(prediction, reference),
    }
