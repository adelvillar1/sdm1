"""
Normalize backend worker output into Prediction wire shapes.

The worker protocol is ours, but the worker itself wraps third-party model
outputs whose shapes vary (classification ensembles, regression values,
different class-name sources). Parsers stay tolerant: several key aliases,
list-form entries, and derived confidence when the worker omits one.

Parsing never raises — malformed output degrades to structured_ok=False with
an error snippet, which the API turns into an error verdict.
"""

from __future__ import annotations

import json
from typing import Any

from .types import Prediction

_PROB_KEYS = ("probabilities", "probs", "probs_by_class", "class_probabilities")
_LABEL_KEYS = ("label", "prediction", "pred", "value", "y_pred")
_CONF_KEYS = ("confidence", "conf", "score")


def parse_worker_output(raw: str) -> tuple[list[Prediction], bool, str | None]:
    """Parse worker stdout. Returns (predictions, structured_ok, error).

    `structured_ok` reflects whether we got a well-formed JSON envelope with
    predictions — mirroring sys1's structured_ok telemetry key so dashboards
    can track parse failures the same way.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        snippet = raw[:200].replace("\n", " ") if raw else "(empty)"
        return [], False, f"worker output is not JSON: {snippet}"
    if not isinstance(payload, dict):
        return [], False, "worker output is not a JSON object"

    if payload.get("ok") is False:
        return [], False, str(payload.get("error") or "worker reported failure")

    entries = payload.get("predictions")
    if entries is None:
        return [], False, "worker output missing 'predictions' key"
    if not isinstance(entries, list):
        return [], False, "'predictions' is not a list"

    predictions: list[Prediction] = []
    for i, entry in enumerate(entries):
        pred = _normalize_entry(entry, default_row=i)
        if pred is not None:
            predictions.append(pred)
    return predictions, True, None


def _normalize_entry(entry: Any, *, default_row: int) -> Prediction | None:
    """One prediction entry; tolerate dict and [row_index, label, probs] forms."""
    row = default_row
    label: Any = None
    probs: dict[str, float] | None = None
    conf: float | None = None

    if isinstance(entry, list):
        if len(entry) >= 2:
            row, label = entry[0], entry[1]
        if len(entry) >= 3 and isinstance(entry[2], dict):
            probs = entry[2]
    elif isinstance(entry, dict):
        row = entry.get("row_index", entry.get("index", default_row))
        for key in _LABEL_KEYS:
            if key in entry and entry[key] is not None:
                label = entry[key]
                break
        for key in _PROB_KEYS:
            if isinstance(entry.get(key), dict):
                probs = entry[key]
                break
        for key in _CONF_KEYS:
            if entry.get(key) is not None:
                conf = _as_float(entry[key])
                break
    else:
        return None

    if probs is not None:
        probs = {str(k): _as_float(v) for k, v in probs.items() if _as_float(v) is not None}
        if not probs:
            probs = None
    if label is not None and not isinstance(label, (str, int, float, bool)):
        label = str(label)
    if conf is None and probs:
        conf = max(probs.values())  # ensemble mean max-probability for classification
    try:
        row = int(row)
    except (TypeError, ValueError):
        row = default_row
    return Prediction(row_index=row, label=label, probabilities=probs, confidence=conf)


def _as_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN guard
