"""Telemetry: JSONL writes, redaction, never-raise, histogram."""

import json

from conftest import read_last_log_record

from sdm1.telemetry import confidence_histogram, log_record
from sdm1.types import Prediction


def test_log_record_writes_jsonl_with_ts(tmp_path):
    path = log_record({"op": "classify", "task": "x"})
    assert path is not None and path.exists()
    rec = json.loads(path.read_text().strip().splitlines()[-1])
    assert rec["op"] == "classify" and "ts" in rec


def test_dataset_shaped_keys_are_redacted(tmp_path):
    rec = log_record({"op": "classify", "rows": [{"secret_cell": 1}], "cells": "x", "values": "y"})
    assert rec is not None
    saved = json.loads(rec.read_text().strip().splitlines()[-1])
    assert saved["rows"] == "[redacted]" and saved["cells"] == "[redacted]" and saved["values"] == "[redacted]"


def test_log_failure_swallowed(tmp_path, monkeypatch):
    # Point SDM1_LOG_DIR at a *file* so mkdir explodes; log_record must return None.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    monkeypatch.setenv("SDM1_LOG_DIR", str(blocker))
    assert log_record({"op": "classify"}) is None


def test_histogram_bins_and_skips_none():
    preds = [
        Prediction(row_index=0, label="a", confidence=0.95),
        Prediction(row_index=1, label="a", confidence=0.05),
        Prediction(row_index=2, label="a", confidence=None),
        Prediction(row_index=3, label="a", confidence=0.55),
    ]
    hist = confidence_histogram(preds, bins=10)
    assert hist["all"][9] == 1 and hist["all"][0] == 1 and hist["all"][5] == 1
    assert sum(hist["all"]) == 3


def test_log_record_via_api_carries_no_dataset_cells(tmp_path, rows):
    from sdm1 import classify
    from sdm1.registry import register
    from sdm1.types import Prediction, Provider

    register(
        Provider(id="fake", name="f", run=lambda c, q, t, tel: [
            Prediction(row_index=i, label="yes", confidence=0.9) for i in range(len(q))
        ], is_available=lambda: True)
    )
    result = classify(rows, target="ok", model="fake", table_name="t")
    assert result.verdict == "pass"
    saved = read_last_log_record(tmp_path)
    blob = json.dumps(saved)
    assert "1111111" not in blob and "2222222" not in blob  # distinctive cell values never logged
