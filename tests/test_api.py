"""Public API: composition, fail-open provider handling, contract shape."""

import hashlib
import json

import pytest
from conftest import CONTRACT_KEYS, read_last_log_record

from sdm1 import classify
from sdm1.types import Prediction, TableTask, WorkerError


def test_classify_success_shape(rows, fake_provider):
    result = classify(rows, target="ok", model="fake", table_name="t")
    assert list(result.as_dict().keys()) == CONTRACT_KEYS
    assert result.verdict == "pass"
    assert result.models_used == ["fake"]
    assert result.task == "ok:classification"
    assert result.n_context_rows == 3 and result.n_query_rows == 2
    assert result.telemetry["fake"] is True
    assert "confidence_histogram" in result.telemetry
    assert json.dumps(result.as_dict())  # serializable


def test_input_sha256_hashes_query_rows(rows, fake_provider):
    result = classify(rows, target="ok", model="fake")
    query_rows = [r for r in rows if r.get("ok") is None]
    expected = hashlib.sha256(json.dumps(query_rows, sort_keys=True, default=str).encode()).hexdigest()
    assert result.input_sha256 == expected


def test_low_confidence_rows_escalate(rows, fake_provider):
    fake_provider["predictions"] = [
        Prediction(row_index=0, label="yes", confidence=0.9),
        Prediction(row_index=1, label="no", confidence=0.3),
    ]
    result = classify(rows, target="ok", model="fake")
    assert result.verdict == "escalate"
    assert result.low_confidence_rows == [1]


def test_request_floor_override_escalates_everything(rows, fake_provider):
    result = classify(rows, target="ok", model="fake", confidence_floor=0.95)
    assert result.verdict == "escalate" and result.low_confidence_rows == [0, 1]


def test_task_instance_floor_is_respected(rows, fake_provider):
    task = TableTask(id="custom", target="ok", confidence_floor=0.95)
    result = classify(rows, task=task, model="fake")
    assert result.task == "custom" and result.verdict == "escalate"
    assert result.telemetry["confidence_floor"] == 0.95


def test_provider_failure_fails_open_as_error(rows, fake_provider):
    fake_provider["error"] = WorkerError("boom")
    result = classify(rows, target="ok", model="fake")
    assert result.verdict == "error"
    assert result.escalated is True
    assert result.models_used == [] and result.predictions == []
    assert "boom" in (result.error or "")


def test_zero_predictions_is_an_error(rows, fake_provider):
    fake_provider["predictions"] = []
    result = classify(rows, target="ok", model="fake")
    assert result.verdict == "error" and "zero predictions" in (result.reason or "")


def test_missing_target_raises():
    with pytest.raises(ValueError, match="target"):
        classify([{"a": 1}], model="fake")


def test_all_rows_labeled_raises(rows):
    labeled = [{**r, "ok": "yes"} for r in rows]
    with pytest.raises(ValueError, match="no query rows"):
        classify(labeled, target="ok", model="fake")


def test_no_labeled_rows_raises(rows):
    unlabeled = [{**r, "ok": None} for r in rows]
    with pytest.raises(ValueError, match="no context rows"):
        classify(unlabeled, target="ok", model="fake")


def test_log_record_written_and_free_of_dataset_cells(tmp_path, rows, fake_provider):
    classify(rows, target="ok", model="fake", table_name="t")
    rec = read_last_log_record(tmp_path)
    assert rec["op"] == "classify"
    blob = json.dumps(rec)
    assert "1111111" not in blob and "2222222" not in blob


def test_log_suppressed_with_log_false(tmp_path, fake_provider, rows):
    result = classify(rows, target="ok", model="fake", log=False)
    assert result.verdict == "pass"
    logs = list((tmp_path / "logs").rglob("events.jsonl"))
    assert logs == []
