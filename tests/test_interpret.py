"""Interpretation: per-backend floors, precedence, escalation semantics."""


from sdm1.interpret import _DEFAULT_FLOORS, effective_floor, interpret
from sdm1.types import Prediction, TableTask

TASK = TableTask(id="ok:classification", target="ok")


def _pred(idx, label="yes", conf=0.9):
    return Prediction(row_index=idx, label=label, probabilities={"yes": conf, "no": 1 - conf}, confidence=conf)


def test_nvidia_sdm_has_its_own_floor_entry():
    # Per-model-class semantics: floors are keyed per backend id, never shared.
    assert "nvidia-sdm" in _DEFAULT_FLOORS


def test_floor_precedence_override_beats_task_beats_default():
    task = TableTask(id="t", target="ok", confidence_floor=0.75)
    assert effective_floor("nvidia-sdm", task, floor_override=0.9) == 0.9
    assert effective_floor("nvidia-sdm", task) == 0.75
    assert effective_floor("nvidia-sdm", None) == _DEFAULT_FLOORS["nvidia-sdm"]


def test_unknown_backend_gets_conservative_default():
    assert effective_floor("future-fit-predict-fm", None) == 0.6


def test_all_rows_clear_floor():
    verdict, escalated, low, _ = interpret([_pred(0), _pred(1)], "nvidia-sdm", TASK)
    assert verdict == "pass" and not escalated and low == []


def test_row_below_floor_escalates_with_index():
    preds = [_pred(0, conf=0.9), _pred(1, conf=0.3)]
    verdict, escalated, low, summary = interpret(preds, "nvidia-sdm", TASK)
    assert verdict == "escalate" and escalated and low == [1]
    assert any("0.30" in line for line in summary)


def test_missing_label_escalates_even_at_high_confidence():
    preds = [_pred(0, label=None, conf=0.95)]
    verdict, _, low, _ = interpret(preds, "nvidia-sdm", TASK)
    assert verdict == "escalate" and low == [0]


def test_regression_rows_without_confidence_never_gate():
    preds = [Prediction(row_index=i, label=3.14, probabilities=None, confidence=None) for i in range(3)]
    verdict, _, low, _ = interpret(preds, "nvidia-sdm", TASK)
    assert verdict == "pass" and low == []


def test_task_floor_override_via_request_level():
    preds = [_pred(0, conf=0.7)]
    task = TableTask(id="t", target="ok", confidence_floor=0.8)
    verdict, _, low, _ = interpret(preds, "nvidia-sdm", task, floor_override=0.65)
    assert verdict == "pass" and low == []
    verdict, _, low, _ = interpret(preds, "nvidia-sdm", task)
    assert verdict == "escalate" and low == [0]
