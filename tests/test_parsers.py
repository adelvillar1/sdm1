"""Worker-output parsing: tolerant shapes, NaN guards, structured_ok."""

from sdm1.parsers import parse_worker_output


def test_happy_path():
    raw = '{"ok": true, "predictions": [{"row_index": 0, "label": "yes", "probabilities": {"yes": 0.9, "no": 0.1}, "confidence": 0.9}]}'
    preds, ok, err = parse_worker_output(raw)
    assert ok and err is None
    assert preds[0].label == "yes" and preds[0].confidence == 0.9


def test_prob_key_aliases():
    raw = '{"predictions": [{"row_index": 0, "label": "a", "probs": {"a": 0.7, "b": 0.3}}]}'
    preds, ok, _ = parse_worker_output(raw)
    assert ok and preds[0].probabilities["a"] == 0.7


def test_confidence_derived_from_probs_when_missing():
    raw = '{"predictions": [{"row_index": 0, "label": "a", "probabilities": {"a": 0.7, "b": 0.3}}]}'
    preds, ok, _ = parse_worker_output(raw)
    assert ok and preds[0].confidence == 0.7


def test_list_form_entries():
    raw = '{"predictions": [[0, "a", {"a": 0.8, "b": 0.2}], [1, "b"]]}'
    preds, ok, _ = parse_worker_output(raw)
    assert ok and preds[0].row_index == 0 and preds[0].confidence == 0.8
    assert preds[1].label == "b" and preds[1].probabilities is None


def test_bad_row_index_falls_back_to_position():
    raw = '{"predictions": [{"row_index": "x", "label": "a"}]}'
    preds, ok, _ = parse_worker_output(raw)
    assert ok and preds[0].row_index == 0


def test_ok_false_envelope():
    preds, ok, err = parse_worker_output('{"ok": false, "error": "CUDA missing"}')
    assert not ok and not preds and "CUDA missing" in err


def test_missing_predictions_key():
    _, ok, err = parse_worker_output('{"ok": true}')
    assert not ok and "predictions" in err


def test_predictions_not_a_list():
    _, ok, _ = parse_worker_output('{"predictions": {"0": {}}}')
    assert not ok


def test_non_json_output():
    _, ok, err = parse_worker_output("Traceback (most recent call last):")
    assert not ok and "not JSON" in err


def test_non_object_json():
    _, ok, _ = parse_worker_output("[1, 2, 3]")
    assert not ok


def test_nan_probabilities_dropped():
    raw = '{"predictions": [{"row_index": 0, "label": "a", "probabilities": {"a": 0.5, "b": "NaN"}}]}'
    preds, ok, _ = parse_worker_output(raw)
    assert ok and "b" not in (preds[0].probabilities or {})


def test_scalar_label_coerced_to_str():
    raw = '{"predictions": [{"row_index": 0, "label": {"bad": "shape"}}]}'
    preds, ok, _ = parse_worker_output(raw)
    assert ok and preds[0].label == "{'bad': 'shape'}"
