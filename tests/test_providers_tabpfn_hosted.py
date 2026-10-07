"""tabpfn-hosted provider: wire flow, auth/quota semantics, telemetry, redaction."""

import json

import pytest

from sdm1.providers import tabpfn_hosted
from sdm1.types import ProviderUnavailableError, TableTask, WorkerError

CONTEXT = [{"size": 1, "color": "red", "ok": "yes"}, {"size": 2, "color": "blue", "ok": "no"}]
QUERY = [{"size": 3, "color": "red", "ok": None}, {"size": 4, "color": "blue", "ok": None}]
TASK = TableTask(id="ok:classification", target="ok", num_estimators=8)


def test_unconfigured_raises_actionable(monkeypatch):
    with pytest.raises(ProviderUnavailableError, match="TABPFN_API_KEY"):
        tabpfn_hosted.run(CONTEXT, QUERY, TASK, {})
    ok, reason = tabpfn_hosted.probe_detail()
    assert not ok and "TABPFN_API_KEY" in reason


def test_run_happy_path_classification(stub_tabpfn):
    telemetry: dict = {}
    preds = tabpfn_hosted.run(CONTEXT, QUERY, TASK, telemetry)
    assert [p.row_index for p in preds] == [0, 1]
    assert preds[0].label == "yes" and preds[0].confidence == 0.9
    assert preds[1].label == "no" and preds[1].confidence == 0.6
    assert preds[0].probabilities == {"yes": 0.9, "no": 0.1}
    # wire flow shape: prepare → PUT x2 → fit → prepare_test → PUT → predict
    paths = [p for p, _ in stub_tabpfn["calls"]]
    assert paths[0].endswith("prepare_train_set_upload")
    assert paths[1].endswith("fit") and paths[2].endswith("prepare_test_set_upload")
    assert paths[3].endswith("predict")
    assert len(stub_tabpfn["put_bodies"]) == 3
    fit_payload = stub_tabpfn["calls"][1][1]
    assert fit_payload == {"train_set_upload_id": "tsu-1", "task": "classification"}
    predict_payload = stub_tabpfn["calls"][3][1]
    assert predict_payload["fitted_train_set_id"] == "fts-1"
    assert predict_payload["task_config"]["task"] == "classification"
    assert predict_payload["task_config"]["predict_params"]["output_type"] == "probas"
    # y_train CSV carries only the target column
    y_body = stub_tabpfn["put_bodies"][1][1].decode()
    assert y_body.splitlines()[0] == "ok" and "red" not in y_body


def test_telemetry_carries_billing_keys(stub_tabpfn):
    telemetry: dict = {}
    tabpfn_hosted.run(CONTEXT, QUERY, TASK, telemetry)
    assert telemetry["billing_model_version"] == "v3.5"
    assert telemetry["execution_mode"] == "standard"
    assert telemetry["cache_outcome"] == "not_requested"
    assert telemetry["n_estimators"] == 8
    assert telemetry["structured_ok"] is True
    assert "wire_latency_ms" in telemetry


def test_dataset_contents_never_in_requests_metadata(stub_tabpfn):
    """Cell values may ride in the CSV bodies only — never in telemetry paths."""
    marker_context = [{"secretcol": "ZZMARKZZ", "ok": "yes"}]
    marker_query = [{"secretcol": "ZZMARKZZ", "ok": None}]
    task = TableTask(id="ok:classification", target="ok")
    telemetry: dict = {}
    tabpfn_hosted.run(marker_context, marker_query, task, telemetry)
    assert "ZZMARKZZ" not in json.dumps(telemetry)
    assert any("ZZMARKZZ" in body.decode() for _, body in stub_tabpfn["put_bodies"])


def test_api_log_redacts_dataset_contents(rows, tmp_path, fake_provider):
    """API-level redaction tripwire: distinctive cell values never reach the log."""
    from sdm1 import classify
    from sdm1.telemetry import log_dir

    marker = "QQUNIQUEQQ"
    table = [{"size": 10, "ok": "yes"}, {"size": 20, "ok": "no"}, {"size": f"{marker}", "ok": None}]
    result = classify(table, target="ok", model="fake")
    assert result.verdict == "pass"
    log_text = "\n".join(p.read_text() for p in log_dir().glob("events.jsonl"))
    assert marker not in log_text


def _raise_http_at_boundary(monkeypatch, code, reset=None):
    """Raise a real HTTPError from urllib so the provider's mapping runs."""
    import io as _io
    import urllib.error as _ue

    monkeypatch.setenv("TABPFN_API_KEY", "tabpfn_sk_test_value")
    headers = {"x-ratelimit-reset": reset} if reset else {}
    err = _ue.HTTPError("https://stub/call", code, "err", headers, _io.BytesIO(b"{}"))

    def fake_urlopen(req, timeout=None):
        raise err

    monkeypatch.setattr(tabpfn_hosted.urllib.request, "urlopen", fake_urlopen)


def test_http_401_maps_to_unavailable(monkeypatch):
    _raise_http_at_boundary(monkeypatch, 401)
    with pytest.raises(ProviderUnavailableError, match="TABPFN_API_KEY"):
        tabpfn_hosted.run(CONTEXT, QUERY, TASK, {})


def test_http_429_names_reset(monkeypatch):
    _raise_http_at_boundary(monkeypatch, 429, reset="2026-10-08T00:00:00Z")
    with pytest.raises(ProviderUnavailableError, match="2026-10-08T00:00:00Z"):
        tabpfn_hosted.run(CONTEXT, QUERY, TASK, {})


def test_http_500_maps_to_worker_error(monkeypatch):
    _raise_http_at_boundary(monkeypatch, 500)
    with pytest.raises(WorkerError, match="500"):
        tabpfn_hosted.run(CONTEXT, QUERY, TASK, {})


def test_forecast_wire_uses_regression_quantiles(stub_tabpfn):
    stub_tabpfn["predict_payload"] = _tabpfn_predict_payload_task("forecast")
    task = TableTask(id="dur:forecast", target="dur", task_type="forecast")
    preds = tabpfn_hosted.run(CONTEXT, QUERY, task, {})
    fit_payload = stub_tabpfn["calls"][1][1]
    assert fit_payload["task"] == "regression"
    predict_payload = stub_tabpfn["calls"][3][1]
    assert predict_payload["task_config"]["task"] == "regression"
    assert predict_payload["task_config"]["predict_params"]["output_type"] == "quantiles"
    # levels x rows: median level (index 4) per row → label
    assert preds[0].label == 45.0
    assert preds[0].extras["quantiles"]["0.1"] == 41.0
    assert preds[0].extras["quantiles"]["0.9"] == 49.0
    assert preds[0].confidence is None  # forecast rows are never floor-gated


def test_anomaly_wire_labels_band_membership(stub_tabpfn):
    stub_tabpfn["predict_payload"] = _tabpfn_predict_payload_task("anomaly")
    # rows carry the observed value in `<target>_actual`; band rows 0-1 near 45,
    # row 2 far outside every band (levels for row 2 span 99-107 → actual 200.0)
    query = [
        {"size": 3, "dur_actual": 45.5, "dur": None},
        {"size": 4, "dur_actual": 45.9, "dur": None},
        {"size": 5, "dur_actual": 200.0, "dur": None},
    ]
    task = TableTask(id="dur:anomaly", target="dur", task_type="anomaly")
    preds = tabpfn_hosted.run(CONTEXT, query, task, {})
    # actual must never be sent as a feature column
    x_body = stub_tabpfn["put_bodies"][2][1].decode()
    assert "dur_actual" not in x_body
    assert preds[0].label == "normal" and preds[0].confidence == 0.8
    assert preds[1].label == "normal"
    assert preds[2].label == "anomaly" and preds[2].confidence == 0.0
    assert preds[2].extras["actual"] == 200.0


def _tabpfn_predict_payload_task(task_type):
    quantiles = [
        [41.0, 42.0, 99.0], [42.0, 43.0, 100.0], [43.0, 44.0, 101.0],
        [44.0, 45.0, 102.0], [45.0, 46.0, 103.0], [46.0, 47.0, 104.0],
        [47.0, 48.0, 105.0], [48.0, 49.0, 106.0], [49.0, 50.0, 107.0],
    ]
    return {
        "prediction": quantiles,
        "metadata": {"billing_model_version": "v3.5", "execution_mode": "standard",
                     "cache_outcome": "not_requested", "n_estimators": 8},
    }


def test_anomaly_missing_actual_falls_back_to_median(stub_tabpfn):
    stub_tabpfn["predict_payload"] = _tabpfn_predict_payload_task("anomaly")
    query = [{"size": 3, "dur": None}]
    task = TableTask(id="dur:anomaly", target="dur", task_type="anomaly")
    preds = tabpfn_hosted.run(CONTEXT, query, task, {})
    assert preds[0].label == 45.0 and preds[0].confidence is None


def test_probe_available_and_cached(stub_tabpfn):
    ok, reason = tabpfn_hosted.probe_detail()
    assert ok and "v3.5" in reason
    calls = {"n": 0}
    real = tabpfn_hosted._authed_json

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    tabpfn_hosted._authed_json = counting
    try:
        tabpfn_hosted.probe_detail()
        assert calls["n"] == 0  # TTL cache
    finally:
        tabpfn_hosted._authed_json = real


def test_regression_median_wire(stub_tabpfn):
    stub_tabpfn["predict_payload"] = {
        "prediction": [45.5, 46.0],
        "metadata": {"billing_model_version": "v3.5", "execution_mode": "standard",
                     "cache_outcome": "not_requested", "n_estimators": 8},
    }
    task = TableTask(id="dur:regression", target="dur", task_type="regression")
    preds = tabpfn_hosted.run(CONTEXT, QUERY, task, {})
    assert preds[0].label == 45.5 and preds[1].label == 46.0
    assert preds[0].confidence is None
