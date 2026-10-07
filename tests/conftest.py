"""Shared fixtures: env isolation, stub worker/models-python, registry snapshot."""

import io
import json
import urllib.error

import pytest

# The public wire contract (TECHNICAL-DOCUMENTATION.md §4). Renaming or
# removing any key is a breaking change; the contract tests are the tripwire.
CONTRACT_KEYS = [
    "table",
    "task",
    "model",
    "models_used",
    "task_type",
    "target",
    "n_context_rows",
    "n_query_rows",
    "num_estimators",
    "predictions",
    "telemetry",
    "verdict",
    "escalated",
    "low_confidence_rows",
    "latency_ms",
    "input_sha256",
    "error",
    "reason",
    "summary",
]

STUB_WORKER = """\
import json, sys
req = json.loads(sys.stdin.read())
n = len(req["rows_query"])
target = req["target"]
classes = sorted({str(r.get(target)) for r in req["rows_context"]})
preds = []
for i in range(n):
    if i % 2 == 0:
        probs = {classes[0]: 0.9, classes[1]: 0.1}
    else:
        probs = {classes[0]: 0.4, classes[1]: 0.6}
    label = max(probs, key=probs.__getitem__)
    preds.append({"row_index": i, "label": label, "probabilities": probs, "confidence": probs[label]})
print(json.dumps({"ok": True, "model_name": req.get("model"), "predictions": preds}))
"""

STUB_WORKER_CRASH = "import sys; sys.stderr.write('boom'); sys.exit(1)\n"

STUB_WORKER_SLOW = "import time, sys; time.sleep(5)\n"

STUB_PYTHON_OK = (
    "#!/usr/bin/env python3\n"
    "import json, runpy, sys\n"
    'if "-c" in sys.argv[1:]:\n'
    '    print(json.dumps({"torch": "9.9.9", "cuda": True, "sdm": "9.9.9"}))\n'
    "else:\n"
    "    sys.argv = sys.argv[1:]\n"  # worker mode: execute the script like a real python would
    "    runpy.run_path(sys.argv[0])\n"
)

STUB_PYTHON_NO_CUDA = (
    "#!/usr/bin/env python3\n"
    "import json, sys\n"
    'if "-c" in sys.argv[1:]:\n'
    '    print(json.dumps({"torch": "9.9.9", "cuda": False, "sdm": "9.9.9"}))\n'
    "else:\n"
    "    sys.exit(3)\n"
)


@pytest.fixture(autouse=True)
def _env_isolation(tmp_path, monkeypatch):
    """Every test gets an isolated log dir and no backend-stack env vars."""
    monkeypatch.setenv("SDM1_LOG_DIR", str(tmp_path / "logs"))
    for var in (
        "SDM1_MODELS_PYTHON",
        "SDM1_MODELS_VENV",
        "SDM1_WORKER_SCRIPT",
        "SDM1_TIMEOUT",
        "TABPFN_API_KEY",
        "SDM1_TABPFN_BASE",
        "SDM1_TABPFN_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)
    from sdm1.providers import nvidia_sdm, tabpfn_hosted

    nvidia_sdm.reset_probe_cache()
    tabpfn_hosted.reset_probe_cache()
    yield
    nvidia_sdm.reset_probe_cache()
    tabpfn_hosted.reset_probe_cache()


@pytest.fixture
def stub_worker(tmp_path, monkeypatch):
    p = tmp_path / "stub_worker.py"
    p.write_text(STUB_WORKER)
    monkeypatch.setenv("SDM1_WORKER_SCRIPT", str(p))
    return str(p)


@pytest.fixture
def stub_models_python(tmp_path, monkeypatch):
    p = tmp_path / "stub_python.py"
    p.write_text(STUB_PYTHON_OK)
    p.chmod(0o755)
    monkeypatch.setenv("SDM1_MODELS_PYTHON", str(p))
    return str(p)


@pytest.fixture
def rows():
    """A tiny table: 3 labeled context rows + 2 query rows (target null)."""
    return [
        {"size": 10, "color": "red", "ok": "yes"},
        {"size": 20, "color": "blue", "ok": "no"},
        {"size": 30, "color": "red", "ok": "yes"},
        {"size": 1111111, "color": "red", "ok": None},
        {"size": 2222222, "color": "blue", "ok": None},
    ]


@pytest.fixture
def clean_registry():
    """Snapshot REGISTRY around a test so fake providers don't leak."""
    from sdm1 import registry

    snapshot = dict(registry.REGISTRY)
    yield registry
    registry.REGISTRY.clear()
    registry.REGISTRY.update(snapshot)


@pytest.fixture
def fake_provider(clean_registry):
    """A fake backend whose predictions are configurable per test."""
    from sdm1.types import Prediction, Provider

    holder: dict = {"predictions": None, "error": None}

    def run(rows_context, rows_query, task, telemetry):
        telemetry["fake"] = True
        if holder["error"] is not None:
            raise holder["error"]
        if holder["predictions"] is None:
            holder["predictions"] = [
                Prediction(row_index=i, label="yes", probabilities={"yes": 0.9, "no": 0.1}, confidence=0.9)
                for i in range(len(rows_query))
            ]
        return holder["predictions"]

    provider = Provider(
        id="fake",
        name="Fake backend",
        run=run,
        is_available=lambda: True,
        is_configured=lambda: True,
        description="test double",
    )
    clean_registry.register(provider)
    return holder


# ── tabpfn-hosted stubs ───────────────────────────────────────────────────────


class FakeHTTPError(urllib.error.HTTPError):
    """HTTPError that carries a reset header for quota tests."""

    def __init__(self, code, body=b"{}", headers=None):
        super().__init__("https://stub", code, "err", headers or {}, io.BytesIO(body))


def _tabpfn_predict_payload(task_type="classification"):
    if task_type == "classification":
        return {
            "prediction": [[0.9, 0.1], [0.4, 0.6]],
            "metadata": {
                "billing_model_version": "v3.5",
                "execution_mode": "standard",
                "cache_outcome": "not_requested",
                "n_estimators": 8,
                "classes": ["yes", "no"],
            },
        }
    quantiles = [[41.0, 42.0, 99.0], [42.0, 43.0, 100.0], [43.0, 44.0, 101.0],
                 [44.0, 45.0, 102.0], [45.0, 46.0, 103.0], [46.0, 47.0, 104.0],
                 [47.0, 48.0, 105.0], [48.0, 49.0, 106.0], [49.0, 50.0, 107.0]]
    prediction = quantiles  # levels x rows
    return {
        "prediction": prediction,
        "metadata": {"billing_model_version": "v3.5", "execution_mode": "standard",
                     "cache_outcome": "not_requested", "n_estimators": 8},
    }


@pytest.fixture
def stub_tabpfn(monkeypatch):
    """Script the hosted wire: env key set, HTTP calls recorded, canned responses.

    Returns a holder dict: calls (path, payload) list, put_bodies list, and
    settable `predict_payload` / `http_error` (code, reset) for failure tests.
    """
    import sdm1.providers.tabpfn_hosted as th

    monkeypatch.setenv("TABPFN_API_KEY", "tabpfn_sk_test_value")
    holder: dict = {"calls": [], "put_bodies": [], "predict_payload": None,
                    "http_error": None, "limits_payload": {"default_model_version": "v3.5",
                                                            "max_model_limit": {"max_classes": 160}}}

    def fake_authed(path, payload, timeout):
        holder["calls"].append((path, payload))
        if holder["http_error"] is not None:
            code, reset = holder["http_error"]
            headers = {"x-ratelimit-reset": reset} if reset else {}
            raise FakeHTTPError(code, headers=headers)
        if path.endswith("get_model_limits"):
            return holder["limits_payload"]
        if path.endswith("prepare_train_set_upload"):
            return {
                "train_set_upload_id": "tsu-1",
                "x_train_info": {"signed_urls": ["https://stub/x"], "required_headers": {"a": "b"}},
                "y_train_info": {"signed_urls": ["https://stub/y"], "required_headers": {"a": "b"}},
            }
        if path.endswith("fit"):
            return {"fitted_train_set_id": "fts-1"}
        if path.endswith("prepare_test_set_upload"):
            return {
                "test_set_upload_id": "tsu-2",
                "x_test_info": {"signed_urls": ["https://stub/xt"], "required_headers": {"a": "b"}},
            }
        if path.endswith("predict"):
            return holder["predict_payload"] or _tabpfn_predict_payload()
        raise AssertionError(f"unexpected path {path}")

    def fake_put(url, body, required_headers, timeout):
        holder["put_bodies"].append((url, body))

    monkeypatch.setattr(th, "_authed_json", fake_authed)
    monkeypatch.setattr(th, "_put_signed", fake_put)
    th.reset_probe_cache()
    return holder


def read_last_log_record(tmp_path) -> dict:
    from pathlib import Path

    logs = sorted((Path(tmp_path) / "logs").rglob("events.jsonl"))
    assert logs, "no events.jsonl written"
    lines = logs[0].read_text().strip().splitlines()
    return json.loads(lines[-1])
