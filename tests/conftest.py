"""Shared fixtures: env isolation, stub worker/models-python, registry snapshot."""

import json

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
    """Every test gets an isolated log dir and no models-stack env vars."""
    monkeypatch.setenv("SDM1_LOG_DIR", str(tmp_path / "logs"))
    for var in ("SDM1_MODELS_PYTHON", "SDM1_MODELS_VENV", "SDM1_WORKER_SCRIPT", "SDM1_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)
    from sdm1.providers import nvidia_sdm

    nvidia_sdm.reset_probe_cache()
    yield
    nvidia_sdm.reset_probe_cache()


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


def read_last_log_record(tmp_path) -> dict:
    from pathlib import Path

    logs = sorted((Path(tmp_path) / "logs").rglob("events.jsonl"))
    assert logs, "no events.jsonl written"
    lines = logs[0].read_text().strip().splitlines()
    return json.loads(lines[-1])
