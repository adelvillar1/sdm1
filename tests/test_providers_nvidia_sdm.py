"""nvidia-sdm provider: config resolution, subprocess protocol, probe cache."""

import json

import pytest

from sdm1.providers import nvidia_sdm
from sdm1.types import ProviderUnavailableError, TableTask, WorkerError

TASK = TableTask(id="ok:classification", target="ok", num_estimators=8)
CONTEXT = [{"size": 1, "ok": "yes"}, {"size": 2, "ok": "no"}]
QUERY = [{"size": 3, "ok": None}, {"size": 4, "ok": None}]


def test_unconfigured_raises_actionable(monkeypatch):
    with pytest.raises(ProviderUnavailableError, match="SDM1_MODELS_PYTHON"):
        nvidia_sdm.run(CONTEXT, QUERY, TASK, {})
    ok, reason = nvidia_sdm.probe_detail()
    assert not ok and "SDM1_MODELS_PYTHON" in reason


def test_models_python_from_venv_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SDM1_MODELS_VENV", str(tmp_path / "modelsvenv"))
    assert nvidia_sdm.models_python() == str(tmp_path / "modelsvenv" / "bin" / "python")


def test_run_happy_path(stub_models_python, stub_worker):
    telemetry: dict = {}
    preds = nvidia_sdm.run(CONTEXT, QUERY, TASK, telemetry)
    assert telemetry["worker_returncode"] == 0 and telemetry["structured_ok"] is True
    assert [p.row_index for p in preds] == [0, 1]
    assert preds[0].confidence == 0.9 and preds[1].confidence == 0.6


def test_run_crashing_worker_raises_worker_error(tmp_path, monkeypatch, stub_models_python):
    crash = tmp_path / "crash_worker.py"
    crash.write_text("import sys; sys.exit(1)\n")
    monkeypatch.setenv("SDM1_WORKER_SCRIPT", str(crash))
    telemetry: dict = {}
    with pytest.raises(WorkerError, match="exited 1"):
        nvidia_sdm.run(CONTEXT, QUERY, TASK, telemetry)
    assert telemetry["worker_returncode"] == 1


def test_run_unparseable_worker_output(tmp_path, monkeypatch, stub_models_python):
    bad = tmp_path / "bad_worker.py"
    bad.write_text("print('this is not json')\n")
    monkeypatch.setenv("SDM1_WORKER_SCRIPT", str(bad))
    telemetry: dict = {}
    with pytest.raises(WorkerError, match="unparseable"):
        nvidia_sdm.run(CONTEXT, QUERY, TASK, telemetry)
    assert telemetry["structured_ok"] is False and telemetry["error_kind"] == "parse"


def test_run_timeout(tmp_path, monkeypatch, stub_models_python):
    slow = tmp_path / "slow_worker.py"
    slow.write_text("import time; time.sleep(5)\n")
    monkeypatch.setenv("SDM1_WORKER_SCRIPT", str(slow))
    monkeypatch.setenv("SDM1_TIMEOUT", "1")
    with pytest.raises(WorkerError, match="timed out"):
        nvidia_sdm.run(CONTEXT, QUERY, TASK, {})


def test_probe_available_and_cached(stub_models_python):
    ok, reason = nvidia_sdm.probe_detail()
    assert ok and "cuda ok" in reason
    # Second call within TTL must come from cache (probe counter unchanged).
    calls = {"n": 0}
    real = nvidia_sdm.probe_stack

    def counting(mp):
        calls["n"] += 1
        return real(mp)

    nvidia_sdm.probe_stack = counting
    try:
        nvidia_sdm.probe_detail()
        assert calls["n"] == 0
    finally:
        nvidia_sdm.probe_stack = real


def test_probe_reports_missing_cuda(tmp_path, monkeypatch):
    p = tmp_path / "no_cuda.py"
    p.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        'print(json.dumps({"torch": "9.9.9", "cuda": False, "sdm": "9.9.9"}))\n'
    )
    p.chmod(0o755)
    monkeypatch.setenv("SDM1_MODELS_PYTHON", str(p))
    ok, reason = nvidia_sdm.probe_detail()
    assert not ok and "CUDA" in reason


def test_request_carries_task_spec(tmp_path, monkeypatch, stub_models_python):
    seen = {}

    class FakeProc:
        returncode = 0
        stdout = json.dumps(
            {"ok": True, "predictions": [{"row_index": 0, "label": "yes", "probabilities": {"yes": 1.0}}]}
        )
        stderr = ""

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["request"] = json.loads(kwargs["input"])
        return FakeProc()

    monkeypatch.setattr(nvidia_sdm.subprocess, "run", fake_run)
    nvidia_sdm.run(CONTEXT, QUERY, TASK, {})
    assert seen["request"]["model"] == "nvidia/Kumo-Tabular"
    assert seen["request"]["target"] == "ok"
    assert seen["request"]["num_estimators"] == 8
    assert seen["request"]["rows_context"] == CONTEXT
