"""CLI: version, models, doctor, classify exit codes."""

import json

from sdm1.cli import main


def test_version(capsys):
    assert main(["version"]) == 0
    assert "0.1.0" in capsys.readouterr().out


def test_models_lists_builtin(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "nvidia-sdm" in out and "Kumo-Tabular" in out


def test_models_json(capsys):
    assert main(["models", "--json"]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert entries["nvidia-sdm"]["meta"]["weights"] == "nvidia/Kumo-Tabular"


def test_doctor_unconfigured(tmp_path, capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "✗" in out and "SDM1_MODELS_PYTHON" in out


def test_doctor_json(tmp_path, capsys):
    assert main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["nvidia-sdm"]["available"] is False


def test_classify_unavailable_backend_exits_2(tmp_path, capsys):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("size,ok\n1,yes\n2,no\n3,\n")
    code = main(["classify", "--input", str(csv_path), "--target", "ok"])
    assert code == 2
    assert "verdict: error" in capsys.readouterr().out


def test_classify_json_output(tmp_path, capsys):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("size,ok\n1,yes\n2,no\n3,\n")
    code = main(["classify", "--input", str(csv_path), "--target", "ok", "--json"])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "error"


def test_classify_missing_target_exits_2_with_stderr(tmp_path, capsys):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("size,ok\n1,yes\n")
    code = main(["classify", "--input", str(csv_path), "--target", ""])
    assert code == 2
    assert "error:" in capsys.readouterr().err
