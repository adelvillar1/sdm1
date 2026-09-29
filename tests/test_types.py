"""Contract freeze: the TableResult wire shape (TECHNICAL-DOCUMENTATION §4)."""

import json

import pytest
from conftest import CONTRACT_KEYS

from sdm1.types import Prediction, TableResult, TableTask

# The public contract. Renaming/removing any of these is a breaking change;
# this test is the tripwire. Adding keys means updating this list deliberately.
# (Key list lives in conftest so the API tests freeze it too.)


def _result() -> TableResult:
    return TableResult(
        table="t",
        task="ok:classification",
        model="nvidia-sdm",
        models_used=["nvidia-sdm"],
        task_type="classification",
        target="ok",
        n_context_rows=3,
        n_query_rows=2,
        num_estimators=8,
        predictions=[
            Prediction(row_index=0, label="yes", probabilities={"yes": 0.9, "no": 0.1}, confidence=0.9)
        ],
    )


def test_as_dict_matches_contract_key_set():
    assert list(_result().as_dict().keys()) == CONTRACT_KEYS


def test_as_dict_is_json_serializable():
    dumped = json.dumps(_result().as_dict())
    assert json.loads(dumped)["predictions"][0]["label"] == "yes"


def test_prediction_as_dict_shape():
    p = Prediction(row_index=1, label="no", probabilities={"yes": 0.2, "no": 0.8}, confidence=0.8).as_dict()
    assert p["row_index"] == 1 and p["confidence"] == 0.8 and p["probabilities"]["no"] == 0.8


class TestTableTaskValidation:
    def test_bad_task_type_rejected(self):
        with pytest.raises(ValueError, match="task_type"):
            TableTask(id="x", target="ok", task_type="clustering")

    def test_empty_target_rejected(self):
        with pytest.raises(ValueError, match="target"):
            TableTask(id="x", target="")

    def test_zero_estimators_rejected(self):
        with pytest.raises(ValueError, match="num_estimators"):
            TableTask(id="x", target="ok", num_estimators=0)

    def test_floor_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="confidence_floor"):
            TableTask(id="x", target="ok", confidence_floor=1.5)

    def test_to_spec_shape(self):
        spec = TableTask(id="t", target="ok", stype_overrides={"size": "numerical"}, num_estimators=4).to_spec()
        assert spec == {
            "target": "ok",
            "task_type": "classification",
            "feature_columns": None,
            "stype_overrides": {"size": "numerical"},
            "num_estimators": 4,
        }
