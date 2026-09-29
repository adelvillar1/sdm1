"""Registry: register/replace/lookup, built-in presence."""

import pytest

import sdm1  # noqa: F401  (import registers the built-in nvidia-sdm backend)
from sdm1.registry import REGISTRY, all_ids, get, register
from sdm1.types import Provider


def test_builtin_registered():
    assert "nvidia-sdm" in REGISTRY
    provider = get("nvidia-sdm")
    assert provider.meta["weights"] == "nvidia/Kumo-Tabular"
    assert provider.meta["weights_license"] == "OpenMDW-1.1"
    assert provider.meta["model_class"] == "in-context"


def test_unknown_id_names_known_ids(clean_registry):
    with pytest.raises(KeyError, match="nvidia-sdm"):
        get("nope")


def test_register_replaces_by_id(clean_registry):
    p1 = Provider(id="x", name="one", run=lambda *a: [], is_available=lambda: True)
    p2 = Provider(id="x", name="two", run=lambda *a: [], is_available=lambda: True)
    register(p1)
    register(p2)
    assert get("x").name == "two"


def test_all_ids_sorted(clean_registry):
    register(Provider(id="zeta", name="z", run=lambda *a: [], is_available=lambda: True))
    register(Provider(id="alpha", name="a", run=lambda *a: [], is_available=lambda: True))
    ids = all_ids()
    assert ids == sorted(ids) and "alpha" in ids and "zeta" in ids
