import sdm1


def test_package_imports_and_has_version():
    assert isinstance(sdm1.__version__, str)
    assert sdm1.__version__.count(".") == 2
