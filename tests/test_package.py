import rckst


def test_package_exposes_version() -> None:
    assert rckst.__version__ == "0.1.0"
