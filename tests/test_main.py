from rckst.__main__ import _parse_val


def test_parse_val() -> None:
    assert _parse_val("0.75") == 0.75
    assert _parse_val("100") == 100
    assert _parse_val("True") is True
    assert _parse_val("False") is False
    assert _parse_val("greedy") == "greedy"
