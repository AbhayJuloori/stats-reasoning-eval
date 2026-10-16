from evalset.grading import first_number, item_correct, last_number, qrdata_correct
from evalset.run_review import parse_verdict
from evalset.sandbox import extract_code


def test_numbers():
    assert first_number("about 12.5% of users") == 0.125
    assert last_number("lift = 0.74\n0.81") == 0.81
    assert last_number("1,234 customers") == 1234
    assert last_number("no digits") is None


def test_qrdata_rule_matches_benchmark():
    assert qrdata_correct("0.205", "0.20", "numerical")
    assert not qrdata_correct("0.25", "0.20", "numerical")
    assert qrdata_correct("B", "B", "multiple_choice")
    assert not qrdata_correct("A", "B", "multiple_choice")


def test_relative_tolerance_and_rounding_floor():
    tol = {"kind": "rel", "rel": 0.03, "decimals": 2}
    assert item_correct("0.75", "0.74", tol)
    assert not item_correct("0.80", "0.74", tol)
    # an answer near zero still gets half a rounding unit of slack
    assert item_correct("0.004", "0.0", tol)


def test_pvalues():
    tol = {"kind": "pvalue"}
    assert item_correct("0.0000", "0.0", tol)
    assert item_correct("3e-08", "0.0", tol)
    assert not item_correct("0.02", "0.0", tol)
    assert item_correct("0.0436", "0.0427", tol)
    assert not item_correct("0.06", "0.0427", tol)


def test_exact():
    assert item_correct("em2", "em2", {"kind": "exact"})
    assert item_correct("3", "3", {"kind": "exact"})
    assert not item_correct("4", "3", {"kind": "exact"})


def test_verdict_parsing():
    assert parse_verdict("VERDICT: FLAWED\nISSUE: x") == "FLAWED"
    assert parse_verdict("**VERDICT:** SOUND") == "SOUND"
    assert parse_verdict("The analysis is flawed.") == "FLAWED"
    assert parse_verdict("hmm") == "UNPARSED"


def test_extract_code_prefers_longest_block():
    text = "```python\nx = 1\n```\nthen\n```python\nimport pandas as pd\nprint(1)\n```"
    assert "import pandas" in extract_code(text)
