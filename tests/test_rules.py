from rules import RULES, lookup_rules


def test_every_rule_has_citation_and_link():
    for r in RULES:
        assert r["id"] and r["citation"] and r["text"]
        assert r["url"].startswith("https://")


def test_experimental_query_finds_1370_4():
    ids = [r["id"] for r in lookup_rules("denied as experimental or investigational therapy")]
    assert "HSC-1370.4(a)" in ids


def test_tie_query_finds_tie_rule():
    ids = [r["id"] for r in lookup_rules("what happens if the reviewers are tied")]
    assert ids[0] == "HSC-1374.33(d)"


def test_medical_necessity_query_finds_criteria():
    ids = [r["id"] for r in lookup_rules("was the service medically necessary")]
    assert "HSC-1374.33(b)" in ids


def test_k_limits_results():
    assert len(lookup_rules("medically necessary experimental", k=1)) == 1


def test_unrelated_query_returns_nothing():
    assert lookup_rules("banana pancake recipe") == []


def test_empty_query_returns_nothing():
    assert lookup_rules("") == []
    assert lookup_rules(None) == []


def test_results_are_copies():
    r = lookup_rules("medically necessary")[0]
    r["text"] = "changed"
    assert lookup_rules("medically necessary")[0]["text"] != "changed"
