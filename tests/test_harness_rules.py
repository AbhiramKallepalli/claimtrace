import json

import harness


class FakeSearcher:
    def search(self, diagnosis, treatment, query, k=5, exclude_id=None):
        return [{"case_id": "A1", "outcome": "overturned", "request": "r", "reasoning": "x"}]


CASE = {"case_id": "T1", "diagnosis": "d", "treatment": "t", "request": "req"}
ANSWER = json.dumps({"prediction": "overturned", "confidence": "medium", "reasons": "ok",
                     "case_ids": ["A1"], "rule_ids": ["HSC-1374.33(b)"]})


def call(name, query="medically necessary"):
    return {"function": {"name": name, "arguments": {"query": query}}}


def scripted(*replies):
    it = iter(replies)
    return lambda messages: next(it)


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "LOG_PATH", str(tmp_path / "log.jsonl"))


def test_dispatch_rules_returns_rule_text():
    out = harness.dispatch(call("lookup_rules", "what if the reviewers are tied"), CASE, FakeSearcher())
    assert "HSC-1374.33(d)" in out


def test_dispatch_rules_no_match_says_so():
    out = harness.dispatch(call("lookup_rules", "banana pancake"), CASE, FakeSearcher())
    assert out == "No matching rule found."


def test_dispatch_rules_needs_query():
    out = harness.dispatch({"function": {"name": "lookup_rules", "arguments": {}}}, CASE, FakeSearcher())
    assert out.startswith("Error")


def test_unknown_tool_still_rejected():
    out = harness.dispatch(call("delete_everything"), CASE, FakeSearcher())
    assert out.startswith("Error: unknown tool")


def test_rules_lookup_alone_does_not_count_as_search(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    model = scripted({"role": "assistant", "tool_calls": [call("lookup_rules")]},
                     {"role": "assistant", "content": ANSWER},
                     {"role": "assistant", "tool_calls": [call("search_cases")]},
                     {"role": "assistant", "content": ANSWER})
    out = harness._agent_loop(CASE, FakeSearcher(), model, None)
    events = [json.loads(l)["event"] for l in open(tmp_path / "log.jsonl")]
    assert "answer_before_search" in events
    assert out["status"] == "ok"


def test_answer_keeps_rule_ids(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    model = scripted({"role": "assistant", "tool_calls": [call("search_cases"), call("lookup_rules")]},
                     {"role": "assistant", "content": ANSWER})
    out = harness._agent_loop(CASE, FakeSearcher(), model, None)
    assert out["rule_ids"] == ["HSC-1374.33(b)"]


def test_rule_ids_default_to_empty_list():
    ans = harness.parse_answer('{"prediction": "upheld", "confidence": "low"}')
    assert ans["rule_ids"] == []


def test_harness_records_rules_it_looked_up(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    plain = json.dumps({"prediction": "upheld", "confidence": "low", "reasons": "x", "case_ids": []})
    model = scripted({"role": "assistant", "tool_calls": [call("search_cases"), call("lookup_rules", "reviewers tied")]},
                     {"role": "assistant", "content": plain})
    out = harness._agent_loop(CASE, FakeSearcher(), model, None)
    assert out["rule_ids"] == []                      # the model did not cite anything
    assert "HSC-1374.33(d)" in out["rules_looked_up"]  # but the harness saw the lookup


def test_no_lookup_means_empty_list(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    model = scripted({"role": "assistant", "tool_calls": [call("search_cases")]},
                     {"role": "assistant", "content": ANSWER})
    out = harness._agent_loop(CASE, FakeSearcher(), model, None)
    assert out["rules_looked_up"] == []
