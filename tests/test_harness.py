import harness


class FakeSearcher:
    def search(self, diagnosis, treatment, query, k=5, exclude_id=None):
        return [{"case_id": 1, "outcome": "overturned", "request": "genomic test", "reasoning": "guidelines support it"}]


CASE = {"case_id": 99, "diagnosis": "Cancer", "treatment": "Diag Imag & Screen", "request": "genomic test"}
FINAL = {"role": "assistant", "content": '{"prediction": "overturned", "confidence": "high", "reasons": "similar cases overturned", "case_ids": [1]}'}
SEARCH = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search_cases", "arguments": {"query": "genomic test"}}}]}


def scripted(*replies):
    it = iter(replies)
    return lambda messages: next(it)


def setup_log(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "LOG_PATH", str(tmp_path / "log.jsonl"))


def test_search_then_answer(tmp_path, monkeypatch):
    setup_log(tmp_path, monkeypatch)
    out = harness.run_agent(CASE, FakeSearcher(), scripted(SEARCH, FINAL))
    assert out["status"] == "ok" and out["prediction"] == "overturned"
    assert (tmp_path / "log.jsonl").read_text().count("\n") >= 3


def test_answer_without_search_is_rejected(tmp_path, monkeypatch):
    setup_log(tmp_path, monkeypatch)
    out = harness.run_agent(CASE, FakeSearcher(), scripted(FINAL, SEARCH, FINAL))
    assert out["status"] == "ok"


def test_unknown_tool_does_not_crash(tmp_path, monkeypatch):
    setup_log(tmp_path, monkeypatch)
    bad = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "delete_everything", "arguments": {}}}]}
    out = harness.run_agent(CASE, FakeSearcher(), scripted(bad, SEARCH, FINAL))
    assert out["status"] == "ok"


def test_bad_arguments_string(tmp_path, monkeypatch):
    setup_log(tmp_path, monkeypatch)
    bad = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search_cases", "arguments": "not json"}}]}
    out = harness.run_agent(CASE, FakeSearcher(), scripted(bad, SEARCH, FINAL))
    assert out["status"] == "ok"


def test_turn_limit_returns_unsure(tmp_path, monkeypatch):
    setup_log(tmp_path, monkeypatch)
    out = harness.run_agent(CASE, FakeSearcher(), lambda m: SEARCH)
    assert out["prediction"] == "unsure" and out["status"] == "turn_limit"


def test_garbage_answer_asks_once_then_unsure(tmp_path, monkeypatch):
    setup_log(tmp_path, monkeypatch)
    junk = {"role": "assistant", "content": "I think it will probably be fine!"}
    out = harness.run_agent(CASE, FakeSearcher(), scripted(SEARCH, junk, junk))
    assert out["prediction"] == "unsure" and out["status"] == "bad_format"


def test_model_down_returns_unsure(tmp_path, monkeypatch):
    setup_log(tmp_path, monkeypatch)
    def boom(messages):
        raise ConnectionError("no model")
    out = harness.run_agent(CASE, FakeSearcher(), boom)
    assert out["status"] == "model_error"


def test_trim_cuts_long_results():
    assert len(harness.trim("x" * 10000)) <= harness.MAX_TOOL_CHARS + 20