"""Tests of the model interface, the budget guard, the call store, and the log."""

import json
from datetime import date
from pathlib import Path

import pytest
import studies.capacity_llm as capacity_llm
from studies.capacity_labeller import AnswerError, Prompt, build_prompt
from studies.capacity_llm import (
    BudgetExceeded,
    BudgetGuard,
    CallStore,
    ClaudeCliModel,
    ModelError,
    ModelReply,
    StubModel,
    assert_under_data,
    labelled_answer,
    parse_stream,
    stream_message,
)

FIRST = date(2024, 1, 1)
LAST = date(2024, 12, 31)
VALID = json.dumps(
    {"plant": "A", "period": [str(FIRST), str(LAST)], "no_change": True, "knots": []}
)


def _prompt(*, text: str = "case file") -> Prompt:
    return build_prompt(text=text, images=[b"png-bytes"])


def _reply(*, cost: float = 0.1, tokens: int = 1000) -> ModelReply:
    return ModelReply(
        text=VALID, input_tokens=tokens, output_tokens=10, cost_usd=cost, model="stub"
    )


def test_the_guard_raises_before_the_hundred_and_fifty_first_call() -> None:
    guard = BudgetGuard(ceiling_usd=1e9)
    for _ in range(150):
        guard.check(prompt=_prompt())
        guard.record(reply=_reply(cost=0.0, tokens=1))
    with pytest.raises(BudgetExceeded, match="151"):
        guard.check(prompt=_prompt())


def test_the_guard_raises_before_two_million_input_tokens() -> None:
    guard = BudgetGuard(ceiling_usd=1e9)
    guard.record(reply=_reply(cost=0.0, tokens=1_999_000))
    with pytest.raises(BudgetExceeded, match="input tokens"):
        guard.check(prompt=_prompt(text="x" * 10_000))


def test_the_guard_raises_when_the_next_call_could_reach_the_cost_ceiling() -> None:
    guard = BudgetGuard(ceiling_usd=1.0)
    guard.check(prompt=_prompt())
    guard.record(reply=_reply(cost=0.45))
    guard.check(prompt=_prompt())
    guard.record(reply=_reply(cost=0.45))
    with pytest.raises(BudgetExceeded, match="spent"):
        guard.check(prompt=_prompt())


def test_the_guard_assumes_a_cost_before_any_call_has_been_measured() -> None:
    with pytest.raises(BudgetExceeded):
        BudgetGuard(ceiling_usd=0.1).check(prompt=_prompt())


def test_the_stream_message_carries_each_image_before_the_text() -> None:
    message = json.loads(stream_message(prompt=_prompt()))
    content = message["message"]["content"]
    assert [part["type"] for part in content] == ["image", "text"]
    assert content[0]["source"]["media_type"] == "image/png"


def _stream(*, model: str = "claude-sonnet-5-5", is_error: bool = False) -> str:
    result = {
        "type": "result",
        "is_error": is_error,
        "result": "hello",
        "total_cost_usd": 0.05,
        "usage": {
            "input_tokens": 2,
            "cache_creation_input_tokens": 700,
            "cache_read_input_tokens": 30,
            "output_tokens": 9,
        },
    }
    assistant = {"type": "assistant", "message": {"model": model}}
    return "\n".join(json.dumps(event) for event in ({"type": "system"}, assistant, result))


def test_the_stream_parser_sums_every_kind_of_input_token_and_reads_the_cost() -> None:
    reply = parse_stream(output=_stream(), requested_model="claude-sonnet-5-5")
    assert (reply.text, reply.input_tokens, reply.output_tokens, reply.cost_usd) == (
        "hello",
        732,
        9,
        0.05,
    )


def test_the_stream_parser_refuses_another_model_or_an_error_result() -> None:
    with pytest.raises(ModelError, match="answered"):
        parse_stream(output=_stream(model="claude-haiku"), requested_model="claude-sonnet-5-5")
    with pytest.raises(ModelError, match="no successful result"):
        parse_stream(output=_stream(is_error=True), requested_model="claude-sonnet-5-5")


def test_the_command_disables_every_tool_and_every_source_of_instructions() -> None:
    command = ClaudeCliModel().command(prompt=_prompt())
    for flag in (
        "--tools",
        "--setting-sources",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--system-prompt",
    ):
        assert flag in command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--model") + 1] == "claude-sonnet-5-5"


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(capacity_llm, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    return tmp_path / "data"


def test_the_log_must_lie_under_data(tmp_path: Path, data_root: Path) -> None:
    assert_under_data(path=data_root / "studies" / "log.jsonl")
    with pytest.raises(ValueError, match="under"):
        assert_under_data(path=tmp_path / "elsewhere" / "log.jsonl")
    with pytest.raises(ValueError, match="under"):
        CallStore(directory=tmp_path / "elsewhere", log_path=tmp_path / "elsewhere" / "log.jsonl")


def _store(*, data_root: Path) -> CallStore:
    return CallStore(directory=data_root / "calls", log_path=data_root / "log.jsonl")


def test_a_repeated_call_is_served_from_the_cache_and_the_log_holds_the_prompt(
    data_root: Path,
) -> None:
    calls = []
    model = StubModel(respond=lambda prompt: (calls.append(prompt), VALID)[1])
    store, guard = _store(data_root=data_root), BudgetGuard()
    arguments = {
        "store": store,
        "model": model,
        "guard": guard,
        "prompt": _prompt(),
        "label": "t",
        "plant": "A",
        "first_day": FIRST,
        "last_day": LAST,
    }
    labelled_answer(sample_index=0, **arguments)
    labelled_answer(sample_index=0, **arguments)
    assert len(calls) == 1
    assert guard.calls == 1
    labelled_answer(sample_index=1, **arguments)
    assert len(calls) == 2
    log = [json.loads(line) for line in (data_root / "log.jsonl").read_text().splitlines()]
    assert len(log) == 2
    assert log[0]["prompt"] == "case file"
    assert log[0]["model"] == "stub"
    assert len(log[0]["image_sha256"][0]) == 64


def test_an_invalid_reply_is_retried_once_and_each_attempt_counts_as_a_call(
    data_root: Path,
) -> None:
    replies = iter(["not json", VALID])
    model = StubModel(respond=lambda prompt: next(replies))
    guard = BudgetGuard()
    answer = labelled_answer(
        store=_store(data_root=data_root), model=model, guard=guard, prompt=_prompt(),
        sample_index=0, label="t", plant="A", first_day=FIRST, last_day=LAST,
    )  # fmt: skip
    assert answer.no_change
    assert guard.calls == 2


def test_two_invalid_replies_raise_the_answer_error(data_root: Path) -> None:
    model = StubModel(respond=lambda prompt: "still not json")
    with pytest.raises(AnswerError):
        labelled_answer(
            store=_store(data_root=data_root), model=model, guard=BudgetGuard(), prompt=_prompt(),
            sample_index=0, label="t", plant="A", first_day=FIRST, last_day=LAST,
        )  # fmt: skip
