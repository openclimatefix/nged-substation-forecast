"""Tests of the model interface, the budget guard, the call store, and the log."""

import json
import subprocess
from datetime import date
from pathlib import Path

import pytest
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

from studies import capacity_llm

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


def test_the_guard_projects_the_next_call_at_the_per_call_cap_and_stops_at_equality() -> None:
    cap = capacity_llm.PER_CALL_BUDGET_USD
    guard = BudgetGuard(ceiling_usd=3.0, spent_usd=3.0 - cap - 0.01)
    guard.check(prompt=_prompt())
    guard = BudgetGuard(ceiling_usd=3.0, spent_usd=3.0 - cap)
    with pytest.raises(BudgetExceeded, match="spent"):
        guard.check(prompt=_prompt())


def test_the_guard_projects_at_the_largest_call_seen_when_that_exceeds_the_cap() -> None:
    with pytest.raises(BudgetExceeded):
        BudgetGuard(ceiling_usd=15.0, spent_usd=13.1, largest_call_usd=2.0).check(prompt=_prompt())
    BudgetGuard(ceiling_usd=15.0, spent_usd=12.9, largest_call_usd=2.0).check(prompt=_prompt())


def test_the_guard_is_not_asked_when_the_answer_is_cached(data_root: Path) -> None:
    model = StubModel(respond=lambda prompt: VALID)
    store = _store(data_root=data_root)
    labelled_answer(
        store=store, model=model, guard=BudgetGuard(), prompt=_prompt(), sample_index=0,
        label="t", plant="A", first_day=FIRST, last_day=LAST,
    )  # fmt: skip
    exhausted = BudgetGuard(spent_usd=100.0)
    labelled_answer(
        store=store, model=model, guard=exhausted, prompt=_prompt(), sample_index=0,
        label="t", plant="A", first_day=FIRST, last_day=LAST,
    )  # fmt: skip
    assert exhausted.calls == 0


def test_a_resumed_guard_starts_from_every_logged_call_including_failed_ones(
    data_root: Path,
) -> None:
    log = data_root / "log.jsonl"
    entries = [
        {"input_tokens": 100, "output_tokens": 5, "cost_usd": 0.5, "model": "m"},
        {
            "input_tokens": 300,
            "output_tokens": 0,
            "cost_usd": 1.0,
            "model": "m",
            "status": "failed",
        },
    ]
    log.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    guard = BudgetGuard.from_log(log_path=log)
    assert (guard.calls, guard.spent_usd, guard.input_tokens, guard.largest_call_usd) == (
        2,
        1.5,
        400,
        1.0,
    )
    assert BudgetGuard.from_log(log_path=data_root / "missing.jsonl").calls == 0


def test_the_stream_message_carries_each_image_before_the_text() -> None:
    message = json.loads(stream_message(prompt=_prompt()))
    content = message["message"]["content"]
    assert [part["type"] for part in content] == ["image", "text"]
    assert content[0]["source"]["media_type"] == "image/png"


def _stream(
    *,
    model: str | None = "claude-sonnet-5-5",
    is_error: bool = False,
    tools: list[str] | None = None,
    cwd: str = "/empty",
) -> str:
    init: dict[str, object] = {
        "type": "system",
        "tools": [] if tools is None else tools,
        "skills": [],
        "mcp_servers": [],
        "cwd": cwd,
    }
    result: dict[str, object] = {
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
    events: list[dict[str, object]] = [init, {"type": "system"}]
    if model is not None:
        events.append({"type": "assistant", "message": {"model": model}})
    events.append(result)
    return "\n".join(json.dumps(event) for event in events)


def test_the_stream_parser_sums_every_kind_of_input_token_and_reads_the_cost() -> None:
    reply = parse_stream(output=_stream(), requested_model="claude-sonnet-5-5")
    assert (reply.text, reply.input_tokens, reply.output_tokens, reply.cost_usd) == (
        "hello",
        732,
        9,
        0.05,
    )


def test_the_stream_parser_refuses_another_model_an_error_result_or_a_missing_model() -> None:
    with pytest.raises(ModelError, match="answered") as wrong:
        parse_stream(output=_stream(model="claude-haiku"), requested_model="claude-sonnet-5-5")
    assert wrong.value.reply is not None
    assert wrong.value.reply.cost_usd == 0.05
    with pytest.raises(ModelError, match="answered"):
        parse_stream(output=_stream(model=None), requested_model="claude-sonnet-5-5")
    with pytest.raises(ModelError, match="no successful result") as failed:
        parse_stream(output=_stream(is_error=True), requested_model="claude-sonnet-5-5")
    assert failed.value.reply is not None
    assert failed.value.reply.input_tokens == 732


def test_the_stream_parser_refuses_a_call_with_tools_or_in_the_wrong_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelError, match="tools"):
        parse_stream(output=_stream(tools=["Bash"]), requested_model="claude-sonnet-5-5")
    with pytest.raises(ModelError, match="empty directory"):
        parse_stream(
            output=_stream(cwd="/home/someone/repo"),
            requested_model="claude-sonnet-5-5",
            expected_cwd=tmp_path,
        )
    parse_stream(
        output=_stream(cwd=str(tmp_path)),
        requested_model="claude-sonnet-5-5",
        expected_cwd=tmp_path,
    )
    with pytest.raises(ModelError, match="no init"):
        parse_stream(
            output="\n".join(_stream().splitlines()[1:]), requested_model="claude-sonnet-5-5"
        )


def test_complete_runs_in_an_empty_directory_and_refuses_a_failed_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        directory = Path(str(kwargs["cwd"]))
        seen["cwd"], seen["was_empty"] = directory, not any(directory.iterdir())
        seen["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            command, 0, stdout=_stream(cwd=str(directory)), stderr=""
        )

    monkeypatch.setattr(capacity_llm.subprocess, "run", fake_run)
    reply = ClaudeCliModel().complete(prompt=_prompt())
    assert reply.text == "hello"
    assert seen["was_empty"] is True
    assert seen["cwd"] != Path.cwd()
    assert seen["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"  # ty: ignore[not-subscriptable]
    monkeypatch.setattr(
        capacity_llm.subprocess,
        "run",
        lambda command, **kw: subprocess.CompletedProcess(command, 1, "", "boom"),
    )
    with pytest.raises(ModelError, match="exited with 1"):
        ClaudeCliModel().complete(prompt=_prompt())


def test_a_billed_failure_is_recorded_in_the_guard_and_the_log_before_it_is_raised(
    data_root: Path,
) -> None:
    billed = ModelReply(text="", input_tokens=900, output_tokens=0, cost_usd=0.4, model="stub")

    def fail(prompt: Prompt) -> str:
        raise ModelError("wrong model", reply=billed)

    def timeout(prompt: Prompt) -> str:
        raise ModelError("timed out")

    for respond, expected_cost in ((fail, 0.4), (timeout, capacity_llm.PER_CALL_BUDGET_USD)):
        guard = BudgetGuard()
        store = CallStore(
            directory=data_root / f"c{expected_cost}",
            log_path=data_root / f"l{expected_cost}.jsonl",
        )
        with pytest.raises(ModelError):
            labelled_answer(
                store=store, model=StubModel(respond=respond), guard=guard, prompt=_prompt(),
                sample_index=0, label="t", plant="A", first_day=FIRST, last_day=LAST,
            )  # fmt: skip
        assert (guard.calls, guard.spent_usd) == (1, expected_cost)
        logged = [
            json.loads(line)
            for line in (data_root / f"l{expected_cost}.jsonl").read_text().splitlines()
        ]
        assert logged[0]["status"] == "failed"
        assert (
            BudgetGuard.from_log(log_path=data_root / f"l{expected_cost}.jsonl").spent_usd
            == expected_cost
        )


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
    monkeypatch.setattr(capacity_llm, "REPO_DATA_DIR", tmp_path / "data")
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


def test_a_later_round_counts_the_log_and_gets_only_its_own_allowance(data_root: Path) -> None:
    log = data_root / "log.jsonl"
    entries = [
        {"input_tokens": 1000, "output_tokens": 5, "cost_usd": 3.0, "model": "m"}
        for _ in range(138)
    ]
    entries = entries[:3]
    log.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    guard = BudgetGuard.for_round(
        log_path=log, round_usd=6.0, round_calls=10, round_input_tokens=5000
    )
    assert guard.spent_usd == 9.0
    assert guard.ceiling_usd == 15.0
    assert (guard.max_calls, guard.max_input_tokens) == (13, 8000)
    guard.check(prompt=_prompt())
    small = BudgetGuard.for_round(
        log_path=log, round_usd=2.5, round_calls=10, round_input_tokens=5000
    )
    assert small.ceiling_usd == 11.5
    with pytest.raises(BudgetExceeded, match="spent"):
        BudgetGuard.for_round(
            log_path=log, round_usd=1.5, round_calls=10, round_input_tokens=5000
        ).check(prompt=_prompt())
