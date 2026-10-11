"""Call a language model for the capacity labeller, with a cache, a log, and a spending guard.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.

**Every call goes through the `LabellerModel` interface, so tests and smoke runs use a stub and
never touch the network.** `ClaudeCliModel` runs Claude Code's print mode (`claude -p`) from an
empty temporary directory with every tool disabled, no settings sources, no skills, no MCP
servers, no session persistence, and the study's fixed system prompt in place of Claude Code's
own. The model therefore has no file to open and no memory to read. The command-line tool has no
temperature flag, so every arm shares its default sampling.

**`BudgetGuard` stops the run before a call that could take it past a ceiling.** It tracks the cost
the command-line tool reports for each call, the call count, and the input tokens, and raises
`BudgetExceeded` before the call that would cross a limit.

**`CallStore` keeps one JSON file per call, keyed by `studies.capacity_labeller.cache_key`, and an
append-only log of every prompt and answer.** The log pairs plant letters with dates that must not
leave the private data store, so its path must lie under `data/` (`assert_under_data`).
"""

import base64
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, Protocol, Self

from studies.capacity_labeller import (
    Answer,
    AnswerError,
    Prompt,
    cache_key,
    parse_answer,
)
from studies.sources import REPO_DATA_DIR

COST_CEILING_USD: Final[float] = 15.0
"""The most the run may spend, in US dollars, as the command-line tool reports cost."""

MAX_CALLS: Final[int] = 150
"""The most calls the run may make, retries included."""

MAX_INPUT_TOKENS: Final[int] = 2_000_000
"""The most input tokens the run may send."""

CHARACTERS_PER_TOKEN: Final[float] = 1.4
"""Numeric tables cost about this many characters per token (the probe: 18,016 characters, 12,557
tokens)."""

TOKENS_PER_IMAGE: Final[int] = 800
"""A generous token estimate for one case-file chart."""

MODEL_ID: Final[str] = "claude-sonnet-5-5"
"""The model every real call requests."""

PER_CALL_BUDGET_USD: Final[float] = 1.0
"""The command-line tool's own per-call cap. The guard projects the next call at this cost or the
largest cost seen, whichever is larger, so the run cannot reach `COST_CEILING_USD`."""

CALL_TIMEOUT_SECONDS: Final[int] = 900
"""How long one call may take."""

CLI_FLAGS: Final[tuple[str, ...]] = (
    "-p",
    "--input-format",
    "stream-json",
    "--output-format",
    "stream-json",
    "--verbose",
    "--tools",
    "",
    "--setting-sources",
    "",
    "--disable-slash-commands",
    "--strict-mcp-config",
    "--no-session-persistence",
)
"""The flags every call passes, besides the model, the system prompt, and the per-call budget."""


class BudgetExceeded(RuntimeError):
    """The next call would take the run past a ceiling."""


class ModelError(RuntimeError):
    """The model call failed or returned something unusable.

    The `reply` attribute holds what the call was billed, when the command-line tool reported a
    result before the failure was found, and `None` when nothing is known, which the caller counts
    at the per-call cap.
    """

    def __init__(self, message: str, *, reply: ModelReply | None = None) -> None:
        """Hold the message and any billed reply.

        Args:
            message: What went wrong.
            reply: The billed reply, if one was reported.
        """
        super().__init__(message)
        self.reply = reply


@dataclass(frozen=True)
class ModelReply:
    """What one call returned.

    Attributes:
        text: The reply text.
        input_tokens: All input tokens, cached or not.
        output_tokens: All output tokens, thinking included.
        cost_usd: The cost the command-line tool reports; 0 for a stub.
        model: The model that answered.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str


class LabellerModel(Protocol):
    """Something that answers a prompt."""

    @property
    def model_id(self) -> str:
        """The exact model identifier, part of the cache key."""
        ...

    def complete(self, *, prompt: Prompt) -> ModelReply:
        """Answer one prompt."""
        ...


@dataclass(frozen=True)
class StubModel:
    """A model that answers from a function, for tests and smoke runs.

    Attributes:
        respond: Maps a prompt to reply text.
        model_id: The identifier stored in the cache key.
    """

    respond: Callable[[Prompt], str]
    model_id: str = "stub"

    def complete(self, *, prompt: Prompt) -> ModelReply:
        """Return the function's reply with token counts estimated from the prompt's size."""
        return ModelReply(
            text=self.respond(prompt),
            input_tokens=estimate_input_tokens(prompt=prompt),
            output_tokens=100,
            cost_usd=0.0,
            model=self.model_id,
        )


def estimate_input_tokens(*, prompt: Prompt) -> int:
    """Estimate a prompt's input tokens from its characters and images.

    Args:
        prompt: The prompt.

    Returns:
        The estimate, deliberately on the high side.
    """
    characters = len(prompt.system) + len(prompt.text)
    return int(characters / CHARACTERS_PER_TOKEN) + TOKENS_PER_IMAGE * len(prompt.images)


def stream_message(*, prompt: Prompt) -> str:
    """Build the one `stream-json` input line: the chart images, then the case-file text.

    Args:
        prompt: The prompt.

    Returns:
        A JSON line.
    """
    content: list[dict[str, object]] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(image).decode(),
            },
        }
        for image in prompt.images
    ]
    content.append({"type": "text", "text": prompt.text})
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


def _billed_reply(*, result: dict[str, object], model: str) -> ModelReply:
    """Read the tokens and cost of a result event, tolerating a missing field."""
    usage = result.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return ModelReply(
        text=str(result.get("result", "")),
        input_tokens=int(
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        ),
        output_tokens=int(usage.get("output_tokens", 0)),
        cost_usd=float(result.get("total_cost_usd", 0.0)),  # ty: ignore[invalid-argument-type]
        model=model,
    )


def _check_isolation(*, init: dict[str, object] | None, expected_cwd: Path | None) -> None:
    """Raise unless the init event shows no tools, skills, or MCP servers, and the empty cwd."""
    if init is None:
        msg = "the output holds no init event, so the isolation cannot be checked"
        raise ModelError(msg)
    for key in ("tools", "skills", "mcp_servers"):
        if init.get(key) != []:
            msg = f"the init event shows {key} = {init.get(key)!r}, not an empty list"
            raise ModelError(msg)
    if expected_cwd is not None and Path(str(init.get("cwd"))).resolve() != expected_cwd.resolve():
        msg = f"the call ran in {init.get('cwd')}, not the empty directory {expected_cwd}"
        raise ModelError(msg)


def parse_stream(
    *, output: str, requested_model: str, expected_cwd: Path | None = None
) -> ModelReply:
    """Read the command-line tool's `stream-json` output.

    Args:
        output: The standard output, one JSON object per line.
        requested_model: The model the call asked for.
        expected_cwd: The empty directory the call ran in, which the init event must name, or
            `None` to skip that check.

    Returns:
        The reply.

    Raises:
        ModelError: If there is no init event showing no tools, skills, or MCP servers (and the
            expected directory), no successful result, no assistant event, or a served model other
            than the one requested. A failure after a result was reported carries that result's
            billed reply.
    """
    result: dict[str, object] | None = None
    init: dict[str, object] | None = None
    served: set[str] = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "system" and "tools" in event:
            init = event
        elif event.get("type") == "assistant":
            served.add(str(event["message"].get("model")))
        elif event.get("type") == "result":
            result = event
    billed = None if result is None else _billed_reply(result=result, model=requested_model)
    if result is None or result.get("is_error"):
        msg = f"the model call returned no successful result: {str(result)[:300]}"
        raise ModelError(msg, reply=billed)
    try:
        _check_isolation(init=init, expected_cwd=expected_cwd)
    except ModelError as error:
        raise ModelError(str(error), reply=billed) from error
    if not served or served - {requested_model}:
        msg = f"asked for {requested_model} but {sorted(served)} answered"
        raise ModelError(msg, reply=billed)
    assert billed is not None
    return billed


@dataclass(frozen=True)
class ClaudeCliModel:
    """Claude Code's print mode, isolated: an empty directory and no tools.

    Attributes:
        model_id: The model to request.
    """

    model_id: str = MODEL_ID

    def command(self, *, prompt: Prompt) -> list[str]:
        """Return the command line, with the system prompt and the per-call budget."""
        return [
            "claude",
            *CLI_FLAGS,
            "--model",
            self.model_id,
            "--system-prompt",
            prompt.system,
            "--max-budget-usd",
            str(PER_CALL_BUDGET_USD),
        ]

    def run_raw(self, *, prompt: Prompt, directory: Path) -> str:
        """Run the command-line tool in `directory` and return its standard output.

        Args:
            prompt: The prompt.
            directory: An empty directory to run in.

        Returns:
            The `stream-json` output.

        Raises:
            ModelError: If the tool times out or exits with an error. A call that fails this way
                may still have been billed, so the error carries no reply and the caller counts
                it at the per-call cap.
        """
        environment = {**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
        try:
            done = subprocess.run(
                self.command(prompt=prompt),
                input=stream_message(prompt=prompt),
                capture_output=True,
                text=True,
                cwd=directory,
                env=environment,
                timeout=CALL_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            msg = f"the model call timed out after {CALL_TIMEOUT_SECONDS} s"
            raise ModelError(msg) from error
        if done.returncode != 0:
            msg = f"claude exited with {done.returncode}: {done.stderr[:300]}"
            raise ModelError(msg)
        return done.stdout

    def complete(self, *, prompt: Prompt) -> ModelReply:
        """Answer one prompt by running the command-line tool in an empty temporary directory.

        Args:
            prompt: The prompt.

        Returns:
            The reply, with the tokens and cost the tool reports.
        """
        with tempfile.TemporaryDirectory(prefix="labeller_empty_") as empty:
            output = self.run_raw(prompt=prompt, directory=Path(empty))
            return parse_stream(
                output=output, requested_model=self.model_id, expected_cwd=Path(empty)
            )


@dataclass
class BudgetGuard:
    """Refuses a call that could cross the cost, call-count, or input-token ceiling.

    Attributes:
        ceiling_usd: The cost ceiling.
        max_calls: The call-count ceiling.
        max_input_tokens: The input-token ceiling.
        spent_usd: The cost reported so far.
        calls: The calls made so far.
        input_tokens: The input tokens sent so far.
        largest_call_usd: The largest single-call cost seen so far.
    """

    ceiling_usd: float = COST_CEILING_USD
    max_calls: int = MAX_CALLS
    max_input_tokens: int = MAX_INPUT_TOKENS
    spent_usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    largest_call_usd: float = field(default=0.0)

    @classmethod
    def from_log(cls, *, log_path: Path) -> Self:
        """Start a guard from what an earlier run already spent, so a resumed run cannot overspend.

        Args:
            log_path: The call log, which may not exist yet.

        Returns:
            A guard whose totals are the log's cost, calls, input tokens, and largest cost.
        """
        guard = cls()
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                entry = json.loads(line)
                guard.record(
                    reply=ModelReply(
                        text="",
                        input_tokens=int(entry["input_tokens"]),
                        output_tokens=int(entry["output_tokens"]),
                        cost_usd=float(entry["cost_usd"]),
                        model=str(entry["model"]),
                    )
                )
        return guard

    @classmethod
    def for_round(
        cls, *, log_path: Path, round_usd: float, round_calls: int, round_input_tokens: int
    ) -> Self:
        """Start a guard for a later round that counts everything already in the log.

        The study's total ceiling still holds (`COST_CEILING_USD`), and the round gets its own
        smaller allowance on top of what was spent before it.

        Args:
            log_path: The call log, holding every earlier round's calls.
            round_usd: The most this round may spend.
            round_calls: The most calls this round may make, retries included.
            round_input_tokens: The most input tokens this round may send.

        Returns:
            A guard whose totals are the log's and whose limits are the earlier totals plus the
            round's allowance, with the cost limit never above `COST_CEILING_USD`.
        """
        guard = cls.from_log(log_path=log_path)
        guard.ceiling_usd = min(COST_CEILING_USD, guard.spent_usd + round_usd)
        guard.max_calls = guard.calls + round_calls
        guard.max_input_tokens = guard.input_tokens + round_input_tokens
        return guard

    def check(self, *, prompt: Prompt) -> None:
        """Raise before a call that could cross a ceiling.

        Args:
            prompt: The prompt about to be sent.

        Raises:
            BudgetExceeded: If the call count, the input tokens, or the projected cost (spent plus
                the per-call cap or the largest call so far, whichever is larger) would reach a
                ceiling.
        """
        if self.calls + 1 > self.max_calls:
            msg = f"the next call would be number {self.calls + 1}, past the {self.max_calls} cap"
            raise BudgetExceeded(msg)
        projected_tokens = self.input_tokens + estimate_input_tokens(prompt=prompt)
        if projected_tokens > self.max_input_tokens:
            msg = (
                f"the next call would take input tokens to {projected_tokens}, "
                f"past {self.max_input_tokens}"
            )
            raise BudgetExceeded(msg)
        next_cost = max(self.largest_call_usd, PER_CALL_BUDGET_USD)
        if self.spent_usd + next_cost >= self.ceiling_usd:
            msg = (
                f"spent ${self.spent_usd:.2f}; another call of ${next_cost:.2f} "
                f"would reach ${self.ceiling_usd:.2f}"
            )
            raise BudgetExceeded(msg)

    def record(self, *, reply: ModelReply) -> None:
        """Add one finished call to the totals.

        Args:
            reply: The call's reply.
        """
        self.calls += 1
        self.spent_usd += reply.cost_usd
        self.input_tokens += reply.input_tokens
        self.largest_call_usd = max(self.largest_call_usd, reply.cost_usd)


def assert_under_data(*, path: Path) -> None:
    """Raise unless a path lies under the repository's `data/` directory (`REPO_DATA_DIR`).

    Args:
        path: The path.

    Raises:
        ValueError: If the path is elsewhere, because the log pairs plant letters with dates and
            must stay in the private data store.
    """
    if REPO_DATA_DIR.resolve() not in path.resolve().parents:
        msg = f"{path} is not under {REPO_DATA_DIR}; the labeller log must stay there"
        raise ValueError(msg)


class CallStore:
    """One JSON file per call under `directory`, and an append-only log at `log_path`."""

    def __init__(self, *, directory: Path, log_path: Path) -> None:
        """Create the store.

        Args:
            directory: Where the per-call files live; created if need be.
            log_path: The log file; must lie under `data/`.
        """
        assert_under_data(path=directory)
        assert_under_data(path=log_path)
        directory.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.log_path = log_path

    def get(self, *, key: str) -> ModelReply | None:
        """Return the stored reply for a key, or `None`."""
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        return ModelReply(**json.loads(path.read_text()))

    def put(
        self, *, key: str, reply: ModelReply, prompt: Prompt, sample_index: int, label: str
    ) -> None:
        """Store a reply and append the full prompt and answer to the log.

        Args:
            key: The cache key.
            reply: The reply.
            prompt: The prompt that produced it.
            sample_index: The sample index (with any retry offset).
            label: A short description of the call, such as the record, plant, and arm.
        """
        path = self.directory / f"{key}.json"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(reply.__dict__))
        temporary.rename(path)
        entry: dict[str, object] = {
            "time": datetime.now(UTC).isoformat(),
            "label": label,
            "key": key,
            "model": reply.model,
            "flags": list(CLI_FLAGS),
            "sample_index": sample_index,
            "system": prompt.system,
            "prompt": prompt.text,
            "image_sha256": [_sha256(image) for image in prompt.images],
            "answer": reply.text,
            "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens,
            "cost_usd": reply.cost_usd,
        }
        self.append_log(entry=entry)

    def append_log(self, *, entry: dict[str, object]) -> None:
        """Append one entry to the log.

        Args:
            entry: A JSON-ready mapping with at least `input_tokens`, `output_tokens`,
                `cost_usd`, and `model`, which `BudgetGuard.from_log` sums.
        """
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

    def log_failure(
        self, *, label: str, prompt: Prompt, error: ModelError, billed: ModelReply, model: str
    ) -> None:
        """Log a call that failed after it may have been billed, so a resume still counts it.

        Args:
            label: A short description of the call.
            prompt: The prompt that was sent.
            error: What went wrong.
            billed: What the call is counted as: the reported result, or the per-call cap.
            model: The model requested.
        """
        self.append_log(
            entry={
                "time": datetime.now(UTC).isoformat(),
                "label": label,
                "status": "failed",
                "error": str(error)[:500],
                "model": model,
                "prompt": prompt.text,
                "image_sha256": [_sha256(image) for image in prompt.images],
                "input_tokens": billed.input_tokens,
                "output_tokens": billed.output_tokens,
                "cost_usd": billed.cost_usd,
            }
        )


def _sha256(data: bytes) -> str:
    """Return the hexadecimal SHA-256 of some bytes."""
    return hashlib.sha256(data).hexdigest()


MAX_ATTEMPTS: Final[int] = 2
"""A reply that fails validation is asked for once more; each attempt counts as a call."""

RETRY_OFFSET: Final[int] = 1000
"""Added to the sample index in the cache key of a retry, so a retry is a new call."""


def labelled_answer(
    *,
    store: CallStore,
    model: LabellerModel,
    guard: BudgetGuard,
    prompt: Prompt,
    sample_index: int,
    label: str,
    plant: str,
    first_day: date,
    last_day: date,
) -> Answer:
    """Return a validated answer, from the cache if the call was made before.

    Args:
        store: The call store.
        model: The model.
        guard: The budget guard; a cached call costs nothing.
        prompt: The prompt.
        sample_index: Which sample of this prompt this is.
        label: A description for the log.
        plant: The plant the case file is about.
        first_day: The first date of the period.
        last_day: The last date of the period.

    Returns:
        The validated answer.

    Raises:
        AnswerError: If every attempt returns an invalid answer.
        BudgetExceeded: From the guard.
    """
    last_error: AnswerError | None = None
    for attempt in range(MAX_ATTEMPTS):
        key = cache_key(
            model=model.model_id, prompt=prompt, sample_index=sample_index + RETRY_OFFSET * attempt
        )
        reply = store.get(key=key)
        if reply is None:
            guard.check(prompt=prompt)
            try:
                reply = model.complete(prompt=prompt)
            except ModelError as error:
                billed = error.reply or ModelReply(
                    text="",
                    input_tokens=estimate_input_tokens(prompt=prompt),
                    output_tokens=0,
                    cost_usd=PER_CALL_BUDGET_USD,
                    model=model.model_id,
                )
                guard.record(reply=billed)
                store.log_failure(
                    label=label, prompt=prompt, error=error, billed=billed, model=model.model_id
                )
                raise
            guard.record(reply=reply)
            store.put(
                key=key,
                reply=reply,
                prompt=prompt,
                sample_index=sample_index + RETRY_OFFSET * attempt,
                label=label,
            )
        try:
            return parse_answer(
                text=reply.text, plant=plant, first_day=first_day, last_day=last_day
            )
        except AnswerError as error:
            last_error = error
    assert last_error is not None
    raise last_error
