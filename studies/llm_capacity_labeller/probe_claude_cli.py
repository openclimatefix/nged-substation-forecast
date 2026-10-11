"""The 2-call probe of `claude -p`: images, token counts, the model flag, and the price of a call.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.

**The probe makes two calls, about US$0.05 in all.** The first sends a 200 by 100 pixel line chart
as a base64 image and asks for the value at the line's right-hand end (the answer is 5). The second
sends an 18,016-character numeric table and asks for the number of data rows (1,080). Both run
through `studies.capacity_llm.ClaudeCliModel`'s flags: an empty directory, no tools, no settings.

With `--summarise` the script makes no call and prints the table from the two saved outputs under
`probe/` in the study's data folder, which is how the page's probe numbers are produced.

Run it with `uv run python studies/llm_capacity_labeller/probe_claude_cli.py --summarise`.
"""

import argparse
import random
import subprocess
import sys
import tempfile

import altair as alt
import polars as pl
import vl_convert as vlc
from labeller_paths import STUDY_DIR
from studies.capacity_labeller import Prompt
from studies.capacity_llm import MODEL_ID, ClaudeCliModel, parse_stream, stream_message

PROBE_DIR = STUDY_DIR / "probe"
"""Where the two saved outputs, and the command-line tool's version, live."""

IMAGE_FILE = "image_call.jsonl"
TABLE_FILE = "table_call.jsonl"


def _table_text() -> str:
    """Build the probe's numeric table: 1,080 rows of month, hour, and a ratio."""
    random.seed(1)
    rows = ["month,hour,ratio"] + [
        f"2024-{month:02d},{hour},{random.uniform(0.8, 1.1):.3f}"
        for month in range(1, 13)
        for hour in range(5, 20)
        for _ in range(6)
    ]
    return "\n".join(rows)


def run_calls() -> None:
    """Make the two calls and save their raw outputs."""
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    chart = alt.Chart(pl.DataFrame({"x": [1, 2, 3, 4], "y": [1, 3, 2, 5]})).mark_line()
    png = vlc.vegalite_to_png(
        chart.encode(x="x:Q", y="y:Q").properties(width=200, height=100).to_dict(),
        scale=1,  # ty: ignore[unresolved-attribute]
    )
    image_prompt = Prompt(
        system="You read charts. Answer briefly.",
        text="What is the y value at the right-hand end of the line? Answer with one number.",
        images=(png,),
    )
    table_prompt = Prompt(
        system="You read tables. Answer briefly.",
        text=_table_text() + '\n\nReply with JSON only: {"n_rows": <number of data rows>}',
        images=(),
    )

    model = ClaudeCliModel()
    for prompt, name in ((image_prompt, IMAGE_FILE), (table_prompt, TABLE_FILE)):
        with tempfile.TemporaryDirectory() as empty:
            done = subprocess.run(
                model.command(prompt=prompt),
                input=stream_message(prompt=prompt),
                capture_output=True,
                text=True,
                cwd=empty,
                check=True,
            )
        (PROBE_DIR / name).write_text(done.stdout)
    (PROBE_DIR / "cli_version.txt").write_text(
        subprocess.run(["claude", "--version"], capture_output=True, text=True, check=True).stdout
    )


def summarise() -> pl.DataFrame:
    """Read the two saved outputs into one row each.

    Returns:
        One row per call: the answer, the tokens, the cost, and the characters per token.
    """
    rows = []
    for name, characters in ((IMAGE_FILE, None), (TABLE_FILE, len(_table_text()) + 66)):
        reply = parse_stream(output=(PROBE_DIR / name).read_text(), requested_model=MODEL_ID)
        rows.append(
            {
                "call": name.removesuffix(".jsonl"),
                "answer": reply.text.strip()[:40],
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "cost_usd": reply.cost_usd,
                "characters_per_input_token": None
                if characters is None
                else characters / reply.input_tokens,
            }
        )
    return pl.DataFrame(rows)


def main() -> int:
    """Make the calls or summarise the saved ones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summarise", action="store_true", help="Print the saved results; no call."
    )
    arguments = parser.parse_args()
    if not arguments.summarise:
        run_calls()
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(summarise())
    print((PROBE_DIR / "cli_version.txt").read_text().strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
