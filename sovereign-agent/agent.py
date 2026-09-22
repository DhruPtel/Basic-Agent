"""
A minimal, self-contained analyst agent.

Give it a research task on the command line. It searches the web, reads the
pages it finds, computes figures with Python, writes an evidence-dense markdown
report, and saves it. Every event is traced to a JSONL file and the terminal.
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import traceback
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import anthropic
import requests
from anthropic.types import Message, ToolParam, ToolResultBlockParam, ToolUseBlock
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv


# --- Configuration ------------------------------------------------------------

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 16_000        # ceiling on a single reply; must exceed THINKING_BUDGET
THINKING_BUDGET = 4_000    # tokens Claude may spend reasoning before it answers
TOOL_TIMEOUT = 30          # seconds before a run_python call is killed
MAX_TURNS = 25             # research + writing needs more turns than plain coding
MAX_REPEATED_ERRORS = 3    # stop after this many identical failures in a row

FETCH_TIMEOUT = 20         # seconds before a fetch_url call gives up
MAX_FETCH_CHARS = 20_000   # trim fetched pages so context stays manageable
SEARCH_RESULTS = 6         # hits returned per web_search
USER_AGENT = "Mozilla/5.0 (compatible; sovereign-agent/1.0)"

COMPLETION_MARKER = "TASK COMPLETE"
TRACE_DIR = Path(__file__).parent / "traces"
REPORT_DIR = Path(__file__).parent / "reports"


# --- System prompt (edit freely) ----------------------------------------------

SYSTEM_PROMPT = f"""\
You are a research analyst. You gather real data from real sources, then write a
tight, evidence-dense report and save it.

Your tools:

- web_search  — find sources. Returns titles, URLs and snippets.
- fetch_url   — read a page's actual text. Snippets are not evidence; fetch.
- run_python  — compute. Growth rates, shares, ratios, sanity checks.
- write_report — save the finished markdown report.

Work in three phases, in order:

1. RESEARCH. Never write from memory — your training data is stale and your
   recollection of numbers is not a source. Search, then fetch the most
   promising pages and read them. Keep going until you hold specific, current
   figures: values, dates, percentages, totals. Derive anything comparative
   with run_python rather than eyeballing it.

2. WRITE. Draft the report in markdown, following the style rules below.

3. SAVE. Call write_report, then tell the user where it landed.

Style — analytical research prose:

- Every claim carries a specific number you actually fetched. "Adoption grew" is
  not a claim; "addresses grew 34% to 2.1M between January and August" is.
- Thesis first. Open each paragraph with the claim, then the evidence for it.
- Analytical, not descriptive. Say why a number matters — what it implies, what
  changed, what it pressures. A list of facts is not analysis.
- Dense. No filler, no throat-clearing, no hedging. Cut any sentence that
  carries neither a number nor an inference.
- Attribute key numbers inline with source and date, e.g.
  "(ethereum.org, fetched 2026-09-21)".
- If a number you need cannot be found, say so plainly in the report. Never
  estimate quietly or fill a gap from memory.

How to work:

- Nothing persists between run_python calls. Every snippet stands on its own,
  carries its own imports, and prints what you need to see.

- Errors, tracebacks, timeouts and dead links are information, not dead ends.
  Read the actual error, say what it tells you, and fix the cause. Never re-run
  identical code hoping for a different result. If a page will not load or a
  search returns nothing useful, try a different source or different wording.

- Verify before you conclude. Cross-check important figures against a second
  source or a second method, and say what you checked.

Finishing:

- When — and only when — the report is written, saved, and every claim in it is
  backed by a figure you fetched, end your final message with this marker on a
  line of its own:

      {COMPLETION_MARKER}

- Follow the marker with two or three sentences on what you found, which sources
  you used, and where the report was saved.

- Do not write the marker if the report is unsaved, thin on real data, or padded
  with claims you could not source. Say what is missing instead.
"""


# --- Tools --------------------------------------------------------------------

class ToolResult(NamedTuple):
    """Whether the call succeeded, and the text Claude gets to read."""
    ok: bool
    output: str


def run_python(code: str) -> ToolResult:
    """Run code in a fresh interpreter; failures come back as text Claude can read."""
    if not code.strip():
        return ToolResult(False, "Error: no code was provided.")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"Error: code did not finish within {TOOL_TIMEOUT}s.")

    parts = []
    if proc.stdout.strip():
        parts.append(proc.stdout.rstrip())
    if proc.stderr.strip():
        parts.append("stderr:\n" + proc.stderr.rstrip())
    if proc.returncode != 0:
        parts.append(f"exit code: {proc.returncode}")

    # The API rejects an empty tool_result, so always say something.
    output = "\n".join(parts) or "(no output — remember to print() your result)"
    return ToolResult(proc.returncode == 0, output)


def web_search(query: str) -> ToolResult:
    """Keyless DuckDuckGo search — titles, URLs and snippets."""
    if not query.strip():
        return ToolResult(False, "Error: the query was empty.")
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=SEARCH_RESULTS))
    except Exception as exc:
        return ToolResult(False, f"Error searching for {query!r}: {exc}")

    if not hits:
        return ToolResult(False, f"No results for {query!r}. Try different wording.")
    return ToolResult(True, "\n\n".join(
        f"{h.get('title', '')}\n{h.get('href', '')}\n{h.get('body', '')}" for h in hits))


def fetch_url(url: str) -> ToolResult:
    """Fetch a page and return its readable text, HTML stripped out."""
    if not url.startswith(("http://", "https://")):
        return ToolResult(False, "Error: url must start with http:// or https://")
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
    except requests.RequestException as exc:
        return ToolResult(False, f"Error fetching {url}: {exc}")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))

    if len(text) > MAX_FETCH_CHARS:
        text = text[:MAX_FETCH_CHARS] + f"\n\n[truncated at {MAX_FETCH_CHARS} chars]"
    return ToolResult(True, text or "(the page had no readable text)")


def write_report(title: str, markdown: str) -> ToolResult:
    """Save a markdown report to reports/ under a timestamped filename."""
    if not markdown.strip():
        return ToolResult(False, "Error: the report body was empty.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "report"
    path = REPORT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug[:60]}.md"
    path.write_text(markdown, encoding="utf-8")
    return ToolResult(True, f"Report saved to {path} ({len(markdown)} chars).")


def execute_tool(block: ToolUseBlock) -> ToolResult:
    """Route one tool call to its handler."""
    args = block.input
    if block.name == "run_python":
        return run_python(str(args.get("code", "")))
    if block.name == "web_search":
        return web_search(str(args.get("query", "")))
    if block.name == "fetch_url":
        return fetch_url(str(args.get("url", "")))
    if block.name == "write_report":
        return write_report(str(args.get("title", "")), str(args.get("markdown", "")))
    return ToolResult(False, f"Error: unknown tool {block.name!r}.")


def render_args(name: str, args: dict) -> str:
    """A readable one-look rendering of a tool call's arguments."""
    if name == "run_python":
        return str(args.get("code", ""))
    if name == "write_report":
        return f"{args.get('title', 'untitled')} — {len(str(args.get('markdown', '')))} chars"
    return " ".join(f"{k}={v!r}" for k, v in args.items())


def error_signature(output: str) -> str:
    """Last meaningful line of a failure — usually the exception and its message."""
    lines = [line.strip() for line in output.splitlines()
             if line.strip() and not line.strip().startswith("exit code:")]
    return lines[-1] if lines else output.strip()


def _tool(name: str, description: str, **properties: dict) -> ToolParam:
    """Build a tool schema where every property is required."""
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
        },
    }


TOOLS: list[ToolParam] = [
    _tool("web_search",
          "Search the web and get back titles, URLs and snippets. Snippets are "
          "leads, not evidence — fetch_url the promising ones to read them.",
          query={"type": "string", "description": "What to search for."}),
    _tool("fetch_url",
          "Fetch a web page and return its readable text with HTML stripped out.",
          url={"type": "string", "description": "Full http:// or https:// URL."}),
    _tool("run_python",
          "Execute a snippet of Python code and get back its output. Each call "
          "runs in a fresh interpreter, so nothing persists between calls — "
          "print() anything you want to see.",
          code={"type": "string", "description": "The Python source to run."}),
    _tool("write_report",
          "Save the finished markdown report to the reports/ directory.",
          title={"type": "string", "description": "Short title, used in the filename."},
          markdown={"type": "string", "description": "The full report in markdown."}),
]


# --- Tracing ------------------------------------------------------------------

class Trace:
    """Appends each event to a JSONL file and prints a readable version."""

    LABELS = {
        "task_start": "TASK",
        "thinking": "THINKING",
        "assistant_message": "CLAUDE",
        "tool_call": "TOOL CALL",
        "tool_result": "TOOL RESULT",
        "final_answer": "FINAL ANSWER",
        "task_complete": "TASK COMPLETE",
        "ended_without_marker": "ENDED — NO COMPLETION MARKER",
        "stuck": "STUCK",
        "limit_reached": "STOPPED",
        "error": "ERROR",
    }

    # Checked in order; the first one present is shown under the heading.
    BODY_FIELDS = ("text", "thinking", "output")

    def __init__(self) -> None:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.path = TRACE_DIR / f"run-{stamp}.jsonl"
        self._file = self.path.open("w", encoding="utf-8")

    def record(self, event: str, **data) -> None:
        """Write one structured entry and show it in the terminal."""
        entry = {"ts": datetime.now().astimezone().isoformat(), "event": event, **data}
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()  # a crashed run still leaves a usable trace
        self._print(entry)

    def _print(self, entry: dict) -> None:
        event = entry["event"]
        label = "TOOL ERROR" if entry.get("status") == "error" else self.LABELS.get(event, event.upper())
        if tool := entry.get("tool"):
            label += f": {tool}"
        if entry.get("after_error"):
            label += " [retry]"
        if turn := entry.get("turn"):
            label += f" (turn {turn})"

        body = next((entry[f] for f in self.BODY_FIELDS if entry.get(f)), "")
        print(f"--- {label} ---")
        if body:
            print(textwrap.indent(body, "    "))
        print()

    def close(self) -> None:
        self._file.close()


# --- The agent loop -----------------------------------------------------------

def split_completion(text: str) -> tuple[bool, str]:
    """Find the marker on a line of its own; return it and the summary that follows."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().strip("*#").strip() == COMPLETION_MARKER:  # tolerate **bold**
            return True, "\n".join(lines[i + 1:]).strip()
    return False, ""


def record_thinking(response: Message, trace: Trace, turn: int) -> None:
    """Trace the reasoning Claude produced before acting."""
    for block in response.content:
        if block.type == "thinking":
            trace.record("thinking", turn=turn, thinking=block.thinking)
        elif block.type == "redacted_thinking":
            trace.record("thinking", turn=turn, thinking="(redacted)")


def run_tool_call(block: ToolUseBlock, trace: Trace, turn: int,
                  retrying: bool) -> tuple[ToolResultBlockParam, ToolResult]:
    """Execute one tool call; return the block to send back and the raw outcome."""
    args = dict(block.input)
    trace.record("tool_call", turn=turn, tool=block.name, after_error=retrying,
                 args=args, text=render_args(block.name, args))

    result = execute_tool(block)
    trace.record("tool_result", turn=turn, tool=block.name,
                 status="ok" if result.ok else "error", output=result.output)

    sent: ToolResultBlockParam = {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": result.output,
        "is_error": not result.ok,  # tells Claude this was a failure
    }
    return sent, result


def run_agent(client: anthropic.Anthropic, task: str, trace: Trace) -> None:
    """Loop until Claude declares completion, gets stuck, or runs out of turns."""
    messages = [{"role": "user", "content": task}]
    trace.record("task_start", text=task, model=MODEL)

    last_error: str | None = None
    repeats = 0

    for turn in range(1, MAX_TURNS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
            messages=messages,
        )
        record_thinking(response, trace, turn)

        # No tool calls means Claude has stopped — did it declare completion?
        if response.stop_reason != "tool_use":
            text = "\n".join(b.text for b in response.content if b.type == "text")
            trace.record("final_answer", text=text, stop_reason=response.stop_reason)

            declared, summary = split_completion(text)
            if declared:
                trace.record("task_complete", ending="complete",
                             text=summary or "(marker given with no summary)")
            else:
                trace.record("ended_without_marker", ending="no_marker",
                             text="Stopped without declaring the task complete.")
            return

        # Replay the full turn — thinking blocks must go back verbatim.
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                trace.record("assistant_message", turn=turn, text=block.text)
            elif block.type == "tool_use":
                sent, result = run_tool_call(block, trace, turn, last_error is not None)
                tool_results.append(sent)

                # Count identical failures in a row; any success resets the count.
                if result.ok:
                    last_error, repeats = None, 0
                else:
                    signature = error_signature(result.output)
                    repeats = repeats + 1 if signature == last_error else 1
                    last_error = signature

        if repeats >= MAX_REPEATED_ERRORS:
            trace.record("stuck", ending="stopped_early",
                         text=f"Same failure {repeats} times in a row — "
                              f"stopping. Last error: {last_error}")
            return

        messages.append({"role": "user", "content": tool_results})

    trace.record("limit_reached", ending="stopped_early",
                 text=f"Hit the {MAX_TURNS}-turn limit with no final answer."
                      + (f" Last error: {last_error}" if last_error else ""))


# --- Entry point --------------------------------------------------------------

def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Put it in the .env file next to agent.py.")

    task = " ".join(sys.argv[1:]).strip()
    if not task:
        sys.exit('Usage: python agent.py "your task here"')

    client = anthropic.Anthropic()
    trace = Trace()
    try:
        run_agent(client, task, trace)
    except Exception as exc:  # any failure becomes a trace event, not a traceback
        trace.record("error", ending="error",
                     text=f"{type(exc).__name__}: {exc}", detail=traceback.format_exc())
    finally:
        trace.close()
        print(f"Trace saved to {trace.path}")


if __name__ == "__main__":
    main()
