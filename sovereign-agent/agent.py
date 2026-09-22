"""
The reusable agent: a think → act → observe loop with tools, tracing and
self-correction. Each role (analyst, hunter, verifier) is an instance of Agent
with its own system prompt, tools and trace file.

Run on its own, this file is the original single analyst agent:
    python agent.py "your research task"
"""

import json
import os
import sys
import textwrap
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import anthropic
from anthropic.types import Message, ToolParam, ToolResultBlockParam, ToolUseBlock
from dotenv import load_dotenv

from tools import Tool, ToolResult, brief_args, error_signature, render_args


# --- Configuration ------------------------------------------------------------

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 16_000        # ceiling on a single reply; must exceed THINKING_BUDGET
THINKING_BUDGET = 4_000    # tokens Claude may spend reasoning before it answers
MAX_TURNS = 25             # default per agent; roles can override
MAX_REPEATED_ERRORS = 3    # stop after this many identical failures in a row

COMPLETION_MARKER = "TASK COMPLETE"
TRACE_DIR = Path(__file__).parent / "traces"
VERBOSE = os.environ.get("VERBOSE", "") not in ("", "0")  # VERBOSE=1: full detail in quiet-by-default runs


# --- Tracing ------------------------------------------------------------------

_print_lock = threading.Lock()  # agents run in parallel; keep printed blocks whole


class Trace:
    """Appends each event to a JSONL file and prints a readable, role-labelled version."""

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
        # Orchestrator events.
        "plan": "PLAN",
        "handoff": "HANDOFF →",
        "handback": "HANDBACK ←",
        "merge": "MERGE",
        "report": "REPORT",
    }

    # Checked in order; the first one present is shown under the heading.
    BODY_FIELDS = ("text", "thinking", "output")

    # Quiet mode prints only these, one line each; the file always gets everything.
    QUIET_EVENTS = {"plan", "handoff", "handback", "merge", "report", "tool_call", "run_end"}

    def __init__(self, path: Path, label: str = "", quiet: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.label = label
        self.quiet = quiet
        self._file = path.open("w", encoding="utf-8")

    def record(self, event: str, **data) -> None:
        """Write one structured entry and show it in the terminal."""
        entry = {"ts": datetime.now().astimezone().isoformat(), "event": event,
                 "agent": self.label or None, **data}
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()  # a crashed run still leaves a usable trace
        self._print(entry)

    def _print(self, entry: dict) -> None:
        event = entry["event"]
        if self.quiet:
            return self._print_brief(entry) if event in self.QUIET_EVENTS else None
        if event == "run_end":  # verbose output already showed the ending
            return
        label = "TOOL ERROR" if entry.get("status") == "error" else self.LABELS.get(event, event.upper())
        if tool := entry.get("tool"):
            label += f": {tool}"
        if peer := entry.get("peer"):
            label += f" {peer}"
        if entry.get("after_error"):
            label += " [retry]"
        if turn := entry.get("turn"):
            label += f" (turn {turn})"
        if self.label:
            label = f"[{self.label}] {label}"

        body = next((entry[f] for f in self.BODY_FIELDS if entry.get(f)), "")
        with _print_lock:
            print(f"--- {label} ---")
            if body:
                print(textwrap.indent(body, "    "))
            print(flush=True)

    def _print_brief(self, entry: dict) -> None:
        """One milestone line, e.g. "[hunter-1] web_search (turn 3): 'query'"."""
        prefix = f"[{self.label}] " if self.label else ""
        if entry["event"] == "tool_call":
            line = f"{entry['tool']} (turn {entry['turn']}): {brief_args(entry['tool'], entry['args'])}"
        elif entry["event"] == "run_end":
            line = entry["text"]
        else:
            line = self.LABELS[entry["event"]] + (f" {entry['peer']}" if entry.get("peer") else "")
            text = entry.get("text", "")
            if "\n" in text:  # e.g. the PLAN's list of angles
                line += ":\n" + textwrap.indent(text, "    ")
            elif text:
                line += f": {text}"
        with _print_lock:
            print(prefix + line, flush=True)

    def close(self) -> None:
        self._file.close()


# --- The agent ----------------------------------------------------------------

class RunResult(NamedTuple):
    """How a run ended, Claude's closing summary, and any items it submitted."""
    ending: str        # complete | no_marker | stopped_early | error
    summary: str
    items: list[dict]


def split_completion(text: str) -> tuple[bool, str]:
    """Find the marker on a line of its own; return it and the summary that follows."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().strip("*#").strip() == COMPLETION_MARKER:  # tolerate **bold**
            return True, "\n".join(lines[i + 1:]).strip()
    return False, ""


class Agent:
    """One Claude-driven worker: loops until it declares completion, gets stuck, or runs out of turns."""

    def __init__(self, client: anthropic.Anthropic, role: str, system_prompt: str,
                 tools: list[Tool], trace: Trace, *, output_tool: ToolParam | None = None,
                 max_turns: int = MAX_TURNS) -> None:
        self.client = client
        self.role = role
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in tools}
        self.trace = trace
        self.output_tool = output_tool  # optional submit tool for structured results
        self.max_turns = max_turns
        self.items: list[dict] = []     # everything submitted through output_tool

    @property
    def schemas(self) -> list[ToolParam]:
        schemas = [t.schema for t in self.tools.values()]
        return schemas + [self.output_tool] if self.output_tool else schemas

    def run(self, task: str) -> RunResult:
        """Run the loop on one task; any crash becomes a traced 'error' ending."""
        try:
            ending, summary = self._loop(task)
        except Exception as exc:
            summary = f"{type(exc).__name__}: {exc}"
            self.trace.record("error", ending="error", text=summary,
                              detail=traceback.format_exc())
            ending = "error"
        count = f" — {len(self.items)} items" if self.output_tool else ""
        self.trace.record("run_end", ending=ending, items=len(self.items), text=ending + count)
        return RunResult(ending, summary, self.items)

    def _loop(self, task: str) -> tuple[str, str]:
        messages = [{"role": "user", "content": task}]
        self.trace.record("task_start", text=task, model=MODEL, role=self.role)

        last_error: str | None = None
        repeats = 0

        for turn in range(1, self.max_turns + 1):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                tools=self.schemas,
                thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
                messages=messages,
            )
            self._record_thinking(response, turn)

            # No tool calls means Claude has stopped — did it declare completion?
            if response.stop_reason != "tool_use":
                return self._finish(response)

            # Replay the full turn — thinking blocks must go back verbatim.
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    self.trace.record("assistant_message", turn=turn, text=block.text)
                elif block.type == "tool_use":
                    sent, result = self._run_tool_call(block, turn, last_error is not None)
                    tool_results.append(sent)

                    # Count identical failures in a row; any success resets the count.
                    if result.ok:
                        last_error, repeats = None, 0
                    else:
                        signature = error_signature(result.output)
                        repeats = repeats + 1 if signature == last_error else 1
                        last_error = signature

            if repeats >= MAX_REPEATED_ERRORS:
                summary = f"Same failure {repeats} times in a row — stopping. Last error: {last_error}"
                self.trace.record("stuck", ending="stopped_early", text=summary)
                return "stopped_early", summary

            messages.append({"role": "user", "content": tool_results})

        summary = (f"Hit the {self.max_turns}-turn limit with no final answer."
                   + (f" Last error: {last_error}" if last_error else ""))
        self.trace.record("limit_reached", ending="stopped_early", text=summary)
        return "stopped_early", summary

    def _finish(self, response: Message) -> tuple[str, str]:
        """Trace the final answer and classify how the run ended."""
        text = "\n".join(b.text for b in response.content if b.type == "text")
        self.trace.record("final_answer", text=text, stop_reason=response.stop_reason)

        declared, summary = split_completion(text)
        if declared:
            summary = summary or "(marker given with no summary)"
            self.trace.record("task_complete", ending="complete", text=summary)
            return "complete", summary
        self.trace.record("ended_without_marker", ending="no_marker",
                          text="Stopped without declaring the task complete.")
        return "no_marker", text

    def _record_thinking(self, response: Message, turn: int) -> None:
        """Trace the reasoning Claude produced before acting."""
        for block in response.content:
            if block.type == "thinking":
                self.trace.record("thinking", turn=turn, thinking=block.thinking)
            elif block.type == "redacted_thinking":
                self.trace.record("thinking", turn=turn, thinking="(redacted)")

    def _run_tool_call(self, block: ToolUseBlock, turn: int,
                       retrying: bool) -> tuple[ToolResultBlockParam, ToolResult]:
        """Execute one tool call; return the block to send back and the raw outcome."""
        args = dict(block.input)
        self.trace.record("tool_call", turn=turn, tool=block.name, after_error=retrying,
                          args=args, text=render_args(block.name, args))

        result = self._execute(block.name, args)
        self.trace.record("tool_result", turn=turn, tool=block.name,
                          status="ok" if result.ok else "error", output=result.output)

        sent: ToolResultBlockParam = {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result.output,
            "is_error": not result.ok,  # tells Claude this was a failure
        }
        return sent, result

    def _execute(self, name: str, args: dict) -> ToolResult:
        """Route one tool call to its handler, or collect a structured submission."""
        if self.output_tool and name == self.output_tool["name"]:
            items = [i for i in args.get("items", []) if isinstance(i, dict)]
            if not items:
                return ToolResult(False, "Error: no items were submitted.")
            self.items.extend(items)
            return ToolResult(True, f"Received {len(items)} items ({len(self.items)} total so far).")
        if tool := self.tools.get(name):
            return tool.handler(args)
        return ToolResult(False, f"Error: unknown tool {name!r}.")


# --- Entry point: the single analyst ------------------------------------------

def make_client() -> anthropic.Anthropic:
    """Load .env and build the API client, or exit with a clear message."""
    load_dotenv(Path(__file__).parent / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Put it in the .env file next to agent.py.")
    return anthropic.Anthropic()


def main() -> None:
    from roles import make_analyst  # imported here: roles.py builds on this module

    task = " ".join(sys.argv[1:]).strip()
    if not task:
        sys.exit('Usage: python agent.py "your task here"')

    client = make_client()
    trace = Trace(TRACE_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl")
    try:
        make_analyst(client, trace).run(task)
    finally:
        trace.close()
        print(f"Trace saved to {trace.path}")


if __name__ == "__main__":
    main()
