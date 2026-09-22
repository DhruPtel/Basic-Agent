"""
sovereign-agent — Step 6: a self-correcting agent that knows when it's done.

Takes a task, lets Claude reason with extended thinking and run Python to work
on it, and loops until Claude answers without calling a tool. Failures are fed
back as diagnosable information, and the run ends when the agent verifies its
work and declares completion. Every event lands in a JSONL trace file.
"""

# --- Standard library ---
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

# --- Third-party ---
from dotenv import load_dotenv
import anthropic


# --- Configuration ------------------------------------------------------------

# Model to send every request to.
MODEL = "claude-sonnet-4-5"

# Ceiling on reply length, not a target. Must exceed THINKING_BUDGET.
MAX_TOKENS = 16000

# Tokens Claude may spend reasoning before it answers.
THINKING_BUDGET = 4000

# Seconds a single run_python call may take before it's killed.
TOOL_TIMEOUT = 30

# Safety net so a misbehaving loop can't run forever.
MAX_TURNS = 10

# Give up after this many identical failures in a row.
MAX_REPEATED_ERRORS = 3

# What the agent writes on its own line to declare the task finished.
COMPLETION_MARKER = "TASK COMPLETE"

# Where JSONL trace files are written.
TRACE_DIR = Path(__file__).parent / "traces"


# ==============================================================================
# SYSTEM PROMPT — the agent's standing instructions. Edit freely.
# ==============================================================================

SYSTEM_PROMPT = f"""\
You are an autonomous problem-solver. You have one tool, run_python, which
executes Python in a fresh interpreter and returns its output.

How to work:

- Think the problem through before you act. Decide what you need to find out,
  then write the smallest piece of code that finds it out.

- Nothing persists between run_python calls. Every snippet must stand on its
  own, carry its own imports, and print() anything you need to see.

- Errors, tracebacks, and timeouts are information, not dead ends. When a call
  fails, read the actual error text, state plainly what it tells you, and fix
  the cause. Never re-run identical code hoping for a different result, and
  never abandon a workable approach just because the first attempt failed.

- If a library is genuinely unavailable or an approach is truly blocked, say so
  explicitly and solve the problem a different way.

- Verify before you conclude. Check your result against a second method, a known
  case, or an edge case, and say what you checked.

- Keep working until the task is genuinely complete. Do not stop at a partial
  result, and do not hand back a plan in place of a finished answer.

- When you are confident, answer in plain language: the result itself, and how
  you got there.

Finishing:

- When — and only when — the task is fully done and you have verified it, end
  your final message with this marker on a line of its own:

      {COMPLETION_MARKER}

- Follow the marker with two or three sentences on what you accomplished and how
  you verified it.

- Do not write the marker if anything is unfinished, unverified, or blocked. Say
  what is still outstanding instead.
"""


# --- Tracing ------------------------------------------------------------------

def indent(text: str) -> str:
    """Indent a block so it reads as nested under its heading."""
    return "\n".join("    " + line for line in text.splitlines())


class Trace:
    """Records each loop event to a JSONL file and prints a readable line."""

    # Terminal heading for each event type.
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

    def __init__(self, directory: Path = TRACE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        # Timestamped filename so concurrent or repeated runs never collide.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.path = directory / f"run-{stamp}.jsonl"
        self._file = self.path.open("w", encoding="utf-8")

    def record(self, event: str, **data) -> None:
        """Write one structured entry to the file and show it in the terminal."""
        entry = {"ts": datetime.now().astimezone().isoformat(), "event": event, **data}
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()  # flush so a crashed run still leaves a usable trace
        self._show(entry)

    def _show(self, entry: dict) -> None:
        """Render one entry as a human-readable terminal block."""
        event = entry["event"]
        # A failed tool result gets its own heading so it's obvious at a glance.
        label = "TOOL ERROR" if entry.get("status") == "error" else self.LABELS.get(event, event.upper())
        if "tool" in entry:
            label += f": {entry['tool']}"
        if entry.get("after_error"):
            label += " [retry]"
        if "turn" in entry:
            label += f" (turn {entry['turn']})"
        body = (entry.get("text") or entry.get("thinking")
                or entry.get("code") or entry.get("output") or "")
        print(f"--- {label} ---")
        if body:
            print(indent(body))
        print()

    def close(self) -> None:
        self._file.close()


# --- The tool -----------------------------------------------------------------

class ToolResult(NamedTuple):
    """Whether the call succeeded, plus the text Claude gets to read."""
    ok: bool
    output: str


def run_python(code: str) -> ToolResult:
    """Run Python in a fresh interpreter; any failure comes back as readable text."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"Error: code did not finish within {TOOL_TIMEOUT}s.")
    except Exception as exc:
        return ToolResult(False, f"Error: could not start the interpreter: {exc}")

    # Stitch stdout, stderr, and exit status into one blob Claude can diagnose.
    parts = []
    if proc.stdout.strip():
        parts.append(proc.stdout.rstrip())
    if proc.stderr.strip():
        parts.append("stderr:\n" + proc.stderr.rstrip())
    if proc.returncode != 0:
        parts.append(f"exit code: {proc.returncode}")
    # Never return an empty string — the API rejects empty tool_result content.
    output = "\n".join(parts) or "(no output — remember to print() your result)"
    return ToolResult(proc.returncode == 0, output)


def execute_tool(block) -> ToolResult:
    """Dispatch one tool_use block, turning a malformed request into a result."""
    if block.name != "run_python":
        return ToolResult(False, f"Error: unknown tool {block.name!r}.")
    code = block.input.get("code")
    if not isinstance(code, str) or not code.strip():
        return ToolResult(False, "Error: no 'code' argument was provided.")
    return run_python(code)


def error_signature(output: str) -> str:
    """Last meaningful line of a failure — usually the exception and message."""
    lines = [l.strip() for l in output.splitlines()
             if l.strip() and not l.strip().startswith("exit code:")]
    return lines[-1] if lines else output.strip()


def split_completion(text: str) -> tuple[bool, str]:
    """Look for the marker on a line of its own; return it and any summary after it."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        # Tolerate markdown decoration like **TASK COMPLETE**.
        if line.strip().strip("*#").strip() == COMPLETION_MARKER:
            return True, "\n".join(lines[i + 1:]).strip()
    return False, ""


# What Claude sees: the tool's name, when to use it, and its argument schema.
TOOLS = [
    {
        "name": "run_python",
        "description": (
            "Execute a snippet of Python code and get back its output. "
            "Each call runs in a fresh interpreter, so nothing persists between "
            "calls — print() anything you want to see."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python source to run."},
            },
            "required": ["code"],
        },
    }
]


# --- The agent loop -----------------------------------------------------------

def record_thinking(response, trace: Trace, turn: int) -> None:
    """Record any thinking blocks from one assistant turn."""
    for block in response.content:
        if block.type == "thinking":
            trace.record("thinking", turn=turn, thinking=block.thinking)
        elif block.type == "redacted_thinking":
            # Encrypted by safety filters — still replayed to the API, just not readable.
            trace.record("thinking", turn=turn, thinking="(redacted)")


def run_agent(client: anthropic.Anthropic, task: str, trace: Trace) -> None:
    """Drive the tool loop until Claude answers, gets stuck, or runs out of turns."""
    # The conversation history. Every turn appends to this list.
    messages = [{"role": "user", "content": task}]
    trace.record("task_start", text=task, model=MODEL)

    # Track repeats of the same failure so we can stop out instead of spinning.
    last_error = None
    repeats = 0

    for turn in range(1, MAX_TURNS + 1):
        # A failed request ends the run cleanly rather than raising.
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
                messages=messages,
            )
        except Exception as exc:
            trace.record("error", ending="error",
                         text=f"API request failed on turn {turn}: {exc}",
                         detail=traceback.format_exc())
            return

        # Show the reasoning before the actions it led to.
        record_thinking(response, trace, turn)

        # No tool calls means Claude has stopped — work out how it ended.
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

        # Append the FULL turn — thinking blocks must be replayed verbatim
        # alongside tool_use, or the next request is rejected.
        messages.append({"role": "assistant", "content": response.content})

        # Run every tool Claude asked for, collecting the results.
        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                trace.record("assistant_message", turn=turn, text=block.text)
            elif block.type == "tool_use":
                # after_error marks this call as a correction of the last failure.
                trace.record("tool_call", turn=turn, tool=block.name,
                             code=block.input.get("code", ""),
                             after_error=last_error is not None)

                result = execute_tool(block)
                trace.record("tool_result", turn=turn, tool=block.name,
                             status="ok" if result.ok else "error", output=result.output)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.output,
                    "is_error": not result.ok,  # flags the failure to the model
                })

                # Count identical failures in a row; any success resets the count.
                if result.ok:
                    last_error, repeats = None, 0
                else:
                    signature = error_signature(result.output)
                    repeats = repeats + 1 if signature == last_error else 1
                    last_error = signature

        # Stop out rather than retry the same broken thing forever.
        if repeats >= MAX_REPEATED_ERRORS:
            trace.record("stuck", ending="stopped_early",
                         text=f"Same failure {repeats} times in a row — "
                              f"stopping. Last error: {last_error}")
            return

        # Send all results back in one user message, then loop.
        messages.append({"role": "user", "content": tool_results})

    stuck_on = f" Last error: {last_error}" if last_error else ""
    trace.record("limit_reached", ending="stopped_early",
                 text=f"Hit the {MAX_TURNS}-turn limit with no final answer.{stuck_on}")


def main() -> None:
    # Load .env into the environment.
    load_dotenv()

    # Read the API key, bail out if it's missing.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY is not set. Put it in the .env file next to agent.py.")

    # Read the task from the command line.
    task = " ".join(sys.argv[1:]).strip()
    if not task:
        sys.exit('Usage: python agent.py "your task here"')

    client = anthropic.Anthropic(api_key=api_key)
    trace = Trace()

    # Backstop: any unexpected error becomes a trace event, not a traceback.
    try:
        run_agent(client, task, trace)
    except Exception as exc:
        trace.record("error", ending="error",
                     text=f"Unexpected failure: {type(exc).__name__}: {exc}",
                     detail=traceback.format_exc())
    finally:
        trace.close()
        print(f"Trace saved to {trace.path}")


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    main()
