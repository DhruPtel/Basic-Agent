"""
The tools an agent can call, each paired with the handler that runs it.
"""

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple
from urllib.parse import parse_qsl, urlparse

import requests
from anthropic.types import ToolParam
from bs4 import BeautifulSoup
from ddgs import DDGS


# --- Configuration ------------------------------------------------------------

TOOL_TIMEOUT = 30          # seconds before a run_python call is killed
FETCH_TIMEOUT = 20         # seconds before a fetch_url call gives up
MAX_FETCH_CHARS = 20_000   # trim fetched pages so context stays manageable
SEARCH_RESULTS = 6         # hits returned per web_search
USER_AGENT = "Mozilla/5.0 (compatible; sovereign-agent/1.0)"
REPORT_DIR = Path(__file__).parent / "reports"


class ToolResult(NamedTuple):
    """Whether the call succeeded, and the text Claude gets to read."""
    ok: bool
    output: str


class Tool(NamedTuple):
    """A tool's schema (what Claude sees) and handler (what actually runs)."""
    schema: ToolParam
    handler: Callable[[dict], ToolResult]

    @property
    def name(self) -> str:
        return self.schema["name"]


# --- Handlers -----------------------------------------------------------------

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
    if is_reddit(url):
        return fetch_reddit(url)
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
    except requests.RequestException as exc:
        return ToolResult(False, f"Error fetching {url}: {exc}")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
    return ToolResult(True, _trim(text) or "(the page had no readable text)")


def _trim(text: str) -> str:
    """Cap a fetched page so context stays manageable."""
    if len(text) > MAX_FETCH_CHARS:
        return text[:MAX_FETCH_CHARS] + f"\n\n[truncated at {MAX_FETCH_CHARS} chars]"
    return text


# --- Reddit: the public .json endpoint instead of scraping HTML ---------------

REDDIT_COMMENTS = 20       # top-level comments kept per post


def is_reddit(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host == "reddit.com" or host.endswith(".reddit.com")


def fetch_reddit(url: str) -> ToolResult:
    """Read a Reddit post (title, body, top comments) or listing (post titles) via .json."""
    parsed = urlparse(url)
    api = f"https://www.reddit.com{parsed.path.rstrip('/') or '/r/all'}.json"
    params = {"raw_json": 1, "limit": 50, "sort": "top", **dict(parse_qsl(parsed.query))}
    try:
        resp = requests.get(api, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT},
                            params=params)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        return ToolResult(False, f"Error fetching Reddit JSON for {url}: {exc}. Reddit "
                                 f"may be blocking or rate-limiting; try another source.")

    try:
        return ToolResult(True, _trim(_reddit_text(data)) or "(no Reddit content)")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        return ToolResult(False, f"Error: unexpected Reddit JSON shape for {url} ({exc!r}).")


def _reddit_text(data) -> str:
    """Readable text from a post ([post, comments]) or a listing (subreddit, search)."""
    if isinstance(data, list) and len(data) == 2:
        post = data[0]["data"]["children"][0]["data"]
        comments = [c["data"] for c in data[1]["data"]["children"] if c.get("kind") == "t1"]
        parts = [f"{post.get('title', '')}\n"
                 f"r/{post.get('subreddit', '?')} · u/{post.get('author', '?')} · "
                 f"{post.get('score', 0)} points · {post.get('num_comments', 0)} comments",
                 post.get("selftext") or post.get("url", ""),
                 f"--- Top comments ({min(len(comments), REDDIT_COMMENTS)}) ---"]
        parts += [f"[{c.get('score', 0)}] u/{c.get('author', '?')}: {c.get('body', '')}"
                  for c in comments[:REDDIT_COMMENTS]]
    else:
        posts = [c["data"] for c in data.get("data", {}).get("children", []) if c.get("kind") == "t3"]
        parts = [f"[{p.get('score', 0)}] {p.get('title', '')}\n"
                 f"https://www.reddit.com{p.get('permalink', '')}" for p in posts]
    return "\n\n".join(p for p in parts if p)


def write_report(title: str, markdown: str) -> ToolResult:
    """Save a markdown report to reports/ under a timestamped filename."""
    if not markdown.strip():
        return ToolResult(False, "Error: the report body was empty.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "report"
    path = REPORT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug[:60]}.md"
    path.write_text(markdown, encoding="utf-8")
    return ToolResult(True, f"Report saved to {path} ({len(markdown)} chars).")


# --- Schemas ------------------------------------------------------------------

def _schema(name: str, description: str, **properties: dict) -> ToolParam:
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


WEB_SEARCH = Tool(
    _schema("web_search",
            "Search the web and get back titles, URLs and snippets. Snippets are "
            "leads, not evidence — fetch_url the promising ones to read them.",
            query={"type": "string", "description": "What to search for."}),
    lambda args: web_search(str(args.get("query", ""))))

FETCH_URL = Tool(
    _schema("fetch_url",
            "Fetch a web page and return its readable text with HTML stripped out.",
            url={"type": "string", "description": "Full http:// or https:// URL."}),
    lambda args: fetch_url(str(args.get("url", ""))))

RUN_PYTHON = Tool(
    _schema("run_python",
            "Execute a snippet of Python code and get back its output. Each call "
            "runs in a fresh interpreter, so nothing persists between calls — "
            "print() anything you want to see.",
            code={"type": "string", "description": "The Python source to run."}),
    lambda args: run_python(str(args.get("code", ""))))

WRITE_REPORT = Tool(
    _schema("write_report",
            "Save the finished markdown report to the reports/ directory.",
            title={"type": "string", "description": "Short title, used in the filename."},
            markdown={"type": "string", "description": "The full report in markdown."}),
    lambda args: write_report(str(args.get("title", "")), str(args.get("markdown", ""))))

RESEARCH_TOOLS = [WEB_SEARCH, FETCH_URL, RUN_PYTHON]


def output_schema(name: str, description: str, item_properties: dict,
                  required: list[str]) -> ToolParam:
    """Schema for a submit tool: Claude hands back structured items through it."""
    return _schema(name, description, items={
        "type": "array",
        "items": {"type": "object", "properties": item_properties, "required": required},
    })


def render_args(name: str, args: dict) -> str:
    """A readable one-look rendering of a tool call's arguments."""
    if name == "run_python":
        return str(args.get("code", ""))
    if name == "write_report":
        return f"{args.get('title', 'untitled')} — {len(str(args.get('markdown', '')))} chars"
    if isinstance(items := args.get("items"), list):
        names = ", ".join(str(i.get("name", "?")) for i in items[:8] if isinstance(i, dict))
        return f"{len(items)} items: {names}" + (" …" if len(items) > 8 else "")
    return " ".join(f"{k}={v!r}" for k, v in args.items())


def error_signature(output: str) -> str:
    """Last meaningful line of a failure — usually the exception and its message."""
    lines = [line.strip() for line in output.splitlines()
             if line.strip() and not line.strip().startswith("exit code:")]
    return lines[-1] if lines else output.strip()
