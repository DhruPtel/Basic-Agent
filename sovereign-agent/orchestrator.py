"""
The orchestrator: coordinates hunters and verifiers on an enumeration directive.

    python orchestrator.py "Find every AI research lab that exists, with a source for each"

Plan angles → run hunters in parallel → merge + dedupe → verify in batches →
save a sourced markdown list. Each agent traces to its own file under
traces/<run>/, and the orchestrator's handoffs go to orchestrator.jsonl.
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import anthropic

from agent import MODEL, TRACE_DIR, VERBOSE, Agent, Budget, RunResult, Trace, make_client, short
from roles import make_hunter, make_verifier
from tools import write_report


# --- Configuration ------------------------------------------------------------

NUM_HUNTERS = 2
VERIFY_BATCH = 30          # candidates per verifier instance; keeps each context small
MAX_PARALLEL = 3           # agents running at once
QUIET = not VERBOSE        # milestones only in the terminal; traces keep full detail
MAX_TOTAL_TOKENS = int(os.environ.get("MAX_TOTAL_TOKENS", 2_500_000))  # whole run, all agents (~$10)
HUNT_SHARE = 0.8           # hunting stops at 80% of the budget, leaving room to verify

FALLBACK_ANGLES = [
    {"label": "Lists and directories",
     "strategy": "Find list pages, directories, rankings, databases and Wikipedia "
                 "categories that enumerate matching items, and mine them thoroughly."},
    {"label": "Long-tail discovery",
     "strategy": "Find individual items the lists miss: search by sub-field, region, "
                 "country, founding year and news coverage, one niche at a time."},
]


# --- Planning -----------------------------------------------------------------

ANGLES_TOOL = {
    "name": "assign_angles",
    "description": "Assign one distinct search angle to each hunter.",
    "input_schema": {
        "type": "object",
        "properties": {"angles": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "A few words naming the angle."},
                "strategy": {"type": "string", "description": "2-3 sentences: what to search and where."},
            },
            "required": ["label", "strategy"],
        }}},
        "required": ["angles"],
    },
}


def usage_fields(budget: Budget) -> dict:
    """Running token total, attached to orchestrator milestones."""
    return {"tokens": budget.total, "token_limit": budget.limit}


def plan_angles(client: anthropic.Anthropic, directive: str, trace: Trace,
                budget: Budget) -> list[dict]:
    """One forced tool call to split the directive into non-overlapping angles."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2_000,
            tools=[ANGLES_TOOL],
            tool_choice={"type": "tool", "name": "assign_angles"},
            messages=[{"role": "user", "content":
                f"Directive: {directive}\n\nSplit the search for this into exactly "
                f"{NUM_HUNTERS} complementary angles with as little overlap as possible, "
                f"so that together they cover everything that could match."}],
        )
        budget.add(response.usage)
        block = next(b for b in response.content if b.type == "tool_use")
        angles = [a for a in block.input["angles"] if a.get("label") and a.get("strategy")]
        if len(angles) >= NUM_HUNTERS:
            angles = angles[:NUM_HUNTERS]
            trace.record("plan", angles=angles, text=_format_angles(angles), **usage_fields(budget))
            return angles
        reason = f"got {len(angles)} usable angles"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"

    trace.record("plan", angles=FALLBACK_ANGLES, fallback=True, **usage_fields(budget),
                 text=f"Planning failed ({reason}); using fallback angles.\n"
                      + _format_angles(FALLBACK_ANGLES))
    return FALLBACK_ANGLES


def _format_angles(angles: list[dict]) -> str:
    return "\n".join(f"{i}. {a['label']}: {a['strategy']}" for i, a in enumerate(angles, 1))


# --- Merging ------------------------------------------------------------------

def name_key(name: str) -> str:
    """Normalise a name for dedup: lowercase, no punctuation or filler words."""
    key = re.sub(r"[^a-z0-9]+", " ", name.lower())
    key = re.sub(r"\b(the|inc|ltd|llc|gmbh|corp|co)\b", " ", key)
    return " ".join(key.split())


def merge(batches: dict[str, list[dict]]) -> list[dict]:
    """Combine candidate lists, deduplicating by normalised name."""
    master: dict[str, dict] = {}
    for found_by, items in batches.items():
        for item in items:
            name, url = str(item.get("name", "")).strip(), str(item.get("source_url", "")).strip()
            if not (key := name_key(name)):
                continue
            entry = master.setdefault(key, {"name": name, "source_urls": [], "notes": [], "found_by": []})
            if url and url not in entry["source_urls"]:
                entry["source_urls"].append(url)
            if (note := str(item.get("note", "")).strip()) and note not in entry["notes"]:
                entry["notes"].append(note)
            if found_by not in entry["found_by"]:
                entry["found_by"].append(found_by)
    return list(master.values())


# --- The orchestrator ---------------------------------------------------------

class Orchestrator:
    """Runs one directive end to end; every agent gets its own trace in this run's folder."""

    def __init__(self, client: anthropic.Anthropic, directive: str) -> None:
        self.client = client
        self.directive = directive
        self.run_dir = TRACE_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S}"
        self.trace = Trace(self.run_dir / "orchestrator.jsonl", "orchestrator", QUIET)
        self.budget = Budget(MAX_TOTAL_TOKENS)  # shared by every agent in this run

    def run(self) -> None:
        self.trace.record("task_start", text=self.directive, model=MODEL, role="orchestrator")

        angles = plan_angles(self.client, self.directive, self.trace, self.budget)
        hunts = self._run_agents([
            (make_hunter, n, self._hunter_task(angle), angle["label"])
            for n, angle in enumerate(angles, 1)
        ])

        candidates = merge({role: result.items for role, result in hunts.items()})
        total = sum(len(r.items) for r in hunts.values())
        self._record("merge", candidates=len(candidates),
                          text=f"{total} raw candidates from {len(hunts)} hunters → "
                               f"{len(candidates)} after dedup by name.")

        # --- Step 2 placeholder: saturation loop ------------------------------
        # Derive fresh angles from what was found (sub-types, regions, names
        # seen on list pages), re-run hunters, merge, and repeat until a round
        # adds (almost) nothing new. For now: a single hunting round.

        batches = [candidates[i:i + VERIFY_BATCH] for i in range(0, len(candidates), VERIFY_BATCH)]
        verdicts = self._run_agents([
            (make_verifier, n, self._verifier_task(batch), f"{len(batch)} candidates")
            for n, batch in enumerate(batches, 1)
        ])

        # Always report, even when the budget cut hunting or verification short.
        self._save_report(angles, hunts, candidates, batches, verdicts)
        b = self.budget
        self.trace.record("usage", input_tokens=b.input, output_tokens=b.output, cost_usd=round(b.cost(), 2),
                     text=f"{b.total:,} tokens ({short(b.input)} in / {short(b.output)} out) "
                          f"· est. ${b.cost():.2f}" + (" · budget reached" if b.exceeded() else ""))
        self.trace.close()
        print(f"Traces saved to {self.run_dir}/")

    def _run_agents(self, jobs: list[tuple]) -> dict[str, RunResult]:
        """Start each (factory, n, task, summary) agent in parallel; trace handoff and handback."""
        def run_one(factory, n, task, summary) -> tuple[str, RunResult]:
            role = f"{factory.__name__.removeprefix('make_')}-{n}"
            share = HUNT_SHARE if role.startswith("hunter") else 1.0
            if self.budget.exceeded(share):  # don't launch new work past the budget
                self._record("budget_stop", peer=role, text=f"Token budget reached — not launching {role}.")
                return role, RunResult("skipped", "token budget reached", [])

            agent_trace = Trace(self.run_dir / f"{role}.jsonl", role, QUIET)
            agent: Agent = factory(self.client, agent_trace, n, budget=self.budget, budget_share=share)
            self._record("handoff", peer=agent.role, trace=str(agent_trace.path), text=summary)
            try:
                result = agent.run(task)
            finally:
                agent_trace.close()
            self._record("handback", peer=agent.role, ending=result.ending,
                              items=len(result.items),
                              text=f"{result.ending} — {len(result.items)} items returned.")
            return agent.role, result

        if not jobs:
            return {}
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            return dict(pool.map(lambda job: run_one(*job), jobs))

    def _record(self, event: str, **data) -> None:
        """Trace an orchestrator event along with the run's token total so far."""
        self.trace.record(event, **data, **usage_fields(self.budget))

    def _hunter_task(self, angle: dict) -> str:
        return (f"Directive: {self.directive}\n\n"
                f"Your angle: {angle['label']}\n{angle['strategy']}\n\n"
                f"Find every matching item reachable through this angle and submit "
                f"each with a source_url via submit_candidates.")

    def _verifier_task(self, batch: list[dict]) -> str:
        return (f"Directive: {self.directive}\n\n"
                f"Verify these {len(batch)} candidates and submit a verdict for each "
                f"via submit_verdicts:\n\n"
                + json.dumps([{k: c[k] for k in ("name", "source_urls", "notes")} for c in batch],
                             indent=1))

    def _save_report(self, angles, hunts, candidates, batches, verdicts) -> None:
        """Write the final sourced list via write_report."""
        verified: dict[str, dict] = {}
        excluded: list[dict] = []
        for result in verdicts.values():
            for v in result.items:
                ok = v.get("status") == "verified" and str(v.get("source_url", "")).startswith("http")
                if ok:
                    verified.setdefault(name_key(str(v.get("name", ""))), v)  # dedupe across batches
                else:
                    excluded.append(v)
        entries = sorted(verified.values(), key=lambda v: str(v.get("name", "")).lower())

        # Candidates a verifier never reached (skipped, or stopped by budget/turn limit).
        unverified = []
        for n, batch in enumerate(batches, 1):
            r = verdicts.get(f"verifier-{n}")
            if r and r.ending == "complete":
                continue
            judged = {name_key(str(v.get("name", ""))) for v in (r.items if r else [])}
            unverified += [c for c in batch if name_key(c["name"]) not in judged]

        lines = [
            f"# {self.directive}", "",
            f"_{len(entries)} verified entries from {len(candidates)} candidates · "
            f"generated {datetime.now():%Y-%m-%d %H:%M} · traces: `{self.run_dir.name}`_", "",
            "## Entries", "",
        ]
        lines += [f"{i}. **{v['name']}**" + (f" — {v['note']}" if v.get("note") else "")
                  + f" ([source]({v['source_url']}))" for i, v in enumerate(entries, 1)]
        if unverified:
            lines += ["", "## Unverified — no verdict (budget or turn limit reached)", ""]
            lines += [f"- {c['name']}" + (f" ([hunter source]({c['source_urls'][0]}))" if c["source_urls"] else "")
                      for c in unverified]
        if excluded:
            lines += ["", "## Excluded", ""]
            lines += [f"- {v.get('name', '?')} — {v.get('status', '?')}: {v.get('note', '')}"
                      for v in excluded]
        lines += ["", "## How this list was built", ""]
        lines += [f"- **{role}** ({angle['label']}): {len(r.items)} candidates, ended *{r.ending}*"
                  for (role, r), angle in zip(hunts.items(), angles)]
        lines += [f"- **{role}**: {len(r.items)} verdicts, ended *{r.ending}*"
                  for role, r in verdicts.items()]
        lines.append(f"- Tokens: {short(self.budget.total)} of {short(self.budget.limit)} budget "
                     f"(est. ${self.budget.cost():.2f})" + (" — budget reached" if self.budget.exceeded() else ""))
        lines.append("- Single hunting round; no saturation loop yet.")

        result = write_report(f"enumeration {self.directive}", "\n".join(lines) + "\n")
        self._record("report", status="ok" if result.ok else "error", entries=len(entries),
                     excluded=len(excluded), unverified=len(unverified), text=result.output)


def main() -> None:
    directive = " ".join(sys.argv[1:]).strip()
    if not directive:
        sys.exit('Usage: python orchestrator.py "your enumeration directive"')
    Orchestrator(make_client(), directive).run()


if __name__ == "__main__":
    main()
