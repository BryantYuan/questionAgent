"""
Token usage auditing - a persistent record of every token used across
every run, ever. Nothing in here gets deleted or reset.

Two files are kept in the logs/ folder:
  - token_audit.json  - the actual data, source of truth, safe to do
    math on. Don't hand-edit this.
  - token_audit.log   - a plain-text summary, fully rewritten after
    every run so the all-time total always appears at the top,
    followed by a daily breakdown below it. This is the one you
    actually open to glance at your usage.

PRICING NOTE: token prices change often (peak/off-peak rates,
promotional discounts, model renames all happen regularly). The rates
below are a snapshot, not a live feed - treat the cost figures this
produces as a rough estimate, and update PRICING yourself whenever
your provider's rates change. Do not treat this as an exact bill.
"""

import json
import datetime
from pathlib import Path

# USD per 1 million tokens. UPDATE THESE as pricing changes - see the
# note above. Using flat off-peak-ish estimates, not a live price feed.
PRICING = {
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}


def _empty_record() -> dict:
    return {
        "all_time": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "runs": 0,
            "tool_calls": {"web_access": 0, "scrape": 0},
        },
        "daily": {},
    }


def _empty_day() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "runs": 0,
        "tool_calls": {"web_access": 0, "scrape": 0},
    }


def _load(audit_json: Path) -> dict:
    if audit_json.exists():
        return json.loads(audit_json.read_text(encoding="utf-8"))
    return _empty_record()


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING.get(model)
    if not rates:
        return 0.0
    return (input_tokens / 1_000_000 * rates["input"]) + (output_tokens / 1_000_000 * rates["output"])


def sum_usage(messages) -> tuple[int, int]:
    """Add up input/output tokens across every AI response in this run."""
    input_tokens = 0
    output_tokens = 0
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            input_tokens += usage.get("input_tokens", 0) or 0
            output_tokens += usage.get("output_tokens", 0) or 0
    return input_tokens, output_tokens


def count_tool_calls(messages) -> dict:
    """Count how many times each tool was actually called in this run."""
    counts = {"web_access": 0, "scrape": 0}
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name")
                if name in counts:
                    counts[name] += 1
    return counts


def record_run(logs_dir: Path, model: str, messages, tool_counts: dict | None = None) -> dict:
    """Add this run's usage to the permanent record and rewrite the
    human-readable summary. Returns the updated all-time totals."""
    audit_json = logs_dir / "token_audit.json"
    audit_log = logs_dir / "token_audit.log"

    data = _load(audit_json)
    input_tokens, output_tokens = sum_usage(messages)
    total_tokens = input_tokens + output_tokens
    cost = _estimate_cost(model, input_tokens, output_tokens)
    if tool_counts is None:
        tool_counts = count_tool_calls(messages)

    at = data["all_time"]
    at["input_tokens"] += input_tokens
    at["output_tokens"] += output_tokens
    at["total_tokens"] += total_tokens
    at["cost_usd"] += cost
    at["runs"] += 1
    for name, count in tool_counts.items():
        at.setdefault("tool_calls", {"web_access": 0, "scrape": 0})
        at["tool_calls"][name] = at["tool_calls"].get(name, 0) + count

    today = datetime.date.today().isoformat()
    day = data["daily"].setdefault(today, _empty_day())
    day["input_tokens"] += input_tokens
    day["output_tokens"] += output_tokens
    day["total_tokens"] += total_tokens
    day["cost_usd"] += cost
    day["runs"] += 1
    for name, count in tool_counts.items():
        day.setdefault("tool_calls", {"web_access": 0, "scrape": 0})
        day["tool_calls"][name] = day["tool_calls"].get(name, 0) + count

    audit_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    lines = [
        "=== ALL-TIME TOTAL ===",
        f"Runs: {at['runs']}",
        f"Tokens: {at['total_tokens']}  (in: {at['input_tokens']}, out: {at['output_tokens']})",
        f"Estimated cost: ${at['cost_usd']:.4f}  (see PRICING note in audit.py)",
        f"Tool calls: web_access={at['tool_calls']['web_access']}, scrape={at['tool_calls']['scrape']}",
        "",
        "=== DAILY BREAKDOWN (most recent first) ===",
    ]
    for date, stats in sorted(data["daily"].items(), reverse=True):
        tc = stats.get("tool_calls", {"web_access": 0, "scrape": 0})
        lines.append(
            f"{date}  runs: {stats['runs']:>3}  tokens: {stats['total_tokens']:>8}  "
            f"cost: ${stats['cost_usd']:.4f}  "
            f"tools: web_access={tc['web_access']}, scrape={tc['scrape']}"
        )

    audit_log.write_text("\n".join(lines), encoding="utf-8")

    return at
