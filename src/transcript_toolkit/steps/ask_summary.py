"""`toolkit ask summarize` — one AI sentence on what an Ask answer's passages say.

The Ask page's "Read AI summary" button runs this. It states what the call will cost and waits
for a yes (the page shows that question as buttons), makes one call, and records it in
.toolkit/cache/ask.jsonl: asking again is free, and `toolkit cost` counts it. What is sent and
what it is estimated to cost are `core/ask_summary.py`'s, shared with the page.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from pydantic import create_model

from ..core import ask, ask_quick, ask_summary
from ..core.cache import JsonlAppender, cache_key
from ..core.console import confirm_or_abort
from ..core.llm import build_schema, call_llm, check_levels, openai_client
from ..errors import ToolkitError
from ..project import Project

STEP = ask_summary.STEP


def run_ask_summary(project: Project, question_json: str, title: str | None = None,
                    yes: bool = False) -> str:
    for added in ask_summary.add_missing_files(project):
        print(f"Added {added} (new in this version of the toolkit).")
    try:
        question = ask_quick.question_from(json.loads(question_json))
    except (ValueError, KeyError) as e:
        raise ToolkitError(f"Not a question the Ask page wrote: {e}") from e
    if not ask_summary.has_positive(question):
        raise ToolkitError("Nothing to summarize: the question only asks what is not there.")
    title = title or ask.readback(question, ask.fields(project))
    prepared = ask_summary.prepare(project, question, title)
    m, cfg = prepared.material, prepared.cfg
    if m.n_read == 0:
        raise ToolkitError("Nothing to summarize: the question has no passages.")

    summary = ask_summary.saved(project, prepared)
    if summary is None:
        check_levels(cfg["reasoning"], cfg["verbosity"])
        est = ask_summary.estimate(project, prepared)
        read = (f"the clearest {m.n_read} of {m.n_found} passages" if m.n_read < m.n_found
                else f"{m.n_read} passage{'s' if m.n_read != 1 else ''}")
        confirm_or_abort(f"AI summary of {read} with {cfg['model']}: "
                         f"{ask_summary.describe(est)}. Go ahead?", yes=yes)
        summary = _call(project, prepared)
    else:
        print("Already summarized; nothing paid for this time.")
    print(f"\nSummary: {summary}")
    return summary


def _call(project: Project, p: ask_summary.Prepared) -> str:
    cfg = p.cfg
    model, reasoning, verbosity = cfg["model"], cfg["reasoning"], cfg["verbosity"]
    schema = build_schema(create_model("AskSummary", summary=(str, ...)), "ask_summary")
    parsed, usage = call_llm(openai_client(project.root), model, reasoning, verbosity, schema,
                             p.instructions, p.material.user_content,
                             cache_key(model, reasoning, verbosity, p.instructions),
                             poll_interval_s=float(cfg.get("poll_interval_s", 4)),
                             max_total_wait_s=float(cfg.get("max_total_wait_s", 300)))
    summary = (parsed.get("summary") or "").strip()
    # One record per paid call: this is what `toolkit cost` adds up.
    JsonlAppender(ask_summary.cache_path(project)).append({
        "cache_key": p.key, "fingerprint": cache_key(model, reasoning, verbosity, p.instructions),
        "n_found": p.material.n_found, "n_read": p.material.n_read, "summary": summary,
        "model": model, "reasoning_effort": reasoning, "verbosity": verbosity,
        "usage": usage, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return summary
