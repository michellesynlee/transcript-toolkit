"""The Ask page's AI summary: one sentence on what an answer's passages say.

The one part of Ask that costs money, so it is built like every other paid call: the call is
made by a real command (`toolkit ask summarize`, steps/ask_summary.py) that states its cost and
waits for a yes; the call is cached in .toolkit/cache/ask.jsonl, so asking the same question of
the same passages again is free and `toolkit cost` counts it. This module is the part the page
and the command share — what is sent, what it is estimated to cost, whether it is already
paid for — so the figure on the button is the figure the command asks about.

When there is nothing to summarize (no matches, or only possible ones) the page says so from
the counts, with no call at all: `without_ai`.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib import resources

import pandas as pd

from ..project import Project
from . import ask
from . import cost as costmod
from .cache import cache_key, latest_records
from .config import load_step_config, require
from .prompts import load_prompt

STEP = "ask"
# One token is about four characters of English. An estimate for the button, never a charge:
# what is paid is what the call reports.
CHARS_PER_TOKEN = 4
# What a project made before AI summaries existed is missing, copied in on first use.
NEW_FILES = (("scaffold/advanced/ask.yaml", "advanced/ask.yaml"),
             ("prompts/ask_summary.md", "prompts/ask_summary.md"))


def add_missing_files(project: Project) -> list[str]:
    """Copy in the defaults this feature needs that the project does not have. Never overwrites."""
    defaults = resources.files("transcript_toolkit") / "defaults"
    added = []
    for src, dest in NEW_FILES:
        path = project.root / dest
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with resources.as_file(defaults / src) as f:
                shutil.copyfile(f, path)
            added.append(dest)
    return added


def settings(project: Project) -> dict:
    cfg = load_step_config(project, STEP)
    require(cfg, ["model", "reasoning", "verbosity", "prompt", "max_clips",
                  "expected_output_tokens"], STEP)
    return cfg


def cache_path(project: Project):
    return project.cache_dir / "ask.jsonl"


@dataclass(frozen=True)
class Material:
    """What one summary call is sent."""
    n_found: int                # passages in the answer
    n_read: int                 # how many of them are sent (the clearest first)
    user_content: str


def material(answer: pd.DataFrame, title: str, max_clips: int) -> Material:
    """The question and the answer's passages, clearest first, as the model reads them."""
    kept = answer.head(max_clips)           # an answer is already ranked clearest first
    blocks = []
    for r in kept.fillna("").astype(str).to_dict("records"):
        head = " · ".join(v for v in (r.get("clip") or r.get("session"), r.get("interview"),
                                      r.get("speaker"), r.get("label")) if v)
        blocks.append(f"[{head}]\n{r['text']}")
    more = (f"There are {len(answer)} passages; you are given the clearest {len(kept)}.\n\n"
            if len(answer) > len(kept) else "")
    content = f"Question: {title}\n\n{more}Passages:\n\n" + "\n\n".join(blocks)
    return Material(len(answer), len(kept), content)


def summarisable(answer: pd.DataFrame) -> bool:
    """Passages with text to read, at least one of them a clear match."""
    return ("text" in answer.columns and not answer.empty
            and ("match" not in answer.columns or (answer["match"] == "clear").any()))


def without_ai(answer: pd.DataFrame) -> str | None:
    """The sentence the counts can give on their own, when there is nothing for a summary."""
    if "text" not in answer.columns:
        return None
    if answer.empty:
        return "Nothing in the transcripts matches this."
    if "match" in answer.columns and not (answer["match"] == "clear").any():
        n = len(answer)
        return (f"No passage is clearly about this; {n} may touch on it. "
                f"Read {'it' if n == 1 else 'them'} below.")
    return None


@dataclass(frozen=True)
class Prepared:
    cfg: dict
    instructions: str
    material: Material
    key: str


def prepare(project: Project, question: ask.Question, title: str,
            answer: pd.DataFrame | None = None) -> Prepared:
    """Everything the call needs, from the question: what both the page and the command read.
    `answer` saves answering the question again when the caller already has."""
    cfg = settings(project)
    instructions = load_prompt(project, cfg["prompt"])
    answer = ask.answer(project, question) if answer is None else answer
    m = material(answer, title, int(cfg["max_clips"]))
    key = cache_key(cfg["model"], cfg["reasoning"], cfg["verbosity"], instructions, m.user_content)
    return Prepared(cfg, instructions, m, key)


def saved(project: Project, prepared: Prepared) -> str | None:
    """The summary, when this exact call has already been paid for."""
    hit = latest_records(cache_path(project), "cache_key").get(prepared.key)
    return hit["summary"] if hit else None


@dataclass(frozen=True)
class Estimate:
    tokens: int
    usd: float | None               # None for a model the price table does not know


def estimate(project: Project, prepared: Prepared) -> Estimate:
    """About what one call will cost. Output is measured from this project's earlier summary
    calls with the same model once there are any, and taken from settings until then."""
    cfg = prepared.cfg
    tokens_in = (len(prepared.instructions) + len(prepared.material.user_content)) // CHARS_PER_TOKEN
    earlier = [r for r in latest_records(cache_path(project), "cache_key").values()
               if r.get("model") == cfg["model"]]
    tokens_out = (sum(r["usage"].get("output_tokens") or 0 for r in earlier) // len(earlier)
                  if earlier else int(cfg["expected_output_tokens"]))
    rates = costmod.pricing().get(cfg["model"])
    usd = costmod.cost(tokens_in, 0, tokens_out, rates["standard"]) if rates else None
    return Estimate(tokens_in + tokens_out, usd)


def describe(e: Estimate) -> str:
    """As the button and the command's question both say it."""
    tokens = f"about {round(e.tokens, -2):,} tokens" if e.tokens >= 100 else f"about {e.tokens} tokens"
    if e.usd is None:
        return f"{tokens}, price unknown"
    return f"{tokens}, ≈ ${e.usd:.2f}" if e.usd >= 0.01 else f"{tokens}, under $0.01"
