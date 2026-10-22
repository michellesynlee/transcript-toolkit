"""The Ask page's AI summary: one sentence on what an answer's passages say.

The one part of Ask that costs money, so it is built like every other paid call: the call is
made by a real command (`toolkit ask summarize`, steps/ask_summary.py) that states its cost and
waits for a yes; the call is cached in .toolkit/cache/ask.jsonl, so asking the same question of
the same passages again is free and `toolkit cost` counts it. This module is the part the page
and the command share — what is sent, what it is estimated to cost, whether it is already
paid for — so the figure on the button is the figure the command asks about.

Every answer also gets a sentence from its counts alone, free and always there: `at_a_glance`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

import pandas as pd

from ..project import Project
from ..steps import freshness
from . import ask
from . import cost as costmod
from .cache import cache_key, latest_records
from .config import load_step_config, require
from .prompts import load_prompt

STEP = "ask"
# One token is about four characters of English. An estimate for the button, never a charge:
# what is paid is what the call reports.
CHARS_PER_TOKEN = 4
add_missing_files = ask.add_missing_files           # shared with the engine's settings


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


PASSAGE_SUBJECTS = ("clips", "paragraphs")


def has_positive(q: ask.Question) -> bool:
    """Whether the question looks for something present. "Who never mentions X" alone does
    not: its passages are everything else those people said, and a summary of that would be
    an answer to a different question."""
    return any(not ask.CONDITIONS[r.condition].negative for r in (*q.rows, *q.groups))


def passages(project: Project, q: ask.Question, answer: pd.DataFrame) -> pd.DataFrame:
    """The passages behind an answer, clearest first. For interviews and speakers, the matching
    passages from those found; for Compare, each group's, labelled in a `group` column. An
    interview "mostly about" something is read through its passages that are about it."""
    if q.verb == "find" and q.subject in PASSAGE_SUBJECTS:
        return answer
    unit = q.subject if q.subject in PASSAGE_SUBJECTS else (
        "clips" if freshness.wrote_its_results(project, "clip") else "paragraphs")
    rows = tuple(replace(r, condition="talks_about") if r.condition == "mostly_talks_about"
                 else r for r in q.rows)
    if q.verb == "compare":
        frames = [ask.answer(project, ask.Question("find", unit, ask.and_each(rows, (g,)),
                                                   q.roles)).assign(group=ask.row_words(g))
                  for g in q.groups]
        return pd.concat(frames, ignore_index=True)
    found = ask.answer(project, ask.Question("find", unit, rows, q.roles))
    if q.verb == "find":                      # interviews or speakers: only theirs
        found = found[found["interview"].isin(answer["interview"])]
    return found.reset_index(drop=True)


def material(found: pd.DataFrame, title: str, max_clips: int) -> Material:
    """The question and its passages, clearest first, as the model reads them. Compare's groups
    share the allowance and are sent group by group."""
    def blocks(frame: pd.DataFrame) -> str:
        out = []
        for r in frame.fillna("").astype(str).to_dict("records"):
            head = " · ".join(v for v in (r.get("clip") or r.get("session"), r.get("interview"),
                                          r.get("speaker"), r.get("label")) if v)
            out.append(f"[{head}]\n{r['text']}")
        return "\n\n".join(out)

    def more(frame: pd.DataFrame, kept: pd.DataFrame) -> str:
        return (f"There are {len(frame)} passages; you are given the clearest {len(kept)}."
                if len(frame) > len(kept) else "")

    if "group" in found.columns:
        groups = list(dict.fromkeys(found["group"]))
        per = max(1, max_clips // len(groups))
        parts, n_read = [], 0
        for g in groups:
            frame = found[found["group"] == g]
            kept = frame.head(per)
            n_read += len(kept)
            note = more(frame, kept)
            parts.append(f"Group: {g}\n" + (f"{note}\n" if note else "") + f"\n{blocks(kept)}")
        content = (f"Question: {title}\n\nThe passages come in {len(groups)} groups to compare."
                   f"\n\n" + "\n\n".join(parts))
        return Material(len(found), n_read, content)
    kept = found.head(max_clips)
    note = more(found, kept)
    content = (f"Question: {title}\n\n" + (f"{note}\n\n" if note else "")
               + "Passages:\n\n" + blocks(kept))
    return Material(len(found), len(kept), content)


def _n(count: int, word: str) -> str:
    return f"{count} {word if count == 1 else word + 's'}"


def at_a_glance(q: ask.Question, answer: pd.DataFrame) -> str:
    """One sentence on an answer, from its counts alone: free, and always there."""
    unit = {"clips": "clip", "paragraphs": "passage"}.get(q.subject, "passage")
    if answer.empty:
        return "Nothing in the transcripts matches this."
    if q.verb == "compare":
        groups = [c for c in answer.columns if c != "interview"]
        return "; ".join(f"{g}: {_n(int(answer[g].sum()), unit)} in "
                         f"{_n(int((answer[g] > 0).sum()), 'interview')}" for g in groups) + "."
    if q.verb == "count":
        total = answer["clear"] + answer["possible"]
        top = answer.iloc[0]
        return (f"{_n(int(total.sum()), unit)} across {_n(int((total > 0).sum()), 'interview')}"
                + (f"; most in {top['interview']} ({int(total.iloc[0])})." if len(answer) > 1
                   else "."))
    if q.subject in PASSAGE_SUBJECTS:
        n, clear = len(answer), int((answer.get("match") == "clear").sum())
        if "match" in answer.columns and clear == 0:
            return f"No {unit} is clearly about this; {n} may touch on it."
        possible = n - clear if "match" in answer.columns else 0
        return (f"{_n(n, unit)} across {_n(answer['interview'].nunique(), 'interview')}"
                + (f"; {clear} clearly about it, {possible} possibly." if possible else "."))
    # interviews or speakers
    n = len(answer)
    who = (_n(n, "interview") if q.subject == "interviews" else
           f"{_n(n, 'speaker')} in {_n(answer['interview'].nunique(), 'interview')}")
    if not has_positive(q):
        return f"{who} {'matches' if n == 1 else 'match'}."
    top = answer.iloc[0]
    most = int(top["clear"] + top["possible"])
    name = top["interview"] if q.subject == "interviews" else f"{top['speaker']} ({top['interview']})"
    if n == 1:
        return f"Only {name} matches ({_n(most, 'passage')})."
    return f"{who} match; {name} has the most ({_n(most, 'passage')})."


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
    m = material(passages(project, question, answer), title, int(cfg["max_clips"]))
    key = cache_key(cfg["model"], cfg["reasoning"], cfg["verbosity"], instructions, m.user_content)
    return Prepared(cfg, instructions, m, key)


def saved(project: Project, prepared: Prepared) -> str | None:
    """The summary, when this exact call has already been paid for."""
    hit = latest_records(cache_path(project), "cache_key").get(prepared.key)
    return tidy(hit["summary"]) if hit else None


# Models sometimes trail an answer with stray brackets or full-width punctuation ("】【。").
# Read-time as well as write-time, so a summary already paid for is shown clean too.
STRAY = re.compile(r"[【】〔〕「」『』。、]")


def tidy(summary: str) -> str:
    text = re.sub(r"\s+", " ", STRAY.sub("", summary)).strip()
    return text if not text or text[-1] in ".!?" else text + "."


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
