"""Quick questions and saved questions: the Ask page's one-click buttons.

A quick question is an Ask question with blanks (`BLANK`) for the user to fill, shipped in
defaults/app/quick_questions.yaml. A saved question is one a user built and kept. It belongs
to its project and lives in .toolkit/saved_questions.json: a shortcut, not a setting, so it is
kept out of config.yaml.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from importlib import resources

import yaml

from ..errors import ToolkitError
from ..project import Project
from .ask import Field, Question, Row, readback

BLANK = "_"
TOPIC = "topic"         # in a template: whichever topic list the project has


@dataclass(frozen=True)
class Quick:
    button: str
    asks: str               # the finished question as it reads, one `{}` per blank
    question: Question
    saved: bool = False


def question_from(d: dict) -> Question:
    def rows(key):
        return tuple(Row(r["field"], r["condition"], tuple(str(v) for v in r["values"]),
                         r.get("join", "and"))
                     for r in d.get(key) or ())
    return Question(d["verb"], d["subject"], rows("rows"),
                    tuple(d.get("roles") or ("Narrator",)), rows("groups"))


def question_to(q: Question) -> dict:
    def rows(rs):
        return [{"field": r.field, "condition": r.condition, "values": list(r.values),
                 "join": r.join} for r in rs]
    return {"verb": q.verb, "subject": q.subject, "rows": rows(q.rows), "roles": list(q.roles),
            "groups": rows(q.groups)}


def builtin() -> list[Quick]:
    path = resources.files("transcript_toolkit") / "defaults" / "app" / "quick_questions.yaml"
    return [Quick(d["button"], d["asks"], question_from(d)) for d in yaml.safe_load(path.read_text())]


# --- saved questions --------------------------------------------------------------------------

def saved_path(project: Project):
    return project.toolkit_dir / "saved_questions.json"


def saved(project: Project) -> list[Quick]:
    path = saved_path(project)
    if not path.exists():
        return []
    return [Quick(d["button"], d["button"], question_from(d["question"]), saved=True)
            for d in json.loads(path.read_text())]


def save(project: Project, question: Question, fields: dict[str, Field]) -> Quick:
    """Keep a finished question as a button, named by how it reads."""
    if any(BLANK in r.values for r in (*question.rows, *question.groups)):
        raise ToolkitError("Fill in every blank before saving the question.")
    button = readback(question, fields)
    kept = saved(project)
    if any(k.button == button for k in kept):
        raise ToolkitError(f"Already saved: {button}")
    new = Quick(button, button, question, saved=True)
    _write(project, [*kept, new])
    return new


def forget(project: Project, button: str) -> None:
    kept = saved(project)
    if not any(k.button == button for k in kept):
        raise ToolkitError(f"No saved question called: {button}")
    _write(project, [k for k in kept if k.button != button])


def _write(project: Project, kept: list[Quick]) -> None:
    saved_path(project).write_text(json.dumps(
        [{"button": k.button, "question": question_to(k.question)} for k in kept], indent=2) + "\n")


# --- filling a template in --------------------------------------------------------------------

def resolve(quick: Quick, fields: dict[str, Field]) -> Question | None:
    """The question with `topic` made this project's first topic list, or None when the project
    cannot ask it yet (a field it needs has not been produced)."""
    q = quick.question
    topics = [k for k, f in fields.items() if k.startswith("topic:") and f.available]

    def concrete(r: Row) -> Row | None:
        key = (topics[0] if topics else None) if r.field == TOPIC else r.field
        f = fields.get(key) if key else None
        if f is None or not f.available or r.condition not in f.conditions_for(q.subject):
            return None
        return replace(r, field=key)

    rows = [concrete(r) for r in q.rows]
    groups = [concrete(g) for g in q.groups]
    if None in rows or None in groups:
        return None
    return replace(q, rows=tuple(rows), groups=tuple(groups))


def fill(question: Question, values: list[str]) -> Question:
    """Put `values` into the blanks, in reading order."""
    it = iter(values)

    def put(r: Row) -> Row:
        return replace(r, values=tuple(next(it) if v == BLANK else v for v in r.values))

    out = replace(question, rows=tuple(put(r) for r in question.rows),
                  groups=tuple(put(g) for g in question.groups))
    if next(it, None) is not None:
        raise ToolkitError("More values than blanks.")
    return out


def blanks(question: Question) -> int:
    return sum(r.values.count(BLANK) for r in (*question.rows, *question.groups))


def wording(question: Question, quicks: list[Quick], fields: dict[str, Field]) -> str:
    """How a question reads above its answer: as the quick question it is, when it is one."""
    for quick in quicks:
        values = _fills(quick.question, question)
        if values is not None:
            return quick.asks.format(*values)
    return readback(question, fields)


def _fills(template: Question, q: Question) -> list[str] | None:
    """The values `q` puts in `template`'s blanks, or None if `q` is not that template filled in."""
    if (template.verb, template.subject, template.roles) != (q.verb, q.subject, q.roles):
        return None
    pairs = list(zip(template.rows, q.rows)) + list(zip(template.groups, q.groups))
    if len(template.rows) != len(q.rows) or len(template.groups) != len(q.groups):
        return None
    values = []
    for t, r in pairs:
        same_field = t.field == r.field or (t.field == TOPIC and r.field.startswith("topic:"))
        if not same_field or t.condition != r.condition or len(t.values) != len(r.values):
            return None
        for tv, v in zip(t.values, r.values):
            if tv == BLANK:
                values.append(v)
            elif tv != v:
                return None
    return values
