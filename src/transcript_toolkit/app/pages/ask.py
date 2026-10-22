"""Ask page: question the collection with a ready-made question, or by building your own.

Two ways in, kept apart: a quick question is a sentence with blanks you fill in place; the
builder is a filter of rows joined by "and" / "or". Either way the answer is `core/ask.py`
filtering the files the pipeline already wrote: no command runs, nothing is paid for, nothing
in the project changes except a saved question. An answer leads back to its transcript: a
paragraph or clip opens at its place, an interview or speaker opens into its own clips.
Export hands the answer to the browser as a download; it never touches outputs/.
"""
from __future__ import annotations

import html
import io
import json
import re
from urllib.parse import quote

import pandas as pd
from nicegui import ui

from ...core import ask, ask_quick
from ...errors import ToolkitError
from ...steps import freshness
from .. import theme
from ..context import CONTEXT
from .common import guard, info, section, shell

HREF = "/ask"
VIEW = "/ask/view"
PREVIEW_ROWS = 50
TEXT_PREVIEW = 300              # characters of a clip's text shown in the preview table

ROLE_CHOICES = {"Narrator": "Narrators", "Interviewer": "Interviewers", "both": "Both"}
NEEDS_WORDS = {"import": "Import", "label": "Label", "topics tag": "Topics",
               "locations tag": "Locations", "locations map": "Locations (map)"}
MATCH_HELP = ("Clear: the clip is about it (the top topic score).\n"
              "Possible: it may touch on it (a score between zero and the top).")
MODES = {"quick": "Quick question", "build": "Build your own"}
FOUND_BY_WORDS = ("contains", "contains_any", "contains_all", "whole_word")


def _blank_row(join: str = "and") -> dict:
    return {"field": "", "condition": "", "values": [], "join": join}


def ask_page() -> None:
    with shell(HREF):
        if CONTEXT.project is None:
            return
        project = CONTEXT.require_project()
        section("Ask", "Ask questions about your interviews. Nothing here costs anything or "
                       "changes your project.")
        if not project.paragraphs_path.exists():
            ui.label("Import transcripts first: questions are answered from them.") \
                .classes("text-sm opacity-70")
            return
        try:
            fields = ask.fields(project)
        except ToolkitError as e:
            guard(e)
            return
        _Asker(project, fields).draw()


class _Asker:
    """The page: a mode switch, the quick question or the builder, and the answer under both."""

    def __init__(self, project, fields: dict[str, ask.Field]):
        self.project = project
        self.fields = fields
        self.clipped = freshness.wrote_its_results(project, "clip")
        self.mode = "quick"
        self.quick: ask_quick.Quick | None = None      # the chosen quick question
        self.blanks: list[str] = []                     # what has been filled into it
        self.state = {"verb": "find", "subject": "clips" if self.clipped else "paragraphs",
                      "roles": ["Narrator"], "rows": [_blank_row()], "groups": []}
        self.results = _Results(project, fields)

    def draw(self) -> None:
        ui.toggle(MODES, value=self.mode, on_change=lambda e: self.switch(e.value)) \
            .props("no-caps").mark("ask-mode")
        self.panel = ui.refreshable(self._panel)
        self.normalise()
        self.panel()
        self.results.draw()

    def switch(self, mode: str) -> None:
        self.mode = mode
        self.panel.refresh()

    def _panel(self) -> None:
        if self.mode == "quick":
            self._quick_panel()
        else:
            self._builder()

    # --- quick questions ----------------------------------------------------------------------

    def _quick_panel(self) -> None:
        offered = [(k, ask_quick.resolve(k, self.fields)) for k in ask_quick.builtin()]
        ready = {k.button: k for k, q in offered if q is not None}
        with ui.card().classes("w-full gap-3"):
            ui.select(list(ready), value=self.quick.button if self.quick else None,
                      label="Choose a question",
                      on_change=lambda e: self.choose(ready.get(e.value))) \
                .props("outlined").classes("w-full max-w-md").mark("ask-quick")
            waiting = len(offered) - len(ready)
            if waiting:
                ui.label(f"{waiting} more once you run " + ", ".join(self.missing()) + ".") \
                    .classes("text-xs opacity-60")
            if self.quick is not None:
                self._sentence()
            self._saved()

    def choose(self, quick: ask_quick.Quick | None) -> None:
        self.quick = quick
        self.blanks = [""] * (ask_quick.blanks(quick.question) if quick else 0)
        self.panel.refresh()

    def _sentence(self) -> None:
        """The chosen question with its blanks as boxes in the sentence itself."""
        q = ask_quick.resolve(self.quick, self.fields)
        blank_fields = [r.field for r in (*q.rows, *q.groups)
                        for v in r.values if v == ask_quick.BLANK]
        parts = self.quick.asks.split("{}")
        with ui.row().classes("items-center gap-2 flex-wrap text-lg"):
            for i, part in enumerate(parts):
                if part.strip():
                    ui.label(part.strip())
                if i < len(blank_fields):
                    f = self.fields[blank_fields[i]]

                    def fill(e, k=i) -> None:
                        self.blanks[k] = e.value or ""
                    if f.values:
                        box = ui.select(list(f.values), value=self.blanks[i] or None,
                                        with_input=True, on_change=fill)
                    else:
                        box = ui.input(placeholder=f.title, value=self.blanks[i], on_change=fill)
                    box.props("dense outlined").classes("min-w-40").mark(f"ask-blank-{i}")
            ui.button("Ask", icon="search", on_click=self.ask_quick) \
                .props("color=primary").mark("ask")

    def _saved(self) -> None:
        kept = ask_quick.saved(self.project)
        if not kept:
            return
        ui.label("Your saved questions").classes("text-sm font-medium tk-divide pt-2 w-full")
        with ui.column().classes("gap-1"):
            for k in kept:
                with ui.row().classes("items-center gap-1"):
                    ui.button(k.button, icon="bookmark",
                              on_click=lambda _, k=k: self.results.ask(k.question)) \
                        .props("flat dense no-caps align=left")
                    ui.button(icon="close", on_click=lambda _, k=k: self.forget(k.button)) \
                        .props("flat round dense size=sm").tooltip("Remove this saved question")

    def ask_quick(self) -> None:
        if any(not v.strip() for v in self.blanks):
            guard(ToolkitError("Fill in every blank first."))
            return
        self.results.ask(ask_quick.fill(ask_quick.resolve(self.quick, self.fields), self.blanks))

    def forget(self, button: str) -> None:
        ask_quick.forget(self.project, button)
        self.panel.refresh()

    # --- the builder --------------------------------------------------------------------------

    def _builder(self) -> None:
        s = self.state
        with ui.card().classes("w-full gap-2"):
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.select(dict(ask.VERBS), value=s["verb"],
                          on_change=lambda e: self.set("verb", e.value)).props("dense outlined")
                ui.select({k: k for k in self.subjects()}, value=s["subject"],
                          on_change=lambda e: self.set("subject", e.value)) \
                    .props("dense outlined")
                ui.label("where").classes("text-sm")
            for i, row in enumerate(s["rows"]):
                self._row_editor(s["rows"], i, row, joinable=i > 0)
            with ui.row().classes("items-center gap-2"):
                ui.button("and…", icon="add", on_click=lambda: self.add("rows", "and")) \
                    .props("flat dense no-caps")
                ui.button("or…", icon="add", on_click=lambda: self.add("rows", "or")) \
                    .props("flat dense no-caps")
                info("“and” narrows: both must be true. “or” widens: either will do.\n"
                     "“A and B or C” means (A and B) or C.")
            if s["verb"] == "compare":
                ui.label("Compare these groups (one column each):").classes("text-sm mt-2")
                for i, row in enumerate(s["groups"]):
                    self._row_editor(s["groups"], i, row, joinable=False,
                                     prefix="vs" if i else "")
                ui.button("another group", icon="add", on_click=lambda: self.add("groups")) \
                    .props("flat dense no-caps")
            with ui.row().classes("items-center gap-2 mt-2"):
                ui.label("Speakers:").classes("text-sm")
                ui.toggle(ROLE_CHOICES, value=self.role_choice(),
                          on_change=lambda e: self.set_roles(e.value)).props("dense no-caps")
            if self.missing():
                ui.label("More to ask about after you run: " + ", ".join(self.missing())) \
                    .classes("text-xs opacity-60")
            with ui.row().classes("gap-2"):
                ui.button("Ask", icon="search", on_click=self.ask_built) \
                    .props("color=primary").mark("ask")
                ui.button("Save as a quick question", icon="bookmark_add", on_click=self.save) \
                    .props("flat dense no-caps")

    def _row_editor(self, rows: list, i: int, row: dict, joinable: bool, prefix: str = "") -> None:
        options = self.field_options()
        with ui.row().classes("items-center gap-2 flex-wrap"):
            if joinable:
                ui.select({"and": "and", "or": "or"}, value=row["join"],
                          on_change=lambda e, r=row: r.__setitem__("join", e.value)) \
                    .props("dense borderless").classes("w-14").mark(f"ask-join-{i}")
            elif prefix:
                ui.label(prefix).classes("text-sm w-14")
            ui.select(options, value=row["field"] or None,
                      on_change=lambda e, r=row: self.set_row(r, "field", e.value)) \
                .props("dense outlined").classes("min-w-32").mark(f"ask-field-{i}")
            f = self.fields.get(row["field"])
            if f is None:
                return
            conds = {c: ask.CONDITIONS[c].words for c in f.conditions_for(self.state["subject"])}
            ui.select(conds, value=row["condition"] or None,
                      on_change=lambda e, r=row: self.set_row(r, "condition", e.value)) \
                .props("dense outlined").mark(f"ask-condition-{i}")
            self._values_editor(row, f, i)
            if len(rows) > 1:
                ui.button(icon="close", on_click=lambda _, rs=rows, k=i: self.remove(rs, k)) \
                    .props("flat round dense size=sm")

    def _values_editor(self, row: dict, f: ask.Field, i: int) -> None:
        want = ask.CONDITIONS[row["condition"]].n_values if row["condition"] else "one"
        if want == "text":
            ui.input(placeholder="words to look for", value=(row["values"] or [""])[0],
                     on_change=lambda e, r=row: r.__setitem__("values", [e.value])) \
                .props("dense outlined").classes("min-w-48").mark(f"ask-words-{i}")
        elif want == "texts":
            ui.input(placeholder="words, separated by commas", value=", ".join(row["values"]),
                     on_change=lambda e, r=row: r.__setitem__(
                         "values", [w.strip() for w in (e.value or "").split(",") if w.strip()])) \
                .props("dense outlined").classes("min-w-64").mark(f"ask-words-{i}")
        elif want == "some":
            ui.select(list(f.values), multiple=True, value=list(row["values"]),
                      on_change=lambda e, r=row: r.__setitem__("values", list(e.value or []))) \
                .props("dense outlined use-chips").classes("min-w-48")
        else:
            for k in range(2 if want == "two" else 1):
                if k:
                    ui.label("alongside").classes("text-sm")
                ui.select(list(f.values), value=(row["values"][k] or None)
                          if k < len(row["values"]) else None, with_input=True,
                          on_change=lambda e, r=row, k=k: self.set_value(r, k, e.value)) \
                    .props("dense outlined").classes("min-w-40").mark(f"ask-value-{i}-{k}")

    def subjects(self) -> list[str]:
        return [s for s in ask.VERB_SUBJECTS[self.state["verb"]]
                if s != "clips" or self.clipped]

    def field_options(self) -> dict[str, str]:
        subject = self.state["subject"]
        return {k: f.title for k, f in self.fields.items()
                if f.available and subject in f.subjects and f.conditions_for(subject)}

    def missing(self) -> list[str]:
        return sorted({NEEDS_WORDS.get(f.needs, f.needs) for f in self.fields.values()
                       if not f.available})

    def role_choice(self) -> str:
        roles = self.state["roles"]
        return "both" if len(roles) > 1 else roles[0]

    def set(self, key: str, value) -> None:
        self.state[key] = value
        self.normalise()
        self.panel.refresh()

    def set_roles(self, choice: str) -> None:
        self.state["roles"] = ["Narrator", "Interviewer"] if choice == "both" else [choice]

    def set_row(self, row: dict, key: str, value) -> None:
        row[key] = value
        if key == "field":
            row["condition"], row["values"] = "", []
        self.normalise()
        self.panel.refresh()

    def set_value(self, row: dict, k: int, value) -> None:
        while len(row["values"]) <= k:
            row["values"].append("")
        row["values"][k] = value or ""

    def add(self, key: str, join: str = "and") -> None:
        self.state[key].append(_blank_row(join))
        self.normalise()
        self.panel.refresh()

    def remove(self, rows: list, k: int) -> None:
        del rows[k]
        self.panel.refresh()

    def normalise(self) -> None:
        """Make every choice one the current verb and subject allow, keeping what still fits."""
        s = self.state
        if s["subject"] not in self.subjects():
            s["subject"] = self.subjects()[0]
        if s["verb"] == "compare":
            while len(s["groups"]) < 2:
                s["groups"].append(_blank_row())
        else:
            s["groups"] = []
        options = self.field_options()
        for row in (*s["rows"], *s["groups"]):
            if row["field"] not in options:
                row["field"] = next(iter(options), "")
                row["condition"], row["values"] = "", []
            f = self.fields.get(row["field"])
            if f is None:
                continue
            conds = f.conditions_for(s["subject"])
            if row["condition"] not in conds:
                row["condition"] = conds[0]
            want = ask.CONDITIONS[row["condition"]].n_values
            n = {"one": 1, "text": 1, "two": 2}.get(want)
            free = want in ("text", "texts")
            vals = [v for v in row["values"] if free or v in f.values or v == ""]
            row["values"] = (vals + [""] * 2)[:n] if n else vals

    def built(self) -> ask.Question:
        s = self.state
        if any(not v for row in (*s["rows"], *s["groups"]) for v in row["values"] or [""]):
            raise ToolkitError("Pick something for every box first.")
        return ask_quick.question_from(s)

    def ask_built(self) -> None:
        try:
            q = self.built()
        except ToolkitError as e:
            guard(e)
            return
        self.results.ask(q)

    def save(self) -> None:
        try:
            ask_quick.save(self.project, self.built(), self.fields)
        except ToolkitError as e:
            guard(e)
            return
        ui.notify("Saved. You will find it under Quick question.", type="positive")


class _Results:
    """An answer, and the trail of answers opened up from it. Shared by the page and the view."""

    def __init__(self, project, fields: dict[str, ask.Field], preview: bool = True):
        self.project = project
        self.fields = fields
        self.preview = preview          # the page shows the first rows; the view shows them all
        self.trail: list[tuple[ask.Question, pd.DataFrame]] = []

    def draw(self) -> None:
        self.dialog = ui.dialog()
        self.body = ui.refreshable(self._body)
        self.body()

    def ask(self, q: ask.Question, keep_trail: bool = False) -> None:
        try:
            df = ask.answer(self.project, q)
        except ToolkitError as e:
            guard(e)
            return
        if not keep_trail:
            self.trail = []
        self.trail.append((q, df))
        self.body.refresh()

    def back(self) -> None:
        self.trail.pop()
        self.body.refresh()

    def _body(self) -> None:
        if not self.trail:
            return
        q, df = self.trail[-1]
        quicks = [*ask_quick.builtin(), *ask_quick.saved(self.project)]
        with ui.card().classes("w-full gap-2"):
            if len(self.trail) > 1:
                ui.button("Back", icon="arrow_back", on_click=self.back) \
                    .props("flat dense no-caps").mark("ask-back")
            ui.label(ask_quick.wording(q, quicks, self.fields)).classes("text-lg font-medium")
            with ui.row().classes("items-center gap-2"):
                ui.label(counts(df)).classes("text-sm opacity-70")
                if "match" in df.columns and (df["match"] == "possible").any():
                    info(MATCH_HELP)
            with ui.row().classes("gap-2"):
                if self.preview:
                    ui.button("Open in new tab", icon="open_in_new",
                              on_click=lambda: ui.navigate.to(view_href(q), new_tab=True)) \
                        .props("flat dense no-caps")
                export_button(df)
            if df.empty:
                ui.label("Nothing matches.").classes("text-sm")
                return
            if q.verb != "compare":
                ui.label("Click a row to read it in its transcript."
                         if "clip" in df.columns or "paragraph" in df.columns
                         else "Click a row to see its clips.").classes("text-xs opacity-60")
            def open_row(i: int) -> None:
                self.open(df.iloc[i].to_dict(), q)

            clickable = q.verb != "compare"
            if self.preview or "text" not in df.columns:
                t = results_table(df.head(PREVIEW_ROWS) if self.preview else df,
                                  clip_text=TEXT_PREVIEW)
                if clickable:
                    t.classes("cursor-pointer")
                    t.on("rowClick", lambda e: open_row(int(e.args[1]["_id"])))
            else:
                results_cards(df, open_row)
            if self.preview and len(df) > PREVIEW_ROWS:
                ui.label(f"Showing the first {PREVIEW_ROWS} of {len(df)}. Open in new tab to "
                         f"see them all.").classes("text-xs opacity-60")

    def open(self, result: dict, q: ask.Question) -> None:
        if "clip" in result or "paragraph" in result:
            self.show_transcript(result, q)
        else:
            self.ask(ask.narrow(self.project, q, result), keep_trail=True)

    def show_transcript(self, result: dict, q: ask.Question) -> None:
        p = ask.passage(self.project, result)
        words = [v for r in q.rows if r.field == "words" and r.condition in FOUND_BY_WORDS
                 for v in r.values]
        self.dialog.clear()
        with self.dialog, ui.card().classes("w-full max-w-3xl gap-1"):
            with ui.row().classes("w-full items-center"):
                ui.label(p.session).classes("text-lg font-medium flex-1")
                ui.button(icon="close", on_click=self.dialog.close).props("flat round dense")
            if result.get("label"):
                ui.label(str(result["label"])).classes("text-sm opacity-70")
            with ui.scroll_area().classes("w-full").style("height: 70vh"):
                for r in p.transcript.itertuples():
                    inside = p.first <= r.paragraph <= p.last
                    with ui.row().classes("w-full gap-2 px-2 py-1 rounded no-wrap"
                                          + (f" {theme.FOUND}" if inside else "")) \
                            .props(f"id=ask-p{r.paragraph}"):
                        ui.label(f"{r.time} {r.speaker}".strip()) \
                            .classes("text-xs opacity-60 w-28 shrink-0")
                        text = ui.html(marked(r.text, words if inside else []), sanitize=False) \
                            .classes("text-sm")
                        if inside:
                            text.mark("ask-passage")
        self.dialog.open()
        ui.run_javascript(f"setTimeout(() => document.getElementById('ask-p{p.first}')"
                          f"?.scrollIntoView({{block: 'center'}}), 100)")


# --- shared helpers ---------------------------------------------------------------------------

def marked(text: str, words: list[str]) -> str:
    """The text, escaped, with the words it was found by marked."""
    out = html.escape(str(text))
    for w in sorted({w for w in words if w}, key=len, reverse=True):
        out = re.sub(f"({re.escape(html.escape(w))})", rf'<mark class="{theme.FOUND}">\1</mark>',
                     out, flags=re.IGNORECASE)
    return out


def counts(df: pd.DataFrame) -> str:
    if "match" not in df.columns:
        return f"{len(df)} interviews"
    clear = int((df["match"] == "clear").sum())
    possible = len(df) - clear
    return f"{len(df)} found" + (f" · {clear} clear, {possible} possible" if possible else "")


def results_table(df: pd.DataFrame, clip_text: int | None = None):
    shown = df.copy()
    if clip_text and "text" in shown.columns:
        shown["text"] = shown["text"].map(
            lambda t: t if len(str(t)) <= clip_text else str(t)[:clip_text].rstrip() + "…")
    shown = shown.fillna("").astype(str)
    columns = [{"name": c, "label": c.capitalize(), "field": c, "align": "left",
                "style": "white-space: normal; min-width: 18rem" if c == "text" else ""}
               for c in shown.columns]
    rows = [{"_id": i, **r} for i, r in enumerate(shown.to_dict("records"))]
    return ui.table(rows=rows, columns=columns, row_key="_id").classes("w-full") \
        .props("dense flat wrap-cells")


def results_cards(df: pd.DataFrame, on_click) -> None:
    """One card per result, its text at full width: for reading, where a table's narrow column
    would wrap a clip into a ribbon."""
    meta = [c for c in df.columns if c not in ("text", "match")]
    for i, r in enumerate(df.fillna("").astype(str).to_dict("records")):
        with ui.card().classes("w-full gap-1 cursor-pointer").on("click", lambda _, i=i: on_click(i)) \
                .mark(f"ask-card-{i}"):
            with ui.row().classes("w-full items-center gap-3 text-xs opacity-70"):
                for c in meta:
                    if r[c]:
                        ui.label(f"{c.capitalize()}: {r[c]}")
                if r.get("match") == "possible":
                    ui.label("possible match").classes("tk-caution font-medium")
            ui.label(r["text"]).classes("text-sm whitespace-pre-line leading-relaxed")


def export_button(df: pd.DataFrame) -> None:
    has_possible = "match" in df.columns and (df["match"] == "possible").any()
    if not has_possible:
        ui.button("Export", icon="download", on_click=lambda: download(df)) \
            .props("flat dense no-caps")
        return
    with ui.button("Export", icon="download").props("flat dense no-caps"):
        with ui.menu():
            ui.menu_item("Clear matches only",
                         on_click=lambda: download(df[df["match"] == "clear"]))
            ui.menu_item("Clear and possible matches", on_click=lambda: download(df))


def download(df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    ui.download.content(buf.getvalue(), "ask_answer.xlsx")


def view_href(question: ask.Question) -> str:
    return f"{VIEW}?q={quote(json.dumps(ask_quick.question_to(question)))}"


def view_page(q: str = "") -> None:
    """One answer, all of it, on a page of its own — still clickable."""
    with shell(HREF):
        if CONTEXT.project is None:
            return
        project = CONTEXT.require_project()
        try:
            question = ask_quick.question_from(json.loads(q))
            fields = ask.fields(project)
        except (ToolkitError, ValueError, KeyError) as e:
            guard(ToolkitError(f"Could not open that answer: {e}"))
            return
        results = _Results(project, fields, preview=False)
        results.draw()
        results.ask(question)


def register() -> None:
    ui.page(HREF, title="Ask — Transcript Toolkit")(ask_page)
    ui.page(VIEW, title="Answer — Transcript Toolkit")(view_page)
