"""Ask page: question the collection by building a sentence, or by clicking a quick question.

Everything is answered by `core/ask.py` from the files the pipeline already wrote: no command
runs, nothing is paid for, nothing in the project changes except a saved question. Export hands
the answer to the browser as a download; it is not a deliverable and never touches outputs/.
"""
from __future__ import annotations

import io
import json
from urllib.parse import quote

import pandas as pd
from nicegui import ui

from ...core import ask, ask_quick
from ...errors import ToolkitError
from ...steps import freshness
from ..context import CONTEXT
from .common import guard, info, section, shell

HREF = "/ask"
VIEW = "/ask/view"
PREVIEW_ROWS = 50
TEXT_PREVIEW = 300              # characters of a clip's text shown in the preview table

ROLE_CHOICES = {"Narrator": "Narrators", "Interviewer": "Interviewers", "both": "Both"}
SUBJECT_WORDS = {"paragraphs": "paragraphs", "clips": "clips", "interviews": "interviews",
                 "speakers": "speakers"}
NEEDS_WORDS = {"import": "Import", "label": "Label", "topics tag": "Topics",
               "locations tag": "Locations", "locations map": "Locations (map)"}
MATCH_HELP = ("Clear: the clip is about it (the top topic score).\n"
              "Possible: it may touch on it (a score between zero and the top).")


def _empty() -> dict:
    return {"verb": "find", "subject": "clips", "roles": ["Narrator"],
            "rows": [{"field": "", "condition": "", "values": []}], "groups": []}


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
    """One page's question and answer. State is a plain dict shaped like a saved question."""

    def __init__(self, project, fields: dict[str, ask.Field]):
        self.project = project
        self.fields = fields
        self.clipped = freshness.wrote_its_results(project, "clip")
        self.state = _empty()
        if not self.clipped:
            self.state["subject"] = "paragraphs"
        self.answer: pd.DataFrame | None = None
        self.question: ask.Question | None = None

    # --- the parts of the page ----------------------------------------------------------------

    def draw(self) -> None:
        self.quick_row = ui.refreshable(self._quick_row)
        self.builder = ui.refreshable(self._builder)
        self.results = ui.refreshable(self._results)
        self.normalise()
        self.quick_row()
        self.builder()
        self.results()

    def quicks(self) -> list[ask_quick.Quick]:
        return [*ask_quick.builtin(), *ask_quick.saved(self.project)]

    def _quick_row(self) -> None:
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-1"):
                ui.label("Quick questions").classes("text-sm font-medium")
                info("Click one to fill in the question below, then pick what goes in the "
                     "blanks. Greyed-out ones need a step you have not run yet.")
            with ui.row().classes("gap-2 flex-wrap"):
                for quick in self.quicks():
                    q = ask_quick.resolve(quick, self.fields)
                    b = ui.button(quick.button, on_click=lambda _, q=q: self.use(q)) \
                        .props("outline dense no-caps" + ("" if q else " disable"))
                    if quick.saved:
                        b.props("icon=bookmark")
                        ui.button(icon="close",
                                  on_click=lambda _, k=quick: self.forget(k.button)) \
                            .props("flat round dense size=sm").tooltip("Remove this saved question")

    def _builder(self) -> None:
        s = self.state
        with ui.card().classes("w-full gap-2"):
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.select({k: v for k, v in ask.VERBS.items()}, value=s["verb"],
                          on_change=lambda e: self.set("verb", e.value)).props("dense outlined")
                ui.select({k: SUBJECT_WORDS[k] for k in self.subjects()}, value=s["subject"],
                          on_change=lambda e: self.set("subject", e.value)) \
                    .props("dense outlined")
                ui.label("where").classes("text-sm")
            for i, row in enumerate(s["rows"]):
                self._row_editor(s["rows"], i, row, prefix="" if i == 0 else "and")
            with ui.row().classes("gap-2"):
                ui.button("and…", icon="add", on_click=lambda: self.add("rows")) \
                    .props("flat dense no-caps")
            if s["verb"] == "compare":
                ui.label("Compare these groups (one column each):").classes("text-sm mt-2")
                for i, row in enumerate(s["groups"]):
                    self._row_editor(s["groups"], i, row, prefix="vs" if i else "")
                ui.button("another group", icon="add", on_click=lambda: self.add("groups")) \
                    .props("flat dense no-caps")
            with ui.row().classes("items-center gap-2 mt-2"):
                ui.label("Speakers:").classes("text-sm")
                ui.toggle(ROLE_CHOICES, value=self.role_choice(),
                          on_change=lambda e: self.set_roles(e.value)).props("dense no-caps")
            missing = sorted({NEEDS_WORDS.get(f.needs, f.needs) for f in self.fields.values()
                              if not f.available})
            if missing:
                ui.label("More to ask about after you run: " + ", ".join(missing)) \
                    .classes("text-xs opacity-60")
            ui.button("Ask", icon="search", on_click=self.run).props("color=primary").mark("ask")

    def _row_editor(self, rows: list, i: int, row: dict, prefix: str) -> None:
        options = self.field_options()
        with ui.row().classes("items-center gap-2 flex-wrap"):
            if prefix:
                ui.label(prefix).classes("text-sm w-8")
            ui.select(options, value=row["field"] or None,
                      on_change=lambda e, r=row: self.set_row(r, "field", e.value)) \
                .props("dense outlined").classes("min-w-32")
            f = self.fields.get(row["field"])
            if f is None:
                return
            conds = {c: ask.CONDITIONS[c].words for c in f.conditions_for(self.state["subject"])}
            ui.select(conds, value=row["condition"] or None,
                      on_change=lambda e, r=row: self.set_row(r, "condition", e.value)) \
                .props("dense outlined")
            self._values_editor(row, f)
            if len(rows) > 1:
                ui.button(icon="close", on_click=lambda _, rs=rows, k=i: self.remove(rs, k)) \
                    .props("flat round dense size=sm")

    def _values_editor(self, row: dict, f: ask.Field) -> None:
        want = ask.CONDITIONS[row["condition"]].n_values if row["condition"] else "one"
        if want == "text":
            ui.input(placeholder="words to look for", value=(row["values"] or [""])[0],
                     on_change=lambda e, r=row: r.__setitem__("values", [e.value])) \
                .props("dense outlined").classes("min-w-48").mark("ask-words")
        elif want == "some":
            ui.select(list(f.values), multiple=True, value=list(row["values"]),
                      on_change=lambda e, r=row: r.__setitem__("values", list(e.value or []))) \
                .props("dense outlined use-chips").classes("min-w-48")
        else:
            for k in range(2 if want == "two" else 1):
                if k:
                    ui.label("alongside" if want == "two" else "").classes("text-sm")
                ui.select(list(f.values), value=(row["values"][k] or None)
                          if k < len(row["values"]) else None, with_input=True,
                          on_change=lambda e, r=row, k=k: self.set_value(r, k, e.value)) \
                    .props("dense outlined").classes("min-w-40")

    def _results(self) -> None:
        if self.answer is None:
            return
        df, q = self.answer, self.question
        with ui.card().classes("w-full gap-2"):
            ui.label(ask_quick.wording(q, self.quicks(), self.fields)) \
                .classes("text-lg font-medium")
            with ui.row().classes("items-center gap-2"):
                ui.label(self.counts(df)).classes("text-sm opacity-70")
                if "match" in df.columns and (df["match"] == "possible").any():
                    info(MATCH_HELP)
            with ui.row().classes("gap-2"):
                ui.button("Save as a quick question", icon="bookmark_add", on_click=self.save) \
                    .props("flat dense no-caps")
                ui.button("Open in new tab", icon="open_in_new",
                          on_click=lambda: ui.navigate.to(view_href(q), new_tab=True)) \
                    .props("flat dense no-caps")
                self._export_button(df)
            if df.empty:
                ui.label("Nothing matches.").classes("text-sm")
                return
            results_table(df.head(PREVIEW_ROWS), clip_text=TEXT_PREVIEW)
            if len(df) > PREVIEW_ROWS:
                ui.label(f"Showing the first {PREVIEW_ROWS} of {len(df)}. Open in new tab to "
                         f"see them all.").classes("text-xs opacity-60")

    def _export_button(self, df: pd.DataFrame) -> None:
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

    # --- what the controls do -----------------------------------------------------------------

    def subjects(self) -> list[str]:
        return [s for s in ask.VERB_SUBJECTS[self.state["verb"]]
                if s != "clips" or self.clipped]

    def field_options(self) -> dict[str, str]:
        subject = self.state["subject"]
        return {k: f.title for k, f in self.fields.items()
                if f.available and subject in f.subjects and f.conditions_for(subject)}

    def role_choice(self) -> str:
        roles = self.state["roles"]
        return "both" if len(roles) > 1 else roles[0]

    def set(self, key: str, value) -> None:
        self.state[key] = value
        self.normalise()
        self.builder.refresh()

    def set_roles(self, choice: str) -> None:
        self.state["roles"] = ["Narrator", "Interviewer"] if choice == "both" else [choice]

    def set_row(self, row: dict, key: str, value) -> None:
        row[key] = value
        if key == "field":
            row["condition"], row["values"] = "", []
        self.normalise()
        self.builder.refresh()

    def set_value(self, row: dict, k: int, value) -> None:
        while len(row["values"]) <= k:
            row["values"].append("")
        row["values"][k] = value or ""

    def add(self, key: str) -> None:
        self.state[key].append({"field": "", "condition": "", "values": []})
        self.normalise()
        self.builder.refresh()

    def remove(self, rows: list, k: int) -> None:
        del rows[k]
        self.builder.refresh()

    def normalise(self) -> None:
        """Make every choice one the current verb and subject allow, keeping what still fits."""
        s = self.state
        if s["subject"] not in self.subjects():
            s["subject"] = self.subjects()[0]
        if s["verb"] == "compare":
            while len(s["groups"]) < 2:
                s["groups"].append({"field": "", "condition": "", "values": []})
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
            vals = [v for v in row["values"] if want == "text" or v in f.values or v == ""]
            row["values"] = (vals + [""] * 2)[:n] if n else vals

    def use(self, question: ask.Question | None) -> None:
        if question is None:
            return
        d = ask_quick.question_to(question)
        for row in (*d["rows"], *d["groups"]):
            row["values"] = ["" if v == ask_quick.BLANK else v for v in row["values"]]
        self.state = d
        self.normalise()
        self.builder.refresh()

    def current(self) -> ask.Question:
        s = self.state
        if any(not v for row in (*s["rows"], *s["groups"]) for v in row["values"] or [""]):
            raise ToolkitError("Pick something for every box first.")
        return ask_quick.question_from(s)

    def run(self) -> None:
        try:
            q = self.current()
            self.answer, self.question = ask.answer(self.project, q), q
        except ToolkitError as e:
            guard(e)
            return
        self.results.refresh()

    def save(self) -> None:
        try:
            ask_quick.save(self.project, self.question, self.fields)
        except ToolkitError as e:
            guard(e)
            return
        ui.notify("Saved. It is now one of your quick questions.", type="positive")
        self.quick_row.refresh()

    def forget(self, button: str) -> None:
        ask_quick.forget(self.project, button)
        self.quick_row.refresh()

    @staticmethod
    def counts(df: pd.DataFrame) -> str:
        if "match" not in df.columns:
            return f"{len(df)} interviews"
        clear = int((df["match"] == "clear").sum())
        possible = len(df) - clear
        return f"{len(df)} found" + (f" · {clear} clear, {possible} possible" if possible else "")


# --- shared by the page and the new-tab view --------------------------------------------------

def results_table(df: pd.DataFrame, clip_text: int | None = None) -> None:
    shown = df.copy()
    if clip_text and "text" in shown.columns:
        shown["text"] = shown["text"].map(
            lambda t: t if len(str(t)) <= clip_text else str(t)[:clip_text].rstrip() + "…")
    shown = shown.fillna("").astype(str)
    columns = [{"name": c, "label": c.capitalize(), "field": c, "align": "left",
                "style": "white-space: normal; min-width: 18rem" if c == "text" else ""}
               for c in shown.columns]
    rows = [{"_id": i, **r} for i, r in enumerate(shown.to_dict("records"))]
    ui.table(rows=rows, columns=columns, row_key="_id").classes("w-full") \
        .props("dense flat wrap-cells")


def download(df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    ui.download.content(buf.getvalue(), "ask_answer.xlsx")


def view_href(question: ask.Question) -> str:
    return f"{VIEW}?q={quote(json.dumps(ask_quick.question_to(question)))}"


def view_page(q: str = "") -> None:
    """One answer, all of it, on a page of its own."""
    with shell(HREF):
        if CONTEXT.project is None:
            return
        project = CONTEXT.require_project()
        try:
            question = ask_quick.question_from(json.loads(q))
            fields = ask.fields(project)
            df = ask.answer(project, question)
        except (ToolkitError, ValueError, KeyError) as e:
            guard(ToolkitError(f"Could not open that answer: {e}"))
            return
        quicks = [*ask_quick.builtin(), *ask_quick.saved(project)]
        ui.label(ask_quick.wording(question, quicks, fields)).classes("text-xl font-medium")
        ui.label(_Asker.counts(df)).classes("text-sm opacity-70")
        if df.empty:
            ui.label("Nothing matches.").classes("text-sm")
        else:
            results_table(df)


def register() -> None:
    ui.page(HREF, title="Ask — Transcript Toolkit")(ask_page)
    ui.page(VIEW, title="Answer — Transcript Toolkit")(view_page)
