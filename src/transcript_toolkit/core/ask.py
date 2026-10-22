"""Ask: questions about a collection, built from a fixed vocabulary and answered by filtering
the deliverables already on disk. No model, no cost, nothing written.

A question is a sentence — verb, subject, then "where" rows of field · condition · values —
plus which speaker roles count. Rows are evaluated on a *base unit*: clips once the collection
is clipped, paragraphs before that (and always for the paragraphs subject). A row gives each
unit a match strength: CLEAR, POSSIBLE or nothing. Only "might talk about" ever gives POSSIBLE:
a topic score between zero and the top of `score_values`. The top score is CLEAR, the rule the
rollup counts by. Rows combine by their weakest match; a subject made of several units
(an interview, a speaker) takes its strongest.

Negative conditions ("never mentions", "is not") apply to the *subject*, not the unit: a
speaker who never mentions Family is one with no unit that does — not one who has a unit that
doesn't. "Mostly talks about" is the interview rollup's tag, so it too is about the subject.

Which file a field reads, and so whether it is there yet, is `steps/freshness.py`'s to say;
the only paths named here are the companion tables written beside those in the same run.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, replace
from importlib import resources
from functools import cached_property

import pandas as pd
import yaml

from ..errors import ToolkitError
from ..project import Project
from ..steps import freshness
from . import tables
from .config import load_root_config, load_step_config
from .ids import narrator_key

CLEAR, POSSIBLE = 2, 1
MATCH_WORDS = {CLEAR: "clear", POSSIBLE: "possible"}

VERBS = {"find": "Find", "count": "Count", "compare": "Compare"}
SUBJECTS = ("paragraphs", "clips", "interviews", "speakers")
# Count and Compare report per interview how many units matched, so they count units.
VERB_SUBJECTS = {"find": SUBJECTS, "count": ("paragraphs", "clips"),
                 "compare": ("paragraphs", "clips")}
ROLES = ("Narrator", "Interviewer", "Other")      # as core/docx.infer_role assigns them


@dataclass(frozen=True)
class Condition:
    words: str              # as the sentence reads after its field: "topic is Education"
    n_values: str           # "one" | "some" | "two" | "text"
    negative: bool = False
    subjects: tuple[str, ...] = SUBJECTS


CONDITIONS = {
    "talks_about": Condition("is", "one"),
    "might_talk_about": Condition("might be", "one"),
    "never_mentions": Condition("is not", "one", negative=True),
    "only_talks_about": Condition("is only", "one", subjects=("clips",)),
    "mostly_talks_about": Condition("is mostly", "one", subjects=("interviews",)),
    "any_of": Condition("is any of", "some"),
    "all_of": Condition("includes all of", "some"),
    "alongside": Condition("includes", "two", subjects=("clips", "speakers")),
    "is": Condition("is", "one"),
    "is_not": Condition("is not", "one", negative=True),
    "contains": Condition("contains", "text"),
    "not_contains": Condition("does not contain", "text", negative=True),
    "contains_any": Condition("contains any of", "texts"),
    "contains_all": Condition("contains all of", "texts"),
    "whole_word": Condition("contains the whole word", "text"),
}
TEXT_CONDITIONS = ("contains", "not_contains", "contains_any", "contains_all", "whole_word")
N_VALUES = {"one": "one value", "text": "one value", "two": "two values",
            "some": "at least one value", "texts": "at least one value"}
# What a negative condition rules a subject out for having.
POSITIVE = {"never_mentions": "might_talk_about", "is_not": "is", "not_contains": "contains"}

PARAGRAPH_FIELDS = ("words", "speaker", "interview", "timing")   # answered from the transcript


@dataclass(frozen=True)
class Field:
    key: str
    title: str
    conditions: tuple[str, ...]
    values: tuple[str, ...] = ()        # the choices offered; empty = free text
    available: bool = True
    needs: str = ""                     # the step to run first, when not available

    def conditions_for(self, subject: str) -> tuple[str, ...]:
        return tuple(c for c in self.conditions if subject in CONDITIONS[c].subjects)

    @property
    def subjects(self) -> tuple[str, ...]:
        if self.key in PARAGRAPH_FIELDS:
            return SUBJECTS
        return tuple(s for s in SUBJECTS if s != "paragraphs")


@dataclass(frozen=True)
class Row:
    field: str
    condition: str
    values: tuple[str, ...]
    join: str = "and"       # how it joins the row before it ("and" binds tighter than "or")


JOINS = ("and", "or")


@dataclass(frozen=True)
class Question:
    verb: str
    subject: str
    rows: tuple[Row, ...] = ()
    roles: tuple[str, ...] = ("Narrator",)
    groups: tuple[Row, ...] = ()        # Compare: one column per group


# What a project made before Ask existed is missing, copied in on first use. Never overwrites.
NEW_FILES = (("scaffold/advanced/ask.yaml", "advanced/ask.yaml"),
             ("prompts/ask_summary.md", "prompts/ask_summary.md"))


def add_missing_files(project: Project) -> list[str]:
    """Copy in the files a project made before Ask lacks, and append to a settings file the
    settings it lacks (each with its comment), leaving everything already there as it is."""
    defaults = resources.files("transcript_toolkit") / "defaults"
    added = []
    for src, dest in NEW_FILES:
        path = project.root / dest
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with resources.as_file(defaults / src) as f:
                shutil.copyfile(f, path)
            added.append(dest)
        elif dest.endswith(".yaml"):
            missing = _missing_settings(path.read_text(), (defaults / src).read_text())
            if missing:
                path.write_text(path.read_text().rstrip("\n") + "\n\n" + missing + "\n")
                added.append(f"{dest} (new settings)")
    return added


def _missing_settings(have_text: str, default_text: str) -> str:
    """The default file's settings that `have_text` lacks, each with the comment above it."""
    have = yaml.safe_load(have_text) or {}
    blocks, comment = [], []
    for line in default_text.splitlines():
        key = re.match(r"^([a-z_]+):", line)
        if key:
            if key.group(1) not in have:
                blocks.append("\n".join([*comment, line]))
            comment = []
        elif line.startswith("#"):
            comment.append(line)
        else:
            comment = []
    return "\n\n".join(blocks)


# --- the data, read once per question ---------------------------------------------------------

class Collection:
    """The deliverables a question reads, loaded on first use. Full runs only, never the demo."""

    def __init__(self, project: Project):
        self.project = project

    def _keyed(self, df: pd.DataFrame) -> pd.DataFrame:
        regex = load_step_config(self.project, "import")["session_regex"]
        return df.assign(interview_key=[narrator_key(i, regex) for i in df["interview_id"]])

    @cached_property
    def paragraphs(self) -> pd.DataFrame:
        return self._keyed(tables.load_paragraphs(self.project))

    @cached_property
    def clipped(self) -> pd.DataFrame:
        """Paragraphs that belong to a clip (procedural ones do not)."""
        p = pd.read_parquet(tables.paragraphs_clipped_path(self.project))
        return self._keyed(p[p["clip_id"].notna() & (p["clip_id"] != "procedural")])

    @cached_property
    def clips(self) -> pd.DataFrame:
        return self._keyed(pd.read_parquet(tables.clips_path(self.project))).set_index("clip_id")

    @cached_property
    def labels(self) -> pd.Series:
        df = pd.read_parquet(self._out(freshness.WRITES["label"][0]))
        return df.set_index("clip_id")["label"]

    @cached_property
    def speaker_min_share(self) -> float:
        add_missing_files(self.project)
        return float(load_step_config(self.project, "ask")["speaker_min_share"])

    @cached_property
    def top_score(self) -> int:
        return max(int(v) for v in load_step_config(self.project, "topics")["score_values"])

    @cached_property
    def untimed(self) -> set[str]:
        return tables.untimed_ids(self.paragraphs)

    def _out(self, rel: str, set_name: str = ""):
        return self.project.outputs_dir / rel.format(set=set_name)

    def topic_long(self, set_name: str) -> pd.DataFrame:
        return pd.read_parquet(self._out("topics/{set}_clip_topics_long.parquet", set_name))

    def topic_scores(self, set_name: str) -> pd.DataFrame:
        """clip_id x topic name -> score."""
        return self.topic_long(set_name).pivot(index="clip_id", columns="topic_name",
                                               values="score")

    def mostly_topics(self, set_name: str) -> pd.Series:
        """interview_key -> the names of the topics its rollup tagged."""
        rel = freshness.DERIVED[("topics", "rollup")][0][0]
        wide = pd.read_parquet(self._out(rel, set_name)).set_index("interview_key")
        long = self.topic_long(set_name)
        name_by_id = dict(zip(long["topic_id"], long["topic_name"]))
        return wide["topics"].map(lambda s: {name_by_id[t] for t in str(s).split("|") if t})

    def places(self, kind: str) -> pd.Series:
        """clip_id -> its regions (`kind="region"`) or countries (`"location"`)."""
        if kind == "region":
            df, col = pd.read_parquet(self._out(freshness.WRITES["locations"][0])), "regions"
        else:
            rel = freshness.DERIVED[("locations", "map")][0][0]
            df, col = pd.read_parquet(self._out(rel)), "countries_final"
        return df.set_index("clip_id")[col].map(_split)

    def mostly_regions(self) -> pd.Series:
        rel = freshness.DERIVED[("locations", "rollup")][0][0]
        return pd.read_parquet(self._out(rel)).set_index("interview_key")["regions"].map(_split)


def _split(s) -> set[str]:
    return {x for x in str(s).split("|") if x and x != "nan"}


# --- the vocabulary this project can use now --------------------------------------------------

def topic_sets(project: Project) -> list[str]:
    return list(((load_root_config(project).get("topics") or {}).get("sets") or {}).keys())


def fields(project: Project) -> dict[str, Field]:
    """Every field, with whether this project can use it yet and the values it offers."""
    data = Collection(project)
    imported = project.paragraphs_path.exists()
    para = data.paragraphs if imported else None
    out: dict[str, Field] = {}

    def add(f: Field) -> None:
        out[f.key] = f

    add(Field("words", "words", TEXT_CONDITIONS, available=imported, needs="import"))
    add(Field("speaker", "speaker", ("is", "is_not"),
              _sorted(para["speaker_label"]) if imported else (), imported, "import"))
    add(Field("interview", "interview", ("is", "is_not"),
              _sorted(para["interview_key"]) if imported else (), imported, "import"))
    add(Field("timing", "timing", ("is",), ("timed", "untimed"), imported, "import"))

    labelled = freshness.wrote_its_results(project, "label")
    add(Field("label", "clip label", TEXT_CONDITIONS, available=labelled, needs="label"))

    sets = topic_sets(project)
    for s in sets:
        tagged = freshness.wrote_its_results(project, "topics", s)
        conds = ("talks_about", "might_talk_about", "never_mentions", "only_talks_about",
                 "any_of", "all_of", "alongside")
        if tagged and freshness.derived_exists(project, "topics", "rollup", s):
            conds += ("mostly_talks_about",)
        add(Field(f"topic:{s}", "topic" if len(sets) == 1 else f"topic ({s})", conds,
                  _sorted(data.topic_long(s)["topic_name"]) if tagged else (),
                  tagged, "topics tag"))

    place_conds = ("talks_about", "never_mentions", "any_of", "all_of")
    regions = freshness.wrote_its_results(project, "locations")
    conds = place_conds
    if regions and freshness.derived_exists(project, "locations", "rollup"):
        conds += ("mostly_talks_about",)
    add(Field("region", "region", conds,
              _sorted(v for vs in data.places("region") for v in vs) if regions else (),
              regions, "locations tag"))
    mapped = freshness.derived_exists(project, "locations", "map")
    add(Field("location", "location", place_conds,
              _sorted(v for vs in data.places("location") for v in vs) if mapped else (),
              mapped, "locations map"))
    return out


def _sorted(values) -> tuple[str, ...]:
    return tuple(sorted({str(v) for v in values if str(v)}, key=str.casefold))


def check(question: Question, available: dict[str, Field]) -> None:
    """Refuse a question this project cannot answer, saying which part and why."""
    q = question
    if q.verb not in VERBS:
        raise ToolkitError(f"Unknown verb {q.verb!r}.")
    if q.subject not in VERB_SUBJECTS[q.verb]:
        raise ToolkitError(f"{VERBS[q.verb]} works on {' or '.join(VERB_SUBJECTS[q.verb])}, "
                           f"not {q.subject}.")
    if not q.roles or set(q.roles) - set(ROLES):
        raise ToolkitError(f"Speaker roles must be some of {', '.join(ROLES)}.")
    if q.verb == "compare" and len(q.groups) < 2:
        raise ToolkitError("Compare needs at least two groups.")
    if q.verb != "compare" and q.groups:
        raise ToolkitError("Only Compare takes groups.")
    for row in (*q.rows, *q.groups):
        if row.join not in JOINS:
            raise ToolkitError(f"Rows join with 'and' or 'or', not {row.join!r}.")
        f = available.get(row.field)
        if f is None:
            raise ToolkitError(f"Unknown field {row.field!r}.")
        if not f.available:
            raise ToolkitError(f"Nothing to ask about {f.title} yet: run {f.needs} first.")
        if q.subject not in f.subjects:
            raise ToolkitError(f"{q.subject.capitalize()} have no {f.title}.")
        if row.condition not in f.conditions_for(q.subject):
            raise ToolkitError(f"{f.title} cannot be '{CONDITIONS[row.condition].words}' "
                               f"for {q.subject}." if row.condition in CONDITIONS
                               else f"Unknown condition {row.condition!r}.")
        want = CONDITIONS[row.condition].n_values
        n = len(row.values)
        if (want in ("one", "text") and n != 1) or (want == "two" and n != 2) \
                or (want in ("some", "texts") and n < 1):
            raise ToolkitError(f"'{CONDITIONS[row.condition].words}' takes {N_VALUES[want]}.")
        if want in ("text", "texts"):
            if not all(v.strip() for v in row.values):
                raise ToolkitError(f"Type the {f.title} to look for.")
        else:
            unknown = [v for v in row.values if v not in f.values]
            if unknown:
                raise ToolkitError(f"Not a {f.title} in this collection: {', '.join(unknown)}.")


# --- answering --------------------------------------------------------------------------------

@dataclass
class _Base:
    unit: str                       # "paragraph" | "clip" | "pair" (a speaker within a clip)
    ids: pd.Index                   # the units
    paras: pd.DataFrame             # paragraphs by the chosen roles (with clip_id for clips)
    pairs: pd.DataFrame | None = None   # "pair": clip_id, interview_key, speaker_label, share


PAIR = "\t"                        # joins a clip id and a speaker label into a pair's id


def _base(data: Collection, q: Question) -> _Base:
    clipped = q.subject != "paragraphs" and freshness.wrote_its_results(data.project, "clip")
    if not clipped:
        paras = data.paragraphs[data.paragraphs["speaker_role"].isin(q.roles)]
        return _Base("paragraph", paras.index, paras)
    paras = data.clipped[data.clipped["speaker_role"].isin(q.roles)]
    if q.subject == "speakers":
        # Who said it, not who was in the clip: each speaker in each clip is a unit of its own,
        # with the share of the clip's words that are theirs.
        total = data.clipped.groupby("clip_id")["word_count"].sum()
        pairs = (paras.groupby(["clip_id", "interview_key", "speaker_label"], as_index=False)
                 ["word_count"].sum())
        pairs["share"] = pairs["word_count"] / pairs["clip_id"].map(total)
        pairs.index = pd.Index(pairs["clip_id"] + PAIR + pairs["speaker_label"])
        paras = paras.assign(_pair=paras["clip_id"] + PAIR + paras["speaker_label"])
        return _Base("pair", pairs.index, paras, pairs)
    ids = data.clips.index[data.clips.index.isin(paras["clip_id"])]
    return _Base("clip", ids, paras)


def _strength(data: Collection, base: _Base, row: Row) -> pd.Series:
    """CLEAR / POSSIBLE / 0 for every unit of the base, for one positive row."""
    if base.unit == "pair":
        return _pair_strength(data, base, row)
    if row.field in PARAGRAPH_FIELDS:
        hit = _paragraph_hit(data, base.paras, row)
        if base.unit == "paragraph":
            return hit.astype(int) * CLEAR
        by_clip = hit.groupby(base.paras["clip_id"]).any()
        return by_clip.reindex(base.ids, fill_value=False).astype(int) * CLEAR
    if row.field == "label":
        text = data.labels.reindex(base.ids).fillna("")
        return _text_hit(text, row).astype(int) * CLEAR
    if row.field.startswith("topic:"):
        return _topic_strength(data, base, row)
    return _place_strength(data.places(row.field).reindex(base.ids), row)


def _pair_strength(data: Collection, base: _Base, row: Row) -> pd.Series:
    """A speaker in a clip. Words count only in their own lines. A topic or place belongs to the
    whole clip, so it is theirs only when they say a fair share of it (`speaker_min_share`):
    an interviewer's long question answered with "yes" is not the narrator talking about it."""
    if row.field in PARAGRAPH_FIELDS:
        hit = _paragraph_hit(data, base.paras, row).groupby(base.paras["_pair"]).any()
        return hit.reindex(base.ids, fill_value=False).astype(int) * CLEAR
    clips = pd.Index(base.pairs["clip_id"].unique())
    per_clip = _strength(data, _Base("clip", clips, base.paras), row)
    s = base.pairs["clip_id"].map(per_clip).fillna(0).astype(int)
    return s.where(base.pairs["share"] >= data.speaker_min_share, 0)


def _paragraph_hit(data: Collection, paras: pd.DataFrame, row: Row) -> pd.Series:
    v = row.values[0]
    if row.field == "words":
        return _text_hit(paras["speech"], row)
    if row.field == "speaker":
        return paras["speaker_label"] == v
    if row.field == "interview":
        return paras["interview_key"] == v
    untimed = paras["interview_id"].isin(data.untimed)
    return untimed if v == "untimed" else ~untimed


def word_pattern(word: str, whole: bool = False) -> str:
    """Words are matched from their start, ignoring case: "art" finds "artist" but not "party",
    "communit" finds "community" and "communities". A word ending in -y also finds its -ies
    plural ("community" finds "communities"). `whole` matches the word exactly, and only it."""
    w = word.strip()
    forms = [re.escape(w)]
    if not whole and len(w) > 2 and w[-1] in "yY":
        forms.append(re.escape(w[:-1]) + "ies")
    return rf"\b(?:{'|'.join(forms)})" + (r"\b" if whole else "")


def _text_hit(text: pd.Series, row: Row) -> pd.Series:
    whole = row.condition == "whole_word"
    hits = [text.str.contains(word_pattern(v, whole), case=False, regex=True)
            for v in row.values]
    if row.condition == "contains_all":
        return pd.concat(hits, axis=1).all(axis=1)
    return pd.concat(hits, axis=1).any(axis=1)          # contains, any of, whole word


def _topic_strength(data: Collection, base: _Base, row: Row) -> pd.Series:
    scores = data.topic_scores(row.field.split(":", 1)[1]).reindex(base.ids).fillna(0)
    clear = scores == data.top_score
    c = row.condition
    if c == "might_talk_about":
        s = scores[row.values[0]]
        return pd.Series(0, index=base.ids).mask(s > 0, POSSIBLE).mask(s == data.top_score, CLEAR)
    if c == "talks_about":
        hit = clear[row.values[0]]
    elif c == "only_talks_about":
        hit = clear[row.values[0]] & (clear.sum(axis=1) == 1)
    elif c == "any_of":
        hit = clear[list(row.values)].any(axis=1)
    else:                                               # all_of, alongside
        hit = clear[list(row.values)].all(axis=1)
    return hit.astype(int) * CLEAR


def _place_strength(places: pd.Series, row: Row) -> pd.Series:
    places = places.map(lambda s: s if isinstance(s, set) else set())
    want = set(row.values)
    if row.condition == "all_of":
        hit = places.map(lambda s: want <= s)
    else:                                               # talks_about, any_of
        hit = places.map(lambda s: bool(want & s))
    return hit.astype(int) * CLEAR


def _subject_keys(base: _Base, subject: str) -> pd.DataFrame:
    """unit -> the subject it belongs to: columns `unit` and `key` (one row per pair)."""
    if subject in ("paragraphs", "clips"):
        return pd.DataFrame({"unit": base.ids, "key": base.ids})
    if base.unit == "pair":
        return pd.DataFrame({"unit": base.ids, "key": list(zip(base.pairs["interview_key"],
                                                               base.pairs["speaker_label"]))})
    cols = ["interview_key"] if subject == "interviews" else ["interview_key", "speaker_label"]
    unit = base.paras.index if base.unit == "paragraph" else base.paras["clip_id"]
    keys = [tuple(r) if len(cols) > 1 else r[0] for r in base.paras[cols].itertuples(index=False)]
    return pd.DataFrame({"unit": list(unit), "key": keys}).drop_duplicates()


def or_groups(rows: tuple[Row, ...]) -> list[tuple[Row, ...]]:
    """Split rows at each "or": "A and B or C" is (A and B) or (C)."""
    groups: list[list[Row]] = []
    for i, row in enumerate(rows):
        if i == 0 or row.join == "or":
            groups.append([])
        groups[-1].append(row)
    return [tuple(g) for g in groups] or [()]


def _matches(data: Collection, q: Question, rows: tuple[Row, ...],
             extra: tuple[Row, ...] = ()) -> tuple[_Base, pd.Series]:
    """The base, and each matching unit's strength (units that do not match are left out).
    `extra` rows are and-ed onto every or-group (Compare's groups)."""
    base = _base(data, q)
    found = [_and_matches(data, q, base, (*g, *extra)) for g in or_groups(rows)]
    strength = pd.concat(found).groupby(level=0).max() if found else pd.Series(dtype=int)
    return base, strength.reindex([u for u in base.ids if u in strength.index])


def _and_matches(data: Collection, q: Question, base: _Base, rows: tuple[Row, ...]) -> pd.Series:
    strength = pd.Series(CLEAR, index=base.ids)
    excluded_units: set = set()
    excluded_subjects: set = set()
    mostly: list[Row] = []
    for row in rows:
        cond = CONDITIONS[row.condition]
        if row.condition == "mostly_talks_about":
            mostly.append(row)
        elif cond.negative:
            pos = _strength(data, base, Row(row.field, POSITIVE[row.condition], row.values))
            hits = set(pos.index[pos > 0])
            if q.subject in ("paragraphs", "clips"):
                excluded_units |= hits
            else:
                pairs = _subject_keys(base, q.subject)
                excluded_subjects |= set(pairs.loc[pairs["unit"].isin(hits), "key"])
        else:
            strength = strength.combine(_strength(data, base, row), min)
    strength = strength[(strength > 0) & ~strength.index.isin(excluded_units)]

    if excluded_subjects or mostly:
        pairs = _subject_keys(base, q.subject)
        keep = ~pairs["key"].isin(excluded_subjects)
        for row in mostly:
            tags = (data.mostly_regions() if row.field == "region"
                    else data.mostly_topics(row.field.split(":", 1)[1]))
            keep &= pairs["key"].map(lambda k: row.values[0] in tags.get(k, set()))
        strength = strength[strength.index.isin(pairs.loc[keep, "unit"])]
    return strength


def answer(project: Project, question: Question) -> pd.DataFrame:
    """The question's answer as a table, strongest matches first. Refuses what `check` refuses."""
    check(question, fields(project))
    data = Collection(project)
    q = question
    if q.verb == "compare":
        return _compare(data, q)
    if q.subject == "clips" and not freshness.wrote_its_results(project, "clip"):
        raise ToolkitError("There are no clips yet: run clip first.")
    base, strength = _matches(data, q, q.rows)
    if q.verb == "count":
        return _per_interview(data, base, strength)
    if q.subject == "paragraphs":
        return _paragraph_rows(data, base, strength)
    if q.subject == "clips":
        return _clip_rows(data, strength)
    return _subject_rows(data, base, strength, q.subject)


def _paragraph_rows(data: Collection, base: _Base, strength: pd.Series) -> pd.DataFrame:
    p = base.paras.loc[strength.index]
    time = p["turn_time_start"].where(p["sub_time_start"].astype(str) == "", p["sub_time_start"])
    out = pd.DataFrame({"interview": p["interview_key"], "session": p["interview_id"],
                        "paragraph": p["paragraph_idx"], "speaker": p["speaker_label"],
                        "role": p["speaker_role"], "time": time, "text": p["speech"],
                        "match": strength.map(MATCH_WORDS)})
    return _ranked(out, strength, ["session", "paragraph"]).reset_index(drop=True)


def _clip_rows(data: Collection, strength: pd.Series) -> pd.DataFrame:
    c = data.clips.loc[strength.index]
    text = (data.clipped.sort_values(["interview_id", "paragraph_idx"])
            .groupby("clip_id")["speech"].agg("\n\n".join))
    out = pd.DataFrame({"clip": c.index, "interview": c["interview_key"],
                        "session": c["interview_id"], "start": c["start_ts"], "end": c["end_ts"]},
                       index=c.index)
    if freshness.wrote_its_results(data.project, "label"):
        out["label"] = data.labels.reindex(c.index)
    out["text"] = text.reindex(c.index)
    out["match"] = strength.map(MATCH_WORDS)
    out["_start"] = c["start_paragraph_idx"]
    return _ranked(out, strength, ["session", "_start"]).drop(columns="_start") \
        .reset_index(drop=True)


def _subject_rows(data: Collection, base: _Base, strength: pd.Series, subject: str) -> pd.DataFrame:
    pairs = _subject_keys(base, subject)
    pairs = pairs[pairs["unit"].isin(strength.index)]
    pairs["s"] = pairs["unit"].map(strength)
    g = pairs.groupby("key")["s"]
    out = pd.DataFrame({"clear": g.apply(lambda s: int((s == CLEAR).sum())),
                        "possible": g.apply(lambda s: int((s == POSSIBLE).sum()))})
    best = g.max()
    if subject == "interviews":
        sessions = data.paragraphs.groupby("interview_key")["interview_id"].nunique()
        out.insert(0, "interview", out.index)
        out.insert(1, "sessions", sessions.reindex(out.index).values)
    else:
        role = data.paragraphs.groupby(["interview_key", "speaker_label"])["speaker_role"].first()
        out.insert(0, "interview", [k[0] for k in out.index])
        out.insert(1, "speaker", [k[1] for k in out.index])
        out.insert(2, "role", [role.get(k, "") for k in out.index])
    out["match"] = best.map(MATCH_WORDS)
    out = out.assign(_best=best).sort_values(["_best", "clear", "possible", "interview"],
                                             ascending=[False, False, False, True])
    return out.drop(columns="_best").reset_index(drop=True)


def _ranked(out: pd.DataFrame, strength: pd.Series, order: list[str]) -> pd.DataFrame:
    return (out.assign(_s=strength).sort_values(["_s", *order], ascending=[False] + [True] * len(order))
            .drop(columns="_s"))


def _per_interview(data: Collection, base: _Base, strength: pd.Series) -> pd.DataFrame:
    """Count: matching units per interview, clear and possible apart."""
    key = (base.paras["interview_key"] if base.unit == "paragraph"
           else data.clips["interview_key"]).reindex(strength.index)
    df = pd.DataFrame({"interview": key, "s": strength})
    out = df.groupby("interview")["s"].agg(clear=lambda s: int((s == CLEAR).sum()),
                                           possible=lambda s: int((s == POSSIBLE).sum()))
    out = out.reset_index().sort_values(["clear", "possible", "interview"],
                                        ascending=[False, False, True])
    return out.reset_index(drop=True)


def _compare(data: Collection, q: Question) -> pd.DataFrame:
    """One row per interview, one column per group: how many units match the rows and it."""
    columns = {}
    for g in q.groups:
        base, strength = _matches(data, q, q.rows, (replace(g, join="and"),))
        columns[row_words(g)] = _per_interview(data, base, strength) \
            .set_index("interview")[["clear", "possible"]].sum(axis=1)
    out = pd.DataFrame(columns).fillna(0).astype(int)
    out = out.loc[out.sum(axis=1).sort_values(ascending=False, kind="stable").index]
    return out.rename_axis("interview").reset_index()


# --- from an answer back to the transcript ----------------------------------------------------

@dataclass(frozen=True)
class Passage:
    session: str
    first: int                  # paragraph_idx of the passage's first and last paragraph
    last: int
    transcript: pd.DataFrame    # the whole session: paragraph, speaker, role, time, text


def passage(project: Project, result: dict) -> Passage:
    """Where a paragraph or clip in an answer sits: its session's transcript, and its span."""
    data = Collection(project)
    if "clip" in result:
        c = data.clips.loc[result["clip"]]
        session, first, last = (c["interview_id"], int(c["start_paragraph_idx"]),
                                int(c["end_paragraph_idx"]))
    else:
        session, first = result["session"], int(result["paragraph"])
        last = first
    p = data.paragraphs[data.paragraphs["interview_id"] == session].sort_values("paragraph_idx")
    time = p["turn_time_start"].where(p["sub_time_start"].astype(str) == "", p["sub_time_start"])
    transcript = pd.DataFrame({"paragraph": p["paragraph_idx"], "speaker": p["speaker_label"],
                               "role": p["speaker_role"], "time": time, "text": p["speech"]})
    return Passage(session, first, last, transcript.reset_index(drop=True))


def and_each(rows: tuple[Row, ...], extra: tuple[Row, ...]) -> tuple[Row, ...]:
    """`extra` and-ed onto every or-group of `rows`: (A or B) and X is (A and X) or (B and X)."""
    out: list[Row] = []
    for group in or_groups(rows):
        joined = [*group, *extra]
        out += [replace(r, join=("or" if out else "and") if i == 0 else "and")
                for i, r in enumerate(joined)]
    return tuple(out)


def narrow(project: Project, question: Question, result: dict) -> Question:
    """An interview or speaker in an answer, opened up: the same question asked of its clips
    (its paragraphs, before clipping). "Mostly talks about" is about the whole interview, so it
    has already done its work and is dropped."""
    pin = [Row("interview", "is", (result["interview"],))]
    if "speaker" in result:
        pin.append(Row("speaker", "is", (result["speaker"],)))
    kept: list[Row] = []
    for group in or_groups(question.rows):
        group = [r for r in group if r.condition != "mostly_talks_about"]
        kept += [replace(r, join="or" if i == 0 and kept else "and") for i, r in enumerate(group)]
    rows = and_each(tuple(kept), tuple(pin)) if kept else tuple(pin)
    clipped = freshness.wrote_its_results(project, "clip")
    return Question("find", "clips" if clipped else "paragraphs", rows, question.roles)


# --- the sentence -----------------------------------------------------------------------------

def row_words(row: Row, titles: dict[str, str] | None = None) -> str:
    title = (titles or {}).get(row.field, row.field.split(":", 1)[0])
    cond = CONDITIONS[row.condition]
    if row.condition == "alongside":
        return f"{title} includes {row.values[0]} alongside {row.values[1]}"
    if cond.n_values in ("text", "texts"):
        return f"{title} {cond.words} " + ", ".join(f'"{v}"' for v in row.values)
    return f"{title} {cond.words} {', '.join(row.values)}"


ROLE_WORDS = {("Narrator",): "narrators", ("Interviewer",): "interviewers"}


def readback(question: Question, available: dict[str, Field] | None = None) -> str:
    """The question in plain words, as shown above its answer."""
    q = question
    titles = {k: f.title for k, f in (available or {}).items()}
    s = f"{VERBS[q.verb]} {q.subject}"
    if q.rows:
        s += " where " + "".join((f" {r.join} " if i else "") + row_words(r, titles)
                                 for i, r in enumerate(q.rows))
    if q.groups:
        s += ": " + " vs ".join(row_words(g, titles) for g in q.groups)
    roles = tuple(r for r in ROLES if r in q.roles)
    s += f" ({ROLE_WORDS.get(roles, ' and '.join(r.lower() + 's' for r in roles))})"
    return s
