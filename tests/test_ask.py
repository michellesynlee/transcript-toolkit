"""The Ask engine on a collection small enough to answer by hand.

Two narrators: Ann (two timed sessions, pooled as `alpha`) and Bob (one untimed interview).
Clip topic scores on the default 0-2 scale:

    clip     Education  Family  Work
    a1_1         2        0      0
    a1_2         0        2      1      (Kenya / East Africa)
    a2_1         2        0      2
    b_1          1        2      0
"""
import pandas as pd
import pytest

from transcript_toolkit.core import ask
from transcript_toolkit.core.ask import Question, Row
from transcript_toolkit.errors import ToolkitError
from transcript_toolkit.project import init_project
from transcript_toolkit.steps.topics.taxonomy import register_topic_set

A1, A2, B = "alpha_20240101_session1", "alpha_20240108_session2", "beta"
TOPICS = {"edu": "Education", "fam": "Family", "work": "Work"}
SCORES = {f"{A1}_0001": (2, 0, 0), f"{A1}_0002": (0, 2, 1),
          f"{A2}_0001": (2, 0, 2), f"{B}_0001": (1, 2, 0)}


def _para(iid, idx, speaker, role, speech, time="00:00:01", clip=None):
    return {"interview_id": iid, "paragraph_idx": idx, "turn_idx": idx,
            "paragraph_idx_in_turn": 0, "turn_time_start": time, "sub_time_start": "",
            "speaker_label": speaker, "speaker_role": role, "speech": speech,
            "word_count": len(speech.split()), "clip_id": clip}


PARAS = [
    _para(A1, 0, "Q", "Interviewer", "Tell me about school.", clip=f"{A1}_0001"),
    _para(A1, 1, "Ann", "Narrator", "I loved school and my teachers.", clip=f"{A1}_0001"),
    _para(A1, 2, "Ann", "Narrator", "My family moved to Kenya.", clip=f"{A1}_0002"),
    _para(A2, 0, "Q", "Interviewer", "And your work?", clip="procedural"),
    _para(A2, 1, "Ann", "Narrator", "I taught at the election office.", clip=f"{A2}_0001"),
    _para(B, 0, "Q", "Interviewer", "Hello.", time="", clip=f"{B}_0001"),
    _para(B, 1, "Bob", "Narrator", "Education mattered to my family.", time="", clip=f"{B}_0001"),
]


def _write(project, rel, df):
    path = project.outputs_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.fixture
def imported(tmp_path):
    project = init_project(str(tmp_path / "ws"))
    pd.DataFrame(PARAS).drop(columns="clip_id").to_parquet(project.paragraphs_path, index=False)
    return project


@pytest.fixture
def tagged(imported):
    project = imported
    paras = pd.DataFrame(PARAS)
    _write(project, "clips/paragraphs_clipped.parquet", paras)
    clips = (paras[paras["clip_id"] != "procedural"].groupby("clip_id")
             .agg(interview_id=("interview_id", "first"),
                  start_paragraph_idx=("paragraph_idx", "min"),
                  end_paragraph_idx=("paragraph_idx", "max"),
                  start_ts=("turn_time_start", "first"), end_ts=("turn_time_start", "last"))
             .reset_index())
    _write(project, "clips/clips.parquet", clips)
    _write(project, "labels/labels.parquet",
           clips.assign(label=["school days", "moving to Kenya", "the election office",
                               "why education mattered"]))

    register_topic_set(project, "main", "topics/main.xlsx")
    long = pd.DataFrame([{"clip_id": cid, "interview_id": cid.rsplit("_", 1)[0], "topic_id": tid,
                          "topic_name": name, "score": s[i]}
                         for cid, s in SCORES.items() for i, (tid, name) in enumerate(TOPICS.items())])
    _write(project, "topics/main_clip_topics_long.parquet", long)
    _write(project, "topics/main_clip_topics_wide.parquet",
           long.pivot(index="clip_id", columns="topic_id", values="score").reset_index())
    _write(project, "topics/main_interview_topics_wide.parquet",
           pd.DataFrame([{"interview_key": "alpha", "topics": "edu|work"},
                         {"interview_key": "beta", "topics": "fam"}]))

    loc = clips[["clip_id", "interview_id"]].assign(regions="", countries_final="")
    loc.loc[loc["clip_id"] == f"{A1}_0002", ["regions", "countries_final"]] = ["East Africa", "Kenya"]
    _write(project, "locations/clip_locations.parquet", loc.drop(columns="countries_final"))
    _write(project, "locations/clip_countries.parquet", loc)
    return project


def _ask(project, verb, subject, *rows, roles=("Narrator",), groups=()):
    """Rows are (field, condition, values) and may add a join: (..., "or")."""
    return ask.answer(project, Question(verb, subject, tuple(Row(*r) for r in rows), roles,
                                        tuple(Row(*g) for g in groups)))


# --- availability -----------------------------------------------------------------------------

def test_before_import_nothing_is_available(tmp_path):
    project = init_project(str(tmp_path / "ws"))
    assert not any(f.available for f in ask.fields(project).values())


def test_after_import_only_the_transcript_fields(imported):
    f = ask.fields(imported)
    assert {k for k, v in f.items() if v.available} == {"words", "speaker", "interview", "timing"}
    assert f["speaker"].values == ("Ann", "Bob", "Q")
    assert f["interview"].values == ("alpha", "beta")      # sessions pooled per narrator


def test_a_field_is_available_exactly_when_its_file_is_there(tagged):
    f = ask.fields(tagged)
    assert f["topic:main"].available and "mostly_talks_about" in f["topic:main"].conditions
    (tagged.outputs_dir / "topics/main_interview_topics_wide.parquet").unlink()
    assert "mostly_talks_about" not in ask.fields(tagged)["topic:main"].conditions
    (tagged.outputs_dir / "topics/main_clip_topics_wide.parquet").unlink()
    topic = ask.fields(tagged)["topic:main"]
    assert not topic.available and topic.needs == "topics tag"


# --- before clipping: paragraphs --------------------------------------------------------------

def test_words_respect_the_speaker_toggle(imported):
    narr = _ask(imported, "find", "paragraphs", ("words", "contains", ("school",)))
    assert list(narr["speaker"]) == ["Ann"]
    both = _ask(imported, "find", "paragraphs", ("words", "contains", ("SCHOOL",)),
                roles=("Narrator", "Interviewer"))
    assert sorted(both["speaker"]) == ["Ann", "Q"]


def test_interviews_before_clipping_come_from_paragraphs(imported):
    out = _ask(imported, "find", "interviews", ("words", "contains", ("family",)))
    assert list(out["interview"]) == ["alpha", "beta"]


def test_clips_before_clipping_is_refused(imported):
    with pytest.raises(ToolkitError, match="clip label yet: run label first"):
        _ask(imported, "find", "interviews", ("label", "contains", ("school",)))
    with pytest.raises(ToolkitError, match="no clips yet"):
        _ask(imported, "find", "clips", ("words", "contains", ("school",)))


# --- topics -----------------------------------------------------------------------------------

def test_talks_about_is_the_top_score_only(tagged):
    out = _ask(tagged, "find", "clips", ("topic:main", "talks_about", ("Education",)))
    assert list(out["clip"]) == [f"{A1}_0001", f"{A2}_0001"]
    assert set(out["match"]) == {"clear"}


def test_might_talk_about_ranks_clear_before_possible(tagged):
    out = _ask(tagged, "find", "clips", ("topic:main", "might_talk_about", ("Education",)))
    assert list(out["clip"]) == [f"{A1}_0001", f"{A2}_0001", f"{B}_0001"]
    assert list(out["match"]) == ["clear", "clear", "possible"]


def test_never_mentions_is_about_the_speaker_not_one_clip(tagged):
    # Ann has clips without Work, but a1_2 might be about it, so she is out.
    out = _ask(tagged, "find", "speakers", ("topic:main", "never_mentions", ("Work",)))
    assert list(out["speaker"]) == ["Bob"]


def test_only_alongside_any_all(tagged):
    only = _ask(tagged, "find", "clips", ("topic:main", "only_talks_about", ("Family",)))
    assert list(only["clip"]) == [f"{A1}_0002", f"{B}_0001"]       # a1_2's Work is only a 1
    along = _ask(tagged, "find", "clips", ("topic:main", "alongside", ("Education", "Work")))
    assert list(along["clip"]) == [f"{A2}_0001"]
    any_ = _ask(tagged, "find", "clips", ("topic:main", "any_of", ("Family", "Work")))
    assert len(any_) == 3
    all_ = _ask(tagged, "find", "clips", ("topic:main", "all_of", ("Education", "Work")))
    assert list(all_["clip"]) == [f"{A2}_0001"]


def test_mostly_talks_about_is_the_rollup_tag(tagged):
    out = _ask(tagged, "find", "interviews", ("topic:main", "mostly_talks_about", ("Family",)))
    assert list(out["interview"]) == ["beta"]


def test_mostly_is_for_interviews_only(tagged):
    with pytest.raises(ToolkitError, match="is mostly"):
        _ask(tagged, "find", "clips", ("topic:main", "mostly_talks_about", ("Family",)))


def test_interviews_rank_by_their_clear_matches(tagged):
    out = _ask(tagged, "find", "interviews", ("topic:main", "might_talk_about", ("Education",)))
    assert out.to_dict("records") == [
        {"interview": "alpha", "sessions": 2, "clear": 2, "possible": 0, "match": "clear"},
        {"interview": "beta", "sessions": 1, "clear": 0, "possible": 1, "match": "possible"}]


def test_rows_combine(tagged):
    out = _ask(tagged, "find", "clips", ("topic:main", "talks_about", ("Education",)),
               ("speaker", "is", ("Ann",)), ("timing", "is", ("timed",)))
    assert len(out) == 2
    out = _ask(tagged, "find", "clips", ("topic:main", "might_talk_about", ("Education",)),
               ("interview", "is_not", ("alpha",)))
    assert list(out["clip"]) == [f"{B}_0001"]


# --- other fields -----------------------------------------------------------------------------

def test_label_region_location(tagged):
    assert list(_ask(tagged, "find", "clips", ("label", "contains", ("ELECTION",)))["clip"]) \
        == [f"{A2}_0001"]
    assert list(_ask(tagged, "find", "clips", ("region", "talks_about", ("East Africa",)))["clip"]) \
        == [f"{A1}_0002"]
    assert list(_ask(tagged, "find", "interviews", ("location", "never_mentions", ("Kenya",)))
                ["interview"]) == ["beta"]


def test_clip_rows_carry_label_and_text(tagged):
    out = _ask(tagged, "find", "clips", ("topic:main", "talks_about", ("Family",)))
    first = out.iloc[0]
    assert first["label"] == "moving to Kenya" and first["text"] == "My family moved to Kenya."
    assert out.iloc[1]["start"] == ""                        # untimed: no time, not a guess


# --- count and compare ------------------------------------------------------------------------

def test_count_per_interview(tagged):
    out = _ask(tagged, "count", "clips", ("topic:main", "might_talk_about", ("Education",)))
    assert out.to_dict("records") == [{"interview": "alpha", "clear": 2, "possible": 0},
                                      {"interview": "beta", "clear": 0, "possible": 1}]


def test_compare_has_a_column_per_group(tagged):
    out = _ask(tagged, "compare", "clips",
               groups=[("topic:main", "talks_about", ("Education",)),
                       ("topic:main", "talks_about", ("Family",))])
    assert out.to_dict("records") == [
        {"interview": "alpha", "topic is Education": 2, "topic is Family": 1},
        {"interview": "beta", "topic is Education": 0, "topic is Family": 1}]


# --- refusals and readback --------------------------------------------------------------------

def test_values_must_exist(tagged):
    with pytest.raises(ToolkitError, match="Not a topic in this collection: Sport"):
        _ask(tagged, "find", "clips", ("topic:main", "talks_about", ("Sport",)))
    with pytest.raises(ToolkitError, match="takes two values"):
        _ask(tagged, "find", "clips", ("topic:main", "alongside", ("Education",)))


def test_readback(tagged):
    q = Question("find", "clips", (Row("topic:main", "might_talk_about", ("Education",)),
                                   Row("words", "contains", ("school",))),
                 ("Narrator", "Interviewer"))
    assert ask.readback(q, ask.fields(tagged)) == \
        'Find clips where topic might be Education and words contains "school" ' \
        '(narrators and interviewers)'


# --- or, and the text conditions --------------------------------------------------------------

def test_or_joins_groups_and_binds_looser_than_and(tagged):
    # (Education and Ann's Kenya clip) is empty, so only the "or" side is left: a1_2 and b_1
    out = _ask(tagged, "find", "clips", ("topic:main", "talks_about", ("Education",)),
               ("words", "contains", ("Kenya",)),
               ("topic:main", "talks_about", ("Family",), "or"))
    assert list(out["clip"]) == [f"{A1}_0002", f"{B}_0001"]
    out = _ask(tagged, "find", "clips", ("topic:main", "talks_about", ("Work",)),
               ("words", "contains", ("Kenya",), "or"))
    assert list(out["clip"]) == [f"{A1}_0002", f"{A2}_0001"]


def test_or_keeps_the_strongest_match(tagged):
    out = _ask(tagged, "find", "clips", ("topic:main", "might_talk_about", ("Education",)),
               ("speaker", "is", ("Bob",), "or"))
    assert dict(zip(out["clip"], out["match"]))[f"{B}_0001"] == "clear"


@pytest.mark.parametrize("condition,values,expected", [
    ("contains", ("school",), ["I loved school and my teachers."]),
    ("not_contains", ("school",), ["My family moved to Kenya.", "I taught at the election office.",
                                   "Education mattered to my family."]),
    ("contains_any", ("Kenya", "election"), ["My family moved to Kenya.",
                                             "I taught at the election office."]),
    ("contains_all", ("my", "family"), ["My family moved to Kenya.",
                                        "Education mattered to my family."]),
    ("whole_word", ("teach",), []),
    ("whole_word", ("taught",), ["I taught at the election office."]),
])
def test_text_conditions(imported, condition, values, expected):
    out = _ask(imported, "find", "paragraphs", ("words", condition, values))
    assert list(out["text"]) == expected


def test_does_not_contain_rules_out_the_speaker(imported):
    out = _ask(imported, "find", "speakers", ("words", "not_contains", ("family",)))
    assert out.empty                                      # both narrators say "family"


def test_readback_says_or(tagged):
    q = Question("find", "clips", (Row("topic:main", "talks_about", ("Work",)),
                                   Row("words", "contains_any", ("Kenya", "Ghana"), "or")))
    assert ask.readback(q) == \
        'Find clips where topic is Work or words contains any of "Kenya", "Ghana" ' \
        '(narrators)'


# --- back to the transcript -------------------------------------------------------------------

def test_a_clip_opens_at_its_place_in_the_session(tagged):
    p = ask.passage(tagged, {"clip": f"{A1}_0002"})
    assert (p.session, p.first, p.last) == (A1, 2, 2)
    assert list(p.transcript["paragraph"]) == [0, 1, 2]          # the whole session, in order


def test_a_paragraph_opens_at_its_place(imported):
    row = _ask(imported, "find", "paragraphs", ("words", "contains", ("election",))).iloc[0]
    p = ask.passage(imported, row.to_dict())
    assert (p.session, p.first, p.last) == (A2, 1, 1)


def test_an_interview_narrows_to_its_own_clips(tagged):
    q = Question("find", "interviews", (Row("topic:main", "talks_about", ("Education",)),
                                        Row("topic:main", "mostly_talks_about", ("Work",)),
                                        Row("speaker", "is", ("Bob",), "or")))
    narrowed = ask.narrow(tagged, q, {"interview": "alpha"})
    assert narrowed.subject == "clips"
    assert [(r.field, r.join) for r in narrowed.rows] == [
        ("topic:main", "and"), ("interview", "and"),          # mostly dropped
        ("speaker", "or"), ("interview", "and")]
    out = ask.answer(tagged, narrowed)
    assert list(out["clip"]) == [f"{A1}_0001", f"{A2}_0001"]
