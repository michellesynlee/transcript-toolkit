"""Quick and saved questions, on the collection in test_ask.py."""
import json

import pytest
from test_ask import A2, imported, tagged  # noqa: F401  (fixtures)

from transcript_toolkit.core import ask, ask_quick
from transcript_toolkit.core.ask import Question, Row
from transcript_toolkit.errors import ToolkitError


def _filled(q: Question, fields) -> Question:
    """Every blank filled with values this collection has."""
    values = []
    for r in (*q.rows, *q.groups):
        offered = fields[r.field].values or ("school",)
        values += list(offered[:r.values.count(ask_quick.BLANK)])
    return ask_quick.fill(q, values)


def test_every_quick_question_is_answerable_once_tagged(tagged):
    fields = ask.fields(tagged)
    for quick in ask_quick.builtin():
        q = ask_quick.resolve(quick, fields)
        assert q is not None, quick.button
        ask.answer(tagged, _filled(q, fields))              # refuses anything malformed


def test_before_tagging_only_the_words_question_is_offered(imported):
    fields = ask.fields(imported)
    offered = [q.button for q in ask_quick.builtin() if ask_quick.resolve(q, fields)]
    assert offered == ["Who says the words…"]


def test_a_filled_quick_question_reads_as_itself(tagged):
    fields = ask.fields(tagged)
    quicks = ask_quick.builtin()
    who = next(q for q in quicks if q.button == "Who talks about…")
    q = ask_quick.fill(ask_quick.resolve(who, fields), ["Education"])
    assert ask_quick.wording(q, quicks, fields) == "Who talks about Education?"
    both = Question(q.verb, q.subject, q.rows, ("Narrator", "Interviewer"))
    assert ask_quick.wording(both, quicks, fields) == \
        "Find speakers where topic is Education (narrators and interviewers)"


def test_fill_needs_exactly_the_blanks(tagged):
    q = Question("find", "clips", (Row("topic:main", "alongside", ("_", "_")),))
    assert ask_quick.blanks(q) == 2
    with pytest.raises(ToolkitError, match="More values"):
        ask_quick.fill(q, ["Education", "Work", "Family"])


def test_saved_questions_live_with_the_project(tagged):
    fields = ask.fields(tagged)
    q = Question("find", "clips", (Row("topic:main", "talks_about", ("Work",)),))
    kept = ask_quick.save(tagged, q, fields)
    assert kept.button == "Find clips where topic is Work (narrators)"
    assert ask_quick.saved(tagged) == [kept]
    on_disk = json.loads((tagged.toolkit_dir / "saved_questions.json").read_text())
    assert on_disk[0]["question"]["rows"][0]["values"] == ["Work"]
    with pytest.raises(ToolkitError, match="Already saved"):
        ask_quick.save(tagged, q, fields)
    ask_quick.forget(tagged, kept.button)
    assert ask_quick.saved(tagged) == []


def test_a_question_with_blanks_cannot_be_saved(tagged):
    q = Question("find", "clips", (Row("topic:main", "talks_about", ("_",)),))
    with pytest.raises(ToolkitError, match="every blank"):
        ask_quick.save(tagged, q, ask.fields(tagged))


# --- typing where a topic was expected --------------------------------------------------------

def test_a_typed_topic_name_is_the_topic_whatever_its_case(tagged):
    fields = ask.fields(tagged)
    q = Question("find", "speakers", (Row("topic:main", "talks_about", ("education",)),))
    assert ask_quick.typed_as_words(q, fields).rows == (
        Row("topic:main", "talks_about", ("Education",)),)


@pytest.mark.parametrize("condition,words", [("talks_about", "contains"),
                                             ("never_mentions", "not_contains"),
                                             ("mostly_talks_about", "contains")])
def test_anything_else_typed_searches_the_words(tagged, condition, words):
    q = Question("find", "speakers", (Row("topic:main", condition, ("Election",)),))
    assert ask_quick.typed_as_words(q, ask.fields(tagged)).rows == (
        Row("words", words, ("Election",)),)


def test_alongside_keeps_the_topic_and_searches_the_word(tagged):
    fields = ask.fields(tagged)
    q = Question("find", "clips", (Row("topic:main", "alongside", ("Work", "election")),))
    typed = ask_quick.typed_as_words(q, fields)
    assert typed.rows == (Row("topic:main", "talks_about", ("Work",)),
                          Row("words", "contains", ("election",)))
    assert list(ask.answer(tagged, typed)["clip"]) == [f"{A2}_0001"]


def test_who_never_mentions_a_typed_word(tagged):
    q = ask_quick.typed_as_words(
        Question("find", "speakers", (Row("topic:main", "never_mentions", ("kenya",)),)),
        ask.fields(tagged))
    assert list(ask.answer(tagged, q)["speaker"]) == ["Bob"]
