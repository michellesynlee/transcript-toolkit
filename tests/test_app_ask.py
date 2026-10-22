"""The Ask page, driven the way a user drives it, on the collection in test_ask.py."""
import asyncio

import pytest
from nicegui import ui
from nicegui.testing import User
from test_ask import A1, imported, tagged  # noqa: F401  (fixtures)

from transcript_toolkit.app.context import CONTEXT
from transcript_toolkit.app.pages.ask import marked, view_href
from transcript_toolkit.core.ask import Question, Row


async def choose(user: User, marker: str, value) -> None:
    """Pick from a select or toggle, then let the page redraw, as a person would."""
    for element in user.find(marker=marker).elements:
        element.set_value(value)
    await asyncio.sleep(0.2)


async def type_into(user: User, marker: str, text: str) -> None:
    user.find(marker=marker).type(text)
    await asyncio.sleep(0.05)


def table_text(user: User) -> str:
    """Everything in the results table: the harness sees elements, not a table's cells."""
    return " ".join(str(v) for t in user.find(ui.table).elements for r in t.rows for v in r.values())


async def click_row(user: User, i: int) -> None:
    user.find(ui.table).trigger("rowClick", [{}, {"_id": i}, i])
    await asyncio.sleep(0.2)


@pytest.fixture
def open_tagged(tagged, monkeypatch):  # noqa: F811
    monkeypatch.setattr(CONTEXT, "project", tagged)
    return tagged


@pytest.fixture
def open_imported(imported, monkeypatch):  # noqa: F811
    monkeypatch.setattr(CONTEXT, "project", imported)
    return imported


# --- quick questions --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_is_last_in_the_navigation_and_opens_on_quick_questions(user: User, open_tagged):
    from transcript_toolkit.app.pages.common import NAV
    assert NAV[-1] == ("/ask", "Ask")
    await user.open("/ask")
    await user.should_see("Choose a question")


@pytest.mark.asyncio
async def test_a_quick_question_is_filled_in_its_own_sentence(user: User, open_tagged):
    await user.open("/ask")
    await choose(user, "ask-quick", "Who talks about…")
    await user.should_see("Who talks about")
    await choose(user, "ask-blank-0", "Education")
    user.find(marker="ask").click()
    await user.should_see("Who talks about Education?")
    assert "Ann" in table_text(user) and "Bob" not in table_text(user)


@pytest.mark.asyncio
async def test_an_empty_blank_is_refused_not_guessed(user: User, open_tagged):
    await user.open("/ask")
    await choose(user, "ask-quick", "Who says the words…")
    user.find(marker="ask").click()
    await user.should_see("Fill in every blank first.")


@pytest.mark.asyncio
async def test_before_tagging_it_says_what_to_run(user: User, open_imported):
    await user.open("/ask")
    await user.should_see("more once you run")
    await choose(user, "ask-quick", "Who says the words…")
    await type_into(user, "ask-blank-0", "school")
    user.find(marker="ask").click()
    await user.should_see("Who says “school”?")
    text = table_text(user)
    assert "I loved school and my teachers." in text
    assert "Tell me about school." not in text                # the interviewer, toggled off


# --- the builder ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_builder_joins_rows_with_or(user: User, open_tagged):
    await user.open("/ask")
    await choose(user, "ask-mode", "build")
    await type_into(user, "ask-words-0", "school")
    user.find("or…").click()
    await asyncio.sleep(0.2)
    await type_into(user, "ask-words-1", "Kenya")
    user.find(marker="ask").click()
    await user.should_see('Find clips where words contains "school" or words contains "Kenya" '
                          '(narrators)')
    text = table_text(user)
    assert f"{A1}_0001" in text and f"{A1}_0002" in text


@pytest.mark.asyncio
async def test_words_offer_more_than_contains(user: User, open_tagged):
    await user.open("/ask")
    await choose(user, "ask-mode", "build")
    await choose(user, "ask-condition-0", "contains_any")
    await type_into(user, "ask-words-0", "Kenya, election")
    user.find(marker="ask").click()
    await user.should_see('Find clips where words contains any of "Kenya", "election" '
                          '(narrators)')


@pytest.mark.asyncio
async def test_a_built_question_saves_into_quick_questions(user: User, open_tagged):
    await user.open("/ask")
    await choose(user, "ask-mode", "build")
    await type_into(user, "ask-words-0", "Kenya")
    user.find("Save as a quick question").click()
    await choose(user, "ask-mode", "quick")
    await user.should_see("Your saved questions")
    await user.should_see('Find clips where words contains "Kenya" (narrators)')


# --- from an answer back to the transcript ----------------------------------------------------

@pytest.mark.asyncio
async def test_a_clip_opens_in_its_transcript(user: User, open_tagged):
    await user.open("/ask")
    await choose(user, "ask-mode", "build")
    await type_into(user, "ask-words-0", "Kenya")
    user.find(marker="ask").click()
    await asyncio.sleep(0.2)
    await click_row(user, 0)
    await user.should_see(A1)                                   # the session it is in
    await user.should_see("Tell me about school.")              # the rest of the transcript
    passage = user.find(marker="ask-passage").elements
    assert len(passage) == 1 and "<mark" in passage.pop().content


@pytest.mark.asyncio
async def test_an_interview_opens_into_its_clips_and_back(user: User, open_tagged):
    await user.open("/ask")
    await choose(user, "ask-quick", "Who mostly talks about…")
    await choose(user, "ask-blank-0", "Work")
    user.find(marker="ask").click()
    await user.should_see("Which interviews are mostly about Work?")
    await click_row(user, 0)
    await user.should_see(marker="ask-back")
    await user.should_see("Find clips where interview is alpha (narrators)")
    user.find(marker="ask-back").click()
    await user.should_see("Which interviews are mostly about Work?")


def test_marking_escapes_the_transcript():
    assert marked("<b>School</b> days", ["school"]) == \
        '&lt;b&gt;<mark class="tk-found">School</mark>&lt;/b&gt; days'


@pytest.mark.asyncio
async def test_the_new_tab_view_shows_the_whole_answer(user: User, open_tagged):
    q = Question("find", "clips", (Row("topic:main", "might_talk_about", ("Education",)),))
    await user.open(view_href(q))
    await user.should_see("Find clips where topic might be Education (narrators)")
    await user.should_see("3 found · 2 clear, 1 possible")
    await user.should_see("Education mattered to my family.")   # the whole text, in a card
    await user.should_see("possible match")
    user.find(marker="ask-card-0").trigger("click")
    await asyncio.sleep(0.2)
    await user.should_see(marker="ask-passage")


@pytest.mark.asyncio
async def test_typing_a_word_into_a_topic_blank_searches_for_it(user: User, open_tagged):
    await user.open("/ask")
    await choose(user, "ask-quick", "Who talks about…")
    await user.should_see("type any word to search the transcripts")
    user.find(marker="ask-blank-0").trigger("inputValue", "ELECTION")
    user.find(marker="ask").click()
    await user.should_see("Who talks about “ELECTION”?")
    assert "Ann" in table_text(user) and "Bob" not in table_text(user)


@pytest.mark.asyncio
async def test_typing_a_topic_name_in_any_case_picks_the_topic(user: User, open_tagged):
    await user.open("/ask")
    await choose(user, "ask-quick", "Who talks about…")
    user.find(marker="ask-blank-0").trigger("inputValue", "family")
    user.find(marker="ask").click()
    await user.should_see("Who talks about Family?")
    assert "Bob" in table_text(user)
