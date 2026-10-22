"""The Ask page, driven the way a user drives it, on the collection in test_ask.py."""
import asyncio

import pytest
from nicegui import ui
from nicegui.testing import User
from test_ask import imported, tagged  # noqa: F401  (fixtures)

from transcript_toolkit.app.context import CONTEXT
from transcript_toolkit.app.pages.ask import view_href
from transcript_toolkit.core.ask import Question, Row


async def pick(user: User, button: str) -> None:
    """Click a quick question and let the builder redraw before touching it, as a person would."""
    user.find(button).click()
    await asyncio.sleep(0.2)


def table_text(user: User) -> str:
    """Everything in the results table: the harness sees elements, not a table's cells."""
    return " ".join(str(v) for t in user.find(ui.table).elements for r in t.rows for v in r.values())


@pytest.fixture
def open_tagged(tagged, monkeypatch):  # noqa: F811
    monkeypatch.setattr(CONTEXT, "project", tagged)
    return tagged


@pytest.fixture
def open_imported(imported, monkeypatch):  # noqa: F811
    monkeypatch.setattr(CONTEXT, "project", imported)
    return imported


@pytest.mark.asyncio
async def test_ask_is_last_in_the_navigation(user: User, open_tagged):
    from transcript_toolkit.app.pages.common import NAV
    assert NAV[-1] == ("/ask", "Ask")
    await user.open("/ask")
    await user.should_see("Quick questions")
    await user.should_see("Who talks about…")


@pytest.mark.asyncio
async def test_a_quick_question_fills_the_builder_and_answers(user: User, open_tagged):
    await user.open("/ask")
    await pick(user, "Who says the words…")
    user.find(marker="ask-words").type("school")
    user.find(marker="ask").click()
    await user.should_see("Who says “school”?")
    text = table_text(user)
    assert "I loved school and my teachers." in text
    assert "Tell me about school." not in text                # the interviewer, toggled off


@pytest.mark.asyncio
async def test_an_empty_blank_is_refused_not_guessed(user: User, open_tagged):
    await user.open("/ask")
    await pick(user, "Who says the words…")
    user.find(marker="ask").click()
    await user.should_see("Pick something for every box first.")


@pytest.mark.asyncio
async def test_before_tagging_it_says_what_to_run(user: User, open_imported):
    await user.open("/ask")
    await user.should_see("More to ask about after you run:")
    await user.should_see("Topics")


@pytest.mark.asyncio
async def test_saving_adds_a_quick_question(user: User, open_tagged):
    await user.open("/ask")
    await pick(user, "Who says the words…")
    user.find(marker="ask-words").type("Kenya")
    user.find(marker="ask").click()
    await user.should_see("Who says “Kenya”?")
    user.find("Save as a quick question").click()
    await user.should_see('Find paragraphs where words contains "Kenya" (narrators)')


@pytest.mark.asyncio
async def test_the_new_tab_view_shows_the_whole_answer(user: User, open_tagged):
    q = Question("find", "clips", (Row("topic:main", "might_talk_about", ("Education",)),))
    await user.open(view_href(q))
    await user.should_see("Find clips where topic might talk about Education (narrators)")
    await user.should_see("3 found · 2 clear, 1 possible")
    assert "why education mattered" in table_text(user)
