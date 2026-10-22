"""The Ask page's AI summary: what is sent, what it costs, that it is paid for once."""
import json

import pandas as pd
import pytest
from test_ask import imported, tagged  # noqa: F401  (fixtures)

from transcript_toolkit.core import ask_quick, ask_summary, cost
from transcript_toolkit.core.ask import Question, Row
from transcript_toolkit.errors import ToolkitError
from transcript_toolkit.steps import ask_summary as step
from transcript_toolkit.steps.cost import spend_report

EDUCATION = Question("find", "clips", (Row("topic:main", "might_talk_about", ("Education",)),))
TITLE = "Find clips where topic might be Education (narrators)"


@pytest.fixture
def fake_model(monkeypatch):
    """Stand in for OpenAI: count the calls, answer with one sentence."""
    calls = []

    def call_llm(client, model, reasoning, verbosity, schema, instructions, content, *a, **k):
        calls.append(content)
        return {"summary": "They loved school."}, {"input_tokens": 900, "output_tokens": 300,
                                                   "reasoning_tokens": 200,
                                                   "cached_input_tokens": 0}

    monkeypatch.setattr(step, "call_llm", call_llm)
    monkeypatch.setattr(step, "openai_client", lambda root: object())
    return calls


def test_the_passages_go_clearest_first_and_say_when_some_were_left_out():
    answer = pd.DataFrame({"clip": ["c1", "c2", "c3"], "interview": ["a", "a", "b"],
                           "text": ["one", "two", "three"], "match": ["clear"] * 3})
    m = ask_summary.material(answer, "Who talks about X?", max_clips=2)
    assert (m.n_found, m.n_read) == (3, 2)
    assert m.user_content.startswith("Question: Who talks about X?")
    assert "There are 3 passages; you are given the clearest 2." in m.user_content
    assert "[c1 · a]\none" in m.user_content and "three" not in m.user_content


def test_nothing_to_summarize_is_said_without_a_call():
    empty = pd.DataFrame({"text": [], "match": []})
    assert ask_summary.without_ai(empty) == "Nothing in the transcripts matches this."
    weak = pd.DataFrame({"text": ["a", "b"], "match": ["possible", "possible"]})
    assert ask_summary.without_ai(weak).startswith("No passage is clearly about this; 2 may")
    assert not ask_summary.summarisable(weak)
    clear = pd.DataFrame({"text": ["a"], "match": ["clear"]})
    assert ask_summary.without_ai(clear) is None and ask_summary.summarisable(clear)


def test_the_estimate_is_the_prompt_and_passages_plus_the_expected_answer(tagged):
    p = ask_summary.prepare(tagged, EDUCATION, TITLE)
    e = ask_summary.estimate(tagged, p)
    tokens_in = (len(p.instructions) + len(p.material.user_content)) // 4
    assert e.tokens == tokens_in + 800
    rates = cost.pricing()["gpt-5.6-luna"]["standard"]
    assert e.usd == pytest.approx(cost.cost(tokens_in, 0, 800, rates))
    assert ask_summary.describe(ask_summary.Estimate(9_437, 0.0213)) == \
        "about 9,400 tokens, ≈ $0.02"
    assert ask_summary.describe(ask_summary.Estimate(1_200, 0.004)) == \
        "about 1,200 tokens, under $0.01"


def test_a_summary_is_paid_for_once(tagged, fake_model, capsys):
    q = json.dumps(ask_quick.question_to(EDUCATION))
    assert step.run_ask_summary(tagged, q, TITLE, yes=True) == "They loved school."
    assert step.run_ask_summary(tagged, q, TITLE, yes=True) == "They loved school."
    assert len(fake_model) == 1
    assert "Already summarized; nothing paid for this time." in capsys.readouterr().out
    p = ask_summary.prepare(tagged, EDUCATION, TITLE)
    assert ask_summary.saved(tagged, p) == "They loved school."
    spent = next(e for e in spend_report(tagged)["steps"] if e["key"] == "ask")
    assert spent["calls"] == 1 and spent["usd"] > 0              # `toolkit cost` counts it


def test_once_measured_the_estimate_uses_this_projects_real_answers(tagged, fake_model):
    step.run_ask_summary(tagged, json.dumps(ask_quick.question_to(EDUCATION)), TITLE, yes=True)
    p = ask_summary.prepare(tagged, EDUCATION, "another wording")
    tokens_in = (len(p.instructions) + len(p.material.user_content)) // 4
    assert ask_summary.estimate(tagged, p).tokens == tokens_in + 300


def test_the_command_asks_about_the_cost_first(tagged, fake_model, monkeypatch):
    asked = []
    monkeypatch.setattr(step, "confirm_or_abort",
                        lambda question, yes=False: asked.append(question))
    step.run_ask_summary(tagged, json.dumps(ask_quick.question_to(EDUCATION)), TITLE)
    assert asked and asked[0].startswith("AI summary of 3 passages with gpt-5.6-luna: about ")
    assert asked[0].endswith(". Go ahead?")


def test_a_question_with_no_passages_is_refused(tagged, fake_model):
    nothing = Question("find", "clips", (Row("words", "contains", ("zebra",)),))
    with pytest.raises(ToolkitError, match="Nothing to summarize"):
        step.run_ask_summary(tagged, json.dumps(ask_quick.question_to(nothing)), yes=True)
    assert fake_model == []


def test_a_project_made_before_summaries_gets_the_new_files(tagged):
    (tagged.advanced_dir / "ask.yaml").unlink()
    (tagged.prompts_dir / "ask_summary.md").write_text("my own wording")
    assert ask_summary.add_missing_files(tagged) == ["advanced/ask.yaml"]
    assert (tagged.prompts_dir / "ask_summary.md").read_text() == "my own wording"


def test_the_prompt_forbids_guessing_pronouns_and_stretching():
    from importlib import resources
    text = (resources.files("transcript_toolkit") / "defaults" / "prompts"
            / "ask_summary.md").read_text()
    assert "Never guess anyone's gender from a name" in text
    assert "say so plainly" in text
