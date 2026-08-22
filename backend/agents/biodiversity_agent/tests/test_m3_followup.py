"""Reading a typed answer to M3's follow-up question.

The dashboard offers options as buttons, but a user may just as well type the
answer - and almost never as the exact option text. These pin what counts as an
answer, what counts as a different question, and what is left unresolved so the
module asks again instead of analysing something nobody requested.
"""

from __future__ import annotations

import pytest

from backend.agents.biodiversity_agent.workers.hotspots.followup import (
    looks_like_a_new_question,
    resolve_choice,
)

SIX = ["amazon", "congo basin", "kenya", "madagascar", "sahara", "southeast asia"]
TWO = ["amazon", "congo basin"]


@pytest.mark.parametrize("reply, expected", [
    # the option, typed out
    ("madagascar", "madagascar"),
    ("Madagascar", "madagascar"),
    ("congo basin", "congo basin"),
    ("Madagascar?", "madagascar"),
    ("madagascar please", "madagascar"),
    # a documented alias
    ("congo", "congo basin"),
    ("SE Asia", "southeast asia"),
    ("amazonia", "amazon"),
    ("the sahara desert", "sahara"),
    ("DRC", "congo basin"),
    # by position, as displayed
    ("1", "amazon"),
    ("3", "kenya"),
    ("6", "southeast asia"),
    # by ordinal
    ("the first one", "amazon"),
    ("second", "congo basin"),
    ("the last one", "southeast asia"),
    # a distinctive word
    ("asia", "southeast asia"),
])
def test_a_typed_reply_resolves(reply: str, expected: str) -> None:
    assert resolve_choice(reply, SIX) == expected


@pytest.mark.parametrize("reply", [
    "",
    "   ",
    "i don't know",
    "whichever you think",
    "the biggest one",
    "7",              # out of range
    "0",              # 1-based, so this is nobody
])
def test_an_unclear_reply_stays_unresolved(reply: str) -> None:
    """Better to ask again than to analyse a region the user did not choose."""

    assert resolve_choice(reply, SIX) is None


def test_a_number_beyond_the_offered_options_is_not_a_choice() -> None:
    assert resolve_choice("5", TWO) is None
    assert resolve_choice("2", TWO) == "congo basin"


def test_a_name_wins_over_a_number_in_the_same_reply() -> None:
    """"top 5 in kenya" names Kenya; the 5 is a parameter, not a choice."""

    assert resolve_choice("top 5 in kenya", SIX) == "kenya"


def test_an_option_not_offered_is_not_resolved() -> None:
    """Only what was offered can be chosen - the rest is a new question."""

    assert resolve_choice("kenya", TWO) is None
    assert looks_like_a_new_question("kenya", TWO) is True


def test_answering_the_question_is_not_a_new_question() -> None:
    """An offered option is an answer, not a new place to look up."""

    assert looks_like_a_new_question("congo", TWO) is False
    # A number is resolved by resolve_choice, which the caller tries first - so
    # the pair, in the order the dashboard uses them, treats "2" as an answer.
    assert resolve_choice("2", TWO) == "congo basin"


def test_gibberish_is_not_an_answer_to_the_options() -> None:
    """It must never select an option, whatever else happens to it.

    It is passed on as a place attempt rather than refused outright: the
    gazetteer either finds nothing - and the answer says so - or finds something
    too small to grid, and the answer names it. Both are more informative than
    "I did not catch that", and neither can be mistaken for an analysis of a
    region the user did not pick.
    """

    assert resolve_choice("hmm", TWO) is None


def test_no_options_means_nothing_to_choose() -> None:
    assert resolve_choice("madagascar", []) is None


# --------------------------------- a reply that answers nothing but names something


@pytest.mark.parametrize("reply", [
    "island pantelleria",
    "Costa Rica",
    "hotspots in Borneo instead",
    "what about Sumatra",
])
def test_naming_something_else_is_tried_not_refused(reply: str) -> None:
    """Answering "I did not catch that" to a named place is the wrong reply.

    The person is naming a place. Whether it works is for the gazetteer to say -
    and if it fails, the answer explains what it found, which is information.
    """

    assert resolve_choice(reply, SIX) is None or reply.lower() in SIX
    assert looks_like_a_new_question(reply, SIX) is True


@pytest.mark.parametrize("reply", [
    "i don't know", "idk", "whichever", "you choose", "give me all", "?",
])
def test_filler_is_met_with_the_question_again(reply: str) -> None:
    """These carry nothing to look up, so asking again is the only honest move."""

    assert resolve_choice(reply, SIX) is None
    assert looks_like_a_new_question(reply, SIX) is False
