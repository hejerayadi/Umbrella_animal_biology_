"""Reading the user's answer to M3's own follow-up question.

When a question is missing its study area - or names more than one - the module
asks instead of guessing, and offers the options it knows. The user may click one,
but they may equally type a reply, and a typed reply is rarely the exact option
text: "1", "the second one", "SE asia", "congo", "amazonia".

This module turns such a reply into one of the offered options, or into ``None``
when it genuinely cannot tell - in which case the caller asks again rather than
running an analysis the user did not request.

Order matters here. Names are tried before numbers, so "top 5 in kenya" resolves
to Kenya rather than to option 5.
"""

from __future__ import annotations

import re

from ..common.config import regions_in_text

# "the first one", "2nd", "last". Indices are into the offered options.
#
# Ordinals only - never the cardinals "one", "two", "three". A cardinal in a
# reply is usually a count, not a position: "the biggest one" and "two hotspots
# please" would otherwise silently select option 1 and option 2. "one" is the
# worst of them, since it ends most of these phrases.
_ORDINALS: dict[str, int] = {
    "first": 0, "1st": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
    "sixth": 5, "6th": 5,
    "last": -1,
}

_WORD = re.compile(r"[a-z]+")
_NUMBER = re.compile(r"\d+")

# Words that carry no choice on their own, so they must not decide one by
# accidentally overlapping an option name.
_STOPWORDS = {"the", "one", "please", "that", "this", "of", "in", "a", "an",
              "and", "or", "region", "area", "hotspots", "biodiversity"}


def resolve_choice(reply: str | None, options: list[str]) -> str | None:
    """The offered option a free-text reply picks, or None if unclear."""

    if not reply or not options:
        return None

    text = reply.strip().lower()

    # 1. the option, typed out (or with the article, or a question mark)
    for option in options:
        if text.strip(" ?.!") == option:
            return option

    # 2. a name M3 recognises - this is what handles "congo", "SE asia",
    #    "amazonia", and any other documented alias.
    for region in regions_in_text(text):
        if region in options:
            return region

    # 3. one option's name contained in the reply, or the reverse
    #    ("madagascar please", "congo" inside "congo basin")
    for option in options:
        if option in text or (len(text) > 2 and text in option):
            return option

    # 4. "the second one", "last"
    words = set(_WORD.findall(text))
    for word, index in _ORDINALS.items():
        if word in words:
            try:
                return options[index]
            except IndexError:
                return None

    # 5. a bare number, 1-based as displayed
    numbers = _NUMBER.findall(text)
    if len(numbers) == 1:
        position = int(numbers[0]) - 1
        if 0 <= position < len(options):
            return options[position]

    # 6. a distinctive word shared with exactly one option ("asia" -> southeast
    #    asia). Ambiguous overlap is deliberately left unresolved.
    meaningful = words - _STOPWORDS
    matches = [option for option in options
               if meaningful & (set(option.split()) - _STOPWORDS)]
    if len(matches) == 1:
        return matches[0]

    return None


# Replies that answer nothing and name nothing. Anything outside this set is
# given to the worker as a fresh attempt, because "island pantelleria" is a
# person naming a place, and answering it with "I did not catch which one you
# meant" is a worse reply than trying it and explaining what came back.
_NOT_AN_ATTEMPT = ("i don't know", "i dont know", "idk", "dunno", "not sure",
                   "no idea", "whichever", "whatever", "any of them", "anything",
                   "all of them", "give me all", "all", "both", "you choose",
                   "you decide", "up to you", "?", "help")


def looks_like_a_new_question(reply: str, options: list[str]) -> bool:
    """True when a reply should be tried as a place rather than re-asked.

    Someone who ignores the follow-up and names something else - a region M3
    knows, or any place at all - should have it looked up. Only the filler
    replies above are met with the question again, because they carry nothing to
    look up.
    """

    text = (reply or "").strip().lower().strip(" ?.!")
    if not text:
        return False
    if text in _NOT_AN_ATTEMPT or any(text == phrase for phrase in _NOT_AN_ATTEMPT):
        return False
    if len(text.split()) > 8:
        # A sentence that long is a question, not an answer to ours - and the
        # worker reads questions.
        return True
    named = regions_in_text(text)
    if named and any(region in options for region in named):
        return False            # that is an answer, handled by resolve_choice
    return True
