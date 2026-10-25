"""A deterministic check that a question is about what the registered data covers.

The tables hold FDA cardiovascular device recalls for 2023 and 2024. A question with no recall-related word, or that
names another year, is refused before any model call or Zetaris query. This is a keyword and year check, not
understanding, and it is not an injection filter (the SQL guard and the evaluator are the protections for that).
"""
import re

from agent.playbook import DRILLDOWN_CODES

YEARS = {"2023", "2024"}
_TOPIC = re.compile(r"\b(?:recalls?|recalled|devices?|cardiovascular|root\s+cause|firms?|"
                    + "|".join(DRILLDOWN_CODES) + r")\b", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
HINT = ("This agent answers questions about FDA cardiovascular device recalls in 2023 and 2024 "
        "(recall records and events, product codes, root causes, recalling firms).")


def scope_problem(question):
    """None when the question is in scope, otherwise a message that says what to ask instead."""
    if not _TOPIC.search(question or ""):
        return HINT + " Try, for example: Why did cardiovascular device recall records rise between 2023 and 2024?"
    other = sorted({year for year in _YEAR.findall(question) if year not in YEARS})
    if other:
        return (f"The registered data covers {' and '.join(sorted(YEARS))} only; {', '.join(other)} is outside it. "
                + HINT)
    return None
