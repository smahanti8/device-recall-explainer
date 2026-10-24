"""Citation gate: a claim stands only if its cited SQL cell equals the stated value.

Every number in an answer must be computed by SQL, never by the model. The
evaluator re-runs each claim's citation and compares the cell with the claim.
"""
import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from agent import tools_zetaris


@dataclass(frozen=True)
class Citation:
    sql: str
    column: str
    row: int = 0


@dataclass(frozen=True)
class Claim:
    text: str
    value: Any
    citation: Optional[Citation] = None


@dataclass(frozen=True)
class Rejection:
    claim: Claim
    reason: str  # no_citation | sql_failed | no_such_cell | scope_mismatch | value_mismatch | no_claims | unclaimed_number
    detail: str


@dataclass
class Verdict:
    rejections: List[Rejection] = field(default_factory=list)

    @property
    def ok(self):
        return not self.rejections


def _equal(claimed, actual):
    """Numbers compare as numbers (Zetaris returns strings); anything else as text."""
    try:
        return math.isclose(float(claimed), float(actual))
    except (TypeError, ValueError):
        return str(claimed).strip() == str(actual).strip()


# Three-letter uppercase words in a claim are read as FDA product codes unless listed here.
_NOT_PRODUCT_CODES = {"FDA", "SQL", "USA", "USD", "MDR", "CFR", "PMA", "CSV", "API", "ALL", "THE", "AND"}
_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(%?)")


def product_codes(text):
    """FDA product codes named in a piece of text (three capitals, minus common acronyms)."""
    return [code for code in re.findall(r"\b[A-Z]{3}\b", text) if code not in _NOT_PRODUCT_CODES]


def _scope_problem(claim):
    """A claim's wording must be reflected in its SQL: 'cardiovascular' and any product code."""
    sql, wanted = claim.citation.sql.lower(), []
    if "cardiovascular" in claim.text.lower():
        wanted.append("cardiovascular")
    wanted += product_codes(claim.text)
    missing = [term for term in wanted if term.lower() not in sql]
    if missing:
        return (f"claim mentions {', '.join(missing)} but the cited SQL never does; "
                "scope the query (join openfda.classification_cardiovascular_table, or filter on the product code)")
    return None


def unclaimed_numbers(text, claims):
    """Numbers written in an answer that no claim backs. Years and single digits are exempt."""
    values = []
    for claim in claims:
        try:
            values.append(float(claim.value))
        except (TypeError, ValueError):
            pass
    missing = []
    for token, percent in _NUMBER.findall(text or ""):
        token = token.rstrip(",")
        number = float(token.replace(",", ""))
        if "." not in token and ("," not in token) and (number < 10 or 1900 <= number <= 2100) and not percent:
            continue
        candidates = [number, number / 100] if percent else [number]
        if not any(math.isclose(v, c, abs_tol=1e-6) for v in values for c in candidates):
            missing.append(token + percent)
    return missing


def _check(claim, run_sql):
    cite = claim.citation
    if cite is None:
        return Rejection(claim, "no_citation", "claim has no SQL citation")
    try:
        rows = run_sql(cite.sql)
    except Exception as error:
        return Rejection(claim, "sql_failed", str(error)[:200])
    try:
        actual = rows[cite.row][cite.column]
    except (IndexError, KeyError):
        return Rejection(claim, "no_such_cell",
                         f"no row {cite.row} / column {cite.column!r} in the result of {cite.sql}")
    problem = _scope_problem(claim)
    if problem:
        return Rejection(claim, "scope_mismatch", problem)
    if not _equal(claim.value, actual):
        return Rejection(claim, "value_mismatch",
                         f"claimed {claim.value!r} but {cite.sql} returned {actual!r}")
    return None


def evaluate(claims, run_sql=None):
    run_sql = run_sql or tools_zetaris.run_sql
    checked = (_check(claim, run_sql) for claim in claims)
    return Verdict([rejection for rejection in checked if rejection])
