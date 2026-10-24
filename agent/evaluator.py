"""Citation gate: a claim stands only if its cited SQL cell equals the stated value.

Every number in an answer must be computed by SQL, never by the model. The
evaluator re-runs each claim's citation and compares the cell with the claim.
"""
import math
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
    reason: str  # no_citation | sql_failed | no_such_cell | value_mismatch | no_claims
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
    if not _equal(claim.value, actual):
        return Rejection(claim, "value_mismatch",
                         f"claimed {claim.value!r} but {cite.sql} returned {actual!r}")
    return None


def evaluate(claims, run_sql=None):
    run_sql = run_sql or tools_zetaris.run_sql
    checked = (_check(claim, run_sql) for claim in claims)
    return Verdict([rejection for rejection in checked if rejection])
