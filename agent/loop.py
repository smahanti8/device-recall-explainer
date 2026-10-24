"""Draft -> evaluate -> retry loop. Rejection reasons go back to the next draft.

The loop owns no model or database logic: `draft_fn` writes a draft with cited
claims, `evaluate_fn` checks them. It stops at the first draft whose claims all
resolve, or after `max_rounds` and reports what it could not support.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional

from agent.evaluator import Claim, Rejection, unclaimed_numbers


@dataclass
class Draft:
    text: str
    claims: List[Claim]


@dataclass
class Result:
    text: Optional[str]  # None when it gave up
    unsupported: List[Claim]
    rounds: int
    trace: List[dict] = field(default_factory=list)


def _rejections(draft, evaluate_fn):
    if not draft.claims:  # an answer with nothing checkable is not evidence
        placeholder = Claim("answer states no verifiable claims", None)
        return [Rejection(placeholder, "no_claims", "every answer needs cited claims")]
    rejections = list(evaluate_fn(draft.claims).rejections)
    for token in unclaimed_numbers(draft.text, draft.claims):
        rejections.append(Rejection(
            Claim(f"unclaimed number {token}", None), "unclaimed_number",
            f"the answer text states {token} but no claim has that value; cite it or remove it"))
    return rejections


def investigate(question, draft_fn, evaluate_fn, max_rounds=3):
    feedback: List[Rejection] = []
    trace: List[dict] = []
    for round_number in range(1, max_rounds + 1):
        draft = draft_fn(question, feedback)
        feedback = _rejections(draft, evaluate_fn)
        trace.append({"round": round_number, "claims": len(draft.claims),
                      "rejected": len(feedback), "reasons": [r.reason for r in feedback]})
        if not feedback:
            return Result(draft.text, [], round_number, trace)
    return Result(None, [r.claim for r in feedback], max_rounds, trace)
