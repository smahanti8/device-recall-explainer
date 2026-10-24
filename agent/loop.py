"""Draft -> evaluate -> retry loop. Rejection reasons go back to the next draft.

The loop owns no model or database logic: `draft_fn` writes a draft with cited
claims, `evaluate_fn` checks them. It stops at the first draft whose claims all
resolve, or after `max_rounds` and reports what it could not support.
A model outage (LLMError) costs one round; other exceptions propagate.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional

from agent.evaluator import Claim, Rejection, unclaimed_numbers
from agent.llm import LLMError


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
    claims: List[Claim] = field(default_factory=list)  # the accepted draft's claims


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
    feedback: List[Rejection] = []  # what the next draft is told
    unsupported: List[Claim] = []   # what the latest round could not support
    trace: List[dict] = []
    for round_number in range(1, max_rounds + 1):
        try:
            draft = draft_fn(question, feedback)
        except LLMError as error:  # an outage costs a round; any other exception is a bug
            unsupported = [Claim(f"model unavailable: {str(error)[:150]}", None)]
            trace.append({"round": round_number, "claims": 0, "rejected": 1,
                          "reasons": ["model_unavailable"]})
            continue
        rejections = _rejections(draft, evaluate_fn)
        trace.append({"round": round_number, "claims": len(draft.claims),
                      "rejected": len(rejections), "reasons": [r.reason for r in rejections]})
        if not rejections:
            return Result(draft.text, [], round_number, trace, claims=draft.claims)
        feedback = rejections
        unsupported = [r.claim for r in rejections]
    return Result(None, unsupported, max_rounds, trace)
