"""Command line: python -m agent.main "<question>" [--memory-dir .hmem] [--no-memory] [--verbose] [--playbook]

Runs the investigation (draft -> evaluate -> retry), prints the trace with the
model receipts, and remembers the claims that were verified.
"""
import argparse
import logging
import re
import sys

from agent.drafter import make_draft_fn
from agent.evaluator import evaluate, unclaimed_numbers
from agent.llm import LLMError
from agent.loop import investigate
from agent.playbook import make_playbook_draft_fn
from agent.memory import Memory

_RECEIPT = re.compile(r"routed_model=(\S+) receipt_id=(\S+)")
_LATENCY = re.compile(r"latency_ms=(\S+)")


class ReceiptCollector(logging.Handler):
    """Context manager that collects routed_model / receipt_id from agent.llm's log lines."""

    def __init__(self, on_receipt=None):
        super().__init__(logging.INFO)
        self.items = []
        self.on_receipt = on_receipt  # called with routed_model, receipt_id and latency_ms

    def emit(self, record):
        message = record.getMessage()
        match = _RECEIPT.search(message)
        if match:
            receipt_id = None if match.group(2) == "None" else match.group(2)
            self.items.append({"routed_model": match.group(1), "receipt_id": receipt_id})
            if self.on_receipt:
                latency = _LATENCY.search(message)
                self.on_receipt({**self.items[-1], "latency_ms": latency.group(1) if latency else None})

    def __enter__(self):
        self._log = logging.getLogger("agent.llm")
        self._level = self._log.level
        if self._level == logging.NOTSET or self._level > logging.INFO:
            self._log.setLevel(logging.INFO)
        self._log.addHandler(self)
        return self

    def __exit__(self, *exc):
        self._log.removeHandler(self)
        self._log.setLevel(self._level)


def format_trace(result, receipts=(), verbose=False):
    lines = ["Trace"]
    for entry in result.trace:
        outcome = (f"{entry['rejected']} rejected ({', '.join(entry['reasons'])})"
                   if entry["rejected"] else "all verified")
        lines.append(f"  Round {entry['round']}: {entry['claims']} claims, {outcome}")
        if verbose:
            lines += [f"      {reason}: {detail}"
                      for reason, detail in zip(entry["reasons"], entry.get("details", []))]
    if result.text:
        lines.append("Verified claims:")
        lines += [f"  - {claim.text} = {claim.value}" for claim in result.claims]
        lines.append(f"Answer: {result.text}")
    else:
        lines.append(f"No verified answer after {result.rounds} round(s). Could not support:")
        lines += [f"  - {claim.text}" + ("" if claim.value is None else f" = {claim.value}")
                  for claim in result.unsupported]
    if receipts:
        lines.append("Model receipts:")
        lines += [f"  {item['routed_model']}  {item['receipt_id']}" for item in receipts]
    return "\n".join(lines)


def _stderr(message):
    print(message, file=sys.stderr, flush=True)  # sys.stderr is looked up at call time


def ask(question, memory=None, *, make_draft=make_draft_fn, make_playbook=make_playbook_draft_fn,
        run_investigation=investigate, evaluate_fn=evaluate, out=print, progress=None,
        verbose=False, playbook=False):
    progress = progress or _stderr
    make = make_playbook if playbook else make_draft
    state = {"round": 0, "turn": 0, "draft": None, "hints": []}

    def recalling(question):
        text = memory.context_for(question)
        state["hints"] = [line for line in text.splitlines() if line.startswith("- ")]
        count = len(state["hints"])
        progress(f"memory: recalled {count} hint{'' if count == 1 else 's'}" if count
                 else "memory: nothing recalled yet")
        return text

    inner = make(context_fn=recalling if memory is not None else None)

    def draft_fn(question, feedback):
        state["round"] += 1
        progress(f"round {state['round']}: drafting")
        try:
            state["draft"] = inner(question, feedback)
        except LLMError:
            progress(f"round {state['round']}: model unavailable")
            raise
        return state["draft"]

    def checked(claims):
        verdict = evaluate_fn(claims)
        reasons = [r.reason for r in verdict.rejections]
        reasons += ["unclaimed_number"] * len(unclaimed_numbers(state["draft"].text, claims))
        outcome = f"{len(reasons)} rejected ({', '.join(reasons)})" if reasons else "all verified"
        progress(f"round {state['round']}: {len(claims)} claims checked, {outcome}")
        return verdict

    def model_turn(item):
        state["turn"] += 1
        try:
            seconds = f"{float(item['latency_ms']) / 1000:.1f}s"
        except (TypeError, ValueError):
            seconds = "?"
        progress(f"  model turn {state['turn']} ({seconds}) {item['routed_model']} {item['receipt_id']}")

    with ReceiptCollector(model_turn) as receipts:
        result = run_investigation(question, draft_fn, checked)
    if state["hints"]:
        out(f"Recalled from earlier sessions ({len(state['hints'])}):")
        for hint in state["hints"]:
            out(f"  {hint}")
    out(format_trace(result, receipts.items, verbose))
    if memory is not None and result.text:
        try:
            out(f"Memory: remembered {len(memory.learn(result.claims, []))} verified claims")
        except Exception as error:  # memory is optional; never lose a verified answer to it
            out(f"note: memory unavailable ({error}); the answer above is unaffected")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m agent.main", description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--memory-dir", default=".hmem")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="show each rejection's reason and detail")
    parser.add_argument("--playbook", action="store_true",
                        help="code runs the fixed queries and the model only explains (faster, more reliable)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    memory = None if args.no_memory else Memory(args.memory_dir)
    return 0 if ask(args.question, memory, verbose=args.verbose, playbook=args.playbook).text else 1


if __name__ == "__main__":
    sys.exit(main())
