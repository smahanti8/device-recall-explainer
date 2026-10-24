"""Command line: python -m agent.main "<question>" [--memory-dir .hmem] [--no-memory]

Runs the investigation (draft -> evaluate -> retry), prints the trace with the
model receipts, and remembers the claims that were verified.
"""
import argparse
import logging
import re
import sys

from agent.drafter import make_draft_fn
from agent.evaluator import evaluate
from agent.loop import investigate
from agent.memory import Memory

_RECEIPT = re.compile(r"routed_model=(\S+) receipt_id=(\S+)")


class ReceiptCollector(logging.Handler):
    """Context manager that collects routed_model / receipt_id from agent.llm's log lines."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.items = []

    def emit(self, record):
        match = _RECEIPT.search(record.getMessage())
        if match:
            receipt_id = None if match.group(2) == "None" else match.group(2)
            self.items.append({"routed_model": match.group(1), "receipt_id": receipt_id})

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


def format_trace(result, receipts=()):
    lines = ["Trace"]
    for entry in result.trace:
        outcome = (f"{entry['rejected']} rejected ({', '.join(entry['reasons'])})"
                   if entry["rejected"] else "all verified")
        lines.append(f"  Round {entry['round']}: {entry['claims']} claims, {outcome}")
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


def ask(question, memory=None, *, make_draft=make_draft_fn, run_investigation=investigate,
        evaluate_fn=evaluate, out=print):
    draft_fn = make_draft(context_fn=memory.context_for if memory is not None else None)
    with ReceiptCollector() as receipts:
        result = run_investigation(question, draft_fn, evaluate_fn)
    out(format_trace(result, receipts.items))
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    memory = None if args.no_memory else Memory(args.memory_dir)
    return 0 if ask(args.question, memory).text else 1


if __name__ == "__main__":
    sys.exit(main())
