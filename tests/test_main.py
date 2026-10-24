"""Tests for agent/main.py (written before the implementation).

    python -m agent.main "<question>" [--memory-dir .hmem] [--no-memory]

    ask(question, memory=None, *, make_draft=make_draft_fn, run_investigation=investigate,
        evaluate_fn=evaluate, out=print) -> Result
        wires memory.context_for into the drafter (context_fn), runs the investigation,
        prints the trace, and on success calls memory.learn(result.claims, []).
        A memory failure is reported ("memory unavailable") and never stops the run.
    format_trace(result, receipts=()) -> str      one block per round, then the answer or what was unsupported
    ReceiptCollector()                             context manager: collects routed_model / receipt_id log lines
    parse_args(argv) -> namespace(question, memory_dir, no_memory)
    investigate() now also returns the accepted claims as Result.claims (see test_evaluator.py).

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import logging
import unittest

from agent.evaluator import Citation, Claim
from agent.loop import Result
from agent.main import ReceiptCollector, ask, format_trace, parse_args

CLAIM = Claim("DSY recall records in 2024", 54, Citation("SELECT 1 AS n", "n"))
TRACE = [
    {"round": 1, "claims": 2, "rejected": 2, "reasons": ["no_citation", "unclaimed_number"]},
    {"round": 2, "claims": 2, "rejected": 0, "reasons": []},
]
GOOD = Result("DSY had 54 recall records.", [], 2, TRACE, claims=[CLAIM])
GAVE_UP = Result(None, [Claim("model unavailable: 502 provider_deadline", None)], 3, TRACE[:1])


class FakeMemory:
    def __init__(self, context="", fail_on_learn=False):
        self.context, self.fail_on_learn, self.learned = context, fail_on_learn, []

    def context_for(self, question):
        return self.context

    def learn(self, claims, rejections):
        if self.fail_on_learn:
            raise RuntimeError("H-MEM: node missing")
        self.learned.append((claims, rejections))
        return [f"mem-{i}" for i, _ in enumerate(claims, 1)]


def run(result, memory=None):
    lines, seen = [], {}

    def make_draft(context_fn=None, **kwargs):
        seen["context_fn"] = context_fn
        return "draft_fn"
    returned = ask("why?", memory, make_draft=make_draft, run_investigation=lambda q, d, e: result,
                   evaluate_fn="evaluate", out=lines.append)
    return returned, lines, seen


class Ask(unittest.TestCase):
    def test_memory_context_is_wired_into_the_drafter(self):
        _, _, seen = run(GOOD, FakeMemory(context="HINTS"))
        self.assertEqual(seen["context_fn"]("why?"), "HINTS")

    def test_no_memory_means_no_context_hook(self):
        _, _, seen = run(GOOD, None)
        self.assertIsNone(seen["context_fn"])

    def test_a_successful_run_is_remembered(self):
        memory = FakeMemory()
        result, lines, _ = run(GOOD, memory)
        self.assertIs(result, GOOD)
        self.assertEqual(memory.learned, [([CLAIM], [])])
        self.assertIn("remembered 1", "\n".join(lines))

    def test_a_run_that_gave_up_remembers_nothing(self):
        memory = FakeMemory()
        run(GAVE_UP, memory)
        self.assertEqual(memory.learned, [])

    def test_a_memory_failure_never_stops_the_run(self):
        result, lines, _ = run(GOOD, FakeMemory(fail_on_learn=True))
        self.assertIs(result, GOOD)
        self.assertIn("memory unavailable", "\n".join(lines))


class Trace(unittest.TestCase):
    def test_each_round_and_the_answer_are_shown(self):
        text = format_trace(GOOD)
        for needle in ("Round 1", "no_citation", "unclaimed_number", "Round 2", "DSY had 54 recall records."):
            self.assertIn(needle, text)

    def test_a_run_that_gave_up_says_what_it_could_not_support(self):
        text = format_trace(GAVE_UP)
        self.assertIn("No verified answer", text)
        self.assertIn("502 provider_deadline", text)

    def test_receipts_are_listed_when_given(self):
        text = format_trace(GOOD, [{"routed_model": "nvidia/n", "receipt_id": "rcpt_1"}])
        self.assertIn("rcpt_1", text)
        self.assertIn("nvidia/n", text)

    def test_receipt_collector_reads_the_llm_log_lines(self):
        log = logging.getLogger("agent.llm")
        log.setLevel(logging.INFO)
        with ReceiptCollector() as receipts:
            log.info("routed_model=%s receipt_id=%s provider_model=%s egress=%s latency_ms=%s",
                     "nvidia/n", "rcpt_1", "meterless/auto", "nvidia", 12)
        self.assertEqual(receipts.items, [{"routed_model": "nvidia/n", "receipt_id": "rcpt_1"}])


class Cli(unittest.TestCase):
    def test_defaults_and_flags(self):
        args = parse_args(["why?"])
        self.assertEqual((args.question, args.memory_dir, args.no_memory), ("why?", ".hmem", False))
        self.assertTrue(parse_args(["why?", "--no-memory"]).no_memory)
        self.assertEqual(parse_args(["why?", "--memory-dir", "/tmp/m"]).memory_dir, "/tmp/m")


if __name__ == "__main__":
    unittest.main()
