"""Tests for token accounting (written before the implementation).

Judges and users should see what a run costs. Counts come from the model responses themselves, so nothing is estimated.

    llm.parse_usage(body) -> {"input_tokens", "output_tokens", "cost_usd"}
        from body["usage"] (prompt_tokens, completion_tokens), falling back to body["meterless"]["receipt"]["usage"]
        (input_tokens, output_tokens); cost_usd from meterless.receipt.cost.actual_usd; zeros when absent
    llm.Reply gains input_tokens, output_tokens, cost_usd (default 0)
    Run-log model_call events (success or replayed) carry input_tokens, output_tokens and cost_usd
    The llm log line ends with: input_tokens=%s output_tokens=%s cost_usd=%s
    main.ReceiptCollector: .tokens -> {"calls", "input", "output", "cost_usd"} summed over the lines it saw; the
        on_receipt dict also carries input_tokens, output_tokens, cost_usd; .items keeps exactly two keys
    main.format_trace(result, receipts=(), verbose=False, tokens=None): a "Tokens:" line when tokens["calls"] > 0
        "Tokens: 2 model call(s), 1,300 in, 70 out; cost reported by Nightshift: $0.0000"
    main.ask prints that line when model calls happened.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import datetime
import glob
import json
import logging
import os
import shutil
import tempfile
import unittest

from agent import llm, runlog
from agent.evaluator import Citation, Claim, Verdict
from agent.llm import Reply, chat, parse_usage
from agent.loop import Draft, Result
from agent.main import ReceiptCollector, ask, format_trace

WITH_USAGE = {"model": "nvidia/n", "usage": {"prompt_tokens": 1200, "completion_tokens": 340},
              "choices": [{"message": {"content": "ready"}}],
              "meterless": {"routed_model": "nvidia/n", "receipt": {"receipt_id": "rcpt_1"}}}
RECEIPT_ONLY = {"model": "nvidia/n", "choices": [{"message": {"content": "ready"}}],
                "meterless": {"routed_model": "nvidia/n",
                              "receipt": {"receipt_id": "rcpt_2", "usage": {"input_tokens": 23, "output_tokens": 151},
                                          "cost": {"actual_usd": 0.0012}}}}
GOOD = Result("ok", [], 1, [], claims=[Claim("c", 1, Citation("SELECT 1 AS n", "n"))])
MESSAGES = [{"role": "user", "content": "hi"}]


def log_line(input_tokens, output_tokens, cost=0.0):
    logging.getLogger("agent.llm").info(
        "routed_model=%s receipt_id=%s provider_model=%s egress=%s latency_ms=%s input_tokens=%s output_tokens=%s cost_usd=%s",
        "nvidia/n", "rcpt_x", "meterless/auto", "nvidia", 100, input_tokens, output_tokens, cost)


class Usage(unittest.TestCase):
    def test_openai_style_usage_is_read_first(self):
        self.assertEqual(parse_usage(WITH_USAGE), {"input_tokens": 1200, "output_tokens": 340, "cost_usd": 0.0})

    def test_the_receipt_is_the_fallback_and_supplies_the_cost(self):
        self.assertEqual(parse_usage(RECEIPT_ONLY), {"input_tokens": 23, "output_tokens": 151, "cost_usd": 0.0012})

    def test_a_response_without_usage_counts_as_zero(self):
        self.assertEqual(parse_usage({"choices": []}), {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})

    def test_chat_returns_the_counts_on_the_reply(self):
        reply = chat(MESSAGES, send=lambda m, t: WITH_USAGE, sleep=lambda s: None)
        self.assertEqual((reply.input_tokens, reply.output_tokens, reply.cost_usd), (1200, 340, 0.0))
        self.assertEqual(Reply("t", "m", None).input_tokens, 0)  # the defaults keep old callers working


class LoggedUsage(unittest.TestCase):
    def test_a_successful_model_call_event_carries_the_counts(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        self.addCleanup(runlog.stop)
        log = runlog.RunLog(directory, now=lambda: datetime.datetime(2026, 10, 25, 12, 0, 0))
        runlog.start(log)
        chat(MESSAGES, send=lambda m, t: WITH_USAGE, sleep=lambda s: None)
        with open(log.path) as handle:
            (event,) = [json.loads(line) for line in handle]
        self.assertEqual((event["input_tokens"], event["output_tokens"], event["cost_usd"]), (1200, 340, 0.0))


class Collector(unittest.TestCase):
    def test_totals_are_summed_and_the_items_keep_their_two_keys(self):
        seen = []
        log = logging.getLogger("agent.llm")
        log.setLevel(logging.INFO)
        with ReceiptCollector(seen.append) as receipts:
            log_line(1000, 50)
            log_line(300, 20, 0.0005)
        self.assertEqual(receipts.tokens, {"calls": 2, "input": 1300, "output": 70, "cost_usd": 0.0005})
        self.assertEqual(receipts.items, [{"routed_model": "nvidia/n", "receipt_id": "rcpt_x"}] * 2)
        self.assertEqual((seen[0]["input_tokens"], seen[0]["output_tokens"], seen[0]["cost_usd"]), (1000, 50, 0.0))

    def test_lines_without_token_fields_count_as_zero(self):
        log = logging.getLogger("agent.llm")
        log.setLevel(logging.INFO)
        with ReceiptCollector() as receipts:
            log.info("routed_model=%s receipt_id=%s provider_model=%s egress=%s latency_ms=%s",
                     "nvidia/n", "rcpt_x", "meterless/auto", "nvidia", 100)
        self.assertEqual(receipts.tokens, {"calls": 1, "input": 0, "output": 0, "cost_usd": 0.0})


class Printed(unittest.TestCase):
    def test_the_trace_ends_with_a_tokens_line_when_calls_happened(self):
        tokens = {"calls": 2, "input": 1300, "output": 70, "cost_usd": 0.0}
        text = format_trace(GOOD, [], tokens=tokens)
        self.assertIn("Tokens: 2 model call(s), 1,300 in, 70 out; cost reported by Nightshift: $0.0000", text)

    def test_no_line_without_calls_or_counts(self):
        self.assertNotIn("Tokens:", format_trace(GOOD, []))
        self.assertNotIn("Tokens:", format_trace(GOOD, [], tokens={"calls": 0, "input": 0, "output": 0, "cost_usd": 0.0}))

    def test_ask_prints_the_line_from_the_model_calls_it_saw(self):
        lines = []

        def make_draft(context_fn=None, **kw):
            def draft_fn(question, feedback):
                log_line(1200, 340)
                return Draft("a", [])
            return draft_fn

        def loop_like(question, draft_fn, evaluate_fn):
            draft_fn(question, [])
            return GOOD
        ask("why?", None, make_draft=make_draft, run_investigation=loop_like, evaluate_fn=lambda claims: Verdict(),
            out=lines.append, progress=lambda message: None)
        self.assertIn("Tokens: 1 model call(s), 1,200 in, 340 out", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
