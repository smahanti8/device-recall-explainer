"""Tests for the run time limit (written before the implementation).

The hackathon guide asks that a single run completes in minutes with internal timeouts. Calls already time out
(model 90 s, Zetaris 60 s), but a whole run had no limit: the exploratory run took about 18 minutes.

    agent/deadline.py
      start(seconds, clock=time.monotonic)   seconds None or 0 means no limit
      stop()                                 clears it
      expired() -> bool                      True once `seconds` have passed on the clock
      remaining() -> float | None            seconds left (never below 0), None when no limit is set
      TimeLimitReached(LLMError)             raised by llm.chat before an attempt when the limit has passed
    loop.investigate: before each round after the first, if the limit has passed, stops with no answer. The trace gets
        {"round": n, "claims": 0, "rejected": 1, "reasons": ["time_limit"], "details": ["run time limit reached"]}
        and the unsupported list holds a claim whose text names the time limit.
    main.ask(..., timeout=None): starts the limit for the run and always stops it afterwards.
    main CLI: --timeout SECONDS (default 600; 0 means no limit), passed to the ask function as timeout=.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import unittest

from agent import deadline
from agent.evaluator import Citation, Claim, Verdict
from agent.llm import LLMError, chat
from agent.loop import Draft, investigate
from agent.main import ask, main

FULL = {"ZETARIS_REST_URL": "u", "ZETARIS_ORG_ID": "1", "ZETARIS_API_KEY": "k", "MODEL_BASE_URL": "u",
        "MODEL_API_KEY": "k", "NEMOTRON_MODEL": "m"}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class DeadlineModule(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.addCleanup(deadline.stop)

    def test_it_expires_after_the_limit(self):
        deadline.start(10, clock=self.clock)
        self.assertFalse(deadline.expired())
        self.assertEqual(deadline.remaining(), 10)
        self.clock.now = 10.5
        self.assertTrue(deadline.expired())
        self.assertEqual(deadline.remaining(), 0)

    def test_no_limit_never_expires(self):
        for seconds in (None, 0):
            deadline.start(seconds, clock=self.clock)
            self.clock.now = 10 ** 6
            self.assertFalse(deadline.expired())
            self.assertIsNone(deadline.remaining())

    def test_stop_clears_it(self):
        deadline.start(1, clock=self.clock)
        deadline.stop()
        self.clock.now = 5
        self.assertFalse(deadline.expired())

    def test_chat_refuses_to_start_a_call_after_the_limit(self):
        deadline.start(1, clock=self.clock)
        self.clock.now = 2
        sent = []
        with self.assertRaises(deadline.TimeLimitReached) as caught:
            chat([{"role": "user", "content": "hi"}], send=lambda m, t: sent.append(1))
        self.assertIsInstance(caught.exception, LLMError)
        self.assertEqual(sent, [])


class InTheLoop(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.addCleanup(deadline.stop)
        self.claims = [Claim("c", 1, Citation("SELECT 1 AS n", "n"))]

    def test_the_loop_stops_before_the_next_round(self):
        deadline.start(10, clock=self.clock)
        calls = []

        def draft_fn(question, feedback):
            calls.append(1)
            self.clock.now += 30  # the round takes longer than the whole limit
            return Draft("a", [])  # no claims, so it is rejected
        result = investigate("q", draft_fn, lambda claims: Verdict())
        self.assertEqual(len(calls), 1)
        self.assertIsNone(result.text)
        self.assertEqual(result.trace[-1]["reasons"], ["time_limit"])
        self.assertEqual(result.trace[-1]["details"], ["run time limit reached"])
        self.assertIn("time limit", result.unsupported[0].text)

    def test_a_run_that_finishes_in_time_is_unaffected(self):
        deadline.start(10, clock=self.clock)
        result = investigate("q", lambda question, feedback: Draft("a", self.claims), lambda claims: Verdict())
        self.assertEqual(result.text, "a")


class InTheCommandLine(unittest.TestCase):
    def passed_timeout(self, argv):
        seen = {}

        def fake_ask(question, memory, **kwargs):
            seen.update(kwargs)
            return type("R", (), {"text": "ok"})()
        main(argv, env=FULL, ask_fn=fake_ask)
        return seen["timeout"]

    def test_the_default_is_600_seconds(self):
        self.assertEqual(self.passed_timeout(["why?", "--no-memory"]), 600)

    def test_it_can_be_changed_or_turned_off(self):
        self.assertEqual(self.passed_timeout(["why?", "--no-memory", "--timeout", "30"]), 30)
        self.assertEqual(self.passed_timeout(["why?", "--no-memory", "--timeout", "0"]), 0)

    def test_ask_runs_with_the_limit_and_clears_it_afterwards(self):
        seen = []

        def run_investigation(question, draft_fn, evaluate_fn):
            seen.append(deadline.remaining())
            from agent.loop import Result
            return Result("ok", [], 1, [], claims=[Claim("c", 1, Citation("SELECT 1 AS n", "n"))])
        ask("why?", None, make_draft=lambda **kw: (lambda q, f: Draft("a", [])), run_investigation=run_investigation,
            evaluate_fn=lambda claims: Verdict(), out=lambda line: None, progress=lambda m: None,
            run_sql=lambda sql: [{"n": "1"}], timeout=600)
        self.assertTrue(0 < seen[0] <= 600)
        self.assertIsNone(deadline.remaining())


if __name__ == "__main__":
    unittest.main()
