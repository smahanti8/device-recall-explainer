"""Tests for the observability line and panel (written before the implementation).

Judges should see, per run, what the agent spent and where: Zetaris queries fetched versus served from the local cache,
which model answered how many times, retries and failures. Counts come from what actually ran.

    agent/observability.py
      summarize(queries, calls) -> {"queries": {"fetched", "cache_hit", "replayed", "errors"},
                                    "models": {model: successful calls, in first-seen order}, "retries", "failed"}
        queries: tools_zetaris.query_log() entries (status success|cache_hit|replayed|error)
        calls:   llm.call_log() entries {"status": success|replayed|retry|failed, "model"} (one per attempt)
      format_line(summary) -> "Observability: Zetaris: 36 queries fetched, 78 served from cache; models: A x1, B x1; retries: 3"
        replayed results read "Zetaris: N results replayed from recordings"; zero parts are omitted; "" when nothing ran
    llm.call_log() / clear_call_log(): every attempt of chat (retries and failures included)
    main.ask: prints the line after the Tokens line and passes the summary to the report as data["observability"];
        report.render shows an "Observability" panel with the same numbers.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import glob
import os
import shutil
import tempfile
import unittest
from unittest import mock

from agent import llm, tools_zetaris
from agent.evaluator import Citation, Claim, Verdict
from agent.llm import TransientError, chat
from agent.loop import Draft, Result
from agent.main import ask
from agent.observability import format_line, summarize
from agent.report import render

A, B = "nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-3-ultra-550b-a55b"
SUMMARY = {"queries": {"fetched": 36, "cache_hit": 78, "replayed": 0, "errors": 0},
           "models": {A: 1, B: 1}, "retries": 3, "failed": 0}
BODY = {"model": A, "choices": [{"message": {"content": "ready"}}],
        "meterless": {"routed_model": A, "receipt": {"receipt_id": "rcpt_1"}}}


def queries(**counts):
    return [{"sql": "SELECT 1", "status": status, "rows": 1} for status, n in counts.items() for _ in range(n)]


class Summarize(unittest.TestCase):
    def test_queries_are_counted_by_status(self):
        summary = summarize(queries(success=36, cache_hit=78, replayed=2, error=1), [])
        self.assertEqual(summary["queries"], {"fetched": 36, "cache_hit": 78, "replayed": 2, "errors": 1})

    def test_models_retries_and_failures_come_from_the_attempts(self):
        calls = [{"status": "retry", "model": None}, {"status": "retry", "model": None}, {"status": "success", "model": A},
                 {"status": "success", "model": B}, {"status": "replayed", "model": A}, {"status": "failed", "model": None}]
        summary = summarize([], calls)
        self.assertEqual(summary["models"], {A: 2, B: 1})
        self.assertEqual((summary["retries"], summary["failed"]), (2, 1))

    def test_nothing_ran_is_all_zeros(self):
        summary = summarize([], [])
        self.assertEqual((summary["models"], summary["retries"], summary["failed"]), ({}, 0, 0))


class FormatLine(unittest.TestCase):
    def test_a_live_run(self):
        self.assertEqual(format_line(SUMMARY), f"Observability: Zetaris: 36 queries fetched, 78 served from cache; "
                                               f"models: {A} x1, {B} x1; retries: 3")

    def test_a_replay_and_zero_parts_are_omitted(self):
        summary = {"queries": {"fetched": 0, "cache_hit": 0, "replayed": 114, "errors": 0}, "models": {A: 2},
                   "retries": 0, "failed": 0}
        self.assertEqual(format_line(summary), f"Observability: Zetaris: 114 results replayed from recordings; models: {A} x2")

    def test_failures_are_shown(self):
        summary = dict(SUMMARY, failed=1, queries=dict(SUMMARY["queries"], errors=2))
        line = format_line(summary)
        self.assertIn("2 failed", line)
        self.assertIn("failed calls: 1", line)

    def test_nothing_ran_gives_no_line(self):
        self.assertEqual(format_line(summarize([], [])), "")


class CallLog(unittest.TestCase):
    def setUp(self):
        llm.clear_call_log()
        self.addCleanup(llm.clear_call_log)

    def test_every_attempt_is_recorded_with_its_status_and_model(self):
        tries = []

        def flaky(messages, max_tokens):
            tries.append(1)
            if len(tries) == 1:
                raise TransientError("502")
            return BODY
        chat([{"role": "user", "content": "hi"}], send=flaky, sleep=lambda s: None)
        self.assertEqual([(c["status"], c["model"]) for c in llm.call_log()], [("retry", None), ("success", A)])
        llm.clear_call_log()
        self.assertEqual(llm.call_log(), [])


class InTheRun(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        tools_zetaris._CACHE.clear()
        self.addCleanup(tools_zetaris._CACHE.clear)
        llm.clear_call_log()
        self.addCleanup(llm.clear_call_log)
        env = mock.patch.dict(os.environ, {"ZETARIS_REST_URL": "http://z.test/api", "ZETARIS_ORG_ID": "1", "ZETARIS_API_KEY": "k-test-1"})
        env.start()
        self.addCleanup(env.stop)
        post = mock.patch("agent.tools_zetaris.requests.post")
        post.start().return_value = mock.Mock(ok=True, json=lambda: {"headers": ["n"], "data": [["54"]]})
        self.addCleanup(post.stop)

    def run_ask(self, **kwargs):
        lines = []

        def make_draft(context_fn=None, **kw):
            def draft_fn(question, feedback):
                tools_zetaris.run_sql("SELECT 54 AS n")   # fetched
                tools_zetaris.run_sql("SELECT 54 AS n")   # served from the cache
                chat([{"role": "user", "content": "hi"}], send=lambda m, t: BODY, sleep=lambda s: None)
                return Draft("a", [])
            return draft_fn

        def loop_like(question, draft_fn, evaluate_fn):
            draft_fn(question, [])
            return Result("ok", [], 1, [], claims=[Claim("c", 54, Citation("SELECT 54 AS n", "n"))])
        ask("why?", None, make_draft=make_draft, run_investigation=loop_like, evaluate_fn=lambda claims: Verdict(),
            out=lines.append, progress=lambda m: None, run_sql=lambda sql: [{"n": "54"}], **kwargs)
        return "\n".join(lines)

    def test_the_run_prints_the_line_with_real_counts(self):
        shown = self.run_ask()
        self.assertIn(f"Observability: Zetaris: 1 queries fetched, 1 served from cache; models: {A} x1", shown)

    def test_the_report_carries_the_same_numbers(self):
        self.run_ask(report_dir=self.dir)
        (path,) = glob.glob(os.path.join(self.dir, "report-*.html"))
        with open(path) as handle:
            page = handle.read()
        for needle in ("Observability", "1 queries fetched", "1 served from cache", A):
            self.assertIn(needle, page)


class ReportPanel(unittest.TestCase):
    def test_the_panel_shows_the_summary(self):
        data = {"question": "q", "answer": None, "claims": [], "rounds": [], "queries": [], "receipts": [], "tokens": {},
                "hints": [], "memory": None, "mode": "live", "timeline": [], "generated_at": "2026-10-25T12:00:00Z",
                "recorded_at": None, "observability": SUMMARY}
        page = render(data)
        for needle in ("Observability", "36 queries fetched", "78 served from cache", f"{A} x1", f"{B} x1", "retries: 3"):
            self.assertIn(needle, page)

    def test_without_a_summary_the_report_still_renders(self):
        data = {"question": "q", "answer": None, "claims": [], "rounds": [], "queries": [], "receipts": [], "tokens": {},
                "hints": [], "memory": None, "mode": "live", "timeline": [], "generated_at": "x", "recorded_at": None}
        self.assertIn("<html", render(data))


if __name__ == "__main__":
    unittest.main()
