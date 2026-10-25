"""Tests for the "show your work" HTML report (written before the implementation).

One self-contained file you open in a browser, with no server and no new packages: the question, the answer with
numbered proofs, the rounds (rejection, research, redraft), every query, receipts, tokens and memory. It works for
live runs and for SAMPLE_MODE replays.

    agent/report.py
      render(data) -> str            a complete HTML page; everything dynamic is escaped; no scripts, no external URLs
      write(data, directory, now=None) -> path | None     report-YYYYmmdd-HHMMSS.html (directory created); None on OSError
      data keys: question, answer (None when unverified), claims [{n, text, value, sql, column, row, cell, status}],
        rounds [{round, claims, rejected, reasons, details}], queries [{sql, status, rows}], receipts
        [{routed_model, receipt_id}], tokens {calls, input, output, cost_usd}, hints [str], memory (str|None),
        mode "live"|"sample"|"record", timeline [str], generated_at, recorded_at
      Answer numbers that equal exactly one claim's value link to that claim ("#claim-N"); values shared by several
      claims are not linked. Each claim is an expandable <details id="claim-N"> with its SQL, column, row, the cell
      read back, and its status.
    tools_zetaris.query_log() -> [{sql, status, rows}] for this process; clear_query_log()
    sample_mode.recorded_at() -> the replayed recordings' timestamp (or None)
    main.ask(..., report_dir=None, run_sql=None, now=None): builds the data from the run (cells read back through
        run_sql, default the Zetaris tool), writes the report and prints "Report: <path>"; a write failure prints
        "note: could not write the report (...)" and never changes the result.
    parse_args: --report (flag) and --report-dir (default "reports"); main passes report_dir only with --report.
    .gitignore: reports/ files ignored except reports/.gitkeep and reports/sample-*.html.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import copy
import datetime
import glob
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from agent import report, sample_mode, tools_zetaris
from agent.evaluator import Citation, Claim, Verdict
from agent.loop import Result
from agent.main import ask, main, parse_args

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHEN = lambda: datetime.datetime(2026, 10, 25, 12, 0, 0)
FULL = {name: "x" for name in ("ZETARIS_REST_URL", "ZETARIS_ORG_ID", "ZETARIS_API_KEY",
                               "MODEL_BASE_URL", "MODEL_API_KEY", "NEMOTRON_MODEL")}


def claim(n, text, value, cell=None, status="verified"):
    return {"n": n, "text": text, "value": value, "sql": f"SELECT {value} AS v", "column": "v", "row": 0,
            "cell": str(value if cell is None else cell), "status": status}


DATA = {
    "question": "Why did cardiovascular device recall records rise between 2023 and 2024?",
    "answer": "Records rose from 281 in 2023 to 482 in 2024. DSY had 1 event.",
    "claims": [claim(1, "Cardiovascular recall records in 2023", 281), claim(2, "Cardiovascular recall records in 2024", 482),
               claim(3, "DSY recall events in 2024", 1), claim(4, "DPT recall events in 2024", 1)],
    "rounds": [{"round": 1, "claims": 4, "rejected": 1, "reasons": ["missing_evidence"],
                "details": ["the answer names DQK but cites no detail"]},
               {"round": 2, "claims": 4, "rejected": 0, "reasons": [], "details": []}],
    "queries": [{"sql": "SELECT q1", "status": "success", "rows": 1}, {"sql": "SELECT q2", "status": "cache_hit", "rows": 1},
                {"sql": "SELECT q3", "status": "replayed", "rows": 2}],
    "receipts": [{"routed_model": "nvidia/nemotron-3.5-lightning-30b-a3b", "receipt_id": "rcpt_req_abc"}],
    "tokens": {"calls": 2, "input": 1300, "output": 70, "cost_usd": 0.0},
    "hints": ["- DSY recall records in 2024: 54 (source: SELECT 1)"],
    "memory": "Memory: 0 new, 22 reinforced, 0 changed",
    "mode": "live",
    "timeline": ["round 1: drafting", "evidence requested: DQK -> 5 facts fetched", "round 2: drafting"],
    "generated_at": "2026-10-25T12:00:00Z",
    "recorded_at": None,
}


class Render(unittest.TestCase):
    def setUp(self):
        self.html = report.render(DATA)

    def test_a_complete_self_contained_page(self):
        self.assertTrue(self.html.lstrip().lower().startswith("<!doctype html>"))
        for needle in ('<html lang="en">', 'name="viewport"', "<title>", "Device Recall Explainer"):
            self.assertIn(needle, self.html)
        self.assertNotRegex(self.html, r'(?:src|href)="https?://')
        self.assertNotIn("<script", self.html.lower())

    def test_everything_dynamic_is_escaped(self):
        evil = "<script>alert(1)</script>"
        data = copy.deepcopy(DATA)
        data.update(question=evil, answer=evil, hints=[evil], timeline=[evil], memory=evil)
        data["claims"][0].update(text=evil, sql=evil)
        data["queries"][0]["sql"] = evil
        page = report.render(data)
        self.assertNotIn("<script", page.lower())
        self.assertIn("&lt;script&gt;", page)

    def test_answer_numbers_link_to_their_claims_unless_the_value_is_shared(self):
        self.assertIn('href="#claim-1"', self.html)
        self.assertIn('href="#claim-2"', self.html)
        self.assertNotIn('href="#claim-3"', self.html)   # the value 1 belongs to claims 3 and 4
        self.assertNotIn('href="#claim-4"', self.html)
        self.assertEqual(len(re.findall(r'href="#claim-', self.html)), 2)  # years are not claim values

    def test_each_claim_is_an_expandable_proof(self):
        for n, text in ((1, "Cardiovascular recall records in 2023"), (3, "DSY recall events in 2024")):
            block = re.search(rf'<details[^>]*id="claim-{n}".*?</details>', self.html, re.S)
            self.assertIsNotNone(block, n)
            for needle in (text, "SELECT", "verified"):
                self.assertIn(needle, block.group(0))
        self.assertEqual(len(re.findall(r'<details[^>]*id="claim-\d+"', self.html)), 4)

    def test_the_rounds_show_the_rejection_the_research_and_the_redraft(self):
        for needle in ("Round 1", "missing_evidence", "the answer names DQK but cites no detail", "Round 2",
                       "all verified", html.escape("evidence requested: DQK -> 5 facts fetched")):
            self.assertIn(needle, self.html)

    def test_every_query_is_listed_with_its_status_and_row_count(self):
        for needle in ("SELECT q1", "SELECT q2", "SELECT q3", "success", "cache_hit", "replayed"):
            self.assertIn(needle, self.html)

    def test_receipts_tokens_and_memory_are_shown(self):
        for needle in ("nvidia/nemotron-3.5-lightning-30b-a3b", "rcpt_req_abc", "1,300", "DSY recall records in 2024: 54",
                       "Memory: 0 new, 22 reinforced, 0 changed"):
            self.assertIn(needle, self.html)

    def test_a_live_report_has_no_sample_banner_and_a_replay_does(self):
        self.assertNotIn("SAMPLE MODE", self.html)
        data = dict(DATA, mode="sample", recorded_at="2026-10-24T22:00:00Z")
        banner = report.render(data)
        for needle in ("SAMPLE MODE", "replayed", "2026-10-24"):
            self.assertIn(needle, banner)

    def test_an_unverified_run_does_not_claim_success(self):
        data = dict(DATA, answer=None, claims=[claim(1, "Unsupported claim", 80, cell="54", status="unsupported")])
        page = report.render(data)
        self.assertIn("No verified answer", page)
        self.assertIn("unsupported", page)
        self.assertNotIn("Verified answer", page)


class Write(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_the_file_is_named_from_the_clock_and_the_directory_is_created(self):
        path = report.write(DATA, os.path.join(self.dir, "reports"), now=WHEN)
        self.assertTrue(path.endswith("report-20261025-120000.html"))
        with open(path) as handle:
            self.assertIn("Device Recall Explainer", handle.read())

    def test_a_failure_returns_none(self):
        blocker = os.path.join(self.dir, "afile")
        open(blocker, "w").close()
        self.assertIsNone(report.write(DATA, os.path.join(blocker, "reports"), now=WHEN))


class QueryLog(unittest.TestCase):
    def setUp(self):
        tools_zetaris._CACHE.clear()
        tools_zetaris.clear_query_log()
        self.addCleanup(tools_zetaris.clear_query_log)
        self.addCleanup(tools_zetaris._CACHE.clear)
        env = mock.patch.dict(os.environ, {"ZETARIS_REST_URL": "http://z.test/api", "ZETARIS_ORG_ID": "1", "ZETARIS_API_KEY": "k-test-1"})
        env.start()
        self.addCleanup(env.stop)
        post = mock.patch("agent.tools_zetaris.requests.post")
        post.start().return_value = mock.Mock(ok=True, json=lambda: {"headers": ["n"], "data": [["54"], ["55"]]})
        self.addCleanup(post.stop)

    def test_each_query_is_recorded_with_status_and_rows(self):
        tools_zetaris.run_sql("SELECT 54 AS n")
        tools_zetaris.run_sql("SELECT 54 AS n")
        self.assertEqual(tools_zetaris.query_log(), [
            {"sql": "SELECT 54 AS n", "status": "success", "rows": 2},
            {"sql": "SELECT 54 AS n", "status": "cache_hit", "rows": 2}])
        tools_zetaris.clear_query_log()
        self.assertEqual(tools_zetaris.query_log(), [])


class InAsk(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(sample_mode.stop)
        self.result = Result("Records rose from 281 to 482.", [], 2,
                             [{"round": 1, "claims": 1, "rejected": 1, "reasons": ["missing_evidence"], "details": ["x"]},
                              {"round": 2, "claims": 1, "rejected": 0, "reasons": [], "details": []}],
                             claims=[Claim("Cardiovascular recall records in 2023", 281, Citation("SELECT 281 AS total", "total"))])

    def run_ask(self, report_dir, result=None, **kwargs):
        lines = []

        def make_draft(context_fn=None, **kw):
            return lambda question, feedback: None
        outcome = ask("Why did recalls rise?", None, make_draft=make_draft,
                      run_investigation=lambda q, d, e: result or self.result, evaluate_fn=lambda claims: Verdict(),
                      out=lines.append, progress=lambda message: None, report_dir=report_dir, now=WHEN,
                      run_sql=lambda sql: [{"total": "281"}], **kwargs)
        return outcome, lines

    def test_a_run_writes_one_report_with_the_question_answer_and_the_cells_read_back(self):
        _, lines = self.run_ask(self.dir)
        (path,) = glob.glob(os.path.join(self.dir, "report-*.html"))
        with open(path) as handle:
            page = handle.read()
        for needle in ("Why did recalls rise?", "Records rose from", "SELECT 281 AS total", "281", "Round 1", "missing_evidence"):
            self.assertIn(needle, page)
        self.assertIn("Report: ", "\n".join(lines))

    def test_without_a_report_dir_nothing_is_written(self):
        before = set(os.listdir(self.dir))
        outcome, lines = self.run_ask(None)
        self.assertIs(outcome, self.result)
        self.assertEqual(set(os.listdir(self.dir)), before)
        self.assertNotIn("Report: ", "\n".join(lines))

    def test_a_write_failure_is_reported_and_never_changes_the_result(self):
        blocker = os.path.join(self.dir, "afile")
        open(blocker, "w").close()
        outcome, lines = self.run_ask(os.path.join(blocker, "reports"))
        self.assertIs(outcome, self.result)
        self.assertIn("note: could not write the report", "\n".join(lines))

    def test_a_replay_report_carries_the_recording_date_and_the_banner(self):
        for name, payload in (("zetaris_results.json", {"queries": {}}), ("model_replies.json", {"replies": {}})):
            with open(os.path.join(self.dir, name), "w") as handle:
                json.dump(dict(payload, source="real run", recorded_at="2026-10-24T22:00:00Z"), handle)
        sample_mode.start_replay(self.dir)
        self.assertEqual(sample_mode.recorded_at(), "2026-10-24T22:00:00Z")
        reports = os.path.join(self.dir, "reports")
        self.run_ask(reports)
        (path,) = glob.glob(os.path.join(reports, "report-*.html"))
        with open(path) as handle:
            page = handle.read()
        for needle in ("SAMPLE MODE", "2026-10-24"):
            self.assertIn(needle, page)

    def test_an_unverified_run_is_reported_honestly(self):
        gave_up = Result(None, [Claim("DQK detail", None)], 3, [{"round": 1, "claims": 0, "rejected": 1,
                                                                 "reasons": ["model_unavailable"], "details": ["502"]}])
        self.run_ask(self.dir, result=gave_up)
        (path,) = glob.glob(os.path.join(self.dir, "report-*.html"))
        with open(path) as handle:
            self.assertIn("No verified answer", handle.read())


class Cli(unittest.TestCase):
    def test_flags_and_defaults(self):
        args = parse_args(["why?"])
        self.assertEqual((args.report, args.report_dir), (False, "reports"))
        self.assertTrue(parse_args(["why?", "--report"]).report)
        self.assertEqual(parse_args(["why?", "--report-dir", "/tmp/r"]).report_dir, "/tmp/r")

    def test_main_passes_report_dir_only_with_the_flag(self):
        seen = []

        def fake_ask(question, memory, **kwargs):
            seen.append(kwargs.get("report_dir"))
            return Result("ok", [], 1, [], claims=[])
        main(["why did recalls rise?", "--no-memory"], env=FULL, ask_fn=fake_ask)
        main(["why did recalls rise?", "--no-memory", "--report"], env=FULL, ask_fn=fake_ask)
        main(["why did recalls rise?", "--no-memory", "--report", "--report-dir", "/tmp/r"], env=FULL, ask_fn=fake_ask)
        self.assertEqual(seen, [None, "reports", "/tmp/r"])


@unittest.skipUnless(shutil.which("git"), "git not available")
class GitIgnoreRules(unittest.TestCase):
    def ignored(self, path):
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0

    def test_reports_are_ignored_but_samples_and_the_keep_file_are_not(self):
        self.assertTrue(self.ignored("reports/report-20261025-120000.html"))
        self.assertFalse(self.ignored("reports/sample-run-1.html"))
        self.assertFalse(self.ignored("reports/.gitkeep"))


class ShippedSampleReports(unittest.TestCase):
    def test_every_shipped_sample_report_is_self_contained_and_verified(self):
        paths = glob.glob(os.path.join(ROOT, "reports", "sample-*.html"))
        if not paths:
            self.skipTest("no sample report shipped yet")
        for path in paths:
            with self.subTest(path=os.path.basename(path)), open(path) as handle:
                page = handle.read()
            self.assertNotRegex(page, r'(?:src|href)="https?://')
            self.assertNotIn("<script", page.lower())
            self.assertIn("verified", page)


if __name__ == "__main__":
    unittest.main()
