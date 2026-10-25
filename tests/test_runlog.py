"""Tests for structured run logs (written before the implementation; replaces the plain-text draft).

The hackathon policy wants logs that reflect real execution: which agent acted, what it received and
produced, which model handled each request, whether calls succeeded, retries. One JSON object per line.

    agent/runlog.py
      RunLog(log_dir, now=None, env=None)   file log_dir/run-YYYYmmdd-HHMMSS.jsonl (dir created on first write)
        .path
        .emit(agent_name, action, *, input_summary="", output_summary="", target_agent=None, model=None,
              status="success", retry_count=0, **extra) -> the event dict
        Every event has: timestamp (UTC, "2026-10-25T12:00:00Z"), agent_name, action, input_summary,
        output_summary, target_agent, model, status, retry_count, plus extra fields.
        Strings over 500 characters are truncated; values equal to a secret in env (variables whose names
        contain KEY, TOKEN, SECRET or PASSWORD) are replaced by "[redacted]". A write failure never raises.
      start(log) / stop() / current(); module-level emit(...) writes to the current log, a no-op when none.

    Events written by the existing modules while a log is current:
      ModelClient   model_call   status success|retry|failed, model = routed model, receipt_id, retry_count
      ZetarisTool   run_sql      status success|cache_hit|error, input_summary = the SQL, output_summary "N row(s)"
      Memory        recall       output_summary "K hint(s)";   learn   output_summary "N new, R reinforced, C changed"
      Orchestrator  run_start (question, playbook, memory, sample_mode)  /  run_end (status verified|gave_up)
      Drafter       round_start
      Evaluator     evaluate_claims  status verified|rejected, reasons in output_summary, target_agent "Drafter" on reject
    ask(..., log_dir=None, now=None) creates the RunLog and makes it current for the run.
    parse_args: --log-dir (default "logs"), --no-log.  main passes log_dir=None with --no-log.
    .gitignore: logs/run-* ignored; logs/.gitkeep and logs/sample-*.jsonl are not.
    Shipped sample logs (logs/sample-*.jsonl) must be real recordings: schema-valid, a run_start with
    sample_mode false, and a model_call to an nvidia/ model.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import datetime
import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from agent import llm, runlog, tools_zetaris
from agent.evaluator import Citation, Claim, Rejection, Verdict
from agent.llm import TransientError, chat
from agent.loop import Draft, Result
from agent.main import ask, main, parse_args
from agent.memory import Memory
from agent.runlog import RunLog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHEN = lambda: datetime.datetime(2026, 10, 25, 12, 0, 0)
FIELDS = {"timestamp", "agent_name", "action", "input_summary", "output_summary", "target_agent",
          "model", "status", "retry_count"}
CLAIM = Claim("DSY recall records in 2024", 54, Citation("SELECT 1 AS n", "n"))
GOOD = Result("DSY had 54 recall records.", [], 1, [], claims=[CLAIM])
GAVE_UP = Result(None, [CLAIM], 3, [])
FULL = {name: "x" for name in ("ZETARIS_REST_URL", "ZETARIS_ORG_ID", "ZETARIS_API_KEY",
                               "MODEL_BASE_URL", "MODEL_API_KEY", "NEMOTRON_MODEL")}
BODY = {"model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "choices": [{"message": {"role": "assistant", "content": "ready"}}],
        "meterless": {"routed_model": "nvidia/nemotron-3.5-lightning-30b-a3b",
                      "provider_model": "meterless/auto", "egress": {"destination": "nvidia"},
                      "receipt": {"receipt_id": "rcpt_req_tjyjvtk5", "usage": {"latency_ms": 1135}}}}
MESSAGES = [{"role": "user", "content": "Reply with the single word: ready"}]


def read_events(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TempDir(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(runlog.stop)

    def started(self):
        log = RunLog(self.dir, now=WHEN)
        runlog.start(log)
        return log

    def events(self, log):
        return read_events(log.path)


class RunLogFile(TempDir):
    def test_the_file_is_named_from_the_clock_and_the_directory_is_created(self):
        nested = os.path.join(self.dir, "logs")
        log = RunLog(nested, now=WHEN)
        log.emit("Tester", "ping")
        self.assertEqual(os.listdir(nested), ["run-20261025-120000.jsonl"])

    def test_each_event_is_one_json_line_with_the_policy_fields(self):
        log = RunLog(self.dir, now=WHEN)
        event = log.emit("Evaluator", "evaluate_claims", input_summary="22 claims", output_summary="all verified",
                         target_agent="Drafter", model="nvidia/n", status="success", retry_count=2, round=1)
        log.emit("Tester", "second")
        events = read_events(log.path)
        self.assertEqual(len(events), 2)
        self.assertTrue(FIELDS <= set(events[0]))
        self.assertEqual(events[0], event)
        self.assertEqual(events[0]["timestamp"], "2026-10-25T12:00:00Z")
        self.assertEqual((events[0]["agent_name"], events[0]["retry_count"], events[0]["round"]), ("Evaluator", 2, 1))
        self.assertIsNone(events[1]["target_agent"])
        self.assertEqual((events[1]["status"], events[1]["retry_count"]), ("success", 0))

    def test_long_text_is_truncated(self):
        log = RunLog(self.dir, now=WHEN)
        log.emit("Tester", "long", input_summary="x" * 5000)
        self.assertLess(len(read_events(log.path)[0]["input_summary"]), 600)

    def test_secrets_are_redacted_and_secret_named_fields_are_dropped(self):
        env = {"MODEL_API_KEY": "sk-test-secret-123", "OTHER": "fine"}
        log = RunLog(self.dir, now=WHEN, env=env)
        log.emit("Tester", "leak", output_summary="the key is sk-test-secret-123 ok", api_key="sk-test-secret-123")
        with open(log.path) as handle:
            text = handle.read()
        self.assertNotIn("sk-test-secret-123", text)
        self.assertIn("[redacted]", text)
        self.assertNotIn("api_key", text)

    def test_a_write_failure_never_raises(self):
        blocker = os.path.join(self.dir, "afile")
        open(blocker, "w").close()
        log = RunLog(os.path.join(blocker, "logs"), now=WHEN)
        event = log.emit("Tester", "ping")
        self.assertEqual(event["action"], "ping")
        self.assertTrue(log.failed)

    def test_module_level_emit_is_a_no_op_without_a_current_log(self):
        runlog.stop()
        self.assertIsNone(runlog.emit("Tester", "ping"))
        log = self.started()
        runlog.emit("Tester", "ping")
        self.assertEqual(len(self.events(log)), 1)


class ModelClientEvents(TempDir):
    def test_a_successful_call_logs_the_routed_model_and_receipt(self):
        log = self.started()
        chat(MESSAGES, send=lambda m, t: BODY, sleep=lambda s: None)
        (event,) = self.events(log)
        self.assertEqual((event["agent_name"], event["action"], event["status"], event["retry_count"]),
                         ("ModelClient", "model_call", "success", 0))
        self.assertEqual(event["model"], "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.assertEqual(event["receipt_id"], "rcpt_req_tjyjvtk5")

    def test_retries_are_logged_with_a_growing_retry_count(self):
        log = self.started()
        calls = []

        def flaky(messages, max_tokens):
            calls.append(1)
            if len(calls) <= 2:
                raise TransientError("502 Bad Gateway")
            return BODY
        chat(MESSAGES, send=flaky, sleep=lambda s: None)
        events = self.events(log)
        self.assertEqual([e["status"] for e in events], ["retry", "retry", "success"])
        self.assertEqual([e["retry_count"] for e in events], [1, 2, 2])
        self.assertIn("502", events[0]["output_summary"])

    def test_giving_up_is_logged_as_failed(self):
        log = self.started()

        def dead(messages, max_tokens):
            raise TransientError("502 Bad Gateway")
        with self.assertRaises(llm.LLMError):
            chat(MESSAGES, send=dead, sleep=lambda s: None, attempts=2)
        self.assertEqual([e["status"] for e in self.events(log)], ["retry", "failed"])


class ZetarisEvents(TempDir):
    def setUp(self):
        super().setUp()
        tools_zetaris._CACHE.clear()
        self.addCleanup(tools_zetaris._CACHE.clear)
        env = mock.patch.dict(os.environ, {"ZETARIS_REST_URL": "http://z.test/api", "ZETARIS_ORG_ID": "1",
                                           "ZETARIS_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)
        post = mock.patch("agent.tools_zetaris.requests.post")
        self.post = post.start()
        self.addCleanup(post.stop)
        self.post.return_value = mock.Mock(ok=True, json=lambda: {"headers": ["n"], "data": [["54"]]})

    def test_a_query_then_a_cache_hit(self):
        log = self.started()
        tools_zetaris.run_sql("SELECT 54 AS n")
        tools_zetaris.run_sql("SELECT 54 AS n")
        first, second = self.events(log)
        self.assertEqual((first["agent_name"], first["action"], first["status"]), ("ZetarisTool", "run_sql", "success"))
        self.assertEqual((first["input_summary"], first["output_summary"]), ("SELECT 54 AS n", "1 row(s)"))
        self.assertEqual(second["status"], "cache_hit")

    def test_a_failed_query_is_logged_as_an_error(self):
        log = self.started()
        self.post.return_value = mock.Mock(ok=False, status_code=500, text="boom")
        with self.assertRaises(RuntimeError):
            tools_zetaris.run_sql("SELECT 1 AS n")
        self.assertEqual(self.events(log)[0]["status"], "error")


class MemoryEvents(TempDir):
    def test_recall_and_learn_are_logged(self):
        log = self.started()
        hit = {"id": "mem-1", "content": "DSY recall records in 2024: 54", "relevance": 0.8, "source": "SELECT 1"}
        replies = iter([{"memories": [hit, hit], "context": ""}, {"id": "mem-1", "status": "new"}])
        memory = Memory(self.dir, run=lambda request: next(replies))
        memory.recall("DSY 2024")
        memory.learn([CLAIM], [])
        recall, learn = self.events(log)
        self.assertEqual((recall["agent_name"], recall["action"], recall["output_summary"]), ("Memory", "recall", "2 hint(s)"))
        self.assertEqual((learn["action"], learn["output_summary"]), ("learn", "1 new, 0 reinforced, 0 changed"))


def run_logged(log_dir, result=GOOD, verdict=None, **kwargs):
    lines = []

    def make_draft(context_fn=None, **kw):
        return lambda question, feedback: Draft("a", [CLAIM])

    def loop_like(question, draft_fn, evaluate_fn):
        evaluate_fn(draft_fn(question, []).claims)
        return result
    outcome = ask("why?", None, make_draft=make_draft, run_investigation=loop_like,
                  evaluate_fn=lambda claims: verdict or Verdict(), out=lines.append,
                  progress=lambda message: None, log_dir=log_dir, now=WHEN, **kwargs)
    return outcome, lines


class OrchestratorEvents(TempDir):
    def logged(self, **kwargs):
        run_logged(self.dir, **kwargs)
        (path,) = glob.glob(os.path.join(self.dir, "run-*.jsonl"))
        return read_events(path)

    def test_a_verified_run_has_start_round_evaluation_and_end(self):
        events = self.logged()
        self.assertEqual([e["action"] for e in events], ["run_start", "round_start", "evaluate_claims", "run_end"])
        start, _, evaluation, end = events
        self.assertEqual((start["agent_name"], start["question"], start["sample_mode"]), ("Orchestrator", "why?", False))
        self.assertEqual((evaluation["agent_name"], evaluation["status"]), ("Evaluator", "verified"))
        self.assertEqual((end["agent_name"], end["status"]), ("Orchestrator", "verified"))

    def test_a_rejection_is_logged_with_its_reason_and_the_handoff_back(self):
        rejected = Verdict([Rejection(CLAIM, "no_citation", "claim has no SQL citation")])
        evaluation = self.logged(verdict=rejected)[2]
        self.assertEqual(evaluation["status"], "rejected")
        self.assertIn("no_citation", evaluation["output_summary"])
        self.assertEqual(evaluation["target_agent"], "Drafter")

    def test_a_run_that_gave_up_ends_as_gave_up(self):
        self.assertEqual(self.logged(result=GAVE_UP)[-1]["status"], "gave_up")

    def test_no_log_dir_means_no_log_and_the_run_is_unchanged(self):
        before = set(os.listdir(self.dir))
        outcome, _ = run_logged(None)
        self.assertIs(outcome, GOOD)
        self.assertEqual(set(os.listdir(self.dir)), before)

    def test_a_write_failure_is_reported_and_never_changes_the_result(self):
        blocker = os.path.join(self.dir, "afile")
        open(blocker, "w").close()
        outcome, lines = run_logged(os.path.join(blocker, "logs"))
        self.assertIs(outcome, GOOD)
        self.assertIn("note: could not write the run log", "\n".join(lines))

    def test_the_log_path_is_printed(self):
        _, lines = run_logged(self.dir)
        self.assertIn("run-20261025-120000.jsonl", "\n".join(lines))


class Cli(unittest.TestCase):
    def test_flags_and_defaults(self):
        args = parse_args(["why?"])
        self.assertEqual((args.log_dir, args.no_log), ("logs", False))
        self.assertTrue(parse_args(["why?", "--no-log"]).no_log)
        self.assertEqual(parse_args(["why?", "--log-dir", "/tmp/x"]).log_dir, "/tmp/x")

    def test_main_passes_the_log_directory_through(self):
        seen = []

        def fake_ask(question, memory, **kwargs):
            seen.append(kwargs["log_dir"])
            return GOOD
        main(["why?", "--no-memory"], env=FULL, ask_fn=fake_ask)
        main(["why?", "--no-memory", "--no-log"], env=FULL, ask_fn=fake_ask)
        main(["why?", "--no-memory", "--log-dir", "/tmp/x"], env=FULL, ask_fn=fake_ask)
        self.assertEqual(seen, ["logs", None, "/tmp/x"])


@unittest.skipUnless(shutil.which("git"), "git not available")
class GitIgnoreRules(unittest.TestCase):
    def ignored(self, path):
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0

    def test_run_logs_are_ignored_but_samples_and_the_keep_file_are_not(self):
        self.assertTrue(self.ignored("logs/run-20261025-120000.jsonl"))
        self.assertFalse(self.ignored("logs/sample-run-1.jsonl"))
        self.assertFalse(self.ignored("logs/.gitkeep"))


class ShippedSampleLogs(unittest.TestCase):
    """Sample logs must be real recordings of a live run, never replayed or written by hand."""

    def test_every_shipped_sample_log_is_a_schema_valid_real_run(self):
        paths = glob.glob(os.path.join(ROOT, "logs", "sample-*.jsonl"))
        if not paths:
            self.skipTest("no sample logs shipped yet")
        for path in paths:
            with self.subTest(path=os.path.basename(path)):
                events = read_events(path)
                for event in events:
                    self.assertTrue(FIELDS <= set(event), event)
                actions = [e["action"] for e in events]
                self.assertEqual((actions[0], actions[-1]), ("run_start", "run_end"))
                self.assertFalse(events[0]["sample_mode"])
                models = [e["model"] for e in events if e["action"] == "model_call" and e["status"] == "success"]
                self.assertTrue(models and all(m.startswith("nvidia/") for m in models), models)


if __name__ == "__main__":
    unittest.main()
