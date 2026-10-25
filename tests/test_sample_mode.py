"""Tests for SAMPLE_MODE (written before the implementation).

Judges may not have Zetaris, Nightshift or keys, so a documented offline mode replays RECORDINGS of a real run.
The default mode is always real. Nothing is faked: the recordings are captured from live calls, and the evaluator,
retry loop and memory run for real on top of them.

    agent/sample_mode.py
      enabled(env=None) -> bool            SAMPLE_MODE in {"1", "true", "yes", "on"} (any case)
      start_recording(dir, now=None) / start_replay(dir) / stop() / mode() -> None | "record" | "replay"
      SampleModeError(RuntimeError)        a message that says what to do ("record ... with --record")
      Files in the directory (written by stop() when recording):
        zetaris_results.json   {"recorded_at", "source": "real run", "queries": {sql: rows}}
        model_replies.json     {"recorded_at", "source": "real run", "model", "replies": {sha256(messages): body}}
    tools_zetaris.run_sql: record = fetch for real and remember; replay = no network, no settings needed,
        SQL matched after the usual trimming; unknown SQL raises SampleModeError.
    llm.chat's default sender: record = call the model for real and remember the full response body;
        replay = no client is built; messages matched by sha256 of their canonical JSON; unknown raises SampleModeError.
    Run-log events in replay carry status "replayed" (ZetarisTool run_sql, ModelClient model_call).
    main: --sample (or SAMPLE_MODE in env), --sample-dir (default "sample_data"), --record DIR.
        --sample needs none of the six settings. A banner on stderr says SAMPLE MODE, that the Zetaris results and the
        model reply are replayed from recordings, and that the evaluator, retry loop and memory run for real.
        --sample with --record is refused (exit 2). Missing sample data or an unrecorded query exits 2 with a clear message.
        The mode is always reset when main returns. The run_start log event has sample_mode true.
    Shipped fixtures: sample_data/ is recorded from a real run; a sample run completes offline with verified claims.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import contextlib
import datetime
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from agent import llm, runlog, sample_mode, tools_zetaris
from agent.evaluator import Citation, Claim
from agent.loop import Result
from agent.main import ask, main
from agent.sample_mode import SampleModeError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "sample_data")
FULL = {name: "x" for name in ("ZETARIS_REST_URL", "ZETARIS_ORG_ID", "ZETARIS_API_KEY",
                               "MODEL_BASE_URL", "MODEL_API_KEY", "NEMOTRON_MODEL")}
CLAIM = Claim("DSY recall records in 2024", 54, Citation("SELECT 1 AS n", "n"))
GOOD = Result("ok", [], 1, [], claims=[CLAIM])
MESSAGES = [{"role": "user", "content": "Reply with the single word: ready"}]
BODY = {"model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "choices": [{"message": {"role": "assistant", "content": "ready"}}],
        "meterless": {"routed_model": "nvidia/nemotron-3.5-lightning-30b-a3b", "provider_model": "meterless/auto",
                      "receipt": {"receipt_id": "rcpt_req_tjyjvtk5"}}}
NO_NETWORK = AssertionError("replay must not touch the network")


class SampleCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(sample_mode.stop)
        self.addCleanup(runlog.stop)
        tools_zetaris._CACHE.clear()
        self.addCleanup(tools_zetaris._CACHE.clear)


class Enabled(unittest.TestCase):
    def test_truthy_values_in_any_case(self):
        for value in ("true", "TRUE", "True", "1", "yes", "on"):
            with self.subTest(value=value):
                self.assertTrue(sample_mode.enabled({"SAMPLE_MODE": value}))

    def test_everything_else_is_off_and_the_default_is_off(self):
        for env in ({}, {"SAMPLE_MODE": ""}, {"SAMPLE_MODE": "false"}, {"SAMPLE_MODE": "0"}, {"SAMPLE_MODE": "no"}):
            with self.subTest(env=env):
                self.assertFalse(sample_mode.enabled(env))


class ZetarisRecordAndReplay(SampleCase):
    def setUp(self):
        super().setUp()
        env = mock.patch.dict(os.environ, {"ZETARIS_REST_URL": "http://z.test/api", "ZETARIS_ORG_ID": "1",
                                           "ZETARIS_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)

    def record_one_query(self):
        with mock.patch("agent.tools_zetaris.requests.post") as post:
            post.return_value = mock.Mock(ok=True, json=lambda: {"headers": ["n"], "data": [["54"]]})
            sample_mode.start_recording(self.dir, now=lambda: datetime.datetime(2026, 10, 25, 12, 0, 0))
            tools_zetaris.run_sql("SELECT 54 AS n")
            sample_mode.stop()
        tools_zetaris._CACHE.clear()

    def test_recording_saves_the_real_result_with_its_provenance(self):
        self.record_one_query()
        with open(os.path.join(self.dir, "zetaris_results.json")) as handle:
            saved = json.load(handle)
        self.assertEqual(saved["queries"], {"SELECT 54 AS n": [{"n": "54"}]})
        self.assertEqual(saved["source"], "real run")
        self.assertTrue(saved["recorded_at"].startswith("2026-10-25"))

    def test_replay_returns_the_recording_without_the_network_or_settings(self):
        self.record_one_query()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("agent.tools_zetaris.requests.post", side_effect=NO_NETWORK):
            sample_mode.start_replay(self.dir)
            self.assertEqual(tools_zetaris.run_sql("  SELECT 54 AS n ;"), [{"n": "54"}])

    def test_an_unrecorded_query_says_how_to_record_it(self):
        self.record_one_query()
        sample_mode.start_replay(self.dir)
        with mock.patch("agent.tools_zetaris.requests.post", side_effect=NO_NETWORK), \
                self.assertRaises(SampleModeError) as raised:
            tools_zetaris.run_sql("SELECT 99 AS n")
        self.assertIn("--record", str(raised.exception))

    def test_replayed_queries_are_logged_as_replayed(self):
        self.record_one_query()
        sample_mode.start_replay(self.dir)
        log = runlog.RunLog(self.dir, now=lambda: datetime.datetime(2026, 10, 25, 12, 0, 0))
        runlog.start(log)
        tools_zetaris.run_sql("SELECT 54 AS n")
        with open(log.path) as handle:
            (event,) = [json.loads(line) for line in handle]
        self.assertEqual((event["agent_name"], event["status"]), ("ZetarisTool", "replayed"))


class ModelRecordAndReplay(SampleCase):
    def record_one_reply(self):
        client = mock.Mock()
        client.chat.completions.create.return_value.model_dump.return_value = BODY
        env = mock.patch.dict(os.environ, {"MODEL_BASE_URL": "http://m.test/v1", "MODEL_API_KEY": "k-test-1234",
                                           "NEMOTRON_MODEL": "meterless/auto"})
        with env, mock.patch("agent.llm.openai.OpenAI", return_value=client):
            sample_mode.start_recording(self.dir)
            llm.chat(MESSAGES)
            sample_mode.stop()

    def test_recording_saves_the_full_response_body_keyed_by_the_messages(self):
        self.record_one_reply()
        key = hashlib.sha256(json.dumps(MESSAGES, sort_keys=True).encode()).hexdigest()
        with open(os.path.join(self.dir, "model_replies.json")) as handle:
            saved = json.load(handle)
        self.assertEqual(saved["replies"][key], BODY)
        self.assertEqual(saved["source"], "real run")

    def test_replay_answers_from_the_recording_without_building_a_client(self):
        self.record_one_reply()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("agent.llm.openai.OpenAI", side_effect=NO_NETWORK):
            sample_mode.start_replay(self.dir)
            reply = llm.chat(MESSAGES)
        self.assertEqual(reply.text, "ready")
        self.assertEqual((reply.routed_model, reply.receipt_id),
                         ("nvidia/nemotron-3.5-lightning-30b-a3b", "rcpt_req_tjyjvtk5"))

    def test_unrecorded_messages_say_how_to_record_them(self):
        self.record_one_reply()
        sample_mode.start_replay(self.dir)
        with mock.patch("agent.llm.openai.OpenAI", side_effect=NO_NETWORK), self.assertRaises(SampleModeError) as raised:
            llm.chat([{"role": "user", "content": "something else"}])
        self.assertIn("--record", str(raised.exception))

    def test_replayed_model_calls_are_logged_as_replayed_with_the_recorded_model(self):
        self.record_one_reply()
        sample_mode.start_replay(self.dir)
        log = runlog.RunLog(self.dir, now=lambda: datetime.datetime(2026, 10, 25, 12, 0, 0))
        runlog.start(log)
        llm.chat(MESSAGES)
        with open(log.path) as handle:
            (event,) = [json.loads(line) for line in handle]
        self.assertEqual((event["status"], event["model"]), ("replayed", "nvidia/nemotron-3.5-lightning-30b-a3b"))


class MainFlags(SampleCase):
    def run_main(self, argv, env, seen=None):
        errors = io.StringIO()

        def fake_ask(question, memory, **kwargs):
            if seen is not None:
                seen.append(sample_mode.mode())
            return GOOD
        with contextlib.redirect_stderr(errors):
            code = main(argv, env=env, ask_fn=fake_ask)
        return code, errors.getvalue()

    def make_sample_dir(self):
        for name, payload in (("zetaris_results.json", {"queries": {}}), ("model_replies.json", {"replies": {}})):
            with open(os.path.join(self.dir, name), "w") as handle:
                json.dump(dict(payload, source="real run", recorded_at="2026-10-25T12:00:00Z"), handle)

    def test_the_sample_flag_needs_no_settings_and_prints_the_banner(self):
        self.make_sample_dir()
        seen = []
        code, shown = self.run_main(["why?", "--sample", "--sample-dir", self.dir, "--no-memory"], env={}, seen=seen)
        self.assertEqual((code, seen, sample_mode.mode()), (0, ["replay"], None))
        for needle in ("SAMPLE MODE", "replayed", "recordings", "evaluator"):
            self.assertIn(needle, shown)

    def test_the_environment_variable_works_like_the_flag(self):
        self.make_sample_dir()
        seen = []
        code, _ = self.run_main(["why?", "--sample-dir", self.dir, "--no-memory"], env={"SAMPLE_MODE": "true"}, seen=seen)
        self.assertEqual((code, seen), (0, ["replay"]))

    def test_the_default_run_is_real(self):
        seen = []
        code, shown = self.run_main(["why?", "--no-memory"], env=FULL, seen=seen)
        self.assertEqual((code, seen), (0, [None]))
        self.assertNotIn("SAMPLE MODE", shown)

    def test_record_mode_is_on_during_the_run_and_files_are_written_at_the_end(self):
        seen = []
        code, _ = self.run_main(["why?", "--no-memory", "--record", self.dir], env=FULL, seen=seen)
        self.assertEqual((code, seen, sample_mode.mode()), (0, ["record"], None))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "zetaris_results.json")))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "model_replies.json")))

    def test_sample_and_record_together_are_refused(self):
        code, shown = self.run_main(["why?", "--sample", "--record", self.dir], env=FULL)
        self.assertEqual(code, 2)
        self.assertIn("--record", shown)

    def test_missing_sample_data_exits_2_with_a_clear_message(self):
        code, shown = self.run_main(["why?", "--sample", "--sample-dir", os.path.join(self.dir, "nope")], env={})
        self.assertEqual((code, sample_mode.mode()), (2, None))
        self.assertIn("sample data", shown)

    def test_an_unrecorded_query_during_the_run_exits_2_with_the_message(self):
        self.make_sample_dir()

        def unrecorded(question, memory, **kwargs):
            raise SampleModeError("no recording for this SQL; record it with --record")
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            code = main(["why?", "--sample", "--sample-dir", self.dir, "--no-memory"], env={}, ask_fn=unrecorded)
        self.assertEqual((code, sample_mode.mode()), (2, None))
        self.assertIn("no recording", errors.getvalue())


@unittest.skipUnless(os.path.exists(os.path.join(SAMPLE_DIR, "zetaris_results.json"))
                     and os.path.exists(os.path.join(SAMPLE_DIR, "model_replies.json")),
                     "record the sample data first: python -m agent.main --playbook --record sample_data ...")
class ShippedSampleData(unittest.TestCase):
    def tearDown(self):
        sample_mode.stop()
        tools_zetaris._CACHE.clear()

    def test_the_recordings_say_where_they_came_from(self):
        for name in ("zetaris_results.json", "model_replies.json"):
            with self.subTest(name=name), open(os.path.join(SAMPLE_DIR, name)) as handle:
                saved = json.load(handle)
            self.assertEqual(saved["source"], "real run")
            self.assertTrue(saved["recorded_at"])

    def test_a_sample_run_completes_offline_with_verified_claims(self):
        question = ("Why did cardiovascular device recall records rise between 2023 and 2024, "
                    "and which product codes drove it?")
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("agent.tools_zetaris.requests.post", side_effect=NO_NETWORK), \
                mock.patch("agent.llm.openai.OpenAI", side_effect=NO_NETWORK):
            sample_mode.start_replay(SAMPLE_DIR)
            result = ask(question, None, playbook=True, out=lambda *_: None, progress=lambda message: None)
        self.assertIsNotNone(result.text)
        self.assertTrue(result.claims)


if __name__ == "__main__":
    unittest.main()
