"""Tests for escalating a redraft to a larger Nemotron model (written before the implementation).

Nightshift 1.2.2 honours an explicit model ID (the probe routed a request for Ultra to Ultra), so after the Evaluator
rejects a draft, the redraft can go to a stronger model. Optional: with no escalation model configured nothing changes.

    llm.chat(..., model=None): model None -> send(messages, max_tokens) exactly as before; a model ->
        send(messages, max_tokens, model=model)
    llm._send(messages, max_tokens, model=None): the request names `model`, else NEMOTRON_MODEL from the environment
    llm.escalation_model(env=None) -> the NEMOTRON_ESCALATION_MODEL value (stripped) or None
    playbook.make_playbook_draft_fn(..., escalation_model=None): the model call that starts a redraft (feedback not empty)
        uses the escalation model (parameter, else the environment); the first draft and in-draft corrections never do.
        It emits a Drafter "escalate_model" log event (model set) and on_event("escalating to <model> for the redraft").
        If the escalated call fails with LLMError the draft retries once on the default model and logs an
        "escalation_fallback" event, so a slow or unavailable larger model can never sink a run.
        With nothing configured, chat is called without a model argument at all.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import datetime
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from agent import llm, runlog
from agent.evaluator import Claim, Rejection
from agent.llm import LLMError, Reply, chat
from agent.playbook import collect_facts, make_playbook_draft_fn

ESC = "nvidia/nemotron-3-ultra-550b-a55b"
MESSAGES = [{"role": "user", "content": "hi"}]
BODY = {"model": "nvidia/n", "choices": [{"message": {"content": "ready"}}],
        "meterless": {"routed_model": "nvidia/n", "receipt": {"receipt_id": "rcpt_1"}}}
ROW = {"total": "10", "diff": "5", "change": "2", "top3": "9", "pct": "60.0", "n": "3", "top": "Equipment maintenance"}
REJECTED = [Rejection(Claim("x", None), "no_citation", "claim has no SQL citation")]


def fake_run(sql):
    if "ORDER BY change DESC LIMIT 5" in sql:
        return [{"code": c, "device": c} for c in ("DSY", "DQO", "OEZ", "DQK", "DPT")]
    return [dict(ROW)]


WHY_ID = next(f.id for f in collect_facts(run_sql=fake_run) if "recall events in 2024" in f.label)


def good(text="one"):
    """A reply that already cites an events fact, so the playbook's own 'why' correction does not call the model again."""
    return json.dumps({"text": text, "facts": ["f1", WHY_ID]})


def recording_chat(*replies, fail_on_model=False):
    calls, queue = [], iter(replies)

    def fake(messages, max_tokens=2000, **kwargs):
        calls.append(dict(kwargs))
        if fail_on_model and kwargs.get("model"):
            raise LLMError("502 provider_deadline")
        return Reply(next(queue), "nvidia/test", None)
    fake.calls = calls
    return fake


class ChatModelArgument(unittest.TestCase):
    def test_without_a_model_send_is_called_exactly_as_before(self):
        seen = []

        def send(messages, max_tokens, **kwargs):
            seen.append(kwargs)
            return BODY
        chat(MESSAGES, send=send, sleep=lambda s: None)
        self.assertEqual(seen, [{}])

    def test_a_model_is_passed_through_to_send(self):
        seen = []

        def send(messages, max_tokens, **kwargs):
            seen.append(kwargs)
            return BODY
        chat(MESSAGES, send=send, sleep=lambda s: None, model=ESC)
        self.assertEqual(seen, [{"model": ESC}])

    def test_the_request_names_the_override_else_the_configured_model(self):
        client = mock.Mock()
        client.chat.completions.create.return_value.model_dump.return_value = BODY
        env = {"MODEL_BASE_URL": "http://m.test/v1", "MODEL_API_KEY": "k-test-1234", "NEMOTRON_MODEL": "meterless/auto"}
        with mock.patch.dict(os.environ, env), mock.patch("agent.llm.openai.OpenAI", return_value=client):
            llm._send(MESSAGES, 100, model=ESC)
            llm._send(MESSAGES, 100)
        named = [call.kwargs["model"] for call in client.chat.completions.create.call_args_list]
        self.assertEqual(named, [ESC, "meterless/auto"])


class Setting(unittest.TestCase):
    def test_unset_or_blank_means_no_escalation(self):
        for env in ({}, {"NEMOTRON_ESCALATION_MODEL": ""}, {"NEMOTRON_ESCALATION_MODEL": "   "}):
            with self.subTest(env=env):
                self.assertIsNone(llm.escalation_model(env))

    def test_a_value_is_returned_trimmed(self):
        self.assertEqual(llm.escalation_model({"NEMOTRON_ESCALATION_MODEL": f" {ESC} "}), ESC)


class PlaybookEscalation(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(runlog.stop)

    def events(self, log):
        with open(log.path) as handle:
            return [json.loads(line) for line in handle]

    def test_the_first_draft_uses_the_default_model_and_the_redraft_the_escalation_model(self):
        chat_fn = recording_chat(good(), good("two"))
        draft_fn = make_playbook_draft_fn(chat=chat_fn, run_sql=fake_run, escalation_model=ESC)
        draft_fn("why?", [])
        draft_fn("why?", REJECTED)
        self.assertEqual(chat_fn.calls, [{}, {"model": ESC}])

    def test_nothing_configured_means_chat_never_gets_a_model_argument(self):
        chat_fn = recording_chat(good(), good("two"))
        with mock.patch.dict(os.environ, {}, clear=True):
            draft_fn = make_playbook_draft_fn(chat=chat_fn, run_sql=fake_run)
            draft_fn("why?", [])
            draft_fn("why?", REJECTED)
        self.assertEqual(chat_fn.calls, [{}, {}])

    def test_the_environment_setting_is_used_when_no_parameter_is_given(self):
        chat_fn = recording_chat(good(), good("two"))
        with mock.patch.dict(os.environ, {"NEMOTRON_ESCALATION_MODEL": ESC}):
            draft_fn = make_playbook_draft_fn(chat=chat_fn, run_sql=fake_run)
            draft_fn("why?", [])
            draft_fn("why?", REJECTED)
        self.assertEqual(chat_fn.calls, [{}, {"model": ESC}])

    def test_an_in_draft_correction_stays_on_the_default_model(self):
        chat_fn = recording_chat("not json at all", good("fixed"))
        draft_fn = make_playbook_draft_fn(chat=chat_fn, run_sql=fake_run, escalation_model=ESC)
        draft_fn("why?", REJECTED)  # redraft: its first call escalates, the JSON correction does not
        self.assertEqual(chat_fn.calls, [{"model": ESC}, {}])

    def test_escalation_is_logged_and_announced(self):
        log = runlog.RunLog(self.dir, now=lambda: datetime.datetime(2026, 10, 25, 12, 0, 0))
        runlog.start(log)
        messages = []
        draft_fn = make_playbook_draft_fn(chat=recording_chat(good()), run_sql=fake_run, escalation_model=ESC,
                                          on_event=messages.append)
        draft_fn("why?", REJECTED)
        (event,) = [e for e in self.events(log) if e["action"] == "escalate_model"]
        self.assertEqual((event["agent_name"], event["model"]), ("Drafter", ESC))
        self.assertIn(f"escalating to {ESC} for the redraft", messages)

    def test_a_failed_escalated_call_falls_back_to_the_default_model(self):
        log = runlog.RunLog(self.dir, now=lambda: datetime.datetime(2026, 10, 25, 12, 0, 0))
        runlog.start(log)
        chat_fn = recording_chat(good("fallback"), fail_on_model=True)
        draft_fn = make_playbook_draft_fn(chat=chat_fn, run_sql=fake_run, escalation_model=ESC)
        draft = draft_fn("why?", REJECTED)
        self.assertEqual(chat_fn.calls, [{"model": ESC}, {}])
        self.assertIn("fallback", draft.text)
        self.assertIn("escalation_fallback", [e["action"] for e in self.events(log)])


if __name__ == "__main__":
    unittest.main()
