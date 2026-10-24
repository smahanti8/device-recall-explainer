"""Tests for agent/llm.py (written before the implementation).

Contract under test:
    chat(messages, max_tokens=2000, *, send=None, sleep=time.sleep, attempts=4) -> Reply
    Reply has .text, .routed_model, .receipt_id (receipt_id may be None).
    `send(messages, max_tokens)` returns the raw response body (a dict) or raises
    TransientError (502, 429, timeout). Anything else raises LLMError.
    parse_receipt(body) -> {"routed_model", "provider_model", "receipt_id", "egress"}.
    At most 2 `send` calls run at once, across all threads.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import os
import threading
import time
import unittest

from dotenv import load_dotenv

from agent.llm import LLMError, TransientError, chat, parse_receipt

load_dotenv()

# Shape captured from a real Nightshift response (see docs/nemotron-evidence.md).
NIGHTSHIFT_BODY = {
    "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "choices": [{"message": {"role": "assistant", "content": "ready"}}],
    "meterless": {
        "provider_model": "meterless/auto",
        "routed_model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "egress": {"destination": "nvidia"},
        "receipt": {"receipt_id": "rcpt_req_tjyjvtk5", "usage": {"latency_ms": 1135}},
    },
}
DIRECT_BODY = {  # no Meterless block: NVIDIA called directly
    "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "choices": [{"message": {"role": "assistant", "content": "ready"}}],
}
MESSAGES = [{"role": "user", "content": "Reply with the single word: ready"}]


class ConcurrencyCap(unittest.TestCase):
    def test_never_more_than_two_calls_at_once(self):
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}

        def slow_send(messages, max_tokens):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.1)
            with lock:
                state["active"] -= 1
            return NIGHTSHIFT_BODY

        threads = [threading.Thread(target=chat, args=(MESSAGES,), kwargs={"send": slow_send})
                   for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(state["peak"], 2)  # capped at 2, and 2 really ran in parallel


class Retries(unittest.TestCase):
    def test_succeeds_after_two_502s_with_growing_backoff(self):
        calls, delays = [], []

        def flaky_send(messages, max_tokens):
            calls.append(1)
            if len(calls) <= 2:
                raise TransientError("502 Bad Gateway")
            return NIGHTSHIFT_BODY

        reply = chat(MESSAGES, send=flaky_send, sleep=delays.append)
        self.assertEqual(reply.text, "ready")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(delays), 2)
        self.assertLess(delays[0], delays[1])

    def test_gives_up_after_attempts_with_clear_error(self):
        calls = []

        def dead_send(messages, max_tokens):
            calls.append(1)
            raise TransientError("502 Bad Gateway")

        with self.assertRaises(LLMError) as ctx:
            chat(MESSAGES, send=dead_send, sleep=lambda s: None, attempts=3)
        self.assertEqual(len(calls), 3)
        self.assertIn("502", str(ctx.exception))

    def test_default_patience_covers_a_short_provider_outage(self):
        calls, delays = [], []

        def down(messages, max_tokens):
            calls.append(1)
            raise TransientError("502 provider_deadline")

        with self.assertRaises(LLMError):
            chat(MESSAGES, send=down, sleep=delays.append)
        self.assertEqual(len(calls), 5)
        self.assertGreaterEqual(sum(delays), 40)  # waits of 3, 6, 12, 24 seconds

    def test_non_transient_error_is_not_retried(self):
        calls = []

        def bad_key_send(messages, max_tokens):
            calls.append(1)
            raise LLMError("401 Unauthorized")

        with self.assertRaises(LLMError):
            chat(MESSAGES, send=bad_key_send, sleep=lambda s: None)
        self.assertEqual(len(calls), 1)


class Receipt(unittest.TestCase):
    def test_parse_nightshift_receipt(self):
        receipt = parse_receipt(NIGHTSHIFT_BODY)
        self.assertEqual(receipt["routed_model"], "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.assertEqual(receipt["provider_model"], "meterless/auto")
        self.assertEqual(receipt["receipt_id"], "rcpt_req_tjyjvtk5")
        self.assertEqual(receipt["egress"], "nvidia")

    def test_direct_route_falls_back_to_model_field(self):
        receipt = parse_receipt(DIRECT_BODY)
        self.assertEqual(receipt["routed_model"], "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.assertIsNone(receipt["receipt_id"])

    def test_every_call_logs_routed_model_and_receipt_id(self):
        with self.assertLogs("agent.llm", level="INFO") as logs:
            chat(MESSAGES, send=lambda m, t: NIGHTSHIFT_BODY)
        line = "\n".join(logs.output)
        self.assertIn("routed_model=nvidia/nemotron-3.5-lightning-30b-a3b", line)
        self.assertIn("receipt_id=rcpt_req_tjyjvtk5", line)


@unittest.skipUnless(all(os.environ.get(k) for k in ("MODEL_BASE_URL", "MODEL_API_KEY", "NEMOTRON_MODEL")),
                     "MODEL_* settings missing in .env")
class LiveCall(unittest.TestCase):
    def test_real_call_answers_and_names_a_nemotron_model(self):
        reply = chat(MESSAGES, max_tokens=2000)
        self.assertIn("ready", reply.text.lower())
        self.assertTrue(reply.routed_model.startswith("nvidia/"), reply.routed_model)
        self.assertNotEqual(reply.routed_model, "meterless/auto")


if __name__ == "__main__":
    unittest.main()
