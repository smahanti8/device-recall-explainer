"""Model access for the agent: one `chat` call with a shared concurrency cap,
retry with backoff, and a logged routing receipt.

Concurrency is capped at 2 across all threads (4 parallel calls caused 502s on
Nightshift). The slot is held only while a request is in flight, never while
sleeping between retries.
"""
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import openai
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("agent.llm")
_SLOTS = threading.BoundedSemaphore(2)
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_BASE_DELAY = 1.0


class LLMError(Exception):
    pass


class TransientError(LLMError):
    """502, 429, timeout: worth retrying."""


@dataclass
class Reply:
    text: str
    routed_model: str
    receipt_id: Optional[str]


def parse_receipt(body):
    """Pull routing evidence from a response body; fall back to the `model` field."""
    meterless = body.get("meterless") or {}
    receipt = meterless.get("receipt") or {}
    return {
        "routed_model": meterless.get("routed_model") or body.get("model"),
        "provider_model": meterless.get("provider_model"),
        "receipt_id": receipt.get("receipt_id"),
        "egress": (meterless.get("egress") or {}).get("destination"),
        "latency_ms": (receipt.get("usage") or {}).get("latency_ms"),
    }


def _send(messages, max_tokens):
    client = openai.OpenAI(
        base_url=os.environ["MODEL_BASE_URL"],
        api_key=os.environ["MODEL_API_KEY"],
        default_headers={"meterless-app": "hackathon-agent", "meterless-scope": "user"},
        max_retries=0,  # retries are ours, so the cap and backoff stay in one place
        timeout=90,
    )
    try:
        response = client.chat.completions.create(
            model=os.environ["NEMOTRON_MODEL"], messages=messages, max_tokens=max_tokens)
    except openai.APIStatusError as error:
        text = f"{error.status_code} {str(error.message)[:200]}"
        raise (TransientError if error.status_code in _TRANSIENT_STATUS else LLMError)(text)
    except (openai.APITimeoutError, openai.APIConnectionError) as error:
        raise TransientError(f"{type(error).__name__}: {str(error)[:200]}")
    return response.model_dump()


def chat(messages, max_tokens=2000, *, send=None, sleep=time.sleep, attempts=4):
    send = send or _send
    last = None
    for attempt in range(attempts):
        try:
            with _SLOTS:
                body = send(messages, max_tokens)
            break
        except TransientError as error:
            last = error
            if attempt < attempts - 1:
                sleep(_BASE_DELAY * 2 ** attempt)
    else:
        raise LLMError(f"gave up after {attempts} attempts: {last}")
    receipt = parse_receipt(body)
    log.info("routed_model=%s receipt_id=%s provider_model=%s egress=%s latency_ms=%s",
             receipt["routed_model"], receipt["receipt_id"], receipt["provider_model"],
             receipt["egress"], receipt["latency_ms"])
    text = body["choices"][0]["message"].get("content") or ""
    if not text:
        raise LLMError("empty reply (reasoning model ran out of tokens?): raise max_tokens")
    return Reply(text=text, routed_model=receipt["routed_model"], receipt_id=receipt["receipt_id"])
