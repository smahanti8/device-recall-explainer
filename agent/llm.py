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

from agent import runlog, sample_mode

load_dotenv()

log = logging.getLogger("agent.llm")
_SLOTS = threading.BoundedSemaphore(2)
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_BASE_DELAY = 3.0
_CALL_LOG = []


class LLMError(Exception):
    pass


class TransientError(LLMError):
    """502, 429, timeout: worth retrying."""


@dataclass
class Reply:
    text: str
    routed_model: str
    receipt_id: Optional[str]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


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


def parse_usage(body):
    """Token counts and cost as the response reports them (zeros when absent)."""
    usage = body.get("usage") or {}
    receipt = (body.get("meterless") or {}).get("receipt") or {}
    from_receipt = receipt.get("usage") or {}
    return {"input_tokens": int(usage.get("prompt_tokens") or from_receipt.get("input_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or from_receipt.get("output_tokens") or 0),
            "cost_usd": float((receipt.get("cost") or {}).get("actual_usd") or 0.0)}


def escalation_model(env=None):
    """The optional larger model a redraft is escalated to (NEMOTRON_ESCALATION_MODEL), or None."""
    value = (os.environ if env is None else env).get("NEMOTRON_ESCALATION_MODEL")
    return str(value).strip() or None if value else None


def call_log():
    return [dict(entry) for entry in _CALL_LOG]


def clear_call_log():
    _CALL_LOG.clear()


def _send(messages, max_tokens, model=None):
    if sample_mode.mode() == "replay":
        return sample_mode.replay_reply(messages)
    client = openai.OpenAI(
        base_url=os.environ["MODEL_BASE_URL"],
        api_key=os.environ["MODEL_API_KEY"],
        default_headers={"meterless-app": "hackathon-agent", "meterless-scope": "user"},
        max_retries=0,  # retries are ours, so the cap and backoff stay in one place
        timeout=90,
    )
    try:
        response = client.chat.completions.create(
            model=model or os.environ["NEMOTRON_MODEL"], messages=messages, max_tokens=max_tokens)
    except openai.APIStatusError as error:
        text = f"{error.status_code} {str(error.message)[:200]}"
        raise (TransientError if error.status_code in _TRANSIENT_STATUS else LLMError)(text)
    except (openai.APITimeoutError, openai.APIConnectionError) as error:
        raise TransientError(f"{type(error).__name__}: {str(error)[:200]}")
    body = response.model_dump()
    sample_mode.record_reply(messages, body)
    return body


def chat(messages, max_tokens=2000, *, send=None, sleep=time.sleep, attempts=5, model=None):
    send = send or _send
    last = None
    last_prompt = messages[-1]["content"][:200] if messages else ""
    summary = f"{len(messages)} message(s), max_tokens={max_tokens}; last: {last_prompt}"
    from agent import deadline  # imported here because agent.deadline imports LLMError from this module
    for attempt in range(attempts):
        if deadline.expired():
            raise deadline.TimeLimitReached("run time limit reached")
        try:
            with _SLOTS:
                body = send(messages, max_tokens) if model is None else send(messages, max_tokens, model=model)
            break
        except TransientError as error:
            last = error
            final = attempt == attempts - 1
            _CALL_LOG.append({"status": "failed" if final else "retry", "model": None})
            runlog.emit("ModelClient", "model_call", input_summary=summary, output_summary=str(error),
                        status="failed" if final else "retry", retry_count=attempt if final else attempt + 1)
            if not final:
                sleep(_BASE_DELAY * 2 ** attempt)
        except LLMError as error:
            _CALL_LOG.append({"status": "failed", "model": None})
            runlog.emit("ModelClient", "model_call", input_summary=summary, output_summary=str(error),
                        status="failed", retry_count=attempt)
            raise
    else:
        raise LLMError(f"gave up after {attempts} attempts: {last}")
    receipt = parse_receipt(body)
    usage = parse_usage(body)
    log.info("routed_model=%s receipt_id=%s provider_model=%s egress=%s latency_ms=%s "
             "input_tokens=%s output_tokens=%s cost_usd=%s",
             receipt["routed_model"], receipt["receipt_id"], receipt["provider_model"],
             receipt["egress"], receipt["latency_ms"], usage["input_tokens"], usage["output_tokens"],
             usage["cost_usd"])
    text = body["choices"][0]["message"].get("content") or ""
    status = "replayed" if sample_mode.mode() == "replay" else "success"
    _CALL_LOG.append({"status": "failed" if not text else status, "model": receipt["routed_model"]})
    if not text:
        runlog.emit("ModelClient", "model_call", input_summary=summary, output_summary="empty reply",
                    model=receipt["routed_model"], status="failed", retry_count=attempt)
        raise LLMError("empty reply (reasoning model ran out of tokens?): raise max_tokens")
    runlog.emit("ModelClient", "model_call", input_summary=summary, output_summary=text,
                model=receipt["routed_model"], status=status, retry_count=attempt,
                receipt_id=receipt["receipt_id"], egress=receipt["egress"], **usage)
    return Reply(text=text, routed_model=receipt["routed_model"], receipt_id=receipt["receipt_id"], **usage)
