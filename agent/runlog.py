"""Structured run logs: one JSON object per line, written from what actually ran.

Each event records which agent acted, what it received and produced, the model that handled a request,
the status, and the retry count. Strings are truncated, secrets are redacted, and a write failure never
raises (it sets `failed`). Modules call the module-level `emit`, a no-op unless a log is current.
"""
import datetime
import json
import os

MAX_TEXT = 500
_SECRET_NAMES = ("KEY", "TOKEN", "SECRET", "PASSWORD")
_current = None


class RunLog:
    def __init__(self, log_dir, now=None, env=None):
        self.dir = str(log_dir)
        self._now = now or datetime.datetime.utcnow
        self.path = os.path.join(self.dir, "run-" + self._now().strftime("%Y%m%d-%H%M%S") + ".jsonl")
        env = os.environ if env is None else env
        self._secrets = [str(value) for name, value in env.items()
                         if any(mark in name.upper() for mark in _SECRET_NAMES) and len(str(value)) >= 6]
        self.failed = False
        self.error = None

    def _clean(self, value):
        if not isinstance(value, str):
            return value
        for secret in self._secrets:
            value = value.replace(secret, "[redacted]")
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "...(truncated)"

    def emit(self, agent_name, action, *, input_summary="", output_summary="", target_agent=None,
             model=None, status="success", retry_count=0, **extra):
        event = {
            "timestamp": self._now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "agent_name": agent_name, "action": action,
            "input_summary": self._clean(input_summary), "output_summary": self._clean(output_summary),
            "target_agent": target_agent, "model": model, "status": status, "retry_count": retry_count,
        }
        for name, value in extra.items():
            if not any(mark in name.upper() for mark in _SECRET_NAMES):
                event[name] = self._clean(value)
        try:
            os.makedirs(self.dir, exist_ok=True)
            with open(self.path, "a") as handle:
                handle.write(json.dumps(event) + "\n")
        except OSError as error:
            self.failed, self.error = True, str(error)
        return event


def start(log):
    global _current
    _current = log


def stop():
    global _current
    _current = None


def current():
    return _current


def emit(agent_name, action, **fields):
    return _current.emit(agent_name, action, **fields) if _current else None
