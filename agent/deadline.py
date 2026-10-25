"""A time limit for one whole run (individual model and Zetaris calls have their own timeouts).

`start(seconds)` begins the limit, `expired()` says whether it has passed, `stop()` clears it. llm.chat refuses to
start a call after the limit, and the loop stops before the next round.
"""
import time

from agent.llm import LLMError

_state = {"end": None, "clock": time.monotonic}


class TimeLimitReached(LLMError):
    pass


def start(seconds, clock=time.monotonic):
    _state["clock"] = clock
    _state["end"] = clock() + seconds if seconds else None


def stop():
    _state["end"] = None
    _state["clock"] = time.monotonic


def remaining():
    if _state["end"] is None:
        return None
    return max(0.0, _state["end"] - _state["clock"]())


def expired():
    left = remaining()
    return left is not None and left <= 0
