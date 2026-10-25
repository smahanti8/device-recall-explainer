"""SAMPLE_MODE: replay recordings of a real run so the project can be seen without Zetaris, Nightshift or keys.

Nothing is faked. The recordings are captured from live calls with --record; in replay the evaluator,
retry loop and memory still run for real on top of them. The default mode is always real.
"""
import copy
import datetime
import hashlib
import json
import os

QUERIES_FILE = "zetaris_results.json"
REPLIES_FILE = "model_replies.json"
_TRUE = {"1", "true", "yes", "on"}
_state = {"mode": None, "dir": None, "queries": {}, "replies": {}, "recorded_at": None, "model": None}


class SampleModeError(RuntimeError):
    pass


def enabled(env=None):
    env = os.environ if env is None else env
    return str(env.get("SAMPLE_MODE") or "").strip().lower() in _TRUE


def mode():
    return _state["mode"]


def recorded_at():
    return _state["recorded_at"]


def _reset():
    _state.update(mode=None, dir=None, queries={}, replies={}, recorded_at=None, model=None)


def _load(directory, name, key):
    try:
        with open(os.path.join(str(directory), name)) as handle:
            return dict(json.load(handle).get(key) or {})
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def start_recording(directory, now=None):
    _reset()
    stamp = (now or datetime.datetime.utcnow)()
    _state.update(mode="record", dir=str(directory), recorded_at=stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  queries=_load(directory, QUERIES_FILE, "queries"),  # earlier recordings stay; new ones replace same keys
                  replies=_load(directory, REPLIES_FILE, "replies"))


def start_replay(directory):
    _reset()
    try:
        with open(os.path.join(directory, QUERIES_FILE)) as handle:
            queries = json.load(handle)
        with open(os.path.join(directory, REPLIES_FILE)) as handle:
            replies = json.load(handle)
    except (OSError, ValueError):
        raise SampleModeError(f"sample data not found in {directory}; record it from a live run with: "
                              f'python -m agent.main --playbook --record {directory} "<question>"')
    _state.update(mode="replay", dir=str(directory), queries=queries.get("queries", {}),
                  replies=replies.get("replies", {}), recorded_at=queries.get("recorded_at"))


def stop():
    if _state["mode"] == "record":
        os.makedirs(_state["dir"], exist_ok=True)
        common = {"recorded_at": _state["recorded_at"], "source": "real run"}
        with open(os.path.join(_state["dir"], QUERIES_FILE), "w") as handle:
            json.dump(dict(common, queries=_state["queries"]), handle, indent=2)
        with open(os.path.join(_state["dir"], REPLIES_FILE), "w") as handle:
            json.dump(dict(common, model=_state["model"], replies=_state["replies"]), handle, indent=2)
    _reset()


def message_key(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def record_query(sql, rows):
    if _state["mode"] == "record":
        _state["queries"][sql] = rows


def replay_query(sql):
    if sql not in _state["queries"]:
        raise SampleModeError("no recording for this SQL; record it from a live run with --record")
    return copy.deepcopy(_state["queries"][sql])


def record_reply(messages, body):
    if _state["mode"] == "record":
        _state["replies"][message_key(messages)] = body
        _state["model"] = _state["model"] or body.get("model")


def replay_reply(messages):
    key = message_key(messages)
    if key not in _state["replies"]:
        raise SampleModeError("no recording for this model prompt; record it from a live run with --record")
    return copy.deepcopy(_state["replies"][key])
