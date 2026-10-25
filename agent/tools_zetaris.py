"""Zetaris SQL tool: run_sql sends ONE read-only statement over Zetaris REST.

Only SELECT/WITH statements are allowed, and only one at a time (Zetaris runs
one complete statement per call). Results are cached by SQL text for the life
of the process: openFDA is rate-limited and cached results keep demo runs and
citations repeatable.
"""
import os
import re
import uuid

import requests
from dotenv import load_dotenv

from agent import runlog, sample_mode

load_dotenv()

_CACHE = {}
_QUERY_LOG = []
_COLUMN_KEYS = ("columns", "columnNames", "column_names", "schema", "headers", "fields")


def _check(sql):
    """Return the statement without a trailing semicolon, or raise ValueError."""
    statement = (sql or "").strip().rstrip(";").strip()
    if not statement:
        raise ValueError("empty SQL statement")
    without_strings = re.sub(r"'(?:[^']|'')*'|\"[^\"]*\"|`[^`]*`", "", statement)
    if ";" in without_strings:
        raise ValueError("one statement per call")
    if not re.match(r"(?is)^\s*(select|with)\b", statement):
        raise ValueError("only SELECT/WITH statements are allowed")
    return statement


def _column_names(payload):
    for key in _COLUMN_KEYS:
        columns = payload.get(key)
        if isinstance(columns, list) and columns:
            return [c.get("name") or c.get("columnName") if isinstance(c, dict) else str(c)
                    for c in columns]
    raise RuntimeError(
        "no column names found in the Zetaris response (keys: "
        f"{sorted(payload)}); run probe_zetaris.py and adjust _column_names"
    )


def _to_dicts(payload):
    names = _column_names(payload)
    return [dict(zip(names, row)) for row in payload.get("data") or []]


def _log(statement, status, summary, rows=0):
    _QUERY_LOG.append({"sql": statement, "status": status, "rows": rows})
    runlog.emit("ZetarisTool", "run_sql", input_summary=statement, output_summary=summary, status=status)


def query_log():
    return [dict(entry) for entry in _QUERY_LOG]


def clear_query_log():
    _QUERY_LOG.clear()


def peek(sql):
    """Rows already fetched (or recorded) for this SQL, without a query, a log event or a network call."""
    statement = _check(sql)
    if sample_mode.mode() == "replay":
        return sample_mode.replay_query(statement)
    return _CACHE[statement]


def run_sql(sql, limit=1000, use_cache=True):
    """Run one read-only statement; return a list of {column: value} dicts."""
    statement = _check(sql)
    if sample_mode.mode() == "replay":
        rows = sample_mode.replay_query(statement)
        _log(statement, "replayed", f"{len(rows)} row(s)", len(rows))
        return rows
    if use_cache and statement in _CACHE:
        _log(statement, "cache_hit", f"{len(_CACHE[statement])} row(s)", len(_CACHE[statement]))
        return _CACHE[statement]
    base = os.environ["ZETARIS_REST_URL"].rstrip("/")
    response = requests.post(
        f"{base}/sql-editor/sqls/run",
        headers={
            "Authorization": "Bearer " + os.environ["ZETARIS_API_KEY"],
            "X-Org-ID": os.environ["ZETARIS_ORG_ID"],
            "X-Request-ID": str(uuid.uuid4()),
        },
        json={"queryId": str(uuid.uuid4()), "sql": statement, "source": "SqlEditor", "limit": limit},
        timeout=60,
    )
    if not response.ok:
        message = f"Zetaris returned {response.status_code}: {response.text[:300]}"
        _log(statement, "error", message)
        raise RuntimeError(message)
    rows = _to_dicts(response.json())
    if use_cache:
        _CACHE[statement] = rows
    sample_mode.record_query(statement, rows)
    _log(statement, "success", f"{len(rows)} row(s)", len(rows))
    return rows
