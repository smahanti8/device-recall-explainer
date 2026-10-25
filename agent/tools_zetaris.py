"""Zetaris SQL tool: run_sql sends ONE read-only statement over Zetaris REST.

Only SELECT/WITH statements are allowed, and only one at a time (Zetaris runs
one complete statement per call). Every table must be an openfda view (or a CTE of the same statement), and write
keywords and Java-calling functions are refused before any request is made. Results are cached by SQL text for the life
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


_REFUSED_WORDS = re.compile(r"\b(insert|update|delete|merge|create|alter|drop|truncate|grant|revoke|call|exec|execute|"
                            r"reflect|java_method)\b", re.I)
_SAFE_TABLE = re.compile(r"^openfda\.[a-z_][a-z0-9_]*$")
_FROM_INSIDE = {"extract", "substring", "substr", "trim", "overlay", "position"}  # FROM here is not a table list
_FROM_ENDS = {"where", "group", "order", "having", "limit", "union", "intersect", "except", "minus", "window",
              "qualify", "lateral", "on", "using", "inner", "left", "right", "full", "cross", "natural", "join",
              "offset", "fetch", "sort", "distribute", "cluster"}


def _bare(statement):
    """The statement without string literals, comments and backticks, for rule checks."""
    text = re.sub(r"'(?:[^']|'')*'|\"[^\"]*\"", "''", statement)
    return re.sub(r"--[^\n]*|/\*.*?\*/", " ", text, flags=re.S).replace("`", "")


def _inside_function(text, index):
    depth = 0
    for j in range(index - 1, -1, -1):
        if text[j] == ")":
            depth += 1
        elif text[j] == "(":
            if depth == 0:
                word = re.search(r"([a-z_]+)\s*$", text[:j])
                return bool(word) and word.group(1) in _FROM_INSIDE
            depth -= 1
    return False


def _table_names(text, start, comma_list):
    """The first token of each item of the FROM or JOIN list that starts at `start`."""
    depth, begin, items, i = 0, start, [], start
    while True:
        done = i >= len(text)
        if not done:
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                done = depth == 0
                depth -= 0 if done else 1
            elif depth == 0 and ch == ",":
                if comma_list:
                    items.append(text[begin:i])
                    begin = i + 1
                else:
                    done = True
            elif depth == 0 and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
                word = re.match(r"[a-z_]+\b", text[i:])
                done = bool(word) and word.group(0) in _FROM_ENDS
        if done:
            items.append(text[begin:i])
            break
        i += 1
    names = []
    for item in items:
        item = item.strip()
        if item and not item.startswith("("):
            names.append(re.match(r"[a-z_][\w.]*", item).group(0) if re.match(r"[a-z_]", item) else item)
    return names


def _check_tables(bare):
    text = bare.lower()
    ctes = set(re.findall(r"(?:\bwith|,)\s*([a-z_]\w*)\s+as\s*\(", text))
    for match in re.finditer(r"\b(from|join)\b", text):
        if _inside_function(text, match.start()):
            continue
        for name in _table_names(text, match.end(), match.group(1) == "from"):
            if not (_SAFE_TABLE.match(name) or name in ctes):
                raise ValueError(f"only openfda tables and views may be queried (got {name!r})")


def _check(sql):
    """Return the statement without a trailing semicolon, or raise ValueError."""
    statement = (sql or "").strip().rstrip(";").strip()
    if not statement:
        raise ValueError("empty SQL statement")
    bare = _bare(statement)
    if ";" in bare:
        raise ValueError("one statement per call")
    if not re.match(r"(?is)^\s*(select|with)\b", statement):
        raise ValueError("only SELECT/WITH statements are allowed")
    refused = _REFUSED_WORDS.search(bare)
    if refused:
        raise ValueError(f"{refused.group(1).lower()} is not allowed in a read-only query")
    _check_tables(bare)
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
