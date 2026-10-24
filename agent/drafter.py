"""Drafter: the model investigates with read-only SQL, then answers with cited claims.

Protocol: one JSON object per model turn.
    {"action": "sql", "sql": "SELECT ..."}   or   {"action": "sql", "sqls": ["SELECT ...", ...]}  (up to 3)
    {"action": "answer", "text": "...", "claims": [{"text", "value", "sql", "column", "row"}]}
The model never computes a number: each claim cites a cell of a query result,
and agent.evaluator re-runs that query to verify it.
"""
import json
import re

from agent import llm, tools_zetaris
from agent.evaluator import Citation, Claim
from agent.loop import Draft

MAX_ROWS = 20
MAX_BATCH = 3

SYSTEM = """/no_think
You investigate US FDA medical device recall data stored in Zetaris. Reply with exactly one JSON object per turn. Your very first reply must already be an "action" object (no plan, no prose).

To run a query: {"action": "sql", "sql": "SELECT ..."}   (one SELECT or WITH statement; results come back as rows)
To run up to 3 queries in one turn: {"action": "sql", "sqls": ["SELECT ...", "SELECT ..."]}
To finish:      {"action": "answer", "text": "<short explanation>", "claims": [{"text": "...", "value": <number>, "sql": "SELECT ...", "column": "<column>", "row": 0}]}

Views (schema openfda; every value comes back as a string):
- classification_cardiovascular_table(product_code, device_name, device_class, specialty, regulation_number): 349 cardiovascular device types
- recall_by_product_2023_table, recall_by_product_2024_table(product_code, recall_count): recall RECORDS per product code, by year the recall started
- recall_rootcause_all_2023_table, recall_rootcause_all_2024_table(root_cause, recall_count): root causes across ALL device recalls
- recall_dsy_2024_table, recall_dqo_2024_table, recall_oez_2024_table, recall_drilldown_2024_table (the union of the three)(cfres_id, res_event_number, product_code, event_date_initiated, recall_status, recalling_firm, root_cause_description, reason_for_recall): record-level 2024 recalls for product codes DSY, DQO and OEZ only
Join recalls to classification on product_code. Query as openfda.<view>.

Example claim: {"text": "Cardiovascular recall records in 2024", "value": 482, "sql": "SELECT SUM(b.recall_count) AS total FROM openfda.classification_cardiovascular_table c JOIN openfda.recall_by_product_2024_table b ON b.product_code = c.product_code", "column": "total", "row": 0}

Playbook for "why did X rise or fall between two years" questions (batch queries with "sqls"; do not explore beyond it):
- Step 1 (one turn): the totals for both years and the top 5 increases by product_code, joined to the classification view for scope.
- Step 2 (one turn): for the top product code(s) that have a record-level view, the root causes, the recalling firms, and COUNT(DISTINCT res_event_number) as events.
- Step 3: answer with cited claims.

Rules:
- Every number in your answer must be a claim whose sql returns it in the named column and row. Let SQL compute differences, totals and shares; never compute them yourself.
- Recall "records" are product-level entries. One recall event (res_event_number) can have many records. Say "records" for counts of rows, and report COUNT(DISTINCT res_event_number) as "events". Never call records "recalls" without that distinction.
- A claim's sql must be scoped to what its text says (scope): a claim about cardiovascular devices must join openfda.classification_cardiovascular_table; a claim about a product code must filter on that code or use its view (for example recall_dsy_2024_table).
- Every number written in "text" must also be a claim with exactly that value; use ROUND in SQL for percentages. Years and single digits are exempt.
- "Under Investigation by firm" is a placeholder root cause, not an explanation.
- Describe what the data shows; make no clinical or safety judgments.
- If a previous draft was rejected, fix exactly the claims listed and keep the rest."""

_RETRY_PROMPT = ("Reply with ONE JSON object only, using the protocol: "
                 '{"action": "sql", "sql": ...} or {"action": "answer", "text": ..., "claims": [...]}.')


def _parse(text):
    """The first JSON object in the reply (models sometimes add a ```json fence), or None."""
    body = re.sub(r"```(?:json)?", "", text or "")
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(body[start:end + 1])
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("action") == "answer":
        return parsed
    if parsed.get("action") == "sql" and (isinstance(parsed.get("sql"), str)
                                          or (isinstance(parsed.get("sqls"), list) and parsed["sqls"])):
        return parsed
    return None


def _rows_message(rows):
    shown = json.dumps(rows[:MAX_ROWS])
    note = f"\n(truncated: showing {MAX_ROWS} of {len(rows)} rows)" if len(rows) > MAX_ROWS else ""
    return f"Result ({len(rows)} rows):\n{shown}{note}"


def _run_one(run_sql, sql):
    try:
        return _rows_message(run_sql(sql))
    except Exception as error:
        return f"Error: {str(error)[:300]}"


def _run_queries(run_sql, step):
    """Run the requested query (or up to MAX_BATCH of them) and build the message for the model."""
    if "sqls" not in step:
        return _run_one(run_sql, step.get("sql", ""))
    queries = [str(q) for q in step["sqls"]]
    parts = [f"Query {i}:\n{_run_one(run_sql, q)}" for i, q in enumerate(queries[:MAX_BATCH], start=1)]
    if len(queries) > MAX_BATCH:
        parts.append(f"(only the first {MAX_BATCH} queries were run)")
    return "\n\n".join(parts)


def _trimmed(messages):
    """System prompt, opening question, and only the latest reply with its results."""
    return messages if len(messages) <= 4 else messages[:2] + messages[-2:]


def _claim(raw):
    cited = raw.get("sql") and raw.get("column")
    citation = Citation(raw["sql"], raw["column"], raw.get("row", 0)) if cited else None
    return Claim(raw.get("text", ""), raw.get("value"), citation)


def _opening(question, feedback):
    if not feedback:
        return question
    rejected = "\n".join(f"- [{r.reason}] {r.claim.text}: {r.detail}" for r in feedback)
    return f"{question}\n\nYour previous draft was rejected:\n{rejected}"


def make_draft_fn(chat=None, run_sql=None, max_steps=6):
    chat = chat or llm.chat
    run_sql = run_sql or tools_zetaris.run_sql

    def draft_fn(question, feedback):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _opening(question, feedback)}]
        for step_number in range(max_steps):
            if step_number == max_steps - 1 and step_number > 0:
                messages[-1]["content"] += ("\n\nThis is your last step: reply now with an answer "
                                            "JSON using only the results you already have.")
            try:
                raw = chat(messages, max_tokens=3000).text
            except llm.LLMError:  # e.g. provider deadline: retry once with a smaller request
                messages = _trimmed(messages)
                raw = chat(messages, max_tokens=3000).text
            messages.append({"role": "assistant", "content": raw})
            step = _parse(raw)
            if step is None:
                messages.append({"role": "user", "content": _RETRY_PROMPT})
            elif step["action"] == "answer":
                return Draft(step.get("text", ""), [_claim(c) for c in step.get("claims") or []])
            else:
                messages.append({"role": "user", "content": _run_queries(run_sql, step)})
        return Draft(f"(no answer within {max_steps} steps)", [])

    return draft_fn
