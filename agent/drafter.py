"""Drafter: the model investigates with read-only SQL, then answers with cited claims.

Protocol: one JSON object per model turn.
    {"action": "sql", "sql": "SELECT ..."}
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

SYSTEM = """/no_think
You investigate US FDA medical device recall data stored in Zetaris. Reply with exactly one JSON object per turn. Your very first reply must already be an "action" object (no plan, no prose).

To run a query: {"action": "sql", "sql": "SELECT ..."}   (one SELECT or WITH statement; results come back as rows)
To finish:      {"action": "answer", "text": "<short explanation>", "claims": [{"text": "...", "value": <number>, "sql": "SELECT ...", "column": "<column>", "row": 0}]}

Views (schema openfda; every value comes back as a string):
- classification_cardiovascular_table(product_code, device_name, device_class, specialty, regulation_number): 349 cardiovascular device types
- recall_by_product_2023_table, recall_by_product_2024_table(product_code, recall_count): recall RECORDS per product code, by year the recall started
- recall_rootcause_all_2023_table, recall_rootcause_all_2024_table(root_cause, recall_count): root causes across ALL device recalls
- recall_dsy_2024_table, recall_dqo_2024_table, recall_oez_2024_table, recall_drilldown_2024_table (the union of the three)(cfres_id, res_event_number, product_code, event_date_initiated, recall_status, recalling_firm, root_cause_description, reason_for_recall): record-level 2024 recalls for product codes DSY, DQO and OEZ only
Join recalls to classification on product_code. Query as openfda.<view>.

Rules:
- Every number in your answer must be a claim whose sql returns it in the named column and row. Let SQL compute differences, totals and shares; never compute them yourself.
- Recall "records" are product-level entries. One recall event (res_event_number) can have many records. Say "records" for counts of rows, and report COUNT(DISTINCT res_event_number) as "events". Never call records "recalls" without that distinction.
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
    return parsed if isinstance(parsed, dict) and parsed.get("action") in ("sql", "answer") else None


def _rows_message(rows):
    shown = json.dumps(rows[:MAX_ROWS])
    note = f"\n(truncated: showing {MAX_ROWS} of {len(rows)} rows)" if len(rows) > MAX_ROWS else ""
    return f"Result ({len(rows)} rows):\n{shown}{note}"


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
            raw = chat(messages, max_tokens=3000).text
            messages.append({"role": "assistant", "content": raw})
            step = _parse(raw)
            if step is None:
                messages.append({"role": "user", "content": _RETRY_PROMPT})
            elif step["action"] == "answer":
                return Draft(step.get("text", ""), [_claim(c) for c in step.get("claims") or []])
            else:
                try:
                    result = _rows_message(run_sql(step.get("sql", "")))
                except Exception as error:
                    result = f"Error: {str(error)[:300]}"
                messages.append({"role": "user", "content": result})
        return Draft(f"(no answer within {max_steps} steps)", [])

    return draft_fn
