"""Playbook mode: code runs the verified queries, the model explains the results.

Each query result becomes a numbered Fact whose label names its scope and whose
citation is the exact SQL that produced it. The model replies with the fact ids
it used; claims are built from those facts, so a citation cannot be malformed
and the evaluator re-checks every number against Zetaris.

Every statement references each openfda view at most once (Zetaris fails when the
same view appears twice in one statement).
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from agent import llm, tools_zetaris
from agent.drafter import MAX_TOKENS, _context, _opening
from agent.evaluator import Citation, Claim
from agent.loop import Draft

log = logging.getLogger("agent.playbook")

RECORD_LEVEL_CODES = ("DSY", "DQO", "OEZ")  # product codes with a record-level 2024 view
_CODE = re.compile(r"^[A-Za-z0-9]{1,6}$")

_FROM = ("FROM openfda.classification_cardiovascular_table c "
         "LEFT JOIN openfda.recall_by_product_2023_table a ON a.product_code = c.product_code "
         "LEFT JOIN openfda.recall_by_product_2024_table b ON b.product_code = c.product_code")
_CHANGE = f"SELECT c.product_code AS code, COALESCE(b.recall_count,0) - COALESCE(a.recall_count,0) AS change {_FROM}"


def _total(year):
    return (f"SELECT SUM(b.recall_count) AS total FROM openfda.classification_cardiovascular_table c "
            f"JOIN openfda.recall_by_product_{year}_table b ON b.product_code = c.product_code")


SYSTEM = """You explain US FDA cardiovascular device recall data. You are given numbered facts, each already computed by SQL. Use only these facts.
Reply with exactly one JSON object: {"text": "<short explanation>", "facts": ["f1", "f2"]} where "facts" lists every fact id whose number or name you state.
- Every number in "text" must equal the value of a fact you list; write percentages with the one decimal shown.
- Records are product-level entries and one recall event can have many records: say "records" or "events" exactly as the facts do.
- Describe what the data shows; make no clinical or safety judgments.
- If a previous draft was rejected, fix exactly what was listed and keep the rest."""


@dataclass(frozen=True)
class Fact:
    id: str
    label: str
    value: Any
    citation: Citation
    note: str = ""


def _number(raw):
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return raw
    return int(number) if number == int(number) else number


def collect_facts(run_sql=None, top_codes=None, top_n=5):
    run_sql = run_sql or tools_zetaris.run_sql
    facts = []

    def add(label, sql, column, note=""):
        try:
            value = _number(run_sql(sql)[0][column])
        except Exception as error:  # one failed fact must not sink the rest
            log.warning("fact %r skipped: %s", label, str(error)[:200])
            return
        facts.append(Fact(f"f{len(facts) + 1}", label, value, Citation(sql, column, 0), note))

    if top_codes is None:
        try:
            ranked = run_sql(f"SELECT c.product_code AS code, c.device_name AS device, "
                             f"COALESCE(b.recall_count,0) - COALESCE(a.recall_count,0) AS change {_FROM} "
                             f"ORDER BY change DESC LIMIT {int(top_n)}")
            top_codes, names = [r["code"] for r in ranked], {r["code"]: r.get("device", "") for r in ranked}
        except Exception as error:
            log.warning("ranking skipped: %s", str(error)[:200])
            top_codes, names = [], {}
    else:
        names = {}
    top_codes = [c for c in top_codes if _CODE.match(str(c))]

    add("Cardiovascular recall records in 2023", _total(2023), "total")
    add("Cardiovascular recall records in 2024", _total(2024), "total")
    add("Increase in cardiovascular recall records, 2023 to 2024",
        f"SELECT SUM(COALESCE(b.recall_count,0)) - SUM(COALESCE(a.recall_count,0)) AS diff {_FROM}", "diff")
    for code in top_codes:
        add(f"{code} increase in recall records, 2023 to 2024",
            f"SELECT COALESCE(b.recall_count,0) - COALESCE(a.recall_count,0) AS change {_FROM} "
            f"WHERE c.product_code = '{code}'", "change", note=names.get(code, ""))
    add("Increase from the top 3 cardiovascular product codes combined",
        f"SELECT SUM(change) AS top3 FROM (SELECT change FROM ({_CHANGE}) t ORDER BY change DESC LIMIT 3) u", "top3")
    add("Share of the cardiovascular increase from the top 3 product codes (%)",
        "SELECT ROUND(100.0 * SUM(CASE WHEN rn <= 3 THEN change ELSE 0 END) / SUM(change), 1) AS pct FROM "
        f"(SELECT change, ROW_NUMBER() OVER (ORDER BY change DESC) AS rn FROM ({_CHANGE}) t) u", "pct")
    for code in (c for c in top_codes if str(c).upper() in RECORD_LEVEL_CODES):
        view = f"openfda.recall_{str(code).lower()}_2024_table"
        add(f"{code} recall records in 2024", f"SELECT COUNT(*) AS n FROM {view}", "n")
        add(f"{code} recall events in 2024", f"SELECT COUNT(DISTINCT res_event_number) AS n FROM {view}", "n")
        add(f"{code} recalling firms in 2024", f"SELECT COUNT(DISTINCT recalling_firm) AS n FROM {view}", "n")
        for what, column in (("root cause", "root_cause_description"), ("recalling firm", "recalling_firm")):
            add(f"{code} top {what} in 2024",
                f"SELECT {column} AS top FROM {view} GROUP BY {column} ORDER BY COUNT(*) DESC LIMIT 1", "top")
    return facts


def _facts_block(facts):
    lines = [f"{f.id}: {f.label} = {f.value}" + (f"  ({f.note})" if f.note else "") for f in facts]
    return "Facts:\n" + "\n".join(lines)


def _parse(text):
    body = re.sub(r"```(?:json)?", "", text or "")
    start, end = body.find("{"), body.rfind("}")
    try:
        parsed = json.loads(body[start:end + 1]) if 0 <= start < end else None
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) and isinstance(parsed.get("text"), str) else None


def make_playbook_draft_fn(chat=None, run_sql=None, context_fn=None):
    chat = chat or llm.chat
    run_sql = run_sql or tools_zetaris.run_sql

    def draft_fn(question, feedback):
        facts = collect_facts(run_sql=run_sql)
        by_id = {f.id: f for f in facts}
        opening = _opening(question, feedback, _context(context_fn, question))
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"{opening}\n\n{_facts_block(facts)}"}]
        for _ in range(2):  # one correction if the reply is not the JSON object
            try:
                raw = chat(messages, max_tokens=MAX_TOKENS).text
            except llm.LLMError:
                raw = chat(messages, max_tokens=MAX_TOKENS).text
            reply = _parse(raw)
            if reply:
                used = [by_id[i] for i in dict.fromkeys(reply.get("facts") or []) if i in by_id]
                return Draft(reply["text"], [Claim(f.label, f.value, f.citation) for f in used])
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": 'Reply with ONE JSON object only: {"text": "...", "facts": ["f1"]}.'}]
        return Draft("(the model did not return the expected JSON)", [])

    return draft_fn
