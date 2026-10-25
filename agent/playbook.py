"""Playbook mode: code runs the verified queries, the model explains the results.

Each query result becomes a numbered Fact whose label names its scope and whose
citation is the exact SQL that produced it. The model replies with the fact ids
it used; claims are built from those facts, so a citation cannot be malformed
and the evaluator checks every number against the result of its cited SQL.

Evidence branch: a draft that names a drill-down product code without citing its detail is rejected as
`missing_evidence`; the next draft first fetches that code's facts (the Researcher step), then redrafts.

Every statement references each openfda view at most once (Zetaris fails when the
same view appears twice in one statement).
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from agent import llm, runlog, tools_zetaris
from agent.drafter import MAX_TOKENS, _context, _opening
from agent.evaluator import Citation, Claim, Rejection
from agent.loop import Draft

log = logging.getLogger("agent.playbook")

RECORD_LEVEL_CODES = ("DSY", "DQO", "OEZ")  # detailed up front
DRILLDOWN_CODES = RECORD_LEVEL_CODES + ("DQK", "DPT", "DRE", "DWF", "OES", "DYF", "OEQ")  # all have a record-level 2024 view
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
- For each product code that has events, root-cause and recalling-firm facts, state how many records came from how many events, the top root cause and the top recalling firm. When a code has few events behind many records, say so: the increase is concentrated in a few events.
- Never write fact ids (like f1) in "text"; they are only for the "facts" list.
- Describe what the data shows; make no clinical or safety judgments.
- A rejection with reason "missing_evidence" means you named a product code without its detail: the facts for that code have now been added, so state its records versus events, its top root cause and its top recalling firm.
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
        for label, sql, column in _detail(str(code).upper()):
            add(label, sql, column)
    return facts


def _detail(code):
    """The five record-level facts for a product code: (label, sql, column)."""
    view = f"openfda.recall_{code.lower()}_2024_table"
    return [
        (f"{code} recall records in 2024", f"SELECT COUNT(*) AS n FROM {view}", "n"),
        (f"{code} recall events in 2024", f"SELECT COUNT(DISTINCT res_event_number) AS n FROM {view}", "n"),
        (f"{code} recalling firms in 2024", f"SELECT COUNT(DISTINCT recalling_firm) AS n FROM {view}", "n"),
        (f"{code} top root cause in 2024",
         f"SELECT root_cause_description AS top FROM {view} GROUP BY root_cause_description ORDER BY COUNT(*) DESC LIMIT 1", "top"),
        (f"{code} top recalling firm in 2024",
         f"SELECT recalling_firm AS top FROM {view} GROUP BY recalling_firm ORDER BY COUNT(*) DESC LIMIT 1", "top"),
    ]


def collect_code_facts(code, run_sql=None, start_id=1):
    """Detail facts for one drill-down code, with ids continuing from start_id. ValueError for an unknown code."""
    if not isinstance(code, str) or code not in DRILLDOWN_CODES:
        raise ValueError(f"no drill-down for {code!r}")
    run_sql = run_sql or tools_zetaris.run_sql
    facts = []
    for label, sql, column in _detail(code):
        try:
            value = _number(run_sql(sql)[0][column])
        except Exception as error:  # one failed fact must not sink the rest
            log.warning("fact %r skipped: %s", label, str(error)[:200])
            continue
        facts.append(Fact(f"f{start_id + len(facts)}", label, value, Citation(sql, column, 0)))
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


_FACT_MARKER = re.compile(r"\s*[(\[]\s*f\d+(?:\s*[,;]\s*f\d+)*\s*[)\]]")
_WHY_LABELS = ("events", "root cause")
_WHY_PROMPT = ("Your answer cites no recall-event or root-cause fact. For each product code that has them, "
               "state records versus events (events) and the top root cause and recalling firm, "
               "citing those facts. Reply with the same JSON object.")
_JSON_PROMPT = 'Reply with ONE JSON object only: {"text": "...", "facts": ["f1"]}.'


_BRACKET_GROUP = re.compile(r"[(\[][^()\[\]]*[)\]]")


def _trim_ids_in_group(match):
    """Inside one bracket, drop ids that sit beside other items: "(+53, f4)" -> "(+53)"."""
    group = re.sub(r"\s*[,;]\s*f\d+\b", "", match.group(0))
    return re.sub(r"\bf\d+\s*[,;]\s*", "", group)


def _without_fact_ids(text):
    """Ids like "(f1)", "[f2, f3]" or "(+53, f4)" are for the claims list, not for readers."""
    return _BRACKET_GROUP.sub(_trim_ids_in_group, _FACT_MARKER.sub("", text))


def _has_why(items):
    return any(label in item for item in items for label in _WHY_LABELS)


def _requested_codes(feedback):
    codes = [r.claim.text.split()[0] for r in feedback
             if r.reason == "missing_evidence" and r.claim.text.endswith(" detail")]
    return list(dict.fromkeys(codes))


def make_playbook_draft_fn(chat=None, run_sql=None, context_fn=None, on_event=None):
    chat = chat or llm.chat
    run_sql = run_sql or tools_zetaris.run_sql
    fetched = {}      # code -> detail facts fetched on request (kept for later rounds)
    flagged = set()   # codes already flagged as missing evidence: each is flagged at most once

    def completeness(draft):
        """Evaluator rule: a named drill-down code with none of its detail cited is missing evidence."""
        rejections = []
        for code in DRILLDOWN_CODES:
            if not re.search(rf"\b{code}\b", draft.text) or code in flagged:
                continue
            if any(claim.text.startswith(f"{code} recall events in 2024") for claim in draft.claims):
                continue
            flagged.add(code)
            rejections.append(Rejection(
                Claim(f"{code} detail", None), "missing_evidence",
                f"the answer names {code} but cites no records, events, root-cause or recalling-firm facts for it; "
                "its drill-down will be fetched"))
        return rejections

    def research(feedback):
        """Researcher: fetch the detail facts the evaluator said were missing."""
        for code in _requested_codes(feedback):
            if code in fetched:
                continue
            try:
                got = collect_code_facts(code, run_sql=run_sql)
            except ValueError:
                got = []
            fetched[code] = got
            runlog.emit("Researcher", "fetch_facts", input_summary=f"{code} drill-down",
                        output_summary=f"{len(got)} facts", target_agent="Drafter",
                        status="success" if got else "error")
            if on_event:
                on_event(f"evidence requested: {code} -> {len(got)} facts fetched")

    def draft_fn(question, feedback):
        research(feedback)
        facts = collect_facts(run_sql=run_sql)
        for got in fetched.values():
            for fact in got:
                facts.append(Fact(f"f{len(facts) + 1}", fact.label, fact.value, fact.citation, fact.note))
        by_id = {f.id: f for f in facts}
        opening = _opening(question, feedback, _context(context_fn, question))
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"{opening}\n\n{_facts_block(facts)}"}]
        last, fixed_json, fixed_why = None, False, False
        for _ in range(3):  # at most one JSON correction and one "why" correction
            try:
                raw = chat(messages, max_tokens=MAX_TOKENS).text
            except llm.LLMError:
                raw = chat(messages, max_tokens=MAX_TOKENS).text
            reply = _parse(raw)
            if reply is None:
                if fixed_json:
                    break
                fixed_json = True
                messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": _JSON_PROMPT}]
                continue
            used = [by_id[i] for i in dict.fromkeys(reply.get("facts") or []) if i in by_id]
            last = Draft(_without_fact_ids(reply["text"]), [Claim(f.label, f.value, f.citation) for f in used])
            if _has_why([f.label for f in facts]) and not _has_why([f.label for f in used]) and not fixed_why:
                fixed_why = True
                messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": _WHY_PROMPT}]
                continue
            return last
        return last or Draft("(the model did not return the expected JSON)", [])

    draft_fn.completeness = completeness
    return draft_fn
