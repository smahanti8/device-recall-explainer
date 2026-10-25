"""A self-contained HTML report of one run: the question, the answer with numbered proofs, the rounds
(rejection, research, redraft), every query, model receipts, tokens and memory.

No server, no scripts, no external URLs. Everything that comes from a run is HTML-escaped.
"""
import datetime
import html
import os
import re

from agent import observability

_CSS = """
:root{--ink:#2B2F33;--panel:#F3F3F1;--coral:#EE5A4A;--green:#3A9E76;--grey:#6F767D;--line:#DADBDC}
*{box-sizing:border-box}body{margin:0;font:16px/1.55 Arial,Helvetica,sans-serif;color:var(--ink);background:#fff}
main{max-width:980px;margin:0 auto;padding:32px 20px 64px}h1{font:700 30px/1.2 Cambria,Georgia,serif;margin:0 0 4px}
h2{font:700 22px/1.3 Cambria,Georgia,serif;margin:36px 0 10px}h3{margin:18px 0 6px;font-size:17px}
.meta{color:var(--grey);font-size:14px}.panel{background:var(--panel);border-radius:10px;padding:14px 18px;margin:10px 0}
.banner{background:var(--ink);color:#fff;border-radius:10px;padding:12px 18px;margin:16px 0;font-weight:700}
.stamp{display:inline-block;border:2px solid;border-radius:8px;padding:0 8px;font:700 12px/22px Arial;letter-spacing:.08em;text-transform:uppercase}
.ok{color:var(--green);border-color:var(--green)}.no{color:var(--coral);border-color:var(--coral)}
.answer{font-size:18px}.answer a{color:var(--coral);font-weight:700;text-decoration:none;border-bottom:2px solid var(--coral)}
details{background:var(--panel);border-radius:10px;margin:8px 0;padding:8px 14px}summary{cursor:pointer;font-weight:700}
pre{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px;font:13px/1.45 "Courier New",monospace}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:var(--ink);color:#fff}code{font:13px "Courier New",monospace}ul{padding-left:20px}
"""

_NUMBER = re.compile(r"(?<![\w.,])(\d+(?:\.\d+)?)(?!\w|,\d)")


def _e(value):
    return html.escape("" if value is None else str(value), quote=True)


def _key(value):
    """The text a numeric claim value is written as in prose, or None when it is not a number."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return str(int(number)) if number == int(number) else str(number)


def _linked_answer(answer, claims):
    counts, owner = {}, {}
    for claim in claims:
        key = _key(claim.get("value")) if claim.get("status") == "verified" else None
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
            owner[key] = claim["n"]
    escaped = _e(answer)
    return _NUMBER.sub(lambda m: f'<a href="#claim-{owner[m.group(1)]}">{m.group(1)}</a>'
                       if counts.get(m.group(1)) == 1 else m.group(0), escaped)


def _banner(data):
    if data.get("mode") == "sample":
        return ('<div class="banner">SAMPLE MODE: Zetaris results and the model reply were replayed from recordings '
                f'made {_e(data.get("recorded_at"))}; the evaluator, retry loop and memory ran for real.</div>')
    if data.get("mode") == "record":
        return '<div class="banner">Recording run: results from this live run were saved for SAMPLE_MODE replays.</div>'
    return ""


def _claims(claims):
    out = []
    for claim in claims:
        ok = claim.get("status") == "verified"
        value = "" if claim.get("value") is None else f" = {_e(claim['value'])}"
        out.append(
            f'<details id="claim-{claim["n"]}"><summary>[{claim["n"]}] {_e(claim.get("text"))}{value} '
            f'<span class="stamp {"ok" if ok else "no"}">{_e(claim.get("status"))}</span></summary>'
            f'<h3>Cited SQL</h3><pre>{_e(claim.get("sql"))}</pre>'
            f'<p>Column <code>{_e(claim.get("column"))}</code>, row <code>{_e(claim.get("row"))}</code>; '
            f'cell read back: <code>{_e(claim.get("cell"))}</code>; status: <strong>{_e(claim.get("status"))}</strong></p>'
            "</details>")
    return "\n".join(out)


def _rounds(rounds):
    out = []
    for entry in rounds:
        outcome = (f'{entry["rejected"]} rejected ({_e(", ".join(entry.get("reasons", [])))})'
                   if entry.get("rejected") else "all verified")
        items = "".join(f"<li><code>{_e(reason)}</code>: {_e(detail)}</li>"
                        for reason, detail in zip(entry.get("reasons", []), entry.get("details", [])))
        out.append(f'<div class="panel"><h3>Round {_e(entry["round"])}</h3><p>{_e(entry["claims"])} claims checked: '
                   f'<strong>{outcome}</strong></p>{"<ul>" + items + "</ul>" if items else ""}</div>')
    return "\n".join(out) or "<p>No rounds were recorded.</p>"


def _queries(queries):
    rows = "".join(f"<tr><td><code>{_e(q['sql'])}</code></td><td>{_e(q['status'])}</td><td>{_e(q['rows'])}</td></tr>"
                   for q in queries)
    return (f"<details><summary>{len(queries)} Zetaris queries</summary><table><tr><th>SQL</th><th>Status</th>"
            f"<th>Rows</th></tr>{rows}</table></details>") if queries else "<p>No Zetaris queries were recorded.</p>"


def _observability(summary):
    found = observability.parts(summary) if summary else []
    if not found:
        return ""
    return "<h2>Observability</h2><ul>" + "".join(f"<li>{_e(part)}</li>" for part in found) + "</ul>"


def render(data):
    claims = data.get("claims") or []
    tokens = data.get("tokens") or {}
    answer = data.get("answer")
    if answer:
        answer_html = (f'<h2>Verified answer</h2><div class="panel answer">{_linked_answer(answer, claims)}</div>'
                       '<p class="meta">Numbers in the answer link to the claim that proves them.</p>')
    else:
        answer_html = ('<h2>No verified answer</h2><div class="panel"><span class="stamp no">unverified</span> '
                       "The run did not produce an answer whose claims all passed the evaluator.</div>")
    receipts = "".join(f"<li><code>{_e(r['routed_model'])}</code> &middot; receipt <code>{_e(r['receipt_id'])}</code></li>"
                       for r in data.get("receipts") or [])
    token_line = (f"{tokens['calls']} model call(s), {tokens.get('input', 0):,} in, {tokens.get('output', 0):,} out; "
                  f"cost reported by Nightshift: ${tokens.get('cost_usd', 0.0):.4f}") if tokens.get("calls") else ""
    hints = "".join(f"<li>{_e(h)}</li>" for h in data.get("hints") or [])
    timeline = "".join(f"<li>{_e(line)}</li>" for line in data.get("timeline") or [])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Device Recall Explainer: run report</title><style>{_CSS}</style></head>
<body><main>
<h1>Device Recall Explainer</h1>
<p class="meta">Run report &middot; generated {_e(data.get("generated_at"))} &middot; mode: {_e(data.get("mode"))}</p>
{_banner(data)}
<h2>Question</h2><div class="panel">{_e(data.get("question"))}</div>
{answer_html}
<h2>Claims ({len(claims)})</h2>
<p class="meta">Each claim cites one cell of one SQL result. Open one to see the query, the cell and the verdict.</p>
{_claims(claims)}
<h2>Rounds</h2>
{_rounds(data.get("rounds") or [])}
<details><summary>Timeline</summary><ul>{timeline}</ul></details>
<h2>Queries</h2>
{_queries(data.get("queries") or [])}
<h2>Model calls</h2>
<ul>{receipts or "<li>No model calls were recorded.</li>"}</ul>
<p>{_e(token_line)}</p>
{_observability(data.get("observability"))}
<h2>Memory</h2>
{"<p>Recalled hints (not evidence):</p><ul>" + hints + "</ul>" if hints else "<p>Nothing was recalled.</p>"}
<p>{_e(data.get("memory") or "")}</p>
</main></body></html>
"""


def write(data, directory, now=None):
    stamp = (now or datetime.datetime.utcnow)()
    path = os.path.join(str(directory), "report-" + stamp.strftime("%Y%m%d-%H%M%S") + ".html")
    try:
        os.makedirs(str(directory), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(render(data))
    except OSError:
        return None
    return path
