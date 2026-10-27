"""Builds the data for the static replay site (docs/, served by GitHub Pages) from the shipped sample runs.

The site cannot run the agent. It replays what is already in the repository: each run log in logs/sample-*.jsonl is
condensed into a short list of steps, and the run's report (reports/sample-*.html) is copied next to it.

    python -m agent.replay_site            writes docs/runs.json, docs/reports/*.html and docs/.nojekyll
"""
import argparse
import html
import json
import os
import re
import shutil

RUNS = [
    ("sample-run-1", "Why did recalls rise?"),
    ("sample-run-2", "Top three product codes"),
    ("sample-run-3", "One product code: DSY"),
    ("sample-exploratory-run-1", "Same question, exploratory mode"),
    ("sample-refusal-1", "A question about something else"),
]
WATCH = {
    "sample-run-1": ("Round 1 names two product codes without their detail, so the evaluator rejects it. A research step "
                     "fetches the detail and the redraft, on the larger model, verifies all 35 claims."),
    "sample-run-2": ("A clean run: one round, no rejections. Compare it with the first run to see what the evaluator "
                     "does when a draft is incomplete."),
    "sample-run-3": ("A narrow question gets a narrow answer: four claims, each with its own SQL. This is where "
                     "records and events are reported separately."),
    "sample-exploratory-run-1": ("The evaluator refused to print an answer it could not verify. In round 2 the model "
                                 "claimed 65.2 but its cited SQL returned 100.0. That is the intended outcome, not a bug."),
    "sample-refusal-1": ("The scope check runs before any model call or query. A refusal costs nothing and says why."),
}
SUB = {"playbook": "Playbook mode", "exploratory": "Exploratory mode", "refused": "Scope check"}
_TAGS = re.compile(r"<[^>]+>")
_TOKENS = re.compile(r"([\d,]+) in, ([\d,]+) out")
_CLAIMS = re.compile(r"<h2>Claims \((\d+)\)</h2>")
_CLAIM = re.compile(r'<details id="claim-\d+"><summary>\[\d+\] (.*?) <span class="stamp \w+">.*?</span></summary>'
                    r"<h3>Cited SQL</h3><pre>(.*?)</pre><p>Column <code>(.*?)</code>, row <code>(.*?)</code>; "
                    r"cell read back: <code>(.*?)</code>", re.S)
_FOOTNOTE = "Numbers in the answer link to the claim that proves them."


def _answer(report_html):
    match = re.search(r"<h2>Verified answer</h2>(.*?)<h2>", report_html, re.S)
    if not match:
        return None
    text = " ".join(html.unescape(_TAGS.sub(" ", match.group(1))).split())
    text = text.replace(_FOOTNOTE, "").strip()
    return re.sub(r"\s+([.,;:%])", r"\1", text) or None


def _steps(events):
    """Condense a run log into [{"kind", "text", "tone"}] and the Zetaris query counts."""
    steps, totals = [], {"fetched": 0, "cached": 0, "failed": 0}
    sql = {"fetched": 0, "cached": 0, "failed": 0}
    retries = 0

    def add(kind, text, tone=""):
        steps.append({"kind": kind, "text": text, "tone": tone})

    def flush():
        if sql["fetched"] or sql["cached"] or sql["failed"]:
            parts = [f"{sql['fetched']} queries fetched"] if sql["fetched"] else []
            if sql["failed"]:
                parts.append(f"{sql['failed']} failed")
            if sql["cached"]:
                parts.append(f"{sql['cached']} served from cache")
            add("zetaris", "Zetaris: " + ", ".join(parts), "dim")
        sql.update(fetched=0, cached=0, failed=0)

    for event in events:
        action, status = event.get("action"), event.get("status")
        if action == "run_sql":
            if status == "cache_hit":
                sql["cached"] += 1
                totals["cached"] += 1
            elif status == "error":
                sql["failed"] += 1
                totals["failed"] += 1
            else:
                sql["fetched"] += 1
                totals["fetched"] += 1
            continue
        flush()
        if action == "run_start":
            add("question", event.get("question") or event.get("input_summary", ""))
        elif action == "round_start":
            add("round", f"round {event['round']}: drafting")
        elif action == "recall":
            add("memory", "memory: " + ("nothing recalled yet" if event["output_summary"].startswith("0 ")
                                        else event["output_summary"].replace("hint(s)", "hints recalled")), "dim")
        elif action == "learn":
            add("memory", "memory: " + event["output_summary"], "dim")
        elif action == "model_call":
            if status == "retry":
                retries += 1
            elif status == "failed":
                add("model", f"model call failed after {retries + 1} attempts (provider error)", "bad")
                retries = 0
            else:
                suffix = f", after {retries} provider retries" if retries else ""
                add("model", f"model call: {event['model']} ({event['receipt_id']}){suffix}", "dim")
                retries = 0
        elif action == "escalate_model":
            add("escalate", f"escalating to {event['model']} for the redraft", "dim")
        elif action == "fetch_facts":
            add("research", f"research: fetched {event['output_summary']} from Zetaris", "dim")
        elif action == "evaluate_claims":
            add("check", f"round {event['round']}: {event['output_summary']}",
                "good" if status == "verified" else "bad")
        elif action == "run_end":
            if status == "verified":
                add("end", f"finished: {event['output_summary']}, verified answer", "good")
            else:
                add("end", f"finished: {event['output_summary']}, no verified answer: every draft was rejected", "bad")
        elif action == "scope_check":
            add("refused", "Out of scope: " + event["output_summary"], "bad")
    flush()
    return steps, totals


def _report_facts(report_html, verified):
    """Token counts, claim count and the first claims (with their SQL) from a saved report."""
    tokens = _TOKENS.search(report_html)
    claims = _CLAIMS.search(report_html)
    found = [{"claim": html.unescape(c[0]), "sql": html.unescape(c[1]), "column": html.unescape(c[2]),
              "row": html.unescape(c[3]), "cell": html.unescape(c[4])} for c in _CLAIM.findall(report_html)]
    return {"tokens": {"in": tokens.group(1), "out": tokens.group(2)} if tokens else None,
            "claims": int(claims.group(1)) if claims and verified else 0,
            "found": found if verified else []}


def _detail(events):
    """Rounds, model calls, models, receipts and per-round verdicts from a run log."""
    calls = [e for e in events if e.get("action") == "model_call" and e.get("status") == "success"]
    end = next((e for e in events if e.get("action") == "run_end"), None)
    rounds = int(re.match(r"\d+", end["output_summary"]).group(0)) if end else 0
    learn = next((e for e in events if e.get("action") == "learn"), None)
    return {"rounds": rounds, "calls": len(calls),
            "models": sorted({e["model"] for e in calls}), "receipts": [e["receipt_id"] for e in calls],
            "rejections": [{"round": e["round"], "text": e["output_summary"], "ok": e.get("status") == "verified"}
                           for e in events if e.get("action") == "evaluate_claims"],
            "memory": learn["output_summary"] if learn else None}


def _run(root, run_id, title, out_dir):
    with open(os.path.join(root, "logs", f"{run_id}.jsonl"), encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    steps, totals = _steps(events)
    refused = any(e.get("action") == "scope_check" for e in events)
    start = next((e for e in events if e.get("action") == "run_start"), None)
    if refused:  # a refusal logs only the scope check; its input is the question
        scope = next(e for e in events if e.get("action") == "scope_check")
        question = scope["input_summary"]
        steps.insert(0, {"kind": "question", "text": question, "tone": ""})
    else:
        question = start["question"]
    end = next((e for e in events if e.get("action") == "run_end"), None)
    verified = bool(end and end.get("status") == "verified")
    report = answer = None
    facts = {"tokens": None, "claims": 0, "found": []}
    if not refused:
        report = f"reports/{run_id}.html"
        os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
        shutil.copyfile(os.path.join(root, "reports", f"{run_id}.html"), os.path.join(out_dir, report))
        with open(os.path.join(out_dir, report), encoding="utf-8") as handle:
            page = handle.read()
        answer = _answer(page) if verified else None
        facts = _report_facts(page, verified)
    detail = _detail(events)
    mode = "refused" if refused else ("playbook" if start.get("playbook") else "exploratory")
    first = facts["found"][0] if facts["found"] else None
    return {"id": run_id, "title": title, "sub": SUB[mode], "question": question, "mode": mode,
            "verified": verified, "refused": refused, "answer": answer,
            "message": steps[-1]["text"] if refused else None, "report": report,
            "steps": steps, "queries": totals, "rounds": detail["rounds"], "claims": facts["claims"],
            "calls": detail["calls"], "tokens": facts["tokens"], "models": detail["models"],
            "receipts": detail["receipts"], "rejections": detail["rejections"], "memory": detail["memory"],
            "proof": first, "queries_sample": [{"note": c["claim"], "sql": c["sql"]} for c in facts["found"][:3]],
            "watch": WATCH[run_id]}


def build(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    runs = [_run(root, run_id, title, out_dir) for run_id, title in RUNS]
    with open(os.path.join(out_dir, "runs.json"), "w", encoding="utf-8") as handle:
        json.dump({"runs": runs}, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    open(os.path.join(out_dir, ".nojekyll"), "w").close()
    return runs


def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(prog="python -m agent.replay_site", description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(root, "docs"))
    args = parser.parse_args(argv)
    runs = build(root, args.out)
    print(f"wrote {len(runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
