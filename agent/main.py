"""Command line: python -m agent.main "<question>" [--memory-dir .hmem] [--no-memory] [--verbose] [--playbook]

Runs the investigation (draft -> evaluate -> retry), prints the trace with the
model receipts, and remembers the claims that were verified.
"""
import argparse
import datetime
import json
import logging
import os
import re
import sys

from agent import deadline, llm, observability, runlog, sample_mode
from agent.drafter import make_draft_fn
from agent.evaluator import Verdict, evaluate, unclaimed_numbers
from agent.llm import LLMError
from agent.loop import investigate
from agent.playbook import make_playbook_draft_fn
from agent import report as run_report
from agent import tools_zetaris
from agent.runlog import RunLog
from agent.sample_mode import SampleModeError
from agent.scope import scope_problem
from agent.memory import Memory

REQUIRED_SETTINGS = ("ZETARIS_REST_URL", "ZETARIS_ORG_ID", "ZETARIS_API_KEY",
                     "MODEL_BASE_URL", "MODEL_API_KEY", "NEMOTRON_MODEL")
MAX_QUESTION = 500
_RECEIPT = re.compile(r"routed_model=(\S+) receipt_id=(\S+)")
_LATENCY = re.compile(r"latency_ms=(\S+)")
_TOKENS = re.compile(r"input_tokens=(\S+) output_tokens=(\S+) cost_usd=(\S+)")


def _number(text, kind=float):
    try:
        return kind(float(text))
    except (TypeError, ValueError):
        return kind(0)


class ReceiptCollector(logging.Handler):
    """Context manager that collects routed_model / receipt_id from agent.llm's log lines."""

    def __init__(self, on_receipt=None):
        super().__init__(logging.INFO)
        self.items = []
        self.tokens = {"calls": 0, "input": 0, "output": 0, "cost_usd": 0.0}
        self.on_receipt = on_receipt  # called with routed_model, receipt_id, latency_ms and the token counts

    def emit(self, record):
        message = record.getMessage()
        match = _RECEIPT.search(message)
        if match:
            receipt_id = None if match.group(2) == "None" else match.group(2)
            self.items.append({"routed_model": match.group(1), "receipt_id": receipt_id})
            counts = _TOKENS.search(message)
            used = (_number(counts.group(1), int), _number(counts.group(2), int), _number(counts.group(3))) if counts else (0, 0, 0.0)
            self.tokens["calls"] += 1
            self.tokens["input"] += used[0]
            self.tokens["output"] += used[1]
            self.tokens["cost_usd"] += used[2]
            if self.on_receipt:
                latency = _LATENCY.search(message)
                self.on_receipt({**self.items[-1], "latency_ms": latency.group(1) if latency else None,
                                 "input_tokens": used[0], "output_tokens": used[1], "cost_usd": used[2]})

    def __enter__(self):
        self._log = logging.getLogger("agent.llm")
        self._level = self._log.level
        if self._level == logging.NOTSET or self._level > logging.INFO:
            self._log.setLevel(logging.INFO)
        self._log.addHandler(self)
        return self

    def __exit__(self, *exc):
        self._log.removeHandler(self)
        self._log.setLevel(self._level)


def format_trace(result, receipts=(), verbose=False, tokens=None):
    lines = ["Trace"]
    for entry in result.trace:
        outcome = (f"{entry['rejected']} rejected ({', '.join(entry['reasons'])})"
                   if entry["rejected"] else "all verified")
        lines.append(f"  Round {entry['round']}: {entry['claims']} claims, {outcome}")
        if verbose:
            lines += [f"      {reason}: {detail}"
                      for reason, detail in zip(entry["reasons"], entry.get("details", []))]
    if result.text:
        lines.append("Verified claims:")
        lines += [f"  - {claim.text} = {claim.value}" for claim in result.claims]
        lines.append(f"Answer: {result.text}")
    else:
        lines.append(f"No verified answer after {result.rounds} round(s). Could not support:")
        lines += [f"  - {claim.text}" + ("" if claim.value is None else f" = {claim.value}")
                  for claim in result.unsupported]
    if receipts:
        lines.append("Model receipts:")
        lines += [f"  {item['routed_model']}  {item['receipt_id']}" for item in receipts]
    if tokens and tokens.get("calls"):
        lines.append(f"Tokens: {tokens['calls']} model call(s), {tokens['input']:,} in, {tokens['output']:,} out; "
                     f"cost reported by Nightshift: ${tokens['cost_usd']:.4f}")
    return "\n".join(lines)


def _stderr(message):
    print(message, file=sys.stderr, flush=True)  # sys.stderr is looked up at call time


def _report_data(result, question, state, receipts, queries, timeline, memory_line, lookup, now):
    """Everything the HTML report shows, read from this run's own objects."""
    claims = []
    for n, claim in enumerate(result.claims, start=1):
        cite = claim.citation
        try:
            cell = lookup(cite.sql)[cite.row][cite.column]
        except Exception:
            cell = "unavailable"
        claims.append({"n": n, "text": claim.text, "value": claim.value, "sql": cite.sql, "column": cite.column,
                       "row": cite.row, "cell": cell, "status": "verified"})
    if not result.text:
        for n, claim in enumerate(result.unsupported, start=len(claims) + 1):
            cite = claim.citation
            claims.append({"n": n, "text": claim.text, "value": claim.value, "sql": cite.sql if cite else "",
                           "column": cite.column if cite else "", "row": cite.row if cite else "",
                           "cell": "", "status": "unsupported"})
    replay = sample_mode.mode() == "replay"
    return {"question": question, "answer": result.text, "claims": claims, "rounds": result.trace,
            "queries": queries, "receipts": receipts.items, "tokens": receipts.tokens, "hints": state["hints"],
            "memory": memory_line, "timeline": timeline,
            "mode": "sample" if replay else "record" if sample_mode.mode() == "record" else "live",
            "generated_at": (now or datetime.datetime.utcnow)().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "recorded_at": sample_mode.recorded_at() if replay else None}


def _ask(question, memory, *, make_draft, make_playbook, run_investigation, evaluate_fn, out, progress,
         verbose, playbook, log, report_dir, run_sql, now, as_json=False):
    real_out = out
    if as_json:  # stdout carries one JSON document; progress stays on stderr
        out = lambda line: None  # noqa: E731
    raw_progress = progress or _stderr
    timeline = []

    def progress(message):  # every progress line is also kept for the report's timeline
        timeline.append(message)
        raw_progress(message)
    make = make_playbook if playbook else make_draft
    state = {"round": 0, "turn": 0, "draft": None, "hints": []}
    memory_line = None
    tools_zetaris.clear_query_log()
    llm.clear_call_log()
    runlog.emit("Orchestrator", "run_start", input_summary=question, question=question, playbook=playbook,
                memory=memory is not None, sample_mode=sample_mode.mode() == "replay",
                recording=sample_mode.mode() == "record")

    def recalling(question):
        text = memory.context_for(question)
        state["hints"] = [line for line in text.splitlines() if line.startswith("- ")]
        count = len(state["hints"])
        progress(f"memory: recalled {count} hint{'' if count == 1 else 's'}" if count
                 else "memory: nothing recalled yet")
        # hints are shown but not sent in record or replay mode, so recorded prompts match replayed ones
        return "" if sample_mode.mode() else text

    inner = make(context_fn=recalling if memory is not None else None, **({"on_event": progress} if playbook else {}))

    def draft_fn(question, feedback):
        state["round"] += 1
        progress(f"round {state['round']}: drafting")
        runlog.emit("Drafter", "round_start", input_summary=question, round=state["round"])
        try:
            state["draft"] = inner(question, feedback)
        except LLMError:
            progress(f"round {state['round']}: model unavailable")
            raise
        return state["draft"]

    def checked(claims):
        verdict = evaluate_fn(claims)
        completeness = getattr(inner, "completeness", None)  # playbook mode: is each named code backed by detail?
        gaps = completeness(state["draft"]) if completeness and state["draft"] else []
        if gaps:
            verdict = Verdict(list(verdict.rejections) + gaps)
        reasons = [r.reason for r in verdict.rejections]
        reasons += ["unclaimed_number"] * len(unclaimed_numbers(state["draft"].text, claims))
        outcome = f"{len(reasons)} rejected ({', '.join(reasons)})" if reasons else "all verified"
        progress(f"round {state['round']}: {len(claims)} claims checked, {outcome}")
        runlog.emit("Evaluator", "evaluate_claims", input_summary=f"{len(claims)} claims", output_summary=outcome,
                    status="rejected" if reasons else "verified", target_agent="Drafter" if reasons else None,
                    round=state["round"], reasons=", ".join(reasons))
        return verdict

    def model_turn(item):
        state["turn"] += 1
        try:
            seconds = f"{float(item['latency_ms']) / 1000:.1f}s"
        except (TypeError, ValueError):
            seconds = "?"
        progress(f"  model turn {state['turn']} ({seconds}) {item['routed_model']} {item['receipt_id']}")

    with ReceiptCollector(model_turn) as receipts:
        result = run_investigation(question, draft_fn, checked)
    queries = tools_zetaris.query_log()
    runlog.emit("Orchestrator", "run_end", output_summary=f"{result.rounds} round(s)",
                status="verified" if result.text else "gave_up", rounds=result.rounds,
                claims=len(result.claims))
    if state["hints"]:
        out(f"Recalled from earlier sessions ({len(state['hints'])}):")
        for hint in state["hints"]:
            out(f"  {hint}")
    out(format_trace(result, receipts.items, verbose, tokens=receipts.tokens))
    summary = observability.summarize(queries, llm.call_log())
    if observability.format_line(summary):
        out(observability.format_line(summary))
    if memory is not None and result.text:
        try:
            ids = memory.learn(result.claims, [])
            report = getattr(memory, "last_learn", None)
            if report:
                memory_line = (f"Memory: {report['new']} new, {report['reinforced']} reinforced, "
                               f"{len(report['changed'])} changed")
                out(memory_line)
                for change in report["changed"]:
                    out(f"  changed since last session: {change['was']}  ->  {change['now']}")
            else:
                memory_line = f"Memory: remembered {len(ids)} verified claims"
                out(memory_line)
        except Exception as error:  # memory is optional; never lose a verified answer to it
            out(f"note: memory unavailable ({error}); the answer above is unaffected")
    if log:
        out(f"note: could not write the run log ({log.error})" if log.failed else f"Log: {log.path}")
    report_path = None
    if report_dir:
        data = _report_data(result, question, state, receipts, queries, timeline, memory_line,
                            run_sql or tools_zetaris.peek, now)
        data["observability"] = summary
        path = report_path = run_report.write(data, report_dir, now=now)
        out(f"Report: {path}" if path else f"note: could not write the report (nothing could be written to {report_dir})")
    if as_json:
        real_out(json.dumps(_json_document(result, question, receipts, summary, log, report_path)))
    return result


def _json_document(result, question, receipts, summary, log, report_path):
    return {"question": question, "verified": bool(result.text), "answer": result.text, "rounds": result.rounds,
            "claims": [{"text": c.text, "value": c.value, "sql": c.citation.sql, "column": c.citation.column,
                        "row": c.citation.row} for c in result.claims],
            "unsupported": [] if result.text else [{"text": c.text, "value": c.value} for c in result.unsupported],
            "trace": result.trace, "receipts": receipts.items, "tokens": receipts.tokens, "observability": summary,
            "log": log.path if log and not log.failed else None, "report": report_path}


def ask(question, memory=None, *, make_draft=make_draft_fn, make_playbook=make_playbook_draft_fn,
        run_investigation=investigate, evaluate_fn=evaluate, out=print, progress=None,
        verbose=False, playbook=False, log_dir=None, now=None, report_dir=None, run_sql=None, timeout=None,
        as_json=False):
    log = RunLog(log_dir, now=now) if log_dir else None
    if log:
        runlog.start(log)
    deadline.start(timeout)
    try:
        return _ask(question, memory, make_draft=make_draft, make_playbook=make_playbook,
                    run_investigation=run_investigation, evaluate_fn=evaluate_fn, out=out, progress=progress,
                    verbose=verbose, playbook=playbook, log=log, report_dir=report_dir,
                    run_sql=run_sql, now=now, as_json=as_json)
    finally:
        deadline.stop()
        if log:
            runlog.stop()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m agent.main", description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--memory-dir", default=".hmem")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="show each rejection's reason and detail")
    parser.add_argument("--report", action="store_true", help="also write a self-contained HTML report of the run")
    parser.add_argument("--report-dir", default="reports", help="where the HTML report is written")
    parser.add_argument("--log-dir", default="logs", help="write a structured run log here")
    parser.add_argument("--no-log", action="store_true", help="do not write a run log")
    parser.add_argument("--sample", action="store_true",
                        help="offline demonstration: replay recordings of a real run (also SAMPLE_MODE=true)")
    parser.add_argument("--sample-dir", default="sample_data", help="where the recordings are read from")
    parser.add_argument("--record", metavar="DIR", help="record this live run's Zetaris results and model replies")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="print one JSON document on stdout (progress stays on stderr)")
    parser.add_argument("--timeout", type=int, default=600, metavar="SECONDS",
                        help="stop the run after this many seconds with no answer (default 600; 0 = no limit)")
    parser.add_argument("--playbook", action="store_true",
                        help="code runs the fixed queries and the model only explains (faster, more reliable)")
    return parser.parse_args(argv)


def missing_settings(env=None):
    env = os.environ if env is None else env
    return [name for name in REQUIRED_SETTINGS if not str(env.get(name, "")).strip()]


BANNER = ("SAMPLE MODE: Zetaris results and the model reply are replayed from recordings of a real run; "
          "the evaluator, retry loop and memory run for real. Run without --sample for a live run.")


def _refuse(args, problem):
    """An out-of-scope question: say so, log it, and run nothing."""
    print(f"Out of scope: {problem}", file=sys.stderr)
    if args.as_json:
        print(json.dumps({"question": args.question, "verified": False, "reason": "out_of_scope", "message": problem}))
    if not args.no_log:
        log = RunLog(args.log_dir)
        runlog.start(log)
        try:
            runlog.emit("Orchestrator", "scope_check", input_summary=args.question, output_summary=problem,
                        status="refused")
        finally:
            runlog.stop()
    return 2


def main(argv=None, env=None, ask_fn=None):
    args = parse_args(argv)
    if len(args.question) > MAX_QUESTION:
        print(f"The question is too long ({len(args.question)} characters; the limit is {MAX_QUESTION}).", file=sys.stderr)
        return 2
    problem = scope_problem(args.question)
    if problem:
        return _refuse(args, problem)
    env_map = os.environ if env is None else env
    sample = args.sample or sample_mode.enabled(env_map)
    if sample and args.record:
        print("--record cannot be combined with sample mode: recordings must come from a real run.",
              file=sys.stderr)
        return 2
    if not sample:
        missing = missing_settings(env)
        if missing:
            print(f"Missing settings: {', '.join(missing)}.\n"
                  "Copy .env.example to .env and fill them in (see Setup in the README), "
                  "or try SAMPLE_MODE=true for an offline demonstration.", file=sys.stderr)
            return 2
    try:
        if sample:
            sample_mode.start_replay(args.sample_dir)
            print(BANNER, file=sys.stderr)
        elif args.record:
            sample_mode.start_recording(args.record)
        memory = None if args.no_memory else Memory(args.memory_dir)
        run = ask_fn or ask
        result = run(args.question, memory, verbose=args.verbose, playbook=args.playbook,
                     log_dir=None if args.no_log else args.log_dir, timeout=args.timeout, as_json=args.as_json,
                     report_dir=args.report_dir if args.report else None)
    except SampleModeError as error:
        print(str(error), file=sys.stderr)
        return 2
    finally:
        sample_mode.stop()
    return 0 if result.text else 1


if __name__ == "__main__":
    sys.exit(main())
