"""Command line: python -m agent.main "<question>" [--memory-dir .hmem] [--no-memory] [--verbose] [--playbook]

Runs the investigation (draft -> evaluate -> retry), prints the trace with the
model receipts, and remembers the claims that were verified.
"""
import argparse
import logging
import os
import re
import sys

from agent import runlog, sample_mode
from agent.drafter import make_draft_fn
from agent.evaluator import Verdict, evaluate, unclaimed_numbers
from agent.llm import LLMError
from agent.loop import investigate
from agent.playbook import make_playbook_draft_fn
from agent.runlog import RunLog
from agent.sample_mode import SampleModeError
from agent.memory import Memory

REQUIRED_SETTINGS = ("ZETARIS_REST_URL", "ZETARIS_ORG_ID", "ZETARIS_API_KEY",
                     "MODEL_BASE_URL", "MODEL_API_KEY", "NEMOTRON_MODEL")
_RECEIPT = re.compile(r"routed_model=(\S+) receipt_id=(\S+)")
_LATENCY = re.compile(r"latency_ms=(\S+)")


class ReceiptCollector(logging.Handler):
    """Context manager that collects routed_model / receipt_id from agent.llm's log lines."""

    def __init__(self, on_receipt=None):
        super().__init__(logging.INFO)
        self.items = []
        self.on_receipt = on_receipt  # called with routed_model, receipt_id and latency_ms

    def emit(self, record):
        message = record.getMessage()
        match = _RECEIPT.search(message)
        if match:
            receipt_id = None if match.group(2) == "None" else match.group(2)
            self.items.append({"routed_model": match.group(1), "receipt_id": receipt_id})
            if self.on_receipt:
                latency = _LATENCY.search(message)
                self.on_receipt({**self.items[-1], "latency_ms": latency.group(1) if latency else None})

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


def format_trace(result, receipts=(), verbose=False):
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
    return "\n".join(lines)


def _stderr(message):
    print(message, file=sys.stderr, flush=True)  # sys.stderr is looked up at call time


def _ask(question, memory, *, make_draft, make_playbook, run_investigation, evaluate_fn, out, progress,
         verbose, playbook, log):
    progress = progress or _stderr
    make = make_playbook if playbook else make_draft
    state = {"round": 0, "turn": 0, "draft": None, "hints": []}
    runlog.emit("Orchestrator", "run_start", input_summary=question, question=question, playbook=playbook,
                memory=memory is not None, sample_mode=sample_mode.mode() == "replay",
                recording=sample_mode.mode() == "record")

    def recalling(question):
        text = memory.context_for(question)
        state["hints"] = [line for line in text.splitlines() if line.startswith("- ")]
        count = len(state["hints"])
        progress(f"memory: recalled {count} hint{'' if count == 1 else 's'}" if count
                 else "memory: nothing recalled yet")
        return text

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
    runlog.emit("Orchestrator", "run_end", output_summary=f"{result.rounds} round(s)",
                status="verified" if result.text else "gave_up", rounds=result.rounds,
                claims=len(result.claims))
    if state["hints"]:
        out(f"Recalled from earlier sessions ({len(state['hints'])}):")
        for hint in state["hints"]:
            out(f"  {hint}")
    out(format_trace(result, receipts.items, verbose))
    if memory is not None and result.text:
        try:
            ids = memory.learn(result.claims, [])
            report = getattr(memory, "last_learn", None)
            if report:
                out(f"Memory: {report['new']} new, {report['reinforced']} reinforced, "
                    f"{len(report['changed'])} changed")
                for change in report["changed"]:
                    out(f"  changed since last session: {change['was']}  ->  {change['now']}")
            else:
                out(f"Memory: remembered {len(ids)} verified claims")
        except Exception as error:  # memory is optional; never lose a verified answer to it
            out(f"note: memory unavailable ({error}); the answer above is unaffected")
    if log:
        out(f"note: could not write the run log ({log.error})" if log.failed else f"Log: {log.path}")
    return result


def ask(question, memory=None, *, make_draft=make_draft_fn, make_playbook=make_playbook_draft_fn,
        run_investigation=investigate, evaluate_fn=evaluate, out=print, progress=None,
        verbose=False, playbook=False, log_dir=None, now=None):
    log = RunLog(log_dir, now=now) if log_dir else None
    if log:
        runlog.start(log)
    try:
        return _ask(question, memory, make_draft=make_draft, make_playbook=make_playbook,
                    run_investigation=run_investigation, evaluate_fn=evaluate_fn, out=out, progress=progress,
                    verbose=verbose, playbook=playbook, log=log)
    finally:
        if log:
            runlog.stop()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m agent.main", description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--memory-dir", default=".hmem")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="show each rejection's reason and detail")
    parser.add_argument("--log-dir", default="logs", help="write a structured run log here")
    parser.add_argument("--no-log", action="store_true", help="do not write a run log")
    parser.add_argument("--sample", action="store_true",
                        help="offline demonstration: replay recordings of a real run (also SAMPLE_MODE=true)")
    parser.add_argument("--sample-dir", default="sample_data", help="where the recordings are read from")
    parser.add_argument("--record", metavar="DIR", help="record this live run's Zetaris results and model replies")
    parser.add_argument("--playbook", action="store_true",
                        help="code runs the fixed queries and the model only explains (faster, more reliable)")
    return parser.parse_args(argv)


def missing_settings(env=None):
    env = os.environ if env is None else env
    return [name for name in REQUIRED_SETTINGS if not str(env.get(name, "")).strip()]


BANNER = ("SAMPLE MODE: Zetaris results and the model reply are replayed from recordings of a real run; "
          "the evaluator, retry loop and memory run for real. Run without --sample for a live run.")


def main(argv=None, env=None, ask_fn=None):
    args = parse_args(argv)
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
                     log_dir=None if args.no_log else args.log_dir)
    except SampleModeError as error:
        print(str(error), file=sys.stderr)
        return 2
    finally:
        sample_mode.stop()
    return 0 if result.text else 1


if __name__ == "__main__":
    sys.exit(main())
