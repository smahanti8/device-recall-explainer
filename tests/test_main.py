"""Tests for agent/main.py (written before the implementation).

    python -m agent.main "<question>" [--memory-dir .hmem] [--no-memory]

    ask(question, memory=None, *, make_draft=make_draft_fn, run_investigation=investigate,
        evaluate_fn=evaluate, out=print) -> Result
        wires memory.context_for into the drafter (context_fn), runs the investigation,
        prints the trace, and on success calls memory.learn(result.claims, []).
        A memory failure is reported ("memory unavailable") and never stops the run.
    format_trace(result, receipts=()) -> str      one block per round, then the answer or what was unsupported
    ReceiptCollector()                             context manager: collects routed_model / receipt_id log lines
    parse_args(argv) -> namespace(question, memory_dir, no_memory)
    format_trace(result, receipts=(), verbose=False): with verbose=True each round also lists every
        rejection as "reason: detail" (needs the loop's trace entries to carry "details").
    parse_args: --verbose and --playbook are flags (default off).
    ask(..., playbook=False, make_playbook=make_playbook_draft_fn): with playbook=True the draft function comes
        from make_playbook(context_fn=...) (code runs the queries, the model explains) instead of make_draft;
        the memory hook, progress lines, trace and learning are identical in both modes.
    ask(..., progress=None): live progress lines while the run happens. progress(message) is called for
        "round N: drafting", "round N: K claims checked, R rejected (reasons)" or "all verified",
        "round N: model unavailable", and "model turn T (S.Ss) <routed_model> <receipt_id>".
        Default: print to the current sys.stderr, so stdout stays the trace only.
    investigate() now also returns the accepted claims as Result.claims (see test_evaluator.py).

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import contextlib
import io
import logging
import unittest

from agent.evaluator import Citation, Claim, Rejection, Verdict
from agent.llm import LLMError
from agent.loop import Draft, Result
from agent.main import ReceiptCollector, ask, format_trace, parse_args

CLAIM = Claim("DSY recall records in 2024", 54, Citation("SELECT 1 AS n", "n"))
TRACE = [
    {"round": 1, "claims": 2, "rejected": 2, "reasons": ["no_citation", "unclaimed_number"],
     "details": ["claim has no SQL citation", "the answer text states 65% but no claim has that value"]},
    {"round": 2, "claims": 2, "rejected": 0, "reasons": []},
]
GOOD = Result("DSY had 54 recall records.", [], 2, TRACE, claims=[CLAIM])
GAVE_UP = Result(None, [Claim("model unavailable: 502 provider_deadline", None)], 3, TRACE[:1])


class FakeMemory:
    def __init__(self, context="", fail_on_learn=False):
        self.context, self.fail_on_learn, self.learned = context, fail_on_learn, []

    def context_for(self, question):
        return self.context

    def learn(self, claims, rejections):
        if self.fail_on_learn:
            raise RuntimeError("H-MEM: node missing")
        self.learned.append((claims, rejections))
        return [f"mem-{i}" for i, _ in enumerate(claims, 1)]


def run(result, memory=None):
    lines, seen = [], {}

    def make_draft(context_fn=None, **kwargs):
        seen["context_fn"] = context_fn
        return "draft_fn"
    returned = ask("why?", memory, make_draft=make_draft, run_investigation=lambda q, d, e: result,
                   evaluate_fn="evaluate", out=lines.append)
    return returned, lines, seen


class Ask(unittest.TestCase):
    def test_memory_context_is_wired_into_the_drafter(self):
        _, _, seen = run(GOOD, FakeMemory(context="HINTS"))
        self.assertEqual(seen["context_fn"]("why?"), "HINTS")

    def test_no_memory_means_no_context_hook(self):
        _, _, seen = run(GOOD, None)
        self.assertIsNone(seen["context_fn"])

    def test_a_successful_run_is_remembered(self):
        memory = FakeMemory()
        result, lines, _ = run(GOOD, memory)
        self.assertIs(result, GOOD)
        self.assertEqual(memory.learned, [([CLAIM], [])])
        self.assertIn("remembered 1", "\n".join(lines))

    def test_a_run_that_gave_up_remembers_nothing(self):
        memory = FakeMemory()
        run(GAVE_UP, memory)
        self.assertEqual(memory.learned, [])

    def test_a_memory_failure_never_stops_the_run(self):
        result, lines, _ = run(GOOD, FakeMemory(fail_on_learn=True))
        self.assertIs(result, GOOD)
        self.assertIn("memory unavailable", "\n".join(lines))


def drive(drafts, verdicts, during=None, **ask_kwargs):
    """Run ask() with a stand-in for the loop: one draft_fn and one evaluate_fn call per round."""
    queue, verdict_queue = list(drafts), list(verdicts)

    def make_draft(context_fn=None, **kwargs):
        def draft_fn(question, feedback):
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            if during:
                during()
            return item
        return draft_fn

    def loop_like(question, draft_fn, evaluate_fn):
        for _ in drafts:
            try:
                draft = draft_fn(question, [])
            except LLMError:
                continue
            evaluate_fn(draft.claims)
        return GOOD

    ask_kwargs.setdefault("out", lambda *_: None)
    return ask("why?", None, make_draft=make_draft, run_investigation=loop_like,
               evaluate_fn=lambda claims: verdict_queue.pop(0), **ask_kwargs)


TWO = Draft("a", [CLAIM, CLAIM])
REJECT_ONE = Verdict([Rejection(CLAIM, "no_citation", "claim has no SQL citation")])


class Progress(unittest.TestCase):
    def test_each_round_is_announced_and_its_verdict_shown(self):
        messages = []
        drive([TWO, TWO], [REJECT_ONE, Verdict()], progress=messages.append)
        text = "\n".join(messages)
        for needle in ("round 1: drafting", "round 1: 2 claims checked, 1 rejected (no_citation)",
                       "round 2: drafting", "round 2: 2 claims checked, all verified"):
            self.assertIn(needle, text)

    def test_a_model_outage_is_announced(self):
        messages = []
        drive([LLMError("502 provider_deadline"), TWO], [Verdict()], progress=messages.append)
        text = "\n".join(messages)
        self.assertIn("round 1: model unavailable", text)
        self.assertIn("round 2: drafting", text)

    def test_model_turns_are_reported_as_they_happen(self):
        messages = []
        log = logging.getLogger("agent.llm")

        def a_model_call():
            log.info("routed_model=%s receipt_id=%s provider_model=%s egress=%s latency_ms=%s",
                     "nvidia/n", "rcpt_1", "meterless/auto", "nvidia", 2500)
        drive([TWO], [Verdict()], during=a_model_call, progress=messages.append)
        line = next(m for m in messages if "model turn 1" in m)
        for needle in ("2.5s", "nvidia/n", "rcpt_1"):
            self.assertIn(needle, line)

    def test_by_default_progress_goes_to_stderr_not_to_the_trace(self):
        printed, errors = [], io.StringIO()
        with contextlib.redirect_stderr(errors):
            drive([TWO], [Verdict()], out=printed.append)
        self.assertIn("round 1: drafting", errors.getvalue())
        self.assertNotIn("round 1: drafting", "\n".join(printed))


class PlaybookMode(unittest.TestCase):
    def go(self, **ask_kwargs):
        calls = []

        def make_draft(context_fn=None, **kwargs):
            calls.append(("exploratory", context_fn))
            return "draft_fn"

        def make_playbook(context_fn=None, **kwargs):
            calls.append(("playbook", context_fn))
            return "draft_fn"
        ask("why?", ask_kwargs.pop("memory", None), make_draft=make_draft, make_playbook=make_playbook,
            run_investigation=lambda q, d, e: GOOD, evaluate_fn="evaluate", out=lambda *_: None,
            progress=lambda m: None, **ask_kwargs)
        return calls

    def test_the_exploratory_drafter_is_the_default(self):
        self.assertEqual([name for name, _ in self.go()], ["exploratory"])

    def test_playbook_mode_swaps_the_drafter_and_keeps_the_memory_hook(self):
        memory = FakeMemory(context="HINTS")
        (name, context_fn), = self.go(playbook=True, memory=memory)
        self.assertEqual(name, "playbook")
        self.assertEqual(context_fn("why?"), "HINTS")

    def test_playbook_mode_still_learns_from_a_verified_run(self):
        memory = FakeMemory()
        self.go(playbook=True, memory=memory)
        self.assertEqual(memory.learned, [([CLAIM], [])])


class Trace(unittest.TestCase):
    def test_each_round_and_the_answer_are_shown(self):
        text = format_trace(GOOD)
        for needle in ("Round 1", "no_citation", "unclaimed_number", "Round 2", "DSY had 54 recall records."):
            self.assertIn(needle, text)

    def test_verbose_adds_each_rejection_detail(self):
        quiet, loud = format_trace(GOOD), format_trace(GOOD, verbose=True)
        self.assertNotIn("claim has no SQL citation", quiet)
        self.assertIn("no_citation: claim has no SQL citation", loud)
        self.assertIn("unclaimed_number: the answer text states 65% but no claim has that value", loud)

    def test_verbose_copes_with_trace_entries_that_have_no_details(self):
        old = Result("ok", [], 1, [{"round": 1, "claims": 1, "rejected": 1, "reasons": ["no_citation"]}], claims=[CLAIM])
        self.assertIn("no_citation", format_trace(old, verbose=True))

    def test_a_run_that_gave_up_says_what_it_could_not_support(self):
        text = format_trace(GAVE_UP)
        self.assertIn("No verified answer", text)
        self.assertIn("502 provider_deadline", text)

    def test_receipts_are_listed_when_given(self):
        text = format_trace(GOOD, [{"routed_model": "nvidia/n", "receipt_id": "rcpt_1"}])
        self.assertIn("rcpt_1", text)
        self.assertIn("nvidia/n", text)

    def test_receipt_collector_reads_the_llm_log_lines(self):
        log = logging.getLogger("agent.llm")
        log.setLevel(logging.INFO)
        with ReceiptCollector() as receipts:
            log.info("routed_model=%s receipt_id=%s provider_model=%s egress=%s latency_ms=%s",
                     "nvidia/n", "rcpt_1", "meterless/auto", "nvidia", 12)
        self.assertEqual(receipts.items, [{"routed_model": "nvidia/n", "receipt_id": "rcpt_1"}])


class Cli(unittest.TestCase):
    def test_defaults_and_flags(self):
        args = parse_args(["why?"])
        self.assertEqual((args.question, args.memory_dir, args.no_memory), ("why?", ".hmem", False))
        self.assertTrue(parse_args(["why?", "--no-memory"]).no_memory)
        self.assertEqual(parse_args(["why?", "--memory-dir", "/tmp/m"]).memory_dir, "/tmp/m")
        self.assertFalse(args.verbose)
        self.assertTrue(parse_args(["why?", "--verbose"]).verbose)
        self.assertFalse(args.playbook)
        self.assertTrue(parse_args(["why?", "--playbook"]).playbook)


if __name__ == "__main__":
    unittest.main()
