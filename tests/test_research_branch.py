"""Tests for the evidence-completeness branch of playbook mode (written before the implementation).

The branch, matching the rubric's pattern (evaluator detects missing evidence, research retrieves new context
from Zetaris, the drafter reruns, the evaluator approves):

    Evaluator: a draft that NAMES a product code with a drill-down available but cites none of its detail
               (records, events, firms, root cause, top firm) is rejected as `missing_evidence`.
    Researcher: on that rejection the next draft first fetches the code's detail facts from Zetaris.
    Drafter:   redrafts with the extended facts. Each code is flagged at most once, so a missing table can never loop.

    agent/playbook.py
      DRILLDOWN_CODES          10 codes: RECORD_LEVEL_CODES (DSY, DQO, OEZ, detailed up front) + DQK, DPT, DRE, DWF, OES, DYF, OEQ
      collect_code_facts(code, run_sql=None, start_id=1) -> [Fact]   five detail facts; ValueError for a bad or unknown code
      make_playbook_draft_fn(chat=None, run_sql=None, context_fn=None, on_event=None) -> draft_fn
        draft_fn(question, feedback): rejections with reason "missing_evidence" (claim text "<CODE> detail") make it
            fetch that code's facts before drafting, emit a Researcher "fetch_facts" log event (target Drafter), and call
            on_event("evidence requested: DQK -> 5 facts fetched")
        draft_fn.completeness(draft) -> [Rejection]   the missing_evidence rejections for this draft
    main.ask: in playbook mode adds draft_fn.completeness(draft) to each verdict (logged and shown like any rejection)
              and passes on_event=progress to make_playbook.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import datetime
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from agent import playbook, runlog
from agent.evaluator import Citation, Claim, Rejection, Verdict
from agent.llm import Reply
from agent.loop import Draft, Result, investigate
from agent.main import ask
from agent.playbook import DRILLDOWN_CODES, RECORD_LEVEL_CODES, collect_code_facts, collect_facts, make_playbook_draft_fn


def setUpModule():
    # these tests script plain chat fakes; escalation has its own tests, and another test module's
    # load_dotenv() can put the setting back at import time, so clear it when the tests actually run
    patch = mock.patch.dict(os.environ)
    patch.start()
    os.environ.pop("NEMOTRON_ESCALATION_MODEL", None)
    unittest.addModuleCleanup(patch.stop)


MORE = ("DQK", "DPT", "DRE", "DWF", "OES", "DYF", "OEQ")
TOP5 = ["DSY", "DQO", "OEZ", "DQK", "DPT"]
ROW = {"total": "10", "diff": "5", "change": "2", "top3": "9", "pct": "60.0", "n": "3", "top": "Equipment maintenance"}


def fake_run(seen=None, broken=()):
    """A run_sql stand-in: the ranking query returns the top five codes, everything else a generic row."""
    def run(sql):
        if seen is not None:
            seen.append(sql)
        if any(view in sql for view in broken):
            raise RuntimeError("TABLE_OR_VIEW_NOT_FOUND")
        if "ORDER BY change DESC LIMIT 5" in sql:
            return [{"code": c, "device": f"{c} device"} for c in TOP5]
        return [dict(ROW)]
    return run


def reply(text, ids):
    return json.dumps({"text": text, "facts": ids})


def scripted(*replies):
    calls, queue = [], iter(replies)

    def chat(messages, max_tokens=2000):
        calls.append([dict(m) for m in messages])
        return Reply(next(queue), "nvidia/test", None)
    chat.calls = calls
    return chat


def ids_for(*labels):
    facts = collect_facts(run_sql=fake_run(), top_codes=TOP5)
    by_label = {f.label: f.id for f in facts}
    return [by_label[label] for label in labels]


def draft_naming(text, *labels):
    return Draft(text, [Claim(label, 1, Citation("SELECT 1 AS n", "n")) for label in labels])


class Codes(unittest.TestCase):
    def test_ten_drilldown_codes_and_the_three_detailed_up_front(self):
        self.assertEqual(RECORD_LEVEL_CODES, ("DSY", "DQO", "OEZ"))
        self.assertEqual(DRILLDOWN_CODES, RECORD_LEVEL_CODES + MORE)

    def test_the_first_round_gives_detail_only_for_the_top_three(self):
        labels = [f.label for f in collect_facts(run_sql=fake_run(), top_codes=TOP5)]
        self.assertIn("DQK increase in recall records, 2023 to 2024", labels)
        self.assertIn("DSY recall events in 2024", labels)
        self.assertNotIn("DQK recall events in 2024", labels)


class CodeFacts(unittest.TestCase):
    def test_five_detail_facts_with_continuing_ids_and_their_own_view(self):
        seen = []
        facts = collect_code_facts("DQK", run_sql=fake_run(seen), start_id=12)
        self.assertEqual([f.id for f in facts], ["f12", "f13", "f14", "f15", "f16"])
        self.assertEqual([f.label for f in facts], [
            "DQK recall records in 2024", "DQK recall events in 2024", "DQK recalling firms in 2024",
            "DQK top root cause in 2024", "DQK top recalling firm in 2024"])
        self.assertTrue(all("openfda.recall_dqk_2024_table" in f.citation.sql for f in facts))

    def test_a_bad_or_unknown_code_is_refused(self):
        for code in ("DQK'; DROP VIEW x", "ABC", "", "dqk "):
            with self.subTest(code=code), self.assertRaises(ValueError):
                collect_code_facts(code, run_sql=fake_run())


class Completeness(unittest.TestCase):
    def setUp(self):
        self.check = make_playbook_draft_fn(chat=scripted(), run_sql=fake_run()).completeness

    def test_a_named_code_without_detail_is_rejected_as_missing_evidence(self):
        draft = draft_naming("DSY rose by 53 and DQK by 25.", "DSY recall events in 2024",
                             "DQK increase in recall records, 2023 to 2024")
        (rejection,) = self.check(draft)
        self.assertEqual(rejection.reason, "missing_evidence")
        self.assertEqual(rejection.claim.text, "DQK detail")
        self.assertIn("DQK", rejection.detail)

    def test_codes_with_detail_or_outside_the_drilldown_list_are_not_flagged(self):
        draft = draft_naming("DSY rose, and so did ABC.", "DSY recall events in 2024")
        self.assertEqual(self.check(draft), [])

    def test_each_code_is_flagged_at_most_once(self):
        draft = draft_naming("DQK rose.", "DQK increase in recall records, 2023 to 2024")
        self.assertEqual(len(self.check(draft)), 1)
        self.assertEqual(self.check(draft), [])


class ResearchOnFeedback(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(runlog.stop)
        self.feedback = [Rejection(Claim("DQK detail", None), "missing_evidence",
                                   "the answer names DQK but cites no detail for it")]
        self.used = ids_for("Cardiovascular recall records in 2023", "DSY recall events in 2024")

    def draft(self, broken=(), events=None):
        seen = []
        chat = scripted(reply("one", self.used), reply("two", self.used))
        fn = make_playbook_draft_fn(chat=chat, run_sql=fake_run(seen, broken), on_event=events.append if events is not None else None)
        fn("why?", [])
        before = len(seen)
        fn("why?", self.feedback)
        return chat, seen, before

    def test_the_second_draft_fetches_the_code_and_the_model_sees_the_new_facts(self):
        chat, seen, before = self.draft()
        self.assertTrue(any("recall_dqk_2024_table" in sql for sql in seen[before:]))
        first, second = chat.calls[0][1]["content"], chat.calls[1][1]["content"]
        self.assertNotIn("DQK recall events in 2024", first)
        self.assertIn("DQK recall events in 2024", second)
        self.assertIn("missing_evidence", second)

    def test_the_researcher_leaves_a_log_event_and_a_progress_message(self):
        log = runlog.RunLog(self.dir, now=lambda: datetime.datetime(2026, 10, 25, 12, 0, 0))
        runlog.start(log)
        messages = []
        self.draft(events=messages)
        with open(log.path) as handle:
            events = [json.loads(line) for line in handle]
        (fetch,) = [e for e in events if e["action"] == "fetch_facts"]
        self.assertEqual((fetch["agent_name"], fetch["target_agent"], fetch["status"]), ("Researcher", "Drafter", "success"))
        self.assertIn("DQK", fetch["input_summary"])
        self.assertEqual(fetch["output_summary"], "5 facts")
        self.assertTrue(any("evidence requested: DQK" in m and "5 facts" in m for m in messages))

    def test_a_missing_table_does_not_break_the_draft_or_loop_the_run(self):
        log = runlog.RunLog(self.dir, now=lambda: datetime.datetime(2026, 10, 25, 12, 0, 0))
        runlog.start(log)
        chat, _, _ = self.draft(broken=("recall_dqk_2024_table",))
        self.assertEqual(len(chat.calls), 2)
        self.assertNotIn("DQK recall events in 2024", chat.calls[1][1]["content"])
        with open(log.path) as handle:
            (fetch,) = [e for e in map(json.loads, handle) if e["action"] == "fetch_facts"]
        self.assertEqual((fetch["status"], fetch["output_summary"]), ("error", "0 facts"))

    def test_the_system_prompt_explains_missing_evidence(self):
        self.assertIn("missing_evidence", playbook.SYSTEM)

    def test_the_prompt_does_not_invite_characterising_the_records_to_events_ratio(self):
        """A live run wrote "43 records across 21 events, reflecting many events per record", which is wrong prose
        the number checks cannot catch. The prompt must not invite such descriptions."""
        self.assertNotIn("few events", playbook.SYSTEM)
        self.assertIn("only when its events fact equals 1", playbook.SYSTEM)
        self.assertIn("do not describe the ratio", playbook.SYSTEM)


class InTheLoop(unittest.TestCase):
    """The whole interaction through the real loop: reject, research, redraft, verify."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(runlog.stop)

    def run_branch(self):
        names_dqk = draft_naming("DSY and DQK drove it.", "DSY recall events in 2024", "DQK increase in recall records, 2023 to 2024")
        complete = draft_naming("DSY and DQK drove it, with DQK detail.", "DSY recall events in 2024", "DQK recall events in 2024")
        drafts, events, progress = iter([names_dqk, complete]), [], []
        flagged = []

        def make_playbook(context_fn=None, on_event=None, **kwargs):
            events.append(on_event)

            def draft_fn(question, feedback):
                return next(drafts)

            def completeness(draft):
                if flagged or "DQK recall events in 2024" in [c.text for c in draft.claims]:
                    return []
                flagged.append("DQK")
                return [Rejection(Claim("DQK detail", None), "missing_evidence", "the answer names DQK but cites no detail")]
            draft_fn.completeness = completeness
            return draft_fn
        result = ask("why?", None, playbook=True, make_playbook=make_playbook, run_investigation=investigate,
                     evaluate_fn=lambda claims: Verdict(), out=lambda *_: None, progress=progress.append,
                     log_dir=self.dir, now=lambda: datetime.datetime(2026, 10, 25, 12, 0, 0))
        with open(os.path.join(self.dir, "run-20261025-120000.jsonl")) as handle:
            return result, progress, events, [json.loads(line) for line in handle]

    def test_round_one_is_rejected_for_missing_evidence_and_round_two_is_verified(self):
        result, progress, _, _ = self.run_branch()
        self.assertEqual(result.rounds, 2)
        self.assertIsNotNone(result.text)
        self.assertEqual([t["reasons"] for t in result.trace], [["missing_evidence"], []])
        self.assertTrue(any("missing_evidence" in line for line in progress))

    def test_the_log_shows_the_evaluator_handing_the_gap_back_to_the_drafter(self):
        _, _, _, events = self.run_branch()
        rejected = [e for e in events if e["action"] == "evaluate_claims" and e["status"] == "rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["target_agent"], "Drafter")
        self.assertIn("missing_evidence", rejected[0]["output_summary"])

    def test_ask_hands_its_progress_function_to_the_playbook(self):
        _, progress, handed, _ = self.run_branch()
        self.assertTrue(callable(handed[0]))
        handed[0]("evidence requested: DQK -> 5 facts fetched")
        self.assertIn("evidence requested: DQK -> 5 facts fetched", progress)


if __name__ == "__main__":
    unittest.main()
