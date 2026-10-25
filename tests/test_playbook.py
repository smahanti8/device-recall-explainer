"""Tests for agent/playbook.py (written before the implementation).

Option B: code runs the verified queries and turns each result into a numbered Fact;
the model only writes the explanation and names the facts it used.

    Fact(id, label, value, citation, note="")      id like "f1"; citation is an agent.evaluator.Citation
    collect_facts(run_sql=None, top_codes=None, top_n=5) -> list[Fact]
        top_codes=None ranks cardiovascular codes by 2023->2024 increase (one extra query)
        Labels (the evaluator's scope check reads them, so they name their scope):
          "Cardiovascular recall records in 2023" / "... in 2024"
          "Increase in cardiovascular recall records, 2023 to 2024"
          "<CODE> increase in recall records, 2023 to 2024"           for every top code
          "Increase from the top 3 cardiovascular product codes combined"
          "Share of the cardiovascular increase from the top 3 product codes (%)"
          "<CODE> recall records in 2024", "<CODE> recall events in 2024",
          "<CODE> recalling firms in 2024", "<CODE> top root cause in 2024",
          "<CODE> top recalling firm in 2024"          only for DSY, DQO, OEZ (record-level views exist)
    make_playbook_draft_fn(chat=None, run_sql=None, context_fn=None) -> draft_fn(question, feedback) -> Draft
        facts are collected once per draft_fn; the model replies {"text": "...", "facts": ["f1", ...]}
        and the claims are the named facts, cited with the fact's own SQL (unknown ids are ignored)

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import collections
import copy
import json
import os
import re
import unittest
from unittest import mock

from dotenv import load_dotenv

from agent.evaluator import Claim, Rejection, evaluate
from agent.llm import Reply
from agent.loop import investigate
from agent.playbook import collect_facts, make_playbook_draft_fn

load_dotenv()
CODES = ["DSY", "ABC", "OEZ"]  # ABC has no record-level view


def setUpModule():
    # these tests script plain chat fakes; escalation has its own tests, and another test module's
    # load_dotenv() can put the setting back at import time, so clear it when the tests actually run
    patch = mock.patch.dict(os.environ)
    patch.start()
    os.environ.pop("NEMOTRON_ESCALATION_MODEL", None)
    unittest.addModuleCleanup(patch.stop)


class AnyColumn(dict):
    """A row whose every column reads "1", so tests do not depend on the playbook's column names."""
    def __missing__(self, key):
        return "1"


def fresh_run():
    def run(sql):
        run.seen.append(sql)
        return [AnyColumn()]
    run.seen = []
    return run


def ranked_run(codes=("DSY", "DQO")):
    """Like fresh_run, but the ranking query (the only one selecting device_name) returns these codes."""
    def run(sql):
        run.seen.append(sql)
        if "device_name" in sql:
            return [AnyColumn(code=code, device="device") for code in codes]
        return [AnyColumn()]
    run.seen = []
    return run


def scripted(*replies):
    calls, queue = [], iter(replies)

    def chat(messages, max_tokens=2000):
        calls.append(copy.deepcopy(messages))
        return Reply(next(queue), "nvidia/test", None)
    chat.calls = calls
    return chat


def reply(text, facts):
    return json.dumps({"text": text, "facts": facts})


class CollectFacts(unittest.TestCase):
    def test_no_statement_references_the_same_view_twice(self):
        run = fresh_run()
        collect_facts(run_sql=run, top_codes=CODES)
        for sql in run.seen:
            views = collections.Counter(re.findall(r"openfda\.[a-z0-9_]+", sql))
            self.assertTrue(all(n == 1 for n in views.values()), sql)

    def test_every_fact_takes_its_value_from_a_query_it_ran(self):
        run = fresh_run()
        facts = collect_facts(run_sql=run, top_codes=CODES)
        self.assertTrue(facts)
        self.assertEqual(len({f.id for f in facts}), len(facts))
        for fact in facts:
            self.assertIn(fact.citation.sql, run.seen)
            self.assertEqual(str(fact.value).replace(".0", ""), "1")

    def test_record_level_facts_exist_only_for_codes_with_a_view(self):
        facts = collect_facts(run_sql=fresh_run(), top_codes=CODES)
        event_labels = [f.label for f in facts if "events" in f.label]
        self.assertEqual(sorted(label.split()[0] for label in event_labels), ["DSY", "OEZ"])
        increases = [f.label for f in facts if "increase in recall records" in f.label]
        self.assertEqual(sorted(label.split()[0] for label in increases), ["ABC", "DSY", "OEZ"])

    def test_the_evaluator_accepts_every_fact_as_a_claim(self):
        """Labels name their scope (cardiovascular, product codes) and the SQL backs it."""
        run = fresh_run()
        facts = collect_facts(run_sql=run, top_codes=CODES)
        claims = [Claim(f.label, f.value, f.citation) for f in facts]
        verdict = evaluate(claims, run_sql=run)
        self.assertTrue(verdict.ok, [(r.reason, r.claim.text) for r in verdict.rejections])


class PlaybookDraft(unittest.TestCase):
    def draft_fn(self, chat, run=None):
        run = run or fresh_run()
        return make_playbook_draft_fn(chat=chat, run_sql=run)

    def test_the_model_sees_every_fact_with_its_id_and_label(self):
        chat = scripted(reply("ok", ["f1"]))
        self.draft_fn(chat)("why?", [])
        shown = json.dumps(chat.calls[0])
        facts = collect_facts(run_sql=fresh_run())
        self.assertTrue(all(f.id in shown and f.label in shown for f in facts))
        self.assertIn("only", chat.calls[0][0]["content"].lower())  # the system prompt restricts it to these facts

    def test_claims_are_built_from_the_named_facts_with_the_facts_own_sql(self):
        run = fresh_run()
        facts = collect_facts(run_sql=run)
        draft = self.draft_fn(scripted(reply("text", [facts[0].id, facts[1].id, "f99"])), run)("why?", [])
        self.assertEqual([(c.text, c.citation) for c in draft.claims],
                         [(f.label, f.citation) for f in facts[:2]])  # f99 is ignored

    def test_previous_rejections_reach_the_model(self):
        rejection = Rejection(Claim("x", 80), "unclaimed_number", "the answer text states 178 but no claim has it")
        chat = scripted(reply("ok", ["f1"]))
        self.draft_fn(chat)("why?", [rejection])
        self.assertIn("unclaimed_number", json.dumps(chat.calls[0]))

    def test_a_clean_answer_passes_in_round_one(self):
        run = fresh_run()
        facts = collect_facts(run_sql=run)
        chat = scripted(reply(f"{facts[0].label} was 1.", [facts[0].id]))
        result = investigate("why?", self.draft_fn(chat, run), lambda claims: evaluate(claims, run_sql=run))
        self.assertEqual(result.rounds, 1)
        self.assertIsNotNone(result.text)

    def test_a_number_with_no_fact_is_rejected_then_fixed(self):
        run = fresh_run()
        facts = collect_facts(run_sql=run)
        chat = scripted(reply("About 178 devices were affected.", [facts[0].id]),
                        reply(f"{facts[0].label} was 1.", [facts[0].id]))
        result = investigate("why?", self.draft_fn(chat, run), lambda claims: evaluate(claims, run_sql=run))
        self.assertEqual(result.trace[0]["reasons"], ["unclaimed_number"])
        self.assertEqual(result.rounds, 2)


class PlaybookWhy(unittest.TestCase):
    """The answer must explain why, not just report totals: records vs events, root cause, firm."""

    def setUp(self):
        self.run = ranked_run()
        facts = collect_facts(run_sql=self.run)
        ids = {f.label: f.id for f in facts}
        self.basic = [facts[0].id]
        self.why = [ids["DSY recall events in 2024"], ids["DSY top root cause in 2024"]]

    def draft(self, chat, run=None):
        return make_playbook_draft_fn(chat=chat, run_sql=run or self.run)("why?", [])

    def test_prompt_asks_for_records_vs_events_root_cause_and_firm(self):
        chat = scripted(reply("ok", self.basic + self.why))
        self.draft(chat)
        system = chat.calls[0][0]["content"].lower()
        for needle in ("each product code", "events", "root cause", "recalling firm", "single event"):
            self.assertIn(needle, system)

    def test_an_answer_without_the_why_facts_gets_one_correction(self):
        chat = scripted(reply("Totals only.", self.basic), reply("With the why.", self.basic + self.why))
        draft = self.draft(chat)
        self.assertEqual(len(chat.calls), 2)
        correction = chat.calls[1][-1]["content"].lower()
        self.assertIn("events", correction)
        self.assertIn("root cause", correction)
        self.assertIn("DSY recall events in 2024", [c.text for c in draft.claims])
        self.assertIn("DSY top root cause in 2024", [c.text for c in draft.claims])

    def test_the_correction_is_asked_only_once(self):
        chat = scripted(reply("a", self.basic), reply("b", self.basic), reply("c", self.basic))
        draft = self.draft(chat)
        self.assertEqual(len(chat.calls), 2)
        self.assertEqual(draft.text, "b")  # the second answer is returned even if it still lacks the why

    def test_no_correction_when_the_why_is_already_cited(self):
        chat = scripted(reply("ok", self.basic + self.why))
        self.draft(chat)
        self.assertEqual(len(chat.calls), 1)

    def test_no_correction_without_record_level_facts(self):
        run = ranked_run(codes=("ABC",))  # no record-level view for ABC
        chat = scripted(reply("ok", self.basic))
        self.draft(chat, run)
        self.assertEqual(len(chat.calls), 1)


class FactIdMarkers(unittest.TestCase):
    """The model sometimes writes "(f1)" after a number. Ids are for the claims, not for readers."""

    def answer(self, raw_text, facts_used=1):
        run = fresh_run()
        facts = collect_facts(run_sql=run)
        chat = scripted(reply(raw_text, [f.id for f in facts[:facts_used]]))
        return make_playbook_draft_fn(chat=chat, run_sql=run)("why?", []), chat

    def test_bracketed_fact_ids_are_removed_from_the_text(self):
        draft, _ = self.answer("Records rose from 281 (f1) to 482 (f2), up 201 [f3] in total (f4, f5).")
        self.assertEqual(draft.text, "Records rose from 281 to 482, up 201 in total.")

    def test_text_without_markers_is_unchanged(self):
        text = "Class 2 devices, the f-series and a 5f2 ratio are left alone."
        self.assertEqual(self.answer(text)[0].text, text)

    def test_ids_inside_mixed_parentheses_are_removed(self):
        for raw, expected in (
            ("DSY rose (+53, f4) and DQO (+49, f5).", "DSY rose (+53) and DQO (+49)."),
            ("DSY rose (f4, +53).", "DSY rose (+53)."),
            ("Up (+53, f4, f5) overall.", "Up (+53) overall."),
            ("Share 65.2% (about two thirds; f10).", "Share 65.2% (about two thirds)."),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(self.answer(raw)[0].text, expected)

    def test_look_alikes_inside_parentheses_are_left_alone(self):
        text = "Models (f-series, 5f2 ratio, F4) are left alone."
        self.assertEqual(self.answer(text)[0].text, text)

    def test_removing_markers_does_not_hide_an_unclaimed_number(self):
        run = fresh_run()
        facts = collect_facts(run_sql=run)
        chat = scripted(reply("The total was 482 (f2).", [facts[0].id]))
        result = investigate("why?", make_playbook_draft_fn(chat=chat, run_sql=run),
                             lambda claims: evaluate(claims, run_sql=run), max_rounds=1)
        self.assertEqual(result.trace[0]["reasons"], ["unclaimed_number"])

    def test_the_prompt_tells_the_model_not_to_write_fact_ids(self):
        chat = self.answer("ok")[1]
        self.assertIn("never write fact ids", chat.calls[0][0]["content"].lower())


@unittest.skipUnless(all(os.environ.get(k) for k in ("ZETARIS_REST_URL", "ZETARIS_ORG_ID", "ZETARIS_API_KEY")),
                     "Zetaris settings missing in .env")
class ThroughZetaris(unittest.TestCase):
    """The values must match the independent openFDA oracle (calls made 2026-10-24)."""

    def test_facts_match_the_oracle(self):
        by_label = {f.label: float(f.value) if str(f.value).replace(".", "").isdigit() else f.value
                    for f in collect_facts()}
        expected = {"Cardiovascular recall records in 2023": 281,
                    "Cardiovascular recall records in 2024": 482,
                    "Increase in cardiovascular recall records, 2023 to 2024": 201,
                    "DSY increase in recall records, 2023 to 2024": 53,
                    "DQO increase in recall records, 2023 to 2024": 49,
                    "OEZ increase in recall records, 2023 to 2024": 29,
                    "Increase from the top 3 cardiovascular product codes combined": 131,
                    "Share of the cardiovascular increase from the top 3 product codes (%)": 65.2,
                    "DSY recall records in 2024": 54,
                    "DSY recall events in 2024": 1,
                    "DSY recalling firms in 2024": 1}
        for label, value in expected.items():
            with self.subTest(label=label):
                self.assertAlmostEqual(by_label[label], value, places=1)


if __name__ == "__main__":
    unittest.main()
