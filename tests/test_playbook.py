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

from dotenv import load_dotenv

from agent.evaluator import Claim, Rejection, evaluate
from agent.llm import Reply
from agent.loop import investigate
from agent.playbook import collect_facts, make_playbook_draft_fn

load_dotenv()
CODES = ["DSY", "ABC", "OEZ"]  # ABC has no record-level view


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
