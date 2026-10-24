"""Tests for agent/evaluator.py and agent/loop.py (written before the implementation).

CLAIM FORMAT (for review)
    Citation(sql, column, row=0)     points at ONE cell of a run_sql result
    Claim(text, value, citation=None) one number the answer states, and where it comes from
    A claim stands only if re-running citation.sql yields a cell equal to value.
    Every stated number must be computed by SQL, never by the model.
    evaluate(claims, run_sql=agent.tools_zetaris.run_sql) -> Verdict
        Verdict.ok; Verdict.rejections -> [Rejection(claim, reason, detail)]
        reason is one of: no_citation, sql_failed, no_such_cell, value_mismatch
    Draft(text, claims)
    investigate(question, draft_fn, evaluate_fn, max_rounds=3) -> Result
        draft_fn(question, feedback) -> Draft   (feedback: list of Rejection, empty at first)
        evaluate_fn(claims) -> Verdict
        Result: text (None if it gave up), unsupported (claims), rounds, trace (one dict per round)

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import os
import unittest

from dotenv import load_dotenv

from agent.evaluator import Citation, Claim, evaluate
from agent.loop import Draft, investigate

load_dotenv()

DSY_RECORDS = "SELECT COUNT(*) AS n FROM openfda.recall_dsy_2024_table"
DSY_EVENTS = "SELECT COUNT(DISTINCT res_event_number) AS n FROM openfda.recall_dsy_2024_table"


def fake_sql(table):
    """A run_sql stand-in: returns canned rows, raises for anything unknown."""
    def run(sql):
        if sql not in table:
            raise ValueError("unknown sql in test")
        return table[sql]
    return run


def claim(value, sql=DSY_RECORDS, column="n", row=0, text="DSY recall records in 2024"):
    return Claim(text, value, Citation(sql, column, row))


class Evaluate(unittest.TestCase):
    sql = staticmethod(fake_sql({DSY_RECORDS: [{"n": "54"}]}))  # Zetaris returns strings

    def test_claim_without_citation_is_rejected(self):
        verdict = evaluate([Claim("DSY recalls rose a lot", 54)], run_sql=self.sql)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.rejections[0].reason, "no_citation")

    def test_true_claim_passes_even_though_zetaris_returns_strings(self):
        self.assertTrue(evaluate([claim(54), claim(54.0)], run_sql=self.sql).ok)

    def test_fabricated_number_is_rejected_with_expected_and_actual(self):
        rejection = evaluate([claim(80)], run_sql=self.sql).rejections[0]
        self.assertEqual(rejection.reason, "value_mismatch")
        self.assertIn("80", rejection.detail)
        self.assertIn("54", rejection.detail)

    def test_unrunnable_sql_is_rejected_not_raised(self):
        verdict = evaluate([claim(1, sql="DROP VIEW openfda.x")], run_sql=self.sql)
        self.assertEqual(verdict.rejections[0].reason, "sql_failed")

    def test_missing_row_or_column_is_rejected(self):
        for bad in (claim(54, row=3), claim(54, column="nope")):
            with self.subTest(claim=bad.text):
                self.assertEqual(evaluate([bad], run_sql=self.sql).rejections[0].reason, "no_such_cell")


class Loop(unittest.TestCase):
    sql = staticmethod(fake_sql({DSY_RECORDS: [{"n": "54"}]}))

    def evaluator(self, claims):
        return evaluate(claims, run_sql=self.sql)

    def test_retries_with_feedback_until_claims_resolve(self):
        seen = []

        def draft_fn(question, feedback):
            seen.append(feedback)
            return Draft("answer", [claim(80)] if not feedback else [claim(54)])

        result = investigate("why?", draft_fn, self.evaluator)
        self.assertEqual(result.text, "answer")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(seen[0], [])
        self.assertEqual(seen[1][0].reason, "value_mismatch")  # the draft saw why it failed

    def test_gives_up_after_max_rounds_and_lists_what_it_could_not_support(self):
        result = investigate("why?", lambda q, f: Draft("answer", [claim(80)]), self.evaluator, max_rounds=3)
        self.assertIsNone(result.text)
        self.assertEqual(result.rounds, 3)
        self.assertEqual([c.value for c in result.unsupported], [80])

    def test_draft_with_no_claims_is_not_accepted(self):
        result = investigate("why?", lambda q, f: Draft("trust me", []), self.evaluator, max_rounds=2)
        self.assertIsNone(result.text)
        self.assertEqual(result.trace[0]["reasons"], ["no_claims"])

    def test_trace_has_one_entry_per_round(self):
        result = investigate("why?", lambda q, f: Draft("a", [claim(80)] if not f else [claim(54)]), self.evaluator)
        self.assertEqual([t["round"] for t in result.trace], [1, 2])
        self.assertEqual(result.trace[0]["rejected"], 1)
        self.assertEqual(result.trace[1]["rejected"], 0)


@unittest.skipUnless(all(os.environ.get(k) for k in ("ZETARIS_REST_URL", "ZETARIS_ORG_ID", "ZETARIS_API_KEY")),
                     "Zetaris settings missing in .env")
class DsyFinding(unittest.TestCase):
    """Through real Zetaris: DSY is 54 recall records but ONE recall event."""

    def test_records_and_events_are_not_confused(self):
        self.assertTrue(evaluate([claim(54)]).ok)
        self.assertTrue(evaluate([claim(1, sql=DSY_EVENTS, text="DSY recall events in 2024")]).ok)
        wrong = evaluate([claim(54, sql=DSY_EVENTS, text="DSY recall events in 2024")])
        self.assertEqual(wrong.rejections[0].reason, "value_mismatch")


if __name__ == "__main__":
    unittest.main()
