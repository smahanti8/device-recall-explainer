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
from agent.llm import LLMError
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

    def test_claim_about_cardiovascular_must_cite_cardiovascular_scope(self):
        wide = "SELECT SUM(recall_count) AS total FROM openfda.recall_by_product_2024_table"
        wrong = Claim("Total cardiovascular recall records in 2024", 3279, Citation(wide, "total"))
        rejection = evaluate([wrong], run_sql=fake_sql({wide: [{"total": "3279"}]})).rejections[0]
        self.assertEqual(rejection.reason, "scope_mismatch")
        self.assertIn("cardiovascular", rejection.detail)
        scoped = ("SELECT SUM(b.recall_count) AS total FROM openfda.classification_cardiovascular_table c "
                  "JOIN openfda.recall_by_product_2024_table b ON b.product_code = c.product_code")
        right = Claim("Total cardiovascular recall records in 2024", 482, Citation(scoped, "total"))
        self.assertTrue(evaluate([right], run_sql=fake_sql({scoped: [{"total": "482"}]})).ok)

    def test_claim_naming_a_product_code_must_cite_that_code(self):
        other = "SELECT COUNT(*) AS n FROM openfda.recall_dqo_2024_table"
        bad = Claim("DSY recall records in 2024", 51, Citation(other, "n"))
        rejection = evaluate([bad], run_sql=fake_sql({other: [{"n": "51"}]})).rejections[0]
        self.assertEqual(rejection.reason, "scope_mismatch")
        self.assertIn("DSY", rejection.detail)

    def test_common_acronyms_are_not_mistaken_for_product_codes(self):
        wide = "SELECT SUM(recall_count) AS total FROM openfda.recall_by_product_2024_table"
        ok = Claim("FDA device recall records in 2024", 3279, Citation(wide, "total"))
        self.assertTrue(evaluate([ok], run_sql=fake_sql({wide: [{"total": "3279"}]})).ok)


TOTAL_2024 = "SELECT SUM(recall_count) AS total FROM openfda.recall_by_product_2024_table"


class Loop(unittest.TestCase):
    sql = staticmethod(fake_sql({DSY_RECORDS: [{"n": "54"}], TOTAL_2024: [{"total": "3279"}]}))

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

    def test_the_result_carries_the_accepted_claims(self):
        result = investigate("why?", lambda q, f: Draft("DSY had 54 recall records.", [claim(54)]), self.evaluator)
        self.assertEqual(result.claims, [claim(54)])
        failed = investigate("why?", lambda q, f: Draft("a", [claim(80)]), self.evaluator, max_rounds=1)
        self.assertEqual(failed.claims, [])

    def test_gives_up_after_max_rounds_and_lists_what_it_could_not_support(self):
        result = investigate("why?", lambda q, f: Draft("answer", [claim(80)]), self.evaluator, max_rounds=3)
        self.assertIsNone(result.text)
        self.assertEqual(result.rounds, 3)
        self.assertEqual([c.value for c in result.unsupported], [80])

    def test_draft_with_no_claims_is_not_accepted(self):
        result = investigate("why?", lambda q, f: Draft("trust me", []), self.evaluator, max_rounds=2)
        self.assertIsNone(result.text)
        self.assertEqual(result.trace[0]["reasons"], ["no_claims"])

    def test_numbers_in_the_text_must_be_claimed(self):
        drafts = iter([
            Draft("DSY had 54 recall records (about 65% of the rise).", [claim(54)]),
            Draft("DSY had 54 recall records.", [claim(54)]),
        ])
        result = investigate("why?", lambda q, f: next(drafts), self.evaluator)
        self.assertEqual(result.trace[0]["reasons"], ["unclaimed_number"])
        self.assertEqual(result.text, "DSY had 54 recall records.")
        self.assertEqual(result.rounds, 2)

    def test_years_single_digits_and_thousands_separators_are_handled(self):
        text = "In 2024 there were 3,279 records across 1 event, a Class 2 device."
        total = Claim("total recall records", 3279, Citation(TOTAL_2024, "total"))
        result = investigate("why?", lambda q, f: Draft(text, [total]), self.evaluator)
        self.assertEqual(result.text, text)
        self.assertEqual(result.rounds, 1)

    def test_a_model_failure_costs_a_round_and_the_run_continues(self):
        calls = []

        def draft_fn(question, feedback):
            calls.append(feedback)
            if len(calls) == 1:
                raise LLMError("gave up after 5 attempts: 502 provider_deadline")
            return Draft("DSY had 54 recall records.", [claim(54)])

        result = investigate("why?", draft_fn, self.evaluator)
        self.assertEqual(result.text, "DSY had 54 recall records.")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.trace[0]["reasons"], ["model_unavailable"])
        self.assertEqual(calls[1], [])  # the outage is not shown to the model as a rejection

    def test_every_round_failing_gives_no_answer_and_says_why(self):
        def draft_fn(question, feedback):
            raise LLMError("gave up after 5 attempts: 502 provider_deadline")

        result = investigate("why?", draft_fn, self.evaluator, max_rounds=2)
        self.assertIsNone(result.text)
        self.assertEqual(result.rounds, 2)
        self.assertIn("502", result.unsupported[0].text)

    def test_other_exceptions_are_bugs_and_are_not_swallowed(self):
        def draft_fn(question, feedback):
            raise RuntimeError("bug in the drafter")

        with self.assertRaises(RuntimeError):
            investigate("why?", draft_fn, self.evaluator)

    def test_trace_entries_carry_each_rejection_detail(self):
        drafts = iter([Draft("a", [claim(80)]), Draft("a", [claim(54)])])
        result = investigate("why?", lambda q, f: next(drafts), self.evaluator)
        self.assertEqual(len(result.trace[0]["details"]), len(result.trace[0]["reasons"]))
        self.assertIn("80", result.trace[0]["details"][0])
        self.assertEqual(result.trace[1]["details"], [])

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
