"""Tests for the gate evaluation (written before the implementation).

A measurement of the deterministic gate (evaluator plus the loop's checks) on drafts it should reject, drafts it
should accept, and drafts it accepts by design. Nothing calls a model or the network: the SQL results are a fixed
fixture. Each case says where it comes from: "seen in a live run" (the mistake was made by the model in a recorded
run) or "constructed" (written for the test). Evidence-completeness (`missing_evidence`) is covered by
tests/test_research_branch.py and is not part of this evaluation.

    agent/gate_eval.py
      Case(name, origin, kind, draft, expect)   kind: "bad" (must be rejected with `expect`), "good" (must be
                                                accepted; expect is None) or "gap" (accepted by design; documented)
      FIXTURE_SQL                               {sql: rows} the fake Zetaris answers from; anything else raises
      CASES                                     the shipped cases
      run(cases=None) -> [{"name", "origin", "kind", "expect", "got": [reasons]}]
      summarize(results) -> {"bad_total", "bad_caught", "good_total", "good_accepted", "gaps", "by_reason": {reason: n}}
          a bad case is caught when `expect` is among the reasons it got; a good case is accepted when it got none;
          gaps counts "gap" cases that were accepted
      format_summary(summary) -> text whose first line reads
          "Gate evaluation: 12 bad drafts, 12 rejected with the expected reason; 3 good drafts, 3 accepted"
          followed by one line per reason and a "Known gaps" line
      main(argv=None, out=print, cases=None) -> 0 when every bad case is caught and every good case accepted, else 1
          (also `python -m agent.gate_eval`)

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import unittest
from unittest import mock

from agent import gate_eval
from agent.evaluator import Citation, Claim
from agent.gate_eval import CASES, FIXTURE_SQL, Case, format_summary, main, run, summarize
from agent.loop import Draft

REASONS = {"no_citation", "sql_failed", "no_such_cell", "scope_mismatch", "value_mismatch", "unclaimed_number",
           "no_claims"}


class ShippedCases(unittest.TestCase):
    def test_every_case_behaves_as_expected(self):
        for result in run():
            with self.subTest(case=result["name"]):
                if result["kind"] == "bad":
                    self.assertIn(result["expect"], result["got"])
                else:
                    self.assertEqual(result["got"], [])  # good drafts and documented gaps are accepted

    def test_every_reason_the_gate_can_give_has_a_bad_case(self):
        self.assertEqual({c.expect for c in CASES if c.kind == "bad"}, REASONS)

    def test_there_are_good_drafts_and_documented_gaps(self):
        self.assertGreaterEqual(len([c for c in CASES if c.kind == "good"]), 3)
        self.assertGreaterEqual(len([c for c in CASES if c.kind == "gap"]), 2)

    def test_names_are_unique_and_origins_are_honest(self):
        self.assertEqual(len({c.name for c in CASES}), len(CASES))
        for case in CASES:
            self.assertIn(case.origin, {"seen in a live run", "constructed"}, case.name)
            self.assertIn(case.kind, {"bad", "good", "gap"}, case.name)

    def test_unsafe_sql_and_injected_text_are_among_the_bad_cases(self):
        names = " ".join(c.name for c in CASES if c.kind == "bad").lower()
        for word in ("write", "other schema", "injected"):
            self.assertIn(word, names)

    def test_nothing_touches_the_network(self):
        with mock.patch("agent.tools_zetaris.requests.post", side_effect=AssertionError("network")):
            run()


class Summary(unittest.TestCase):
    def test_counts_match_the_cases(self):
        summary = summarize(run())
        self.assertEqual(summary["bad_total"], len([c for c in CASES if c.kind == "bad"]))
        self.assertEqual(summary["bad_caught"], summary["bad_total"])
        self.assertEqual(summary["good_accepted"], summary["good_total"])
        self.assertEqual(summary["gaps"], len([c for c in CASES if c.kind == "gap"]))
        self.assertEqual(set(summary["by_reason"]), REASONS)

    def test_the_text_leads_with_the_headline(self):
        text = format_summary(summarize(run()))
        first = text.splitlines()[0]
        self.assertTrue(first.startswith("Gate evaluation: "), first)
        self.assertIn("rejected with the expected reason", first)
        self.assertIn("accepted", first)
        for reason in REASONS:
            self.assertIn(reason, text)
        self.assertIn("Known gaps", text)


class TheHarnessCanFail(unittest.TestCase):
    """A measurement that cannot report a miss would prove nothing."""

    def good_draft(self):
        return next(c for c in CASES if c.kind == "good").draft

    def test_a_bad_case_that_actually_passes_is_reported_as_not_caught(self):
        miss = Case("wrongly labelled", "constructed", "bad", self.good_draft(), "no_citation")
        summary = summarize(run([miss]))
        self.assertEqual((summary["bad_total"], summary["bad_caught"]), (1, 0))

    def test_a_good_case_that_is_rejected_is_reported(self):
        claim = Claim("Cardiovascular recall records in 2024", 482, Citation(next(iter(FIXTURE_SQL)), "n"))
        broken = Case("wrongly good", "constructed", "good", Draft("482 records", [Claim(claim.text, claim.value)]), None)
        summary = summarize(run([broken]))
        self.assertEqual((summary["good_total"], summary["good_accepted"]), (1, 0))

    def test_main_exits_one_when_a_case_misbehaves_and_zero_otherwise(self):
        lines = []
        miss = Case("wrongly labelled", "constructed", "bad", self.good_draft(), "no_citation")
        self.assertEqual(main([], out=lines.append, cases=[miss]), 1)
        self.assertEqual(main([], out=lines.append), 0)
        self.assertTrue(any(line.startswith("Gate evaluation: ") for line in lines))


if __name__ == "__main__":
    unittest.main()
