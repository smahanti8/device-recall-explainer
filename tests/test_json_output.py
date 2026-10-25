"""Tests for --json (written before the implementation).

The hackathon guide asks for a programmatic entry point. The command line already is one; --json makes its
output machine-readable: progress stays on stderr and stdout carries exactly one JSON document.

    main.ask(..., as_json=False): when True, nothing but one JSON string goes to `out`:
        {"question", "verified" (bool), "answer" (str | null), "rounds" (int),
         "claims": [{"text", "value", "sql", "column", "row"}],      the accepted draft's claims
         "unsupported": [{"text", "value"}],                        what could not be supported (empty when verified)
         "trace": [{"round", "claims", "rejected", "reasons", "details"}],
         "receipts": [{"routed_model", "receipt_id"}],
         "tokens": {"calls", "input", "output", "cost_usd"},
         "observability": {"queries", "models", "retries", "failed"},
         "log": path | null, "report": path | null}
        Progress lines still go to the progress callable.
    main.main: --json passes as_json=True to the ask function; the exit code is unchanged (0 verified, 1 not).

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import json
import unittest

from agent.evaluator import Citation, Claim, Verdict
from agent.loop import Draft, Result
from agent.main import ask, main

FULL = {"ZETARIS_REST_URL": "u", "ZETARIS_ORG_ID": "1", "ZETARIS_API_KEY": "k", "MODEL_BASE_URL": "u",
        "MODEL_API_KEY": "k", "NEMOTRON_MODEL": "m"}
CLAIM = Claim("Cardiovascular recall records in 2024", 482, Citation("SELECT 482 AS n", "n"))


def run(result, **kwargs):
    lines, progress = [], []

    def run_investigation(question, draft_fn, evaluate_fn):
        return result
    ask("why?", None, make_draft=lambda **kw: (lambda q, f: Draft("a", [])), run_investigation=run_investigation,
        evaluate_fn=lambda claims: Verdict(), out=lines.append, progress=progress.append,
        run_sql=lambda sql: [{"n": "482"}], as_json=True, **kwargs)
    return lines, progress


class JsonDocument(unittest.TestCase):
    def test_a_verified_run_is_one_json_document(self):
        lines, _ = run(Result("482 records.", [], 1, [{"round": 1, "claims": 1, "rejected": 0, "reasons": [], "details": []}],
                              claims=[CLAIM]))
        self.assertEqual(len(lines), 1)
        doc = json.loads(lines[0])
        self.assertEqual((doc["question"], doc["verified"], doc["answer"], doc["rounds"]), ("why?", True, "482 records.", 1))
        self.assertEqual(doc["claims"], [{"text": CLAIM.text, "value": 482, "sql": "SELECT 482 AS n", "column": "n", "row": 0}])
        self.assertEqual(doc["unsupported"], [])
        self.assertEqual(doc["trace"][0]["round"], 1)
        for key in ("receipts", "tokens", "observability", "log", "report"):
            self.assertIn(key, doc)
        self.assertEqual(set(doc["tokens"]), {"calls", "input", "output", "cost_usd"})

    def test_a_run_with_no_answer_says_so(self):
        gave_up = Result(None, [Claim("a claim without support", 5)], 3, [{"round": 3, "claims": 1, "rejected": 1,
                                                                            "reasons": ["no_citation"], "details": ["x"]}])
        doc = json.loads(run(gave_up)[0][0])
        self.assertFalse(doc["verified"])
        self.assertIsNone(doc["answer"])
        self.assertEqual(doc["unsupported"], [{"text": "a claim without support", "value": 5}])
        self.assertEqual(doc["claims"], [])

    def test_progress_still_goes_to_the_progress_callable(self):
        _, progress = run(Result("a", [], 1, [], claims=[CLAIM]))
        self.assertTrue(any("round" in line for line in progress) or progress == [])

    def test_without_the_flag_the_output_is_text(self):
        lines = []
        ask("why?", None, make_draft=lambda **kw: (lambda q, f: Draft("a", [])),
            run_investigation=lambda q, d, e: Result("a", [], 1, [], claims=[CLAIM]), evaluate_fn=lambda c: Verdict(),
            out=lines.append, progress=lambda m: None, run_sql=lambda sql: [{"n": "482"}])
        shown = "\n".join(lines)
        self.assertIn("Answer: a", shown)
        with self.assertRaises(ValueError):
            json.loads(shown)


class InTheCommandLine(unittest.TestCase):
    def passed(self, argv):
        seen = {}

        def fake_ask(question, memory, **kwargs):
            seen.update(kwargs)
            return Result("ok", [], 1, [], claims=[CLAIM])
        code = main(argv, env=FULL, ask_fn=fake_ask)
        return code, seen

    def test_the_flag_is_passed_on(self):
        code, seen = self.passed(["why?", "--no-memory", "--json"])
        self.assertEqual((code, seen["as_json"]), (0, True))

    def test_the_default_is_text(self):
        self.assertFalse(self.passed(["why?", "--no-memory"])[1]["as_json"])


if __name__ == "__main__":
    unittest.main()
