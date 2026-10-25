"""Replays every shipped sample question offline (no services, no keys) and checks the recorded outputs.

The recordings in sample_data/ come from real runs (see logs/sample-run-*.jsonl); the evaluator, retry loop and
report run for real on top of them. If a recording is replaced or removed, this test says which question broke.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import os
import unittest

from agent.main import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = [
    ("Why did cardiovascular device recall records rise between 2023 and 2024, and which product codes drove it?",
     35, 2, ["281", "482", "201", "131", "65.2%"]),
    ("Which three product codes drove the increase in cardiovascular device recalls from 2023 to 2024, and what share "
     "of the increase do they account for?", 20, 1, ["131", "65.2%", "DSY", "DQO", "OEZ"]),
    ("For product code DSY, how many 2024 recall records were there, how many distinct recall events and firms, and what "
     "was the top root cause?", 4, 1, ["54", "1 event", "1 firm", "Equipment maintenance"]),
]


def replay(question):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([question, "--playbook", "--json", "--no-log", "--no-memory",
                     "--sample-dir", os.path.join(ROOT, "sample_data")], env={"SAMPLE_MODE": "true"})
    return code, json.loads(out.getvalue()), err.getvalue()


class SampleExamples(unittest.TestCase):
    def test_every_sample_question_replays_to_a_verified_answer(self):
        for question, claims, rounds, words in SAMPLES:
            with self.subTest(question=question[:50]):
                code, doc, stderr = replay(question)
                self.assertEqual(code, 0)
                self.assertTrue(doc["verified"])
                self.assertEqual((len(doc["claims"]), doc["rounds"]), (claims, rounds))
                for word in words:
                    self.assertIn(word, doc["answer"])
                self.assertIn("SAMPLE MODE", stderr)

    def test_the_samples_are_three_different_questions_with_different_answers(self):
        answers = [replay(question)[1]["answer"] for question, *_ in SAMPLES]
        self.assertEqual(len(set(answers)), 3)

    def test_each_sample_has_its_real_log_and_report(self):
        for n in (1, 2, 3):
            self.assertTrue(os.path.exists(os.path.join(ROOT, "logs", f"sample-run-{n}.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(ROOT, "reports", f"sample-run-{n}.html")))


if __name__ == "__main__":
    unittest.main()
