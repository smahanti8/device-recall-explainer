"""Tests for agent/memory.py (written before the implementation).

Memory wraps the vendored H-MEM engine (engines/hmem, TypeScript) behind a tiny
JSON protocol, so the Python agent can remember what it has already verified.

    Memory(persist_dir, run=None)         run(request) -> response dict; default calls agent/memory_cli.ts
    remember(claim) -> id                 stores one verified claim; the cited SQL is the record's source
    recall(question, top_n=5) -> [Recollection(id, content, relevance, source)]
    context_for(question) -> str          hints for the drafter, "" when nothing is recalled
    learn(claims, rejections) -> [id]     stores accepted claims only, then sends "helpful" feedback
    mark_wrong(memory_id)                 sends "wrong" feedback (a recalled hint contradicted fresh SQL)
    feedback(memory_id, kind)             kind in {"helpful", "not_helpful", "wrong"}, else ValueError

Memory is a HINT, never evidence: a claim must still cite fresh SQL.
Requests: {"op": "add"|"query"|"feedback", "persistDir": ..., ...}.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from agent.evaluator import Citation, Claim, Rejection
from agent.memory import Memory, Recollection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_DIR = os.path.join(ROOT, "engines", "hmem", "reference")
SQL = "SELECT COUNT(*) AS n FROM openfda.recall_dsy_2024_table"
CLAIM = Claim("DSY recall records in 2024", 54, Citation(SQL, "n"))
OTHER = Claim("DQO recall records in 2024", 51, Citation(SQL.replace("dsy", "dqo"), "n"))
HIT = {"id": "mem-1", "content": "DSY recall records in 2024: 54", "relevance": 0.8, "source": SQL}


def fake_run(*responses):
    queue, sent = list(responses), []

    def run(request):
        sent.append(request)
        return queue.pop(0) if queue else {}
    run.sent = sent
    return run


class Protocol(unittest.TestCase):
    def test_remember_stores_the_claim_with_its_sql_as_source(self):
        run = fake_run({"id": "mem-1"})
        memory_id = Memory("/tmp/hm", run=run).remember(CLAIM)
        request = run.sent[0]
        self.assertEqual((memory_id, request["op"], request["persistDir"]), ("mem-1", "add", "/tmp/hm"))
        record = request["record"]
        self.assertEqual(record["source"], SQL)
        self.assertEqual((record["type"], record["confidence"]), ("factual", 0.9))
        self.assertTrue({"product_code:DSY", "year:2024"} <= set(record["tags"]))
        self.assertIn("DSY recall records in 2024", record["content"])
        self.assertIn("54", record["content"])

    def test_cardiovascular_claims_are_tagged(self):
        run = fake_run({"id": "mem-1"})
        Memory("/tmp/hm", run=run).remember(Claim("Cardiovascular recall records in 2024", 482, Citation(SQL, "n")))
        self.assertIn("cardiovascular", run.sent[0]["record"]["tags"])

    def test_a_claim_without_a_citation_is_never_stored(self):
        run = fake_run()
        with self.assertRaises(ValueError):
            Memory("/tmp/hm", run=run).remember(Claim("no source", 1))
        self.assertEqual(run.sent, [])

    def test_recall_sends_a_query_and_parses_the_hits(self):
        run = fake_run({"memories": [HIT], "context": ""})
        hits = Memory("/tmp/hm", run=run).recall("DSY 2024")
        self.assertEqual(run.sent[0]["op"], "query")
        self.assertEqual((run.sent[0]["query"], run.sent[0]["topN"]), ("DSY 2024", 5))
        self.assertEqual(hits, [Recollection("mem-1", "DSY recall records in 2024: 54", 0.8, SQL)])

    def test_context_for_labels_hints_as_not_evidence(self):
        text = Memory("/tmp/hm", run=fake_run({"memories": [HIT], "context": ""})).context_for("DSY 2024")
        for needle in ("not evidence", "DSY recall records in 2024: 54", SQL):
            self.assertIn(needle, text)
        self.assertEqual(Memory("/tmp/hm", run=fake_run({"memories": [], "context": ""})).context_for("x"), "")

    def test_learn_stores_only_accepted_claims_then_marks_them_helpful(self):
        run = fake_run({"id": "mem-1"}, {"ok": True})
        rejected = [Rejection(OTHER, "value_mismatch", "claimed 51 but ... returned '50'")]
        ids = Memory("/tmp/hm", run=run).learn([CLAIM, OTHER], rejected)
        self.assertEqual(ids, ["mem-1"])
        self.assertEqual([r["op"] for r in run.sent], ["add", "feedback"])
        self.assertEqual((run.sent[1]["id"], run.sent[1]["kind"]), ("mem-1", "helpful"))

    def test_mark_wrong_and_feedback_validation(self):
        run = fake_run({"ok": True})
        Memory("/tmp/hm", run=run).mark_wrong("mem-7")
        self.assertEqual((run.sent[0]["op"], run.sent[0]["id"], run.sent[0]["kind"]), ("feedback", "mem-7", "wrong"))
        with self.assertRaises(ValueError):
            Memory("/tmp/hm", run=fake_run()).feedback("mem-7", "great")


@unittest.skipUnless(os.path.isdir(os.path.join(ENGINE_DIR, "node_modules")) and shutil.which("npx"),
                     "run `npm ci --ignore-scripts` in engines/hmem/reference first")
class WithTheRealEngine(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_a_second_session_recalls_what_the_first_stored(self):
        memory_id = Memory(self.dir).remember(CLAIM)           # session 1 (its own Node process)
        hits = Memory(self.dir).recall("DSY recall records 2024")  # session 2: fresh objects, state from disk
        self.assertTrue(any(h.id == memory_id and h.source == SQL and "54" in h.content for h in hits), hits)

    def test_the_audit_ledger_records_creation_and_feedback(self):
        memory = Memory(self.dir)
        memory.mark_wrong(memory.remember(CLAIM))
        with open(os.path.join(self.dir, "ledger.jsonl")) as ledger:
            entries = [json.loads(line) for line in ledger if line.strip()]
        self.assertGreaterEqual(len(entries), 2)

    def test_the_vendored_engine_still_passes_its_conformance_suite(self):
        done = subprocess.run(["npx", "tsx", "../conformance/runner.ts"], cwd=ENGINE_DIR,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stdout[-400:])
        self.assertIn("13/13 checks passed", done.stdout)


if __name__ == "__main__":
    unittest.main()
