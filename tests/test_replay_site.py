"""Tests for the static replay site (written before the implementation).

GitHub Pages serves docs/ as static files, so the site cannot run the agent. It replays the five sample runs
already in the repository (logs/sample-*.jsonl and reports/sample-*.html): a build step condenses each run log into
a short list of steps, and docs/index.html plays them back, then shows the verdict, the answer and the full report.

    replay_site.build(root, out_dir) -> list of run dicts, and writes
        <out_dir>/runs.json            {"runs": [...]}
        <out_dir>/reports/<id>.html    a copy of each run's report (not for the refusal, which has none)
        <out_dir>/.nojekyll            so Pages serves the files as they are
    run dict: {"id", "title", "question", "mode" ("playbook" | "exploratory" | "refused"), "verified" (bool),
               "refused" (bool), "answer" (str | None), "message" (str | None), "report" (path | None),
               "steps": [{"kind", "text"}], "queries": {"fetched", "cached"}}
    Step kinds: question, round, memory, zetaris, model, escalate, research, check (rejected | verified), end, refused.
    Consecutive Zetaris queries become one step ("36 queries fetched, 86 served from cache"), so a run is ~10-25 steps.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import json
import os
import re
import tempfile
import unittest

from agent import replay_site

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "docs", "index.html")
IDS = ["sample-run-1", "sample-run-2", "sample-run-3", "sample-exploratory-run-1", "sample-refusal-1"]


class Built(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.runs = replay_site.build(ROOT, cls.tmp.name)
        cls.by_id = {run["id"]: run for run in cls.runs}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def read(self, *parts):
        with open(os.path.join(self.tmp.name, *parts), encoding="utf-8") as handle:
            return handle.read()


class TestBuild(Built):
    def test_all_five_samples_in_a_fixed_order(self):
        self.assertEqual([run["id"] for run in self.runs], IDS)
        self.assertEqual(json.loads(self.read("runs.json"))["runs"], self.runs)

    def test_writes_nojekyll_and_a_copy_of_every_report(self):
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, ".nojekyll")))
        for run in self.runs:
            if run["refused"]:
                self.assertIsNone(run["report"])
            else:
                self.assertEqual(run["report"], f"reports/{run['id']}.html")
                self.assertIn("<html", self.read(run["report"]).lower())

    def test_building_twice_gives_identical_output(self):
        with tempfile.TemporaryDirectory() as other:
            replay_site.build(ROOT, other)
            with open(os.path.join(other, "runs.json"), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), self.read("runs.json"))

    def test_nothing_private_in_the_published_data(self):
        text = self.read("runs.json")
        for needle in ("/Users/", "Bearer", "nvapi", "sk-", "@gmail", "api_key", "API_KEY"):
            self.assertNotIn(needle, text, needle)
        for run in self.runs:
            if run["report"]:
                page = self.read(run["report"])
                for needle in ("/Users/", "Bearer", "nvapi", "@gmail"):
                    self.assertNotIn(needle, page, (run["id"], needle))


class TestRuns(Built):
    def test_the_playbook_run_shows_rejection_research_escalation_and_verification(self):
        run = self.by_id["sample-run-1"]
        self.assertEqual((run["mode"], run["verified"], run["refused"]), ("playbook", True, False))
        self.assertIn("281", run["answer"])
        self.assertIn("482", run["answer"])
        texts = [step["text"] for step in run["steps"]]
        kinds = [step["kind"] for step in run["steps"]]
        self.assertTrue(any("2 rejected (missing_evidence, missing_evidence)" in t for t in texts), texts)
        self.assertEqual(kinds.count("research"), 2)
        self.assertTrue(any(s["kind"] == "escalate" and "nemotron-3-ultra" in s["text"] for s in run["steps"]))
        checks = [s for s in run["steps"] if s["kind"] == "check"]
        self.assertEqual([c["text"] for c in checks][-1], "round 2: all verified")
        self.assertEqual([s["kind"] for s in run["steps"] if s["kind"] != "memory"][-1], "end")

    def test_zetaris_queries_are_condensed_and_counted(self):
        run = self.by_id["sample-run-1"]
        self.assertEqual(run["queries"], {"fetched": 36, "cached": 86, "failed": 0})
        self.assertEqual(self.by_id["sample-exploratory-run-1"]["queries"], {"fetched": 49, "cached": 0, "failed": 4})  # as in the README
        self.assertLess(len(run["steps"]), 40)
        self.assertGreater(len([s for s in run["steps"] if s["kind"] == "zetaris"]), 0)

    def test_every_playbook_sample_is_verified_with_an_answer(self):
        for run_id in ("sample-run-1", "sample-run-2", "sample-run-3"):
            run = self.by_id[run_id]
            self.assertTrue(run["verified"], run_id)
            self.assertTrue(run["answer"], run_id)

    def test_the_exploratory_run_gave_up_and_shows_no_answer(self):
        run = self.by_id["sample-exploratory-run-1"]
        self.assertEqual((run["mode"], run["verified"], run["answer"]), ("exploratory", False, None))
        rejections = [s["text"] for s in run["steps"] if s["kind"] == "check"]
        self.assertEqual(len(rejections), 3)
        self.assertTrue(all("rejected" in t for t in rejections), rejections)
        end = [s for s in run["steps"] if s["kind"] == "end"]
        self.assertEqual(len(end), 1)
        self.assertIn("no verified answer", end[0]["text"].lower())

    def test_provider_retries_are_shown_not_hidden(self):
        run = self.by_id["sample-exploratory-run-1"]
        self.assertTrue(any("retr" in s["text"].lower() for s in run["steps"] if s["kind"] == "model"))

    def test_the_refusal_is_one_step_with_the_scope_message(self):
        run = self.by_id["sample-refusal-1"]
        self.assertEqual((run["mode"], run["refused"], run["verified"]), ("refused", True, False))
        self.assertEqual(run["question"], "What is the capital of France?")
        self.assertIn("Out of scope", run["message"])
        self.assertEqual([s["kind"] for s in run["steps"]], ["question", "refused"])


class TestPage(unittest.TestCase):
    def source(self):
        with open(PAGE, encoding="utf-8") as handle:
            return handle.read()

    def test_the_page_says_it_is_a_replay_of_recorded_runs(self):
        text = self.source().lower()
        self.assertIn("replay", text)
        self.assertIn("recorded", text)

    def test_run_text_is_never_inserted_as_html(self):
        source = self.source()
        for banned in ("innerHTML", "insertAdjacentHTML", "document.write", "eval("):
            self.assertNotIn(banned, source)

    def test_everything_is_loaded_from_this_site_only(self):
        source = self.source()
        self.assertIn("runs.json", source)
        self.assertEqual(re.findall(r'src="(https?://[^"]+)"', source), [])  # nothing external is loaded
        allowed = ("https://github.com/smahanti8/", "https://pmc.ncbi.nlm.nih.gov/", "https://www.fda.gov/")
        links = re.findall(r'href="(https?://[^"]+)"', source)
        self.assertTrue(links)
        self.assertEqual([m for m in links if not m.startswith(allowed)], [])
        self.assertNotIn("cdn.", source)


if __name__ == "__main__":
    unittest.main()
