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


class TestRunDetail(Built):
    """Fields the run page shows: stats, receipts, rejections, an example proof (all read from the run logs and reports)."""

    def test_the_playbook_runs_carry_their_real_numbers(self):
        one, two, three = (self.by_id[i] for i in ("sample-run-1", "sample-run-2", "sample-run-3"))
        self.assertEqual((one["rounds"], one["claims"], one["calls"]), (2, 35, 2))
        self.assertEqual(one["tokens"], {"in": "5,078", "out": "4,402"})
        self.assertEqual(one["receipts"], ["rcpt_req_dv9gxx4q", "rcpt_req_63urdzfk"])
        self.assertEqual(one["models"], ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-lightning-30b-a3b"])
        self.assertEqual((two["rounds"], two["claims"], two["calls"]), (1, 20, 1))
        self.assertEqual(two["tokens"], {"in": "3,611", "out": "3,759"})
        self.assertEqual(two["receipts"], ["rcpt_req_lvbucqnr"])
        self.assertEqual((three["rounds"], three["claims"], three["calls"]), (1, 4, 1))
        self.assertEqual(three["tokens"], {"in": "3,610", "out": "480"})

    def test_the_exploratory_run_and_the_refusal(self):
        explore = self.by_id["sample-exploratory-run-1"]
        self.assertEqual((explore["rounds"], explore["claims"], explore["calls"]), (3, 0, 12))
        self.assertEqual(explore["tokens"], {"in": "50,624", "out": "34,623"})
        self.assertEqual(len(explore["receipts"]), 12)
        refusal = self.by_id["sample-refusal-1"]
        self.assertEqual((refusal["rounds"], refusal["claims"], refusal["calls"]), (0, 0, 0))
        self.assertEqual((refusal["tokens"], refusal["models"], refusal["receipts"]), (None, [], []))

    def test_rejections_are_listed_by_round(self):
        one = self.by_id["sample-run-1"]["rejections"]
        self.assertEqual(one, [{"round": 1, "text": "2 rejected (missing_evidence, missing_evidence)", "ok": False},
                               {"round": 2, "text": "all verified", "ok": True}])
        explore = self.by_id["sample-exploratory-run-1"]["rejections"]
        self.assertEqual([r["round"] for r in explore], [1, 2, 3])
        self.assertTrue(all(not r["ok"] for r in explore))
        self.assertEqual(self.by_id["sample-refusal-1"]["rejections"], [])

    def test_a_verified_run_has_an_example_proof_from_its_first_claim(self):
        proof = self.by_id["sample-run-1"]["proof"]
        self.assertIn("281", proof["claim"])
        self.assertTrue(proof["sql"].startswith("SELECT SUM(b.recall_count)"))
        self.assertEqual((proof["column"], proof["row"], proof["cell"]), ("total", "0", "281"))
        for run_id in ("sample-run-2", "sample-run-3"):
            self.assertTrue(self.by_id[run_id]["proof"]["sql"].startswith("SELECT"), run_id)
        for run_id in ("sample-exploratory-run-1", "sample-refusal-1"):
            self.assertIsNone(self.by_id[run_id]["proof"], run_id)

    def test_sample_sql_and_editorial_notes(self):
        sample = self.by_id["sample-run-1"]["queries_sample"]
        self.assertEqual(len(sample), 3)
        self.assertTrue(all(q["sql"].startswith("SELECT") and q["note"] for q in sample))
        self.assertEqual(self.by_id["sample-refusal-1"]["queries_sample"], [])
        for run in self.runs:
            self.assertTrue(run["watch"], run["id"])
            self.assertTrue(run["sub"], run["id"])


class TestRecallWisePage(unittest.TestCase):
    """The page as designed: name, tab title, favicon, plain-language hero, empty replay list, footer disclaimer."""

    @classmethod
    def setUpClass(cls):
        with open(PAGE, encoding="utf-8") as handle:
            cls.source = handle.read()
        cls.visible = re.sub(r"<script.*?</script>", "", cls.source, flags=re.S)

    def test_title_and_name(self):
        self.assertIn("<title>RecallWise The Device Recall Explainer</title>", self.source)
        self.assertIn("RecallWise", self.visible)
        self.assertIn("The Device Recall Explainer", self.visible)

    def test_favicon_is_linked_and_valid(self):
        self.assertIn('rel="icon"', self.source)
        self.assertIn('href="favicon.svg"', self.source)
        import xml.dom.minidom
        xml.dom.minidom.parse(os.path.join(ROOT, "docs", "favicon.svg"))

    def test_hero_is_plain_language(self):
        self.assertIn("Cardiovascular device recalls rose from 281 to 482 in one year. Why?", self.visible)
        self.assertIn("reasonable probability of serious adverse health consequences or death", self.visible)
        self.assertIn("This agent explains", self.visible)
        self.assertNotIn("Cardiovascular means the heart", self.visible)
        self.assertNotIn("Every number, traced to the SQL", self.visible)

    def test_one_short_button_up_top_and_no_live_disclaimer_above_the_footer(self):
        self.assertIn("See it in action", self.visible)
        self.assertNotIn("Replay a recorded run", self.visible)
        head, _, footer = self.visible.partition("<footer")
        self.assertTrue(footer, "the page needs a <footer>")
        self.assertNotIn("nothing runs live", head.lower())
        self.assertIn("nothing runs live", footer.lower())

    def test_replay_list_is_empty_until_replay_is_pressed(self):
        self.assertIn("Press Replay to watch the agent work", self.visible + self.source)
        self.assertIn("hint.hidden = started", self.source)  # the hint shows until a replay starts
        self.assertNotIn("round 1: drafting", self.visible)  # no step text is pre-filled in the page
        self.assertIn('{id: "log", "aria-live": "polite"}', self.source)  # the list is created empty

    def test_plain_words_for_the_data_layer(self):
        self.assertIn("data lookups", self.source.lower())  # the panels are rendered by the script from runs.json
        self.assertIn("via Zetaris", self.source)
        self.assertNotIn("Zetaris queries", self.source)

    def test_phone_layout_rules(self):
        """At phone width the tiles and stats stack, the chip labels sit on their own line and chips can wrap."""
        css = re.search(r"@media \(max-width:720px\)\{(.*?)\}\s*@media", self.source, re.S)
        self.assertTrue(css, "a phone media query is needed")
        rules = css.group(1)
        for needed in (".tiles{grid-template-columns:1fr}", ".stats{grid-template-columns:repeat(2,minmax(0,1fr))}",
                       ".cl{width:100%}", ".chip{border-radius:14px;flex-wrap:wrap}"):
            self.assertIn(needed, rules)

    def test_the_page_does_not_jump_when_a_replay_runs(self):
        """The step list has a fixed height (it scrolls inside its box), so the sections below never move."""
        self.assertIn("#log{height:440px;overflow-y:auto", self.source)
        self.assertIn("#hint{height:440px", self.source)
        self.assertIn(".body{padding:24px;min-height:", self.source)
        self.assertIn("log.scrollTop = log.scrollHeight", self.source)  # scroll the box, not the page
        self.assertNotIn("scrollIntoView({block:", self.source)
        self.assertIn(".qwrap{flex:1 1 560px;min-width:0;min-height:108px}", self.source)  # room for a three-line question
        self.assertIn("log.hidden = !started", self.source)  # hint and list never take space at the same time

    def test_header_uses_the_soft_slate(self):
        self.assertIn("#4A5157", self.source)


if __name__ == "__main__":
    unittest.main()
