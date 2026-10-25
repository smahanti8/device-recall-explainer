"""Tests for agent/setup_data.py (written before the implementation).

The policy asks for automated data loading, so a judge (or you) can register the Zetaris tables with one command
instead of pasting statements into the SQL Editor.

    python -m agent.setup_data [--sql-dir sql] [--delay 15] [--dry-run] [--verify]

    statements(text) -> [str]        split SQL text into statements: full-line "--" comments are dropped, and a
                                     semicolon inside quotes or backticks does not split
    SQL_FILES                        the files, in load order: openfda_spike, openfda_register, openfda_drilldown,
                                     openfda_drilldown_more
    load(sql_dir, delay=15, sleep=time.sleep, run=None, out=print, dry_run=False) -> report
        report = {"ran": n, "existed": n, "failed": None | {"statement": str, "message": str}}
        run(sql) -> None on success; raises SetupError(message) on failure. A message containing "already exists"
        counts as existed and the load continues. Any other failure stops the load (later statements depend on
        earlier ones) and is reported. sleep(delay) follows each CREATE LIGHTNING REST TABLE (each one calls openFDA,
        which rate-limits) and nothing else. dry_run prints the statements, runs nothing and never sleeps.
    verify(run_sql=None, out=print) -> {"ok": [views], "missing": [views]}  COUNT(*) on the key views
    main(argv=None, env=None, load_fn=None) -> exit code: 2 with a friendly message when ZETARIS_REST_URL,
        ZETARIS_ORG_ID or ZETARIS_API_KEY is missing (--dry-run needs none); 1 when the load failed; 0 otherwise.
        Nothing printed ever contains a setting's value.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import contextlib
import io
import os
import unittest

from agent.setup_data import SQL_FILES, SetupError, load, main, statements, verify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQL_DIR = os.path.join(ROOT, "sql")
ZETARIS_ENV = {"ZETARIS_REST_URL": "http://z.test/api", "ZETARIS_ORG_ID": "1", "ZETARIS_API_KEY": "secret-key-123"}
MORE_CODES = ("DQK", "DPT", "DRE", "DWF", "OES", "DYF", "OEQ")


class Statements(unittest.TestCase):
    def test_comments_and_blank_lines_are_dropped(self):
        text = "-- header\n\nCREATE A;\n-- note\nCREATE B;\n"
        self.assertEqual(statements(text), ["CREATE A", "CREATE B"])

    def test_multi_line_statements_stay_whole(self):
        self.assertEqual(statements("CREATE T\n  FROM X\n  REQUEST(a);\n"), ["CREATE T\n  FROM X\n  REQUEST(a)"])

    def test_a_semicolon_inside_quotes_or_backticks_does_not_split(self):
        text = 'CREATE T REQUEST(endpoint "http://x/a;b", c `d;e`);\nCREATE U;'
        self.assertEqual(len(statements(text)), 2)

    def test_commented_out_statements_are_not_executed(self):
        self.assertEqual(statements("-- CREATE HIDDEN;\nCREATE SHOWN;"), ["CREATE SHOWN"])


class ShippedFiles(unittest.TestCase):
    def read(self, name):
        with open(os.path.join(SQL_DIR, name)) as handle:
            return statements(handle.read())

    def test_the_load_order_is_spike_register_drilldown_more(self):
        self.assertEqual([os.path.basename(f) for f in SQL_FILES],
                         ["openfda_spike.sql", "openfda_register.sql", "openfda_drilldown.sql", "openfda_drilldown_more.sql"])

    def test_every_statement_in_every_file_is_a_create(self):
        for name in SQL_FILES:
            with self.subTest(name=name):
                parsed = self.read(os.path.basename(name))
                self.assertTrue(parsed)
                self.assertTrue(all(s.upper().startswith("CREATE ") for s in parsed), parsed[:2])

    def test_the_second_drilldown_file_adds_a_table_and_a_view_for_seven_more_codes(self):
        parsed = self.read("openfda_drilldown_more.sql")
        self.assertEqual(len(parsed), 14)
        for code in MORE_CODES:
            with self.subTest(code=code):
                table = next(s for s in parsed if f"TABLE recall_{code.lower()}_2024 " in s)
                self.assertIn(f"product_code:{code}+AND+event_date_initiated:[2024-01-01+TO+2024-12-31]", table)
                self.assertTrue(any(f"VIEW recall_{code.lower()}_2024_table " in s for s in parsed))


class Load(unittest.TestCase):
    def setUp(self):
        self.ran, self.slept = [], []

    def run_ok(self, sql):
        self.ran.append(sql)

    def do_load(self, run=None, **kwargs):
        return load(SQL_DIR, delay=15, sleep=self.slept.append, run=run or self.run_ok, out=lambda *_: None, **kwargs)

    def test_statements_run_in_file_order(self):
        report = self.do_load()
        self.assertEqual(report["ran"], len(self.ran))
        self.assertIsNone(report["failed"])
        self.assertIn("CREATE LIGHTNING DATABASE OPENFDA_REST", self.ran[0])
        self.assertIn("recall_oez_2024_table", " ".join(self.ran))
        self.assertTrue(self.ran[-1].startswith("CREATE SCHEMASTORE VIEW recall_oeq_2024_table"))

    def test_already_exists_is_counted_and_the_load_continues(self):
        def run(sql):
            self.ran.append(sql)
            if len(self.ran) <= 3:
                raise SetupError('A table named "x" already exists in database "OPENFDA_REST".')
        report = self.do_load(run=run)
        self.assertEqual(report["existed"], 3)
        self.assertGreater(report["ran"], 3)
        self.assertIsNone(report["failed"])

    def test_any_other_failure_stops_the_load_and_is_reported(self):
        def run(sql):
            self.ran.append(sql)
            if len(self.ran) == 4:
                raise SetupError("Zetaris returned 500: boom")
        report = self.do_load(run=run)
        self.assertEqual(len(self.ran), 4)
        self.assertEqual(report["failed"]["message"], "Zetaris returned 500: boom")
        self.assertEqual(report["failed"]["statement"], self.ran[3])

    def test_only_rest_table_creations_are_followed_by_the_delay(self):
        self.do_load()
        tables = sum(1 for s in self.ran if s.startswith("CREATE LIGHTNING REST TABLE"))
        self.assertGreater(tables, 10)
        self.assertEqual(self.slept, [15] * tables)

    def test_dry_run_lists_statements_and_runs_nothing(self):
        lines = []
        report = load(SQL_DIR, delay=15, sleep=self.slept.append, run=self.run_ok, out=lines.append, dry_run=True)
        self.assertEqual((self.ran, self.slept), ([], []))
        self.assertGreater(report["ran"], 0)  # how many statements would run
        self.assertTrue(any("recall_dsy_2024" in line for line in lines))


class Verify(unittest.TestCase):
    def test_reports_which_views_answer_and_which_are_missing(self):
        def run_sql(sql):
            if "recall_dqk_2024_table" in sql:
                raise RuntimeError("TABLE_OR_VIEW_NOT_FOUND")
            return [{"n": "5"}]
        result = verify(run_sql=run_sql, out=lambda *_: None)
        self.assertIn("openfda.classification_cardiovascular_table", result["ok"])
        self.assertEqual(result["missing"], ["openfda.recall_dqk_2024_table"])


class Main(unittest.TestCase):
    def run_main(self, argv, env, load_fn=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv, env=env, load_fn=load_fn)
        return code, out.getvalue() + err.getvalue()

    def test_missing_settings_exit_2_with_a_friendly_message(self):
        code, shown = self.run_main([], env={"ZETARIS_REST_URL": "x"})
        self.assertEqual(code, 2)
        for needle in ("ZETARIS_ORG_ID", "ZETARIS_API_KEY", ".env.example"):
            self.assertIn(needle, shown)

    def test_dry_run_needs_no_settings(self):
        code, shown = self.run_main(["--dry-run"], env={})
        self.assertEqual(code, 0)
        self.assertIn("recall_dsy_2024", shown)

    def test_a_failed_load_exits_1_and_a_good_one_exits_0(self):
        failed = lambda *a, **k: {"ran": 2, "existed": 0, "failed": {"statement": "CREATE X", "message": "boom"}}
        fine = lambda *a, **k: {"ran": 2, "existed": 1, "failed": None}
        self.assertEqual(self.run_main([], env=ZETARIS_ENV, load_fn=failed)[0], 1)
        self.assertEqual(self.run_main([], env=ZETARIS_ENV, load_fn=fine)[0], 0)

    def test_the_failure_message_names_the_statement_and_no_secret_is_printed(self):
        failed = lambda *a, **k: {"ran": 2, "existed": 0, "failed": {"statement": "CREATE X", "message": "boom"}}
        _, shown = self.run_main([], env=ZETARIS_ENV, load_fn=failed)
        self.assertIn("CREATE X", shown)
        self.assertIn("boom", shown)
        self.assertNotIn("secret-key-123", shown)


if __name__ == "__main__":
    unittest.main()
