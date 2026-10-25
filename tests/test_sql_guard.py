"""Tests for the hardened SQL guard (written before the implementation).

A probe showed the old guard (one statement, must start with SELECT or WITH) accepted a write behind a WITH, a read from
any other schema the Zetaris key can see, and the Spark function reflect(). In playbook mode the model never writes SQL,
but in exploratory mode it does, and openFDA text fields can carry injected instructions. The guard runs before any
query reaches Zetaris; it protects Zetaris, not the model bill (the model has already drafted the SQL by then).

    tools_zetaris._check(sql) -> the statement, or raises ValueError. As before: one statement, SELECT or WITH only.
      New rules (words are matched whole, case-insensitively, outside string literals and comments):
        - every table or view in a FROM or JOIN (including comma-separated lists) must be openfda.<name>, or a CTE name
          defined in the same statement, or a parenthesised subquery whose own FROM obeys the same rule
        - these words are refused anywhere: insert, update, delete, merge, create, alter, drop, truncate, grant, revoke,
          call, exec, execute, reflect, java_method
        - a word that merely contains a refused word (update_count) or sits inside a string literal is fine
    main: a question longer than 500 characters is refused up front with exit code 2 and a message; ask is never called.
    evaluator: a claim whose cited SQL breaks the guard is rejected as sql_failed, without a query being sent.
    The queries the agent itself writes (playbook facts, every drill-down code) all pass the guard.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import contextlib
import io
import unittest
from unittest import mock

from agent import tools_zetaris
from agent.evaluator import Citation, Claim, evaluate
from agent.main import main
from agent.playbook import DRILLDOWN_CODES, collect_code_facts, collect_facts
from agent.tools_zetaris import _check, run_sql

ALLOWED = [
    "SELECT 1 AS n",
    "SELECT COUNT(*) AS n FROM openfda.recall_by_product_2024_table",
    "select a.product_code from OPENFDA.classification_cardiovascular_table a",
    "SELECT n FROM `openfda`.`recall_by_product_2024_table`",
    "WITH a AS (SELECT product_code FROM openfda.recall_by_product_2023_table), "
    "b AS (SELECT product_code FROM openfda.recall_by_product_2024_table) "
    "SELECT a.product_code FROM a JOIN b ON a.product_code = b.product_code",
    "SELECT s.n FROM (SELECT COUNT(*) AS n FROM openfda.recall_by_product_2024_table) s",
    "SELECT update_count FROM openfda.recall_by_product_2024_table",
    "SELECT n FROM openfda.recall_by_product_2024_table WHERE note = 'please delete; drop this'",
    "SELECT 1 AS n -- trailing comment",
]

REFUSED = {
    "a write behind a WITH": "WITH a AS (SELECT 1 AS n) INSERT INTO openfda.t SELECT * FROM a",
    "another schema": "SELECT * FROM pokeapi.some_table",
    "another schema in a union": "SELECT 1 UNION SELECT x FROM secret.t",
    "another schema in a join": "SELECT * FROM openfda.a x JOIN secret.b y ON x.k = y.k",
    "another schema in a comma list": "SELECT * FROM openfda.a, secret.b",
    "an unqualified table": "SELECT * FROM recall_by_product_2024_table",
    "another schema in a subquery": "SELECT * FROM (SELECT * FROM secret.t) s",
    "another schema in a CTE body": "WITH a AS (SELECT * FROM secret.t) SELECT * FROM a",
    "reflect": "SELECT reflect('java.lang.System', 'getenv') FROM openfda.t",
    "java_method": "SELECT java_method('java.lang.System', 'getenv') FROM openfda.t",
    "an update": "WITH a AS (SELECT 1 AS n) UPDATE openfda.t SET n = 1",
    "a delete": "WITH a AS (SELECT 1 AS n) DELETE FROM openfda.t",
    "a drop": "DROP TABLE openfda.t",
    "a second statement": "SELECT 1; DROP TABLE openfda.t",
    "mixed case": "WITH a AS (SELECT 1 AS n) InSeRt INTO openfda.t SELECT * FROM a",
}


class Guard(unittest.TestCase):
    def test_allowed_statements_pass(self):
        for sql in ALLOWED:
            with self.subTest(sql=sql):
                self.assertEqual(_check(sql), sql.strip().rstrip(";").strip())

    def test_unsafe_statements_are_refused_with_a_value_error(self):
        for name, sql in REFUSED.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    _check(sql)

    def test_the_message_names_the_rule(self):
        with self.assertRaises(ValueError) as caught:
            _check("SELECT * FROM pokeapi.some_table")
        self.assertIn("openfda", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            _check("WITH a AS (SELECT 1 AS n) INSERT INTO openfda.t SELECT * FROM a")
        self.assertIn("insert", str(caught.exception).lower())


class BeforeTheNetwork(unittest.TestCase):
    def setUp(self):
        tools_zetaris._CACHE.clear()
        self.addCleanup(tools_zetaris._CACHE.clear)

    def test_an_unsafe_statement_never_reaches_zetaris(self):
        with mock.patch("agent.tools_zetaris.requests.post") as post:
            with self.assertRaises(ValueError):
                run_sql(REFUSED["a write behind a WITH"])
            post.assert_not_called()

    def test_the_evaluator_rejects_a_claim_that_cites_unsafe_sql(self):
        claim = Claim("Records in 2024", 482, Citation("SELECT n FROM secret.t", "n"))
        with mock.patch("agent.tools_zetaris.requests.post") as post:
            verdict = evaluate([claim])
            post.assert_not_called()
        self.assertEqual([r.reason for r in verdict.rejections], ["sql_failed"])


class TheAgentsOwnQueries(unittest.TestCase):
    def test_every_playbook_statement_passes_the_guard(self):
        seen = []

        def run(sql):
            seen.append(sql)
            return [{"code": "DSY", "device": "d", "n": "1", "total": "1", "diff": "1", "change": "1", "top3": "1",
                     "pct": "1", "top": "t"}]
        collect_facts(run_sql=run)
        for code in DRILLDOWN_CODES:
            collect_code_facts(code, run_sql=run)
        self.assertTrue(seen)
        for sql in seen:
            with self.subTest(sql=sql[:80]):
                _check(sql)


class QuestionLength(unittest.TestCase):
    FULL = {"ZETARIS_REST_URL": "u", "ZETARIS_ORG_ID": "1", "ZETARIS_API_KEY": "k", "MODEL_BASE_URL": "u",
            "MODEL_API_KEY": "k", "NEMOTRON_MODEL": "m"}

    def test_an_overlong_question_is_refused_before_anything_runs(self):
        called = []
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            code = main(["x" * 501, "--no-memory"], env=self.FULL, ask_fn=lambda *a, **k: called.append(1))
        self.assertEqual(code, 2)
        self.assertEqual(called, [])
        self.assertIn("500", errors.getvalue())

    def test_a_normal_question_is_accepted(self):
        called = []

        def fake_ask(question, memory, **kwargs):
            called.append(question)
            return type("R", (), {"text": "ok"})()
        self.assertEqual(main(["x" * 500, "--no-memory"], env=self.FULL, ask_fn=fake_ask), 0)
        self.assertEqual(len(called), 1)


if __name__ == "__main__":
    unittest.main()
