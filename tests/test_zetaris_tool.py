"""Tests for agent/tools_zetaris.py (written before the implementation).

run_sql(sql) sends ONE read-only SQL statement to Zetaris over REST and returns
a list of dicts (column name -> value). Integration tests need Zetaris running
and the ZETARIS_* variables in .env; unit tests do not.

Expected numbers come from an independent oracle: direct openFDA calls made on
2026-10-24 (not from Zetaris), so a Zetaris/tool bug cannot hide itself.

Run: .venv/bin/python -m unittest discover -s tests -v
"""
import os
import unittest

from dotenv import load_dotenv

from agent.tools_zetaris import run_sql

load_dotenv()
HAVE_ZETARIS = all(os.environ.get(k) for k in ("ZETARIS_REST_URL", "ZETARIS_ORG_ID", "ZETARIS_API_KEY"))
integration = unittest.skipUnless(HAVE_ZETARIS, "Zetaris settings missing in .env")

JOIN_TOP3 = """
SELECT c.product_code, COALESCE(b.recall_count, 0) - COALESCE(a.recall_count, 0) AS change
FROM openfda.classification_cardiovascular_table c
LEFT JOIN openfda.recall_by_product_2023_table a ON a.product_code = c.product_code
LEFT JOIN openfda.recall_by_product_2024_table b ON b.product_code = c.product_code
ORDER BY change DESC LIMIT 3
"""

DSY_SHAPE = """
SELECT COUNT(*) AS records, COUNT(DISTINCT res_event_number) AS events,
       COUNT(DISTINCT recalling_firm) AS firms
FROM openfda.recall_dsy_2024_table
"""


def one(sql):
    rows = run_sql(sql)
    assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
    return rows[0]


class SafetyRules(unittest.TestCase):
    """Pure unit tests: the tool refuses before any network call."""

    def test_rejects_multiple_statements(self):
        with self.assertRaises(ValueError):
            run_sql("SELECT 1; SELECT 2")

    def test_rejects_non_select(self):
        for sql in ("DROP VIEW openfda.x", "CREATE SCHEMASTORE CONTAINER x",
                    "INSERT INTO t VALUES (1)", "DELETE FROM t"):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                run_sql(sql)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            run_sql("   ")


@integration
class ThroughZetaris(unittest.TestCase):
    def test_select_one(self):
        self.assertEqual(str(one("SELECT 1 AS one")["one"]), "1")

    def test_layer_counts(self):
        for view, expected in (("classification_cardiovascular_table", 349),
                               ("recall_by_product_2023_table", 587),
                               ("recall_by_product_2024_table", 626)):
            with self.subTest(view=view):
                n = one(f"SELECT COUNT(*) AS n FROM openfda.{view}")["n"]
                self.assertEqual(int(n), expected)

    def test_top_three_increases(self):
        rows = run_sql(JOIN_TOP3)
        got = [(r["product_code"], int(r["change"])) for r in rows]
        self.assertEqual(got, [("DSY", 53), ("DQO", 49), ("OEZ", 29)])

    def test_dsy_is_one_event(self):
        row = one(DSY_SHAPE)
        self.assertEqual((int(row["records"]), int(row["events"]), int(row["firms"])), (54, 1, 1))


if __name__ == "__main__":
    unittest.main()
