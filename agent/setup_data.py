"""Load the Zetaris tables and views the agent queries, with one command.

    python -m agent.setup_data [--sql-dir sql] [--delay 15] [--dry-run] [--verify]

Statements run in file order. A statement whose object already exists is skipped; any other failure stops
the load, because later statements depend on earlier ones. REST table creations are followed by a pause:
each one calls openFDA, which rate-limits keyless clients. No setting's value is ever printed.
"""
import argparse
import os
import sys
import time
import uuid

import requests
from dotenv import load_dotenv

load_dotenv()

SQL_FILES = ("sql/openfda_spike.sql", "sql/openfda_register.sql",
             "sql/openfda_drilldown.sql", "sql/openfda_drilldown_more.sql")
DRILLDOWN_VIEWS = tuple(f"openfda.recall_{code}_2024_table"
                        for code in ("dsy", "dqo", "oez", "dqk", "dpt", "dre", "dwf", "oes", "dyf", "oeq"))
VERIFY_VIEWS = ("openfda.classification_cardiovascular_table", "openfda.recall_by_product_2023_table",
                "openfda.recall_by_product_2024_table", "openfda.recall_rootcause_all_2023_table",
                "openfda.recall_rootcause_all_2024_table") + DRILLDOWN_VIEWS
REQUIRED = ("ZETARIS_REST_URL", "ZETARIS_ORG_ID", "ZETARIS_API_KEY")


class SetupError(Exception):
    pass


def statements(text):
    """Split SQL text into statements: full-line comments are dropped; quoted semicolons do not split."""
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("--"))
    found, current, quote = [], [], None
    for char in body:
        if quote:
            current.append(char)
            quote = None if char == quote else quote
        elif char in "\"'`":
            quote = char
            current.append(char)
        elif char == ";":
            statement = "".join(current).strip()
            if statement:
                found.append(statement)
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        found.append(tail)
    return found


def _post(sql, env):
    base = env["ZETARIS_REST_URL"].rstrip("/")
    response = requests.post(
        f"{base}/sql-editor/sqls/run",
        headers={"Authorization": "Bearer " + env["ZETARIS_API_KEY"], "X-Org-ID": env["ZETARIS_ORG_ID"],
                 "X-Request-ID": str(uuid.uuid4())},
        json={"queryId": str(uuid.uuid4()), "sql": sql, "source": "SqlEditor", "limit": 1},
        timeout=120,
    )
    if not response.ok:
        raise SetupError(f"Zetaris returned {response.status_code}: {response.text[:300]}")


def _label(sql):
    return sql.splitlines()[0][:110]


def load(sql_dir, delay=15, sleep=time.sleep, run=None, out=print, dry_run=False, env=None):
    env = os.environ if env is None else env
    run = run or (lambda sql: _post(sql, env))
    report = {"ran": 0, "existed": 0, "failed": None}
    for name in SQL_FILES:
        with open(os.path.join(sql_dir, os.path.basename(name))) as handle:
            parsed = statements(handle.read())
        for sql in parsed:
            if dry_run:
                out(f"would run: {_label(sql)}")
                report["ran"] += 1
                continue
            try:
                run(sql)
            except SetupError as error:
                if "already exists" in str(error):
                    report["existed"] += 1
                    out(f"exists: {_label(sql)}")
                    continue
                report["failed"] = {"statement": sql, "message": str(error)}
                out(f"FAILED: {_label(sql)}: {error}")
                return report
            report["ran"] += 1
            out(f"ok: {_label(sql)}")
            if sql.upper().startswith("CREATE LIGHTNING REST TABLE"):
                sleep(delay)
    return report


def verify(run_sql=None, out=print):
    from agent import tools_zetaris
    run_sql = run_sql or tools_zetaris.run_sql
    result = {"ok": [], "missing": []}
    for view in VERIFY_VIEWS:
        try:
            rows = run_sql(f"SELECT COUNT(*) AS n FROM {view}")
            result["ok"].append(view)
            out(f"verified: {view} ({rows[0]['n']} rows)")
        except Exception as error:
            result["missing"].append(view)
            out(f"missing: {view} ({str(error)[:80]})")
    return result


def _parse(argv):
    parser = argparse.ArgumentParser(prog="python -m agent.setup_data", description="Register the Zetaris tables and views.")
    parser.add_argument("--sql-dir", default="sql")
    parser.add_argument("--delay", type=float, default=15, help="seconds to wait after each REST table creation")
    parser.add_argument("--dry-run", action="store_true", help="list the statements without running them")
    parser.add_argument("--verify", action="store_true", help="afterwards, check that the key views answer")
    return parser.parse_args(argv)


def main(argv=None, env=None, load_fn=None):
    args = _parse(argv)
    env_map = os.environ if env is None else env
    if not args.dry_run:
        missing = [name for name in REQUIRED if not str(env_map.get(name, "")).strip()]
        if missing:
            print(f"Missing settings: {', '.join(missing)}.\n"
                  "Copy .env.example to .env and fill them in (see Setup in the README).", file=sys.stderr)
            return 2
    report = (load_fn or load)(args.sql_dir, delay=args.delay, dry_run=args.dry_run, env=env_map)
    if report["failed"]:
        print(f"Load stopped at: {_label(report['failed']['statement'])}\n{report['failed']['message']}", file=sys.stderr)
        return 1
    print(f"{report['ran']} statement(s) {'listed' if args.dry_run else 'run'}, {report['existed']} already existed.")
    if args.verify and not args.dry_run:
        return 1 if verify()["missing"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
