"""Tests for the local web page (written before the implementation).

A thin page over the command line, standard library only. The server never runs the agent itself: it starts
`python -m agent.main ... --json --report` as a subprocess, so the CLI stays the one entry point and every run
gets clean process state (sample mode, SQL cache, run log and deadline are process-wide).

    web.build_command(question, playbook, sample, report_dir, log_dir, memory_dir, timeout=600, python=None) -> list
        [python, "-m", "agent.main", "--json", "--report", flags..., "--", question]
        The question comes after "--", so a question that starts with "-" can never be read as a flag.
    web.make_server(port=0, runner=web.run_cli, report_dir="reports", log_dir="logs", memory_dir=".hmem")
        A ThreadingHTTPServer bound to 127.0.0.1 only.
        runner(cmd, on_line) -> (exit_code, stdout_text); on_line gets each stderr line as it appears.
    GET /                  the page (agent/static/index.html)
    POST /run              JSON body {"question", "playbook", "sample"}; answers with a server-sent event stream:
                             event: progress   data: one stderr line
                             event: result     data: the CLI's JSON document (also for an unverified or refused run)
                             event: error      data: a message, when the CLI printed no JSON document
                           400 for an empty question or one over 500 characters, 409 while another run is active.
    GET /report/<name>     a report written by a run: only <report_dir>/<name>.html, no path separators.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import http.client
import json
import os
import re
import tempfile
import threading
import unittest

from agent import web

QUESTION = ("Why did cardiovascular device recall records rise between 2023 and 2024, "
            "and which product codes drove it?")
PAGE = os.path.join(os.path.dirname(web.__file__), "static", "index.html")


def fake_runner(lines=(), code=0, stdout='{"verified": true}', gate=None, started=None):
    calls = []

    def run(cmd, on_line):
        calls.append(cmd)
        if started:
            started.set()
        if gate:
            gate.wait(5)
        for line in lines:
            on_line(line)
        return code, stdout
    run.calls = calls
    return run


class Served:
    """A server on an ephemeral port, running in a thread, closed after the test."""

    def __init__(self, test, **kwargs):
        self.server = web.make_server(0, **kwargs)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        test.addCleanup(self.close)

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=60)
        conn.request(method, path, body=None if body is None else json.dumps(body),
                     headers={"Content-Type": "application/json", **(headers or {})})
        response = conn.getresponse()
        data = response.read().decode()
        conn.close()
        return response.status, data

    def run(self, **body):
        return self.request("POST", "/run", {"question": QUESTION, **body})


def events(text):
    """[(event, data)] from a server-sent event stream."""
    found = []
    for block in text.strip().split("\n\n"):
        name = re.search(r"^event: (.+)$", block, re.M)
        data = re.search(r"^data: (.*)$", block, re.M)
        if name and data:
            found.append((name.group(1), data.group(1)))
    return found


class TestCommand(unittest.TestCase):
    def build(self, question=QUESTION, **kw):
        kw.setdefault("playbook", False)
        kw.setdefault("sample", False)
        return web.build_command(question, report_dir="r", log_dir="l", memory_dir="m", python="py", **kw)

    def test_the_question_comes_after_a_double_dash_so_it_cannot_be_a_flag(self):
        cmd = self.build("--timeout 0 why did recalls rise?")
        self.assertEqual(cmd[-2:], ["--", "--timeout 0 why did recalls rise?"])
        self.assertEqual(cmd[:3], ["py", "-m", "agent.main"])

    def test_always_asks_for_json_and_a_report_in_the_given_directories(self):
        cmd = self.build()
        self.assertIn("--json", cmd)
        self.assertIn("--report", cmd)
        self.assertEqual(cmd[cmd.index("--report-dir") + 1], "r")
        self.assertEqual(cmd[cmd.index("--log-dir") + 1], "l")
        self.assertEqual(cmd[cmd.index("--memory-dir") + 1], "m")

    def test_checkboxes_map_to_flags(self):
        self.assertNotIn("--playbook", self.build())
        self.assertNotIn("--sample", self.build())
        cmd = self.build(playbook=True, sample=True)
        self.assertIn("--playbook", cmd)
        self.assertIn("--sample", cmd)

    def test_the_run_timeout_is_passed_through(self):
        self.assertEqual(self.build(timeout=120)[self.build(timeout=120).index("--timeout") + 1], "120")


class TestServer(unittest.TestCase):
    def test_binds_to_loopback_only(self):
        served = Served(self, runner=fake_runner())
        self.assertEqual(served.server.server_address[0], "127.0.0.1")

    def test_serves_the_page_with_a_form(self):
        status, body = Served(self, runner=fake_runner()).request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("<form", body)
        self.assertIn("<textarea", body)

    def test_empty_and_overlong_questions_are_rejected_before_anything_starts(self):
        runner = fake_runner()
        served = Served(self, runner=runner)
        self.assertEqual(served.request("POST", "/run", {"question": "   "})[0], 400)
        self.assertEqual(served.request("POST", "/run", {"question": "x" * 501})[0], 400)
        self.assertEqual(served.request("POST", "/run", {})[0], 400)
        self.assertEqual(served.request("POST", "/run", None)[0], 400)
        self.assertEqual(runner.calls, [])

    def test_progress_lines_stream_in_order_then_one_result(self):
        runner = fake_runner(lines=["round 1: drafting", "round 1: 4 claims checked, all verified"],
                             stdout='{"verified": true, "answer": "A [1]"}')
        status, body = Served(self, runner=runner).run(playbook=True, sample=True)
        self.assertEqual(status, 200)
        self.assertEqual(events(body), [("progress", "round 1: drafting"),
                                        ("progress", "round 1: 4 claims checked, all verified"),
                                        ("result", '{"verified": true, "answer": "A [1]"}')])
        self.assertIn("--playbook", runner.calls[0])
        self.assertIn("--sample", runner.calls[0])

    def test_an_unverified_or_refused_run_still_returns_its_document(self):
        refusal = '{"question": "q", "verified": false, "reason": "out_of_scope", "message": "not about devices"}'
        for code in (1, 2):
            status, body = Served(self, runner=fake_runner(code=code, stdout=refusal)).run()
            self.assertEqual(events(body), [("result", refusal)])

    def test_no_json_document_is_reported_as_an_error_not_a_crash(self):
        status, body = Served(self, runner=fake_runner(lines=["Missing settings: X."], code=2, stdout="")).run()
        self.assertEqual([name for name, _ in events(body)], ["progress", "error"])
        status, body = Served(self, runner=fake_runner(stdout="not json")).run()
        self.assertEqual([name for name, _ in events(body)], ["error"])

    def test_refuses_requests_that_did_not_come_from_its_own_page(self):
        """The server can spend the user's API keys, so a web page on another origin must not be able to drive it."""
        runner = fake_runner()
        served = Served(self, runner=runner)
        # DNS rebinding: a foreign hostname pointing at 127.0.0.1
        self.assertEqual(served.request("POST", "/run", {"question": QUESTION}, {"Host": "evil.example"})[0], 403)
        self.assertEqual(served.request("GET", "/", None, {"Host": "evil.example"})[0], 403)
        # cross-site form posts cannot send application/json without a preflight the server never grants
        self.assertEqual(served.request("POST", "/run", {"question": QUESTION},
                                        {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(runner.calls, [])
        port = served.server.server_address[1]
        self.assertEqual(served.request("GET", "/", None, {"Host": f"localhost:{port}"})[0], 200)

    def test_a_second_run_while_one_is_active_gets_409(self):
        gate, started = threading.Event(), threading.Event()
        served = Served(self, runner=fake_runner(gate=gate, started=started))
        first = threading.Thread(target=served.run)
        first.start()
        try:
            self.assertTrue(started.wait(10), "the first run never started")
            self.assertEqual(served.run()[0], 409)
        finally:
            gate.set()
            first.join(10)
        self.assertEqual(served.run()[0], 200)  # free again once the first run ended


class TestReports(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        with open(os.path.join(self.dir.name, "report-1.html"), "w") as handle:
            handle.write("<html>report</html>")
        with open(os.path.join(self.dir.name, "notes.txt"), "w") as handle:
            handle.write("secret")
        self.served = Served(self, runner=fake_runner(), report_dir=self.dir.name)

    def test_serves_a_report_from_the_report_directory(self):
        self.assertEqual(self.served.request("GET", "/report/report-1.html"), (200, "<html>report</html>"))

    def test_rejects_traversal_other_extensions_and_missing_files(self):
        for path in ("/report/../notes.txt", "/report/..%2Fnotes.txt", "/report/notes.txt",
                     "/report/%2e%2e/%2e%2e/etc/passwd", "/report/missing.html", "/report/"):
            self.assertEqual(self.served.request("GET", path)[0], 404, path)


class TestPage(unittest.TestCase):
    def test_run_text_is_never_inserted_as_html(self):
        with open(PAGE, encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("innerHTML", source)
        self.assertNotIn("insertAdjacentHTML", source)
        self.assertNotIn("document.write", source)

    def test_the_page_names_the_replay_mode_so_it_never_looks_live(self):
        with open(PAGE, encoding="utf-8") as handle:
            source = handle.read().lower()
        self.assertIn("replay", source)


class TestRealCommandLine(unittest.TestCase):
    """The real CLI in sample mode: no services, no keys. Logs, reports and memory go to a temporary directory."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.served = Served(self, report_dir=os.path.join(self.tmp.name, "reports"),
                             log_dir=os.path.join(self.tmp.name, "logs"),
                             memory_dir=os.path.join(self.tmp.name, "hmem"))

    def test_a_recorded_question_comes_back_verified_with_a_readable_report(self):
        status, body = self.served.run(playbook=True, sample=True)
        found = events(body)
        self.assertEqual(found[-1][0], "result", found[-3:])
        document = json.loads(found[-1][1])
        self.assertTrue(document["verified"])
        self.assertGreater(len(document["claims"]), 0)
        self.assertTrue(any(name == "progress" and "round 1" in data for name, data in found))
        report = os.path.basename(document["report"])
        status, page = self.served.request("GET", f"/report/{report}")
        self.assertEqual(status, 200)
        self.assertIn("SAMPLE MODE", page)

    def test_an_out_of_scope_question_is_refused_with_a_message(self):
        status, body = self.served.request("POST", "/run", {"question": "What is the capital of France?",
                                                            "sample": True})
        document = json.loads(events(body)[-1][1])
        self.assertFalse(document["verified"])
        self.assertEqual(document["reason"], "out_of_scope")


if __name__ == "__main__":
    unittest.main()
