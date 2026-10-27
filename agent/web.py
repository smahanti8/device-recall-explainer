"""A local page over the command line: ask a question in the browser and watch the same run.

Standard library only. The server never runs the agent itself: it starts `python -m agent.main ... --json --report`
as a subprocess, so the command line stays the one entry point and every run gets clean process state.
It binds to 127.0.0.1, runs one question at a time and only answers requests that came from its own page,
because a live run spends the user's API keys.

    python -m agent.web [--port 8765]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent.main import MAX_QUESTION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
REPORT_NAME = re.compile(r"^[\w-][\w.-]*\.html$")
PAGE_CSP = ("default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
            "connect-src 'self'; frame-src 'self'")
REPORT_CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:"


def build_command(question, playbook, sample, report_dir, log_dir, memory_dir, timeout=600, python=None):
    cmd = [python or sys.executable, "-m", "agent.main", "--json", "--report", "--report-dir", report_dir,
           "--log-dir", log_dir, "--memory-dir", memory_dir, "--timeout", str(timeout)]
    if playbook:
        cmd.append("--playbook")
    if sample:
        cmd.append("--sample")
    return cmd + ["--", question]  # after "--" the question can never be read as a flag


def run_cli(cmd, on_line):
    """Run the command line; each stderr line goes to on_line as it appears. Returns (exit code, stdout)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)
    out = []
    reader = threading.Thread(target=lambda: out.append(proc.stdout.read()))
    reader.start()
    for line in proc.stderr:
        on_line(line.rstrip("\r\n"))
    reader.join()
    return proc.wait(), "".join(out)


def _document(stdout):
    """The one JSON document the CLI printed, or None."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    try:
        document = json.loads(lines[-1])
    except (IndexError, ValueError):
        return None
    return document if isinstance(document, dict) else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status, body, content_type, csp):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Security-Policy", csp)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status, message):
        self._send(status, json.dumps({"error": message}), "application/json", "default-src 'none'")

    def _own_host(self):
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def do_GET(self):
        if not self._own_host():
            return self._error(403, "unexpected Host header")
        path = self.path.split("?", 1)[0]
        if path == "/":
            with open(os.path.join(STATIC, "index.html"), "rb") as handle:
                return self._send(200, handle.read(), "text/html; charset=utf-8", PAGE_CSP)
        if path.startswith("/report/"):
            name = path[len("/report/"):]
            full = os.path.join(self.server.report_dir, name)
            if REPORT_NAME.match(name) and os.path.isfile(full):
                with open(full, "rb") as handle:
                    return self._send(200, handle.read(), "text/html; charset=utf-8", REPORT_CSP)
        self._error(404, "not found")

    def do_POST(self):
        if not self._own_host():
            return self._error(403, "unexpected Host header")
        if self.path != "/run":
            return self._error(404, "not found")
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self._error(415, "send application/json")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
            question = body["question"].strip()
        except (ValueError, TypeError, KeyError, AttributeError):
            return self._error(400, "send a JSON object with a question")
        if not question or len(question) > MAX_QUESTION:
            return self._error(400, f"the question must be 1 to {MAX_QUESTION} characters")
        if not self.server.busy.acquire(blocking=False):
            return self._error(409, "another question is running; wait for it to finish")
        try:
            self._stream(question, bool(body.get("playbook")), bool(body.get("sample")))
        finally:
            self.server.busy.release()

    def _event(self, name, data):
        try:
            self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode("utf-8"))
            self.wfile.flush()
        except OSError:  # the browser went away; the run still finishes and is logged
            pass

    def _stream(self, question, playbook, sample):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Security-Policy", "default-src 'none'")
        self.end_headers()
        server = self.server
        cmd = build_command(question, playbook, sample, server.report_dir, server.log_dir, server.memory_dir)
        last = []

        def on_line(line):
            last.append(line)
            self._event("progress", line)
        try:
            code, stdout = server.runner(cmd, on_line)
        except OSError as error:
            return self._event("error", f"could not start the run: {error}")
        document = _document(stdout)
        if document is None:
            return self._event("error", last[-1] if last else f"the run ended with exit code {code} and no result")
        self._event("result", json.dumps(document))


def make_server(port=0, runner=run_cli, report_dir="reports", log_dir="logs", memory_dir=".hmem"):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.runner = runner
    server.busy = threading.Lock()
    server.report_dir, server.log_dir, server.memory_dir = (os.path.join(ROOT, d) for d in
                                                              (report_dir, log_dir, memory_dir))
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m agent.web", description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = make_server(args.port)
    print(f"Device Recall Explainer: http://127.0.0.1:{args.port}/   (Ctrl+C to stop)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
