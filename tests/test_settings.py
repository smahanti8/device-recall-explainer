"""Tests for the settings check in agent/main.py (written before the implementation).

    REQUIRED_SETTINGS                 the six variables the agent reads, in .env.example order
    missing_settings(env=None) -> []  the required names that are unset or blank (env defaults to os.environ)
    main(argv=None, env=None, ask_fn=ask) -> exit code
        When anything is missing: print a message to stderr that names each missing variable and points at
        .env.example, return 2, and call neither the model nor Zetaris (ask_fn is never called).
        Otherwise run ask_fn(question, memory, verbose=..., playbook=...) and return 0 for a verified answer.
        --help needs no settings.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import contextlib
import io
import unittest

from agent.loop import Result
from agent.main import REQUIRED_SETTINGS, main, missing_settings

EXPECTED = ["ZETARIS_REST_URL", "ZETARIS_ORG_ID", "ZETARIS_API_KEY",
            "MODEL_BASE_URL", "MODEL_API_KEY", "NEMOTRON_MODEL"]
FULL = {name: "x" for name in EXPECTED}


def never_called(*args, **kwargs):
    raise AssertionError("ask must not run when settings are missing")


class MissingSettings(unittest.TestCase):
    def test_the_six_settings_match_env_example_order(self):
        self.assertEqual(list(REQUIRED_SETTINGS), EXPECTED)

    def test_nothing_set_lists_all_six(self):
        self.assertEqual(missing_settings({}), EXPECTED)

    def test_blank_values_count_as_missing(self):
        env = dict(FULL, ZETARIS_ORG_ID="  ", MODEL_API_KEY="")
        self.assertEqual(missing_settings(env), ["ZETARIS_ORG_ID", "MODEL_API_KEY"])

    def test_everything_set_is_fine(self):
        self.assertEqual(missing_settings(FULL), [])


class Main(unittest.TestCase):
    def test_a_missing_setting_stops_with_a_friendly_message_before_any_work(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            code = main(["why did recalls rise?"], env={"ZETARIS_REST_URL": "x"}, ask_fn=never_called)
        shown = errors.getvalue()
        self.assertEqual(code, 2)
        for needle in ("MODEL_BASE_URL", "NEMOTRON_MODEL", ".env.example"):
            self.assertIn(needle, shown)
        self.assertNotIn("Traceback", shown)
        self.assertNotIn("ZETARIS_REST_URL", shown)  # that one was set, so it is not reported

    def test_it_runs_when_everything_is_set(self):
        calls = []

        def fake_ask(question, memory, **kwargs):
            calls.append((question, memory, kwargs))
            return Result("ok", [], 1, [], claims=[])
        self.assertEqual(main(["why did recalls rise?", "--no-memory", "--playbook"], env=FULL, ask_fn=fake_ask), 0)
        question, memory, kwargs = calls[0]
        self.assertEqual((question, memory), ("why did recalls rise?", None))
        self.assertTrue(kwargs["playbook"])

    def test_an_answer_that_could_not_be_verified_exits_1(self):
        gave_up = lambda question, memory, **kwargs: Result(None, [], 3, [])
        self.assertEqual(main(["why did recalls rise?", "--no-memory"], env=FULL, ask_fn=gave_up), 1)

    def test_help_needs_no_settings(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            main(["--help"], env={}, ask_fn=never_called)
        self.assertEqual(stopped.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
