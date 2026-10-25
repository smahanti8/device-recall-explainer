"""Tests for recording several questions into one sample folder (written before the implementation).

The hackathon guide asks for at least three sample inputs and outputs. Recordings are keyed by SQL text and by the
hash of the model prompt, so several questions can share one folder, but --record used to overwrite the folder.

    sample_mode.start_recording(dir): when the folder already holds recordings, they are loaded first, so stop()
        writes the old and the new entries together. A new entry replaces an old one with the same key.
        recorded_at is the time of the latest recording. A missing or unreadable file starts that file empty.

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
"""
import datetime
import json
import os
import shutil
import tempfile
import unittest

from agent import sample_mode

FIRST = datetime.datetime(2026, 10, 25, 6, 28, 0)
SECOND = datetime.datetime(2026, 10, 25, 9, 0, 0)


def record(directory, sql, rows, messages, text, when):
    sample_mode.start_recording(directory, now=lambda: when)
    sample_mode.record_query(sql, rows)
    sample_mode.record_reply(messages, {"model": "m", "choices": [{"message": {"content": text}}]})
    sample_mode.stop()


class MergeOnRecord(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(sample_mode.stop)

    def test_a_second_recording_keeps_the_first(self):
        record(self.dir, "SELECT 1 AS n", [{"n": "1"}], [{"role": "user", "content": "q1"}], "a1", FIRST)
        record(self.dir, "SELECT 2 AS n", [{"n": "2"}], [{"role": "user", "content": "q2"}], "a2", SECOND)
        sample_mode.start_replay(self.dir)
        self.assertEqual(sample_mode.replay_query("SELECT 1 AS n"), [{"n": "1"}])
        self.assertEqual(sample_mode.replay_query("SELECT 2 AS n"), [{"n": "2"}])
        first = sample_mode.replay_reply([{"role": "user", "content": "q1"}])
        second = sample_mode.replay_reply([{"role": "user", "content": "q2"}])
        self.assertEqual((first["choices"][0]["message"]["content"], second["choices"][0]["message"]["content"]), ("a1", "a2"))

    def test_the_same_key_is_replaced_by_the_newer_recording(self):
        record(self.dir, "SELECT 1 AS n", [{"n": "old"}], [{"role": "user", "content": "q"}], "old", FIRST)
        record(self.dir, "SELECT 1 AS n", [{"n": "new"}], [{"role": "user", "content": "q"}], "new", SECOND)
        sample_mode.start_replay(self.dir)
        self.assertEqual(sample_mode.replay_query("SELECT 1 AS n"), [{"n": "new"}])
        self.assertEqual(sample_mode.replay_reply([{"role": "user", "content": "q"}])["choices"][0]["message"]["content"], "new")

    def test_recorded_at_is_the_latest_recording(self):
        record(self.dir, "SELECT 1 AS n", [{"n": "1"}], [{"role": "user", "content": "q1"}], "a1", FIRST)
        record(self.dir, "SELECT 2 AS n", [{"n": "2"}], [{"role": "user", "content": "q2"}], "a2", SECOND)
        sample_mode.start_replay(self.dir)
        self.assertEqual(sample_mode.recorded_at(), "2026-10-25T09:00:00Z")

    def test_an_unreadable_file_starts_empty_instead_of_failing(self):
        with open(os.path.join(self.dir, sample_mode.QUERIES_FILE), "w") as handle:
            handle.write("not json")
        record(self.dir, "SELECT 2 AS n", [{"n": "2"}], [{"role": "user", "content": "q2"}], "a2", SECOND)
        with open(os.path.join(self.dir, sample_mode.QUERIES_FILE)) as handle:
            self.assertEqual(list(json.load(handle)["queries"]), ["SELECT 2 AS n"])

    def test_a_new_folder_still_records_normally(self):
        fresh = os.path.join(self.dir, "new")
        record(fresh, "SELECT 1 AS n", [{"n": "1"}], [{"role": "user", "content": "q1"}], "a1", FIRST)
        sample_mode.start_replay(fresh)
        self.assertEqual(sample_mode.replay_query("SELECT 1 AS n"), [{"n": "1"}])


if __name__ == "__main__":
    unittest.main()
