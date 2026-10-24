"""Tests for agent/drafter.py (written before the implementation).

make_draft_fn(chat=agent.llm.chat, run_sql=agent.tools_zetaris.run_sql, max_steps=6)
returns draft_fn(question, feedback) -> Draft, for use with agent.loop.investigate.

PROTOCOL (the model replies with one JSON object per turn, optionally in a ```json fence)
    {"action": "sql", "sql": "SELECT ..."}                      run one read-only query
    {"action": "answer", "text": "...", "claims": [
        {"text": "...", "value": 54, "sql": "SELECT ...", "column": "n", "row": 0}]}
The drafter runs requested SQL and feeds the rows (or the error) back as the next user
message. The model never computes a number itself: each claim cites a SQL cell.
If the model never answers within max_steps, the draft has no claims (the loop rejects it).

Run: .venv/bin/python -W ignore -m unittest discover -s tests -v
Live test (slow, real model + Zetaris): RUN_LIVE_AGENT=1 .venv/bin/python -W ignore -m unittest tests.test_drafter -v
"""
import copy
import itertools
import json
import os
import unittest

from agent.drafter import make_draft_fn
from agent.evaluator import Rejection, Claim, Citation
from agent.llm import LLMError, Reply

SQL = "SELECT COUNT(*) AS n FROM openfda.recall_dsy_2024_table"
ASK = json.dumps({"action": "sql", "sql": SQL})
SQL2 = "SELECT COUNT(DISTINCT res_event_number) AS n FROM openfda.recall_dsy_2024_table"
ANSWER = json.dumps({"action": "answer", "text": "DSY had 54 recall records in 2024.",
                     "claims": [{"text": "DSY recall records in 2024", "value": 54,
                                 "sql": SQL, "column": "n", "row": 0}]})


def scripted(*replies):
    """A fake chat: returns the replies in order and records every message list it was given."""
    calls, queue = [], iter(replies)

    def chat(messages, max_tokens=2000):
        calls.append(copy.deepcopy(messages))
        item = next(queue)
        if isinstance(item, Exception):
            raise item
        return Reply(item, "nvidia/test", None)
    chat.calls = calls
    return chat


def rows_54(sql):
    rows_54.seen.append(sql)
    return [{"n": "54"}]


def fresh_rows():
    rows_54.seen = []
    return rows_54


class Drafter(unittest.TestCase):
    def test_runs_requested_sql_then_returns_cited_claims(self):
        chat, run = scripted(ASK, ANSWER), fresh_rows()
        draft = make_draft_fn(chat=chat, run_sql=run)("why?", [])
        self.assertEqual(run.seen, [SQL])
        self.assertEqual(draft.text, "DSY had 54 recall records in 2024.")
        self.assertEqual(draft.claims, [Claim("DSY recall records in 2024", 54, Citation(SQL, "n", 0))])
        self.assertIn("54", json.dumps(chat.calls[1]))  # the rows were fed back to the model

    def test_sql_error_is_fed_back_not_raised(self):
        def broken(sql):
            raise ValueError("only SELECT/WITH statements are allowed")
        chat = scripted(ASK, ANSWER)
        make_draft_fn(chat=chat, run_sql=broken)("why?", [])
        self.assertIn("only SELECT/WITH", json.dumps(chat.calls[1]))

    def test_malformed_reply_is_asked_again(self):
        chat = scripted("sure, here you go!", ANSWER)
        draft = make_draft_fn(chat=chat, run_sql=fresh_rows())("why?", [])
        self.assertEqual(len(draft.claims), 1)
        self.assertIn("JSON", chat.calls[1][-1]["content"])

    def test_json_fence_is_accepted(self):
        draft = make_draft_fn(chat=scripted("```json\n" + ANSWER + "\n```"), run_sql=fresh_rows())("why?", [])
        self.assertEqual(len(draft.claims), 1)

    def test_claim_without_sql_gets_no_citation(self):
        bare = json.dumps({"action": "answer", "text": "x", "claims": [{"text": "t", "value": 1}]})
        draft = make_draft_fn(chat=scripted(bare), run_sql=fresh_rows())("why?", [])
        self.assertIsNone(draft.claims[0].citation)

    def test_gives_up_after_max_steps_with_no_claims(self):
        chat = scripted(*itertools.repeat(ASK, 10))
        draft = make_draft_fn(chat=chat, run_sql=fresh_rows(), max_steps=3)("why?", [])
        self.assertEqual(draft.claims, [])
        self.assertEqual(len(chat.calls), 3)

    def test_long_results_are_truncated_before_going_back_to_the_model(self):
        chat = scripted(ASK, ANSWER)
        make_draft_fn(chat=chat, run_sql=lambda sql: [{"n": str(i)} for i in range(500)])("why?", [])
        fed_back = json.dumps(chat.calls[1])
        self.assertIn("truncated", fed_back)
        self.assertNotIn("499", fed_back)  # rows 50..499 were cut (json.dumps would escape quotes, so test the digits)

    def test_last_step_tells_the_model_to_answer_now(self):
        chat = scripted(*itertools.repeat(ASK, 10))
        make_draft_fn(chat=chat, run_sql=fresh_rows(), max_steps=3)("why?", [])
        self.assertIn("last step", chat.calls[-1][-1]["content"])
        self.assertNotIn("last step", chat.calls[0][-1]["content"])

    def test_system_prompt_turns_off_thinking(self):
        chat = scripted(ANSWER)
        make_draft_fn(chat=chat, run_sql=fresh_rows())("why?", [])
        self.assertTrue(chat.calls[0][0]["content"].startswith("/no_think"))

    # --- fewer, faster turns -------------------------------------------------------
    def test_several_queries_in_one_turn_run_together(self):
        two = json.dumps({"action": "sql", "sqls": [SQL, SQL2]})
        chat, run = scripted(two, ANSWER), fresh_rows()
        make_draft_fn(chat=chat, run_sql=run)("why?", [])
        self.assertEqual(run.seen, [SQL, SQL2])
        fed_back = chat.calls[1][-1]["content"]
        self.assertIn("Query 1", fed_back)
        self.assertIn("Query 2", fed_back)
        self.assertEqual(len(chat.calls), 2)  # one model turn for both queries

    def test_a_batch_is_capped_at_three_queries(self):
        many = json.dumps({"action": "sql", "sqls": [f"SELECT {i} AS n" for i in range(5)]})
        chat, run = scripted(many, ANSWER), fresh_rows()
        make_draft_fn(chat=chat, run_sql=run)("why?", [])
        self.assertEqual(len(run.seen), 3)
        self.assertIn("only the first 3", chat.calls[1][-1]["content"])

    def test_system_prompt_has_a_playbook_for_rise_questions(self):
        chat = scripted(ANSWER)
        make_draft_fn(chat=chat, run_sql=fresh_rows())("why?", [])
        content = chat.calls[0][0]["content"].lower()
        for needle in ("playbook", "step 1", "step 2", "step 3", "distinct"):
            self.assertIn(needle, content)

    # --- recovering from a model timeout ---------------------------------------------
    def test_a_failed_turn_is_retried_once_with_a_trimmed_history(self):
        chat = scripted(ASK, ASK, LLMError("502 provider_deadline"), ANSWER)
        draft = make_draft_fn(chat=chat, run_sql=fresh_rows())("why?", [])
        self.assertEqual(len(draft.claims), 1)
        self.assertEqual(len(chat.calls), 4)
        failed, retried = chat.calls[2], chat.calls[3]
        self.assertEqual(len(failed), 6)   # system, question, 2 x (reply + result)
        self.assertEqual(len(retried), 4)  # system, question, latest reply + result only
        self.assertEqual(retried[0]["role"], "system")
        self.assertIn("why?", retried[1]["content"])
        self.assertEqual(retried[-1], failed[-1])  # the latest results are kept

    def test_a_second_failure_in_a_row_is_raised(self):
        chat = scripted(LLMError("502 provider_deadline"), LLMError("502 provider_deadline"), ANSWER)
        with self.assertRaises(LLMError):
            make_draft_fn(chat=chat, run_sql=fresh_rows())("why?", [])
        self.assertEqual(len(chat.calls), 2)

    # --- optional memory context (hints, never evidence) ---------------------------
    HINT = "Previously verified in earlier sessions (hints, not evidence): DSY had 54 records."

    def test_memory_context_follows_the_question_and_the_feedback(self):
        rejection = Rejection(Claim("x", 80), "value_mismatch", "claimed 80 but ...")
        chat = scripted(ANSWER)
        make_draft_fn(chat=chat, run_sql=fresh_rows(), context_fn=lambda q: self.HINT)("why?", [rejection])
        opening = chat.calls[0][1]["content"]
        self.assertTrue(opening.startswith("why?"))
        self.assertLess(opening.index("value_mismatch"), opening.index(self.HINT))

    def test_empty_or_missing_context_changes_nothing(self):
        plain = scripted(ANSWER)
        make_draft_fn(chat=plain, run_sql=fresh_rows())("why?", [])
        for context in ("", None):
            with self.subTest(context=context):
                chat = scripted(ANSWER)
                make_draft_fn(chat=chat, run_sql=fresh_rows(), context_fn=lambda q: context)("why?", [])
                self.assertEqual(chat.calls[0], plain.calls[0])

    def test_a_raising_context_fn_does_not_break_drafting(self):
        def broken(question):
            raise RuntimeError("memory store unreadable")
        draft = make_draft_fn(chat=scripted(ANSWER), run_sql=fresh_rows(), context_fn=broken)("why?", [])
        self.assertEqual(len(draft.claims), 1)

    def test_context_is_requested_once_per_draft_and_survives_a_trimmed_retry(self):
        asked = []
        chat = scripted(ASK, ASK, LLMError("502 provider_deadline"), ANSWER)
        make_draft_fn(chat=chat, run_sql=fresh_rows(),
                      context_fn=lambda q: asked.append(q) or self.HINT)("why?", [])
        self.assertEqual(asked, ["why?"])
        self.assertIn(self.HINT, chat.calls[3][1]["content"])  # the trimmed history keeps the opening message

    def test_system_prompt_says_recalled_hints_are_not_evidence(self):
        chat = scripted(ANSWER)
        make_draft_fn(chat=chat, run_sql=fresh_rows())("why?", [])
        content = chat.calls[0][0]["content"].lower()
        for needle in ("recalled hints", "not evidence", "fresh sql"):
            self.assertIn(needle, content)

    def test_previous_rejections_reach_the_model(self):
        rejection = Rejection(Claim("x", 80), "value_mismatch", "claimed 80 but SELECT ... returned '54'")
        chat = scripted(ANSWER)
        make_draft_fn(chat=chat, run_sql=fresh_rows())("why?", [rejection])
        first = json.dumps(chat.calls[0])
        self.assertIn("value_mismatch", first)
        self.assertIn("claimed 80", first)

    def test_system_prompt_names_the_views_and_the_records_vs_events_rule(self):
        chat = scripted(ANSWER)
        make_draft_fn(chat=chat, run_sql=fresh_rows())("why?", [])
        system = chat.calls[0][0]
        self.assertEqual(system["role"], "system")
        for needle in ("recall_by_product_2024_table", "recall_dsy_2024_table", "SELECT", "events", "records"):
            self.assertIn(needle, system["content"])

    def test_system_prompt_has_an_example_claim_and_a_scope_rule(self):
        chat = scripted(ANSWER)
        make_draft_fn(chat=chat, run_sql=fresh_rows())("why?", [])
        content = chat.calls[0][0]["content"].lower()
        for needle in ("example claim", "scope", "join"):
            self.assertIn(needle, content)


@unittest.skipUnless(os.environ.get("RUN_LIVE_AGENT"), "set RUN_LIVE_AGENT=1 (slow: real model + Zetaris)")
class LiveAgent(unittest.TestCase):
    def test_explains_the_cardiovascular_rise_with_verified_claims(self):
        from agent.evaluator import evaluate
        from agent.loop import investigate
        question = ("Why did cardiovascular device recall records rise between 2023 and 2024, "
                    "and which product codes drove it?")
        result = investigate(question, make_draft_fn(), evaluate, max_rounds=3)
        self.assertIsNotNone(result.text, f"gave up; unsupported: {result.unsupported}")
        self.assertIn("DSY", result.text)


if __name__ == "__main__":
    unittest.main()
