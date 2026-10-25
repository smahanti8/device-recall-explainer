# Nemotron through Meterless Nightshift: run evidence

Requests are sent to the Nightshift runtime (OpenAI-compatible endpoint) with
`provider_model = meterless/auto`. The receipt in the response body
(`meterless.receipt`, schema `meterless.compute_receipt.v1`) records which
model actually answered.

Captured with `probe_receipt.py` (prompt: "Reply with the single word: ready").

| Captured (GMT) | provider_model | routed_model | egress | latency_ms | receipt_id |
|---|---|---|---|---|---|
| Sat, 24 Oct 2026 12:51:47 | meterless/auto | nvidia/nemotron-3.5-lightning-30b-a3b | nvidia | 4029 | rcpt_req_6r1i91j8 |
| Sat, 24 Oct 2026 12:53:53 | meterless/auto | nvidia/nemotron-3.5-lightning-30b-a3b | nvidia | 1135 | rcpt_req_tjyjvtk5 |

Additional fields from the first receipt:

- `winning_route`: kind `cloud`, provider `nvidia`, tier `A`
- `routing`: policy `learned-router`, winner `nvidia/nemotron-3.5-lightning-30b-a3b`
- `usage`: 23 input tokens, 151 output tokens (reasoning overhead on a one-word reply)
- `cross_check`: status `passed` (no destructive ops, no PII or credential in the response)
- `egress.destination`: `nvidia`

The response's own `model` field also reads `nvidia/nemotron-3.5-lightning-30b-a3b`.

Reproduce: `.venv/bin/python probe_receipt.py` (add `--full` for every receipt field).
No keys or tokens appear in this file or in the probe output.

## Real agent calls (playbook mode)

Receipts from the agent answering "Why did cardiovascular device recall records rise between 2023
and 2024, and which product codes drove it?" Every claim in the answer was checked by the evaluator against
the stored result of its cited SQL before the run counted as verified.

| Run (24 Oct 2026, UTC-4) | Entry point | Receipt | routed_model | Claims verified | Model turn (wall clock) |
|---|---|---|---|---|---|
| 22:45 | playbook through the same investigate/evaluate code, driven by a script | rcpt_req_kv8b3e9e | nvidia/nemotron-3.5-lightning-30b-a3b | 10, round 1 | 58 s |
| 22:52 | `python -m agent.main --playbook --no-memory` | rcpt_req_uomhx4mb | nvidia/nemotron-3.5-lightning-30b-a3b | 22, round 1 | 25.9 s |

For `rcpt_req_kv8b3e9e` the logged receipt also shows `provider_model = meterless/auto` and
`egress = nvidia`. The command line prints the routed model and the receipt id for each call.

About latency: the `latency_ms` field is as reported by Nightshift. Some replies are served from
a cache of identical prompts, and the value 45001 recurs across different calls, so it is not a
measure of model speed. The wall-clock column above is what the run actually took.

Reproduce: `.venv/bin/python -m agent.main --playbook --verbose --no-memory "<question>"`.

## Memory demo: two runs per session, memory on

Receipts from the command line's "Model receipts" output, from two sessions of two runs each:
`python -m agent.main --playbook --verbose "<question>"`, run twice with the same memory
directory, which was cleared before each session. Every call was routed to
`nvidia/nemotron-3.5-lightning-30b-a3b`. Session 2 ran after the answer-text cleanup in commit 569ee6f.

| Session | Run | Memory at the start | Claims verified | Receipts |
|---|---|---|---|---|
| 1 | 1 | nothing recalled yet | 22, round 1 | rcpt_req_4eyajant |
| 1 | 2 | recalled 4 hints | 25, round 1 | rcpt_req_vvdfl9va, rcpt_req_w9u17fet |
| 2 | 1 | nothing recalled yet | 22, round 1 | rcpt_req_sx5b9lbm |
| 2 | 2 | recalled 4 hints | 22, round 1 | rcpt_req_xkqh4x5l, rcpt_req_dzso08xp |

The second run of each session makes two model calls. Recalled hints are labelled as hints, not
evidence: every number in the answer still cites its own SQL, and the evaluator checks it against that query's result.

## Escalation run: Lightning, then Ultra (25 Oct 2026)

`python -m agent.main --playbook --verbose --report --record sample_data "<question>"` with
`NEMOTRON_ESCALATION_MODEL=nvidia/nemotron-3-ultra-550b-a55b`, memory cleared first. Round 1 was drafted on
Lightning and 2 claims were rejected as `missing_evidence` (DQK and DPT named without detail). The evidence
branch fetched their detail from Zetaris and the redraft was sent to Ultra; round 2 verified all 35 claims.
Both calls were routed as requested, and both receipts report `egress = nvidia`. Log: `logs/sample-run-1.jsonl`.

| Call (UTC) | Round | Receipt | routed_model | Tokens in / out |
|---|---|---|---|---|
| 06:28:21 | 1 | rcpt_req_dv9gxx4q | nvidia/nemotron-3.5-lightning-30b-a3b | 2,392 / 2,672 |
| 06:28:46 | 2 (redraft) | rcpt_req_63urdzfk | nvidia/nemotron-3-ultra-550b-a55b | 2,686 / 1,730 |

Totals as printed by the run: 2 model calls, 5,078 in, 4,402 out; 36 Zetaris queries fetched, 86 served from
the in-run cache. The cost field in the receipts reads $0.0000, so no cost is claimed.

## Exploratory run: no verified answer (25 Oct 2026)

`python -m agent.main --verbose --report --no-memory "<question>"` (the default mode, where the model drives the
investigation). Three rounds, 12 model calls, all routed to `nvidia/nemotron-3.5-lightning-30b-a3b`; the evaluator
rejected every draft, so no answer was printed. Log: `logs/sample-exploratory-run-1.jsonl`. The first call's
receipt carries no `egress` value and reports 27 tokens in, so it may have come from Nightshift's cache.

Receipts, in order: rcpt_req_9rsd4u40, rcpt_req_bdivv8rq, rcpt_req_mupwtpry, rcpt_req_z9374tgb,
rcpt_req_00w4jv9t, rcpt_req_qp79mns2 (round 1); rcpt_req_w3g2pdqn, rcpt_req_vthwv528, rcpt_req_82yjjt6m,
rcpt_req_1i3dlyv2, rcpt_req_txbbgufv (round 2); rcpt_req_67dh6cl4 (round 3).

Totals as printed by the run: 50,624 tokens in, 34,623 out; 49 Zetaris queries fetched, 4 failed; 8 provider
retries and 1 failed call.
