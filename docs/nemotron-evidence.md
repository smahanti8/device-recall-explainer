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

TODO: append a receipt from a real agent feature call once the agent exists.
