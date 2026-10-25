# Device Recall Explainer

An agent that answers "why did this change?" questions about FDA medical device recalls,
and refuses to state any number it cannot trace to a query.

Example question: *Why did cardiovascular device recall records rise between 2023 and 2024,
and which product codes drove it?*

The data is public openFDA recall, classification and clearance data, registered in Zetaris as
SQL tables. The agent investigates with read-only SQL, then writes an answer in which every
number is a **claim that cites a cell of a SQL result**. An evaluator re-runs each citation and
rejects the draft if a claim has no citation, cites the wrong scope, or does not match the cell.
Rejection reasons go back to the model for another attempt.

> Output is descriptive statistics over public data. It is not medical or regulatory advice.
> openFDA's own notice applies: https://open.fda.gov/terms/

## Status

| Part | State |
|---|---|
| Data layer (openFDA in Zetaris) | Registered; counts checked against direct openFDA calls |
| SQL tool, evaluator, retry loop | Covered by unit tests, including checks through live Zetaris |
| Drafter (model drives the investigation) | Covered by unit tests with a scripted model |
| Memory (H-MEM engine) | Covered by unit tests, the engine's conformance suite and a two-process recall test against the real engine |
| Model access through Meterless Nightshift | Verified with live calls and saved receipts (`docs/nemotron-evidence.md`) |
| End-to-end run of `python -m agent.main` | Not yet recorded: earlier live runs hit provider timeouts |

<!-- Update the last row once a full run has been recorded. -->

## How it works

```
question
   |  (+ hints recalled from earlier sessions, labelled "not evidence")
   v
drafter --- read-only SQL --> Zetaris --> openFDA tables
   |   ^                                  (recalls, classification, root causes)
   |   '--- rejection reasons -----------.
   v                                     |
draft: text + claims {value, sql, column, row}
   |
   v
evaluator: re-run every claim's SQL, compare the cell, check scope and unclaimed numbers
   |
   v
accepted --> printed trace + model receipts --> verified claims stored in memory
```

**Evaluator rules** (`agent/evaluator.py`, `agent/loop.py`). A claim is rejected with one reason:

| Reason | Meaning |
|---|---|
| `no_citation` | The claim has no SQL citation |
| `sql_failed` | The citation could not run (a non-`SELECT` is refused up front) |
| `no_such_cell` | The cited row or column does not exist in the result |
| `scope_mismatch` | The wording names "cardiovascular" or a product code that the SQL never mentions |
| `value_mismatch` | The cell differs from the claimed value |
| `unclaimed_number` | A number in the answer text is not backed by any claim (years and single digits are exempt) |
| `no_claims` | The draft states nothing checkable |
| `model_unavailable` | The model failed after retries; the round is recorded and the run continues |

Every number must come from SQL, never from the model. A *recall record* is a product-level entry,
and one recall *event* can produce dozens of records. The drafter is told to report both, and the tests pin it
down (one product code in 2024 has 54 records but a single event).

**Two ways to draft.** By default the model investigates on its own: it asks for read-only SQL, reads
the results, and answers with cited claims (`agent/drafter.py`). With `--playbook`, code runs a fixed set
of queries first (`agent/playbook.py`): each result becomes a numbered fact whose label names its scope and
whose citation is the exact SQL that produced it. The model only replies with the fact ids it used, so a
citation cannot be malformed, and the evaluator still re-checks every number against Zetaris. The playbook
covers one question shape (cardiovascular recalls, 2023 against 2024); the exploratory mode is general.

**Model access** (`agent/llm.py`). All calls go to an OpenAI-compatible endpoint, normally the
Meterless Nightshift runtime. At most 2 calls run at once, provider errors are retried 5 times
(waits of 3, 6, 12 and 24 seconds), and every call logs `routed_model` and `receipt_id`.

## Meterless Engine: H-MEM

**Engine used:** H-MEM (persistent memory), the Meterless reference implementation, vendored
unmodified in `engines/hmem/` (Apache-2.0; see `engines/hmem/NOTICE.md` for the upstream commit).

**Where it is used in the agent**

| Step | Code | What happens |
|---|---|---|
| Before drafting | `agent/main.py` -> `Memory.context_for(question)` -> drafter's `context_fn` | Verified claims from earlier sessions are recalled and added to the model's first message, labelled "hints, not evidence" |
| After a verified answer | `agent/main.py` -> `Memory.learn(claims, [])` | Each verified claim is stored with its cited SQL as the record's required `source`, tagged by product code and year, then confirmed with `helpful` feedback |
| Always | `agent/memory.py` -> `agent/memory_cli.mts` -> `engines/hmem/reference` | A small Node bridge (JSON in, JSON out) calls the engine's `add`, `query` and `feedback` |

**Effect:** a second session recalls what the first one verified (for example "DSY recall records
in 2024: 54", with its SQL), so the agent starts from known baselines. Recalled hints are never
evidence: every number in an answer must still cite fresh SQL, which the evaluator re-runs.

**State and audit:** `.hmem/state.json` holds the records and `.hmem/ledger.jsonl` is the engine's
append-only audit ledger of every add and feedback.

**How we check it**
- the engine's own conformance suite passes 13/13 (`tests/test_memory.py` runs it from this repo)
- a test stores a claim in one Node process and recalls it from another, with state read from disk
- a test confirms the ledger records creation and feedback

**See it work:** run the command from "Setup" twice with the same `--memory-dir` (default `.hmem`),
then look at the "Memory:" line of the first run and the recalled hints sent in the second.

## Data

Registered by `sql/` (run in the Zetaris SQL Editor, one statement at a time, about 15 seconds
apart, because keyless openFDA rate-limits):

1. `sql/openfda_spike.sql` steps 0 to 3: the database, the container and a first table
2. `sql/openfda_register.sql`: cardiovascular device types, recalls per product code, root causes (2023, 2024)
3. `sql/openfda_drilldown.sql`: record-level 2024 recalls for three product codes

Tables are fixed slices of the API (one REST table is one request), so the agent reasons over
what is registered, not the whole dataset. Queries use `openfda.<view>`; the system prompt lists them.

## Setup

Prerequisites: Docker Desktop with the Zetaris platform running locally, Meterless Nightshift with an
NVIDIA key and a Nemotron lane, Python 3.9+, Node.js (tested with 26).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # fill in your own values; never commit .env
(cd engines/hmem/reference && npm ci --ignore-scripts)
```

Register the data as described above, then:

```bash
.venv/bin/python -m agent.main "Why did cardiovascular device recall records rise between 2023 and 2024, and which product codes drove it?"
```

Options: `--memory-dir DIR` (default `.hmem`), `--no-memory`, `--verbose` (show each rejection's detail) and
`--playbook` (see below). Progress lines go to stderr while it runs. Exit code is 0 for a verified answer.

## Tests

```bash
.venv/bin/python -W ignore -m unittest discover -s tests -v
```

Tests that need live services skip themselves when their settings are missing. The end-to-end
agent test runs only with `RUN_LIVE_AGENT=1`. Expected values in the Zetaris tests come from direct
openFDA calls, so a bug in a view cannot hide behind its own output.

## Layout

```
agent/        tools_zetaris.py  evaluator.py  loop.py  drafter.py  llm.py  memory.py  memory_cli.mts  main.py
sql/          Zetaris table and view definitions
tests/        unit tests and live-service checks
engines/hmem  vendored H-MEM reference and conformance suite (unmodified)
docs/         nemotron-evidence.md (saved model receipts)
probe_*.py    one-call scripts that print the shape of a Zetaris or Nightshift response
smoke_test.py checks Zetaris and the model route from .env
```

## Sponsor technology

- **Zetaris:** serves the openFDA data as SQL tables; the agent's only data access is read-only SQL through its REST API.
- **Meterless:** Nightshift is the model provider (routing receipts recorded per call); H-MEM provides memory across sessions.
- **NVIDIA:** a Nemotron model (default `nvidia/nemotron-3.5-lightning-30b-a3b`) answers the agent's turns; receipts in `docs/nemotron-evidence.md` show which model was routed to.

## Pre-existing components

- `engines/hmem/` is the H-MEM reference implementation and conformance suite from
  https://github.com/Meterless/Meterless (Apache-2.0), vendored unmodified; see `engines/hmem/NOTICE.md`.
  The suite passes 13/13 from this repo. Everything else was written during the build window.
- The SQL files follow the REST-table pattern of Zetaris's public getting-started recipes (https://github.com/zetaris/getting-started).
- Open-source libraries: `openai`, `python-dotenv`, `requests` (Python); `tsx`, `vitest` (dev tools in the vendored engine).

## Known limits

- Playbook mode answers one question shape; the exploratory mode is general but has been slower and less reliable in live runs.
- Model turns are slow (tens of seconds each) and the provider sometimes times out; the loop records a failed round and continues.
- The scope check is a word check on the claim text, not a semantic one.
- The reference H-MEM embedder is a deterministic hash, so recall relies on tags and keywords.
- The agent can only investigate what the registered tables cover.
