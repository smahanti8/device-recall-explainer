# Device Recall Explainer

An agent that answers "why did this change?" questions about FDA medical device recalls,
and refuses to state any number it cannot trace to a query.

Example question: *Why did cardiovascular device recall records rise between 2023 and 2024,
and which product codes drove it?*

The data is real, public openFDA recall and classification data (no patient data), registered in Zetaris as
SQL tables. The agent investigates with read-only SQL, then writes an answer in which every
number is a **claim that cites a cell of a SQL result**. An evaluator checks each claim against the result of its cited SQL and
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
| Playbook mode (code runs the verified queries, the model explains) | Covered by unit tests with a scripted model, plus a live check that the fact values match direct openFDA calls (281, 482, +201, DSY 53, DQO 49, OEZ 29, top-3 sum 131, share 65.2%) |
| Memory (H-MEM engine) | Covered by unit tests, the engine's conformance suite and a two-process recall test against the real engine |
| Model access through Meterless Nightshift | Verified with live calls and saved receipts (`docs/nemotron-evidence.md`) |
| End-to-end run, playbook mode (`--playbook`) | Run twice with memory. Run 1: 22 claims verified in one round, nothing recalled, `Memory: 22 new, 0 reinforced, 0 changed` (receipt `rcpt_req_sx5b9lbm`). Run 2: 4 hints recalled from run 1 and listed, 22 claims verified in one round, `Memory: 0 new, 22 reinforced, 0 changed` (receipts `rcpt_req_xkqh4x5l`, `rcpt_req_dzso08xp`). Some model replies were served from Nightshift's cache (0.0s), so latencies are not a measure of model speed |
| End-to-end run with the evidence branch and escalation (`--playbook`, recorded for `SAMPLE_MODE`) | Round 1: 25 claims, 2 rejected as `missing_evidence` (the answer named DQK and DPT without their detail). The researcher fetched both codes' detail from Zetaris (5 facts each), and round 2, redrafted on the larger model, verified all 35 claims. Two model calls (receipts `rcpt_req_dv9gxx4q` on Lightning, `rcpt_req_63urdzfk` on Ultra), 5,078 tokens in and 4,402 out; 36 Zetaris queries fetched and 86 served from the in-run cache; `Memory: 35 new, 0 reinforced, 0 changed`. Log: `logs/sample-run-1.jsonl`; report: `reports/sample-run-1.html` |
| End-to-end run, exploratory mode (default) | Ran all 3 rounds live and produced **no verified answer**, and the evaluator refused to print an unverified one. 12 model calls (50,624 tokens in, 34,623 out), 49 Zetaris queries fetched, 4 failed, 8 provider retries. Round 1: 8 claims, 3 rejected (two `sql_failed`, one `scope_mismatch`). Round 2: 15 claims, 1 rejected (`value_mismatch`: the claimed 65.2 did not match the cited SQL, which returned 100.0). Round 3: 25 claims, 4 rejected (one `sql_failed`, three `unclaimed_number`). Log: `logs/sample-exploratory-run-1.jsonl`; report: `reports/sample-exploratory-run-1.html`. Earlier live runs of this mode ended in provider timeouts and 502 errors |

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
evaluator: fetch the result of every claim's cited SQL, compare the cell, check scope and unclaimed numbers
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
citation cannot be malformed, and the evaluator still checks every number against the result of its cited SQL. The playbook
covers one question shape (cardiovascular recalls, 2023 against 2024); the exploratory mode is general.

**Model access** (`agent/llm.py`). All calls go to an OpenAI-compatible endpoint, normally the
Meterless Nightshift runtime. At most 2 calls run at once, provider errors are retried 5 times
(waits of 3, 6, 12 and 24 seconds), and every call logs `routed_model` and `receipt_id`.

## Cost, token and reliability controls

Design choices that keep a run small, safe, cheap to inspect and repeatable. Each one is in code and covered by tests.

| Choice | What it does | Where |
|---|---|---|
| **Facts instead of raw rows** | In playbook mode the model sees a short numbered list of facts, not query results, and answers with fact ids. Exploratory mode truncates each result it shows the model | `agent/playbook.py`, `agent/drafter.py` |
| **Schema in the prompt** | The system prompts list the registered views and columns, so the model does not spend turns discovering the schema | `agent/drafter.py`, `agent/playbook.py` |
| **Two model sizes** | Nemotron 3.5 Lightning is the default. Only a redraft after a rejection goes to the larger model named in `NEMOTRON_ESCALATION_MODEL` (optional); if that call fails, the default model is used | `agent/llm.py`, `agent/playbook.py` |
| **Query cache** | Identical SQL within a run is fetched once and then served from memory, which saves openFDA rate-limit quota | `agent/tools_zetaris.py` |
| **Token and cost line** | Every model call's input tokens, output tokens and the cost Nightshift reports are summed and printed as `Tokens:` | `agent/main.py`, `agent/llm.py` |
| **Observability line** | Each run prints Zetaris queries fetched versus served from cache, calls per model, retries and failed calls. The numbers are counted from what ran | `agent/observability.py` |
| **SQL guard** | Before any request reaches Zetaris, `run_sql` accepts one `SELECT`/`WITH` statement whose tables are all `openfda` views (or CTEs of the same statement), and refuses write keywords (`insert`, `update`, `delete`, `merge`, `create`, `alter`, `drop`, `truncate`, `grant`, `revoke`, `call`, `exec`) and the Java-calling functions `reflect` and `java_method`. In playbook mode the model never writes SQL; in exploratory mode it does. The guard protects Zetaris, not the model bill, because the model has already written the SQL by then. A question over 500 characters is refused up front | `agent/tools_zetaris.py`, `agent/main.py` |
| **Bounded retries** | At most 2 concurrent model calls and 5 attempts per call with backoff; a failed call costs one round, not the run | `agent/llm.py`, `agent/loop.py` |
| **Structured run log** | A JSONL event per step (agent, action, model, status, retry count, token counts), with secrets redacted | `agent/runlog.py` |
| **HTML report** | `--report` writes one self-contained page: the answer with each number linked to its proof, rounds, queries, receipts, tokens and the observability panel | `agent/report.py` |

Not done, on purpose: a rolling summary of earlier turns (runs take one to three rounds, so there is nothing to roll), and model routing through Nightshift's `deep`, `fast` or `fusion` modes (they route to the default lane; see "Decisions and failures").

## Running it: terminal or report

The agent is a command line program. The terminal run is the real thing: progress lines, the trace, the model receipts, the token and observability lines. Adding `--report` also writes an HTML page for the same run, which is easier to read than the terminal output and needs no server:

```bash
.venv/bin/python -m agent.main "<question>" --playbook --report
open reports/report-*.html      # newest file in reports/
```

The report is generated from a finished run; it is not a web application and cannot start a run.

### Sample mode: no services needed

`SAMPLE_MODE=true` (or `--sample`) replays a recorded real run, so the agent can be tried without Zetaris, Nightshift or any key:

```bash
SAMPLE_MODE=true .venv/bin/python -m agent.main "Why did cardiovascular device recall records rise between 2023 and 2024, and which product codes drove it?" --playbook --report
```

What is replayed: the Zetaris results (`sample_data/zetaris_results.json`) and the model replies (`sample_data/model_replies.json`), both captured from the real run described in "Status" with `--record sample_data`. What still runs for real: the evaluator, the rejection and retry loop, the evidence branch, memory and the report. A banner states that the run is a replay, and the report carries it too. Prompts are matched to recordings by hash, so the replay only works for this question and these options; anything else stops with an explanation. A fresh clone with no `.env` was checked to replay this run, ending with the same 35 verified claims.

### Run in Docker

The image needs only Docker: it installs Python, Node and the dependencies at build time and defaults to the offline replay.

```bash
docker build -t device-recall-explainer .
docker run --rm device-recall-explainer          # replays the recorded run, no keys or services needed
docker run --rm --env-file .env -e SAMPLE_MODE=false device-recall-explainer "<question>" --playbook   # live run
docker run --rm --entrypoint python device-recall-explainer -W ignore -m unittest discover -s tests   # tests
```

The build and the full test suite were run in the container on Python 3.11 (284 tests, live ones skipped); the code is
also tested locally on Python 3.9. `.env` is never copied into the image (see `.dockerignore`); a live run reads the
settings from `--env-file`. A live run from the container needs a Zetaris and a Nightshift endpoint it can reach, and on
Docker Desktop for Mac `localhost` inside the container is the container itself, so point `ZETARIS_REST_URL` and
`MODEL_BASE_URL` at `host.docker.internal` instead.

To record again from live services: `--record sample_data` on a live run.

## Agent roles and how they interact

| Role | What it is | Interaction |
|---|---|---|
| **Drafter** | One model (Nemotron through Nightshift), either exploring with read-only SQL or, in playbook mode, explaining numbered facts | Produces a draft: text plus claims, each citing a SQL cell |
| **Evaluator** | Deterministic code, not a model | Checks every claim and returns rejection reasons; the loop hands them back to the drafter for up to three rounds |
| **Memory** | H-MEM, the Meterless engine | Supplies recalled hints before drafting; stores the claims that were verified |

This is a drafter, an evaluator and a memory in a retry loop. It is not a multi-agent swarm: the evaluator is code, and playbook mode is a fixed pipeline (queries, one model call, evaluator). In both recorded runs the evaluator accepted the first draft; the rejection and retry paths are covered by tests.

## Meterless Engine: H-MEM

**Engine used:** H-MEM (persistent memory), the Meterless reference implementation, vendored
unmodified in `engines/hmem/` (Apache-2.0; see `engines/hmem/NOTICE.md` for the upstream commit).

**Where it is used in the agent**

| Step | Code | What happens |
|---|---|---|
| Before drafting | `agent/main.py` -> `Memory.context_for(question)` -> drafter's `context_fn` | Verified claims from earlier sessions are recalled and added to the model's first message, labelled "hints, not evidence" |
| After a verified answer | `agent/main.py` -> `Memory.learn(claims, [])` | Each verified claim is upserted with its cited SQL as the record's required `source`, tagged by product code and year. Same text and value: the existing record is reinforced with `helpful` feedback (no duplicate). New value: a new record supersedes the old one, which is marked `wrong`, and the run prints "changed since last session" |
| Always | `agent/memory.py` -> `agent/memory_cli.mts` -> `engines/hmem/reference` | A small Node bridge (JSON in, JSON out) calls the engine's `add`, `query` and `feedback` |

**Effect:** a second session recalls what the first one verified (for example "DSY recall records
in 2024: 54", with its SQL), so the agent starts from known baselines. The run reports the outcome, for example `Memory: 0 new, 22 reinforced, 0 changed`. Recalled hints are never
evidence: every number in an answer must still cite its own SQL, and the evaluator checks the claim against that query's result.

**State and audit:** `.hmem/state.json` holds the records and `.hmem/ledger.jsonl` is the engine's
append-only audit ledger of every add and feedback.

**How we check it**
- the engine's own conformance suite passes 13/13 (`tests/test_memory.py` runs it from this repo)
- a test stores a claim in one Node process and recalls it from another, with state read from disk
- a test confirms the ledger records creation and feedback
- tests confirm that storing the same claim twice keeps one record, and that a changed value supersedes the old one

**See it work:** run the command from "Setup" twice with the same `--memory-dir` (default `.hmem`),
then look at the "Memory:" line of the first run and the recalled hints sent in the second.

## Data

One command registers everything in Zetaris (it needs `ZETARIS_*` in `.env`):

```bash
.venv/bin/python -m agent.setup_data --verify
```

It runs the files in `sql/` in order, skips anything that already exists, stops on any other error, and waits 15 seconds after each REST table because keyless openFDA rate-limits. `--dry-run` lists the statements without running them, and `--verify` checks that the key views answer.

1. `sql/openfda_spike.sql`: the database, the container and a first table
2. `sql/openfda_register.sql`: cardiovascular device types, recalls per product code, root causes (2023, 2024)
3. `sql/openfda_drilldown.sql`: record-level 2024 recalls for three product codes
4. `sql/openfda_drilldown_more.sql`: record-level 2024 recalls for seven more product codes

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

Register the data (see "Data" above: `.venv/bin/python -m agent.setup_data --verify`), then:

```bash
.venv/bin/python -m agent.main "Why did cardiovascular device recall records rise between 2023 and 2024, and which product codes drove it?"
```

Options: `--memory-dir DIR` (default `.hmem`), `--no-memory`, `--verbose` (show each rejection's detail),
`--playbook` (see "Two ways to draft" above), `--report` (write an HTML report), `--json` (print one JSON document on
stdout instead of text) and `--timeout SECONDS` (stop the run with no answer after this long; default 600, 0 means no
limit). Progress lines go to stderr while it runs. Exit code is 0 for a verified answer, 1 for no verified answer and 2
for a usage or settings problem.

**Programmatic use.** `--json` makes the command line a machine-readable entry point. stdout carries exactly one
document with `question`, `verified`, `answer`, `rounds`, `claims` (each with its SQL, column and row), `unsupported`,
`trace`, `receipts`, `tokens`, `observability`, `log` and `report`:

```bash
SAMPLE_MODE=true .venv/bin/python -m agent.main "<question>" --playbook --json 2>/dev/null | python3 -m json.tool
```

**Run time.** A playbook run takes minutes, and the model (90 s) and Zetaris (60 s) calls have their own timeouts. The
run-level limit is checked between rounds and before every model attempt, so a stuck run ends with a clear "time limit"
and no answer instead of running on. The exploratory mode took about 18 minutes in its recorded run, which the default
limit would now cut off.

## Tests

```bash
.venv/bin/python -W ignore -m unittest discover -s tests -v
```

Tests that need live services skip themselves when their settings are missing. The end-to-end
agent test runs only with `RUN_LIVE_AGENT=1`. Expected values in the Zetaris tests come from direct
openFDA calls, so a bug in a view cannot hide behind its own output.

## Layout

```
agent/        tools_zetaris.py  evaluator.py  loop.py  drafter.py  playbook.py  llm.py  memory.py  memory_cli.mts  main.py
              observability.py  report.py  runlog.py  sample_mode.py  setup_data.py
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

## Decisions and failures

Major decisions, with what was rejected and what each one costs. Failures are listed because they shaped the design.

| # | Decision | Why, and the trade-off |
|---|---|---|
| 1 | **Data: public openFDA recall and classification records**, registered in Zetaris as REST tables | Real, public, structured and keyless, with no patient data. Zetaris cannot read files on a laptop, which ruled out local CSVs. Trade-off: one REST table is one request, so the agent works over registered slices, and keyless openFDA rate-limits |
| 2 | **Every number is a claim that cites a SQL cell**, checked by an evaluator | Rejected: free-text answers, and a second model as judge (not deterministic). Trade-off: eight rejection reasons to maintain, and the model must produce citations |
| 3 | **A scope check and an unclaimed-number check** were added to the evaluator | Live traces showed the model labelling an all-device total "cardiovascular" and writing figures in prose that no claim backed. Trade-off: the scope check is a word check on the claim text, not meaning |
| 4 | **Report recall records and distinct recall events separately** | One product code in 2024 has 54 records but a single event, so "recalls rose" is ambiguous. Trade-off: more claims per answer |
| 5 | **Two drafting modes: exploratory (default) and playbook (`--playbook`)** | The exploratory model-driven mode has not produced a verified answer in live runs (see below). In playbook mode code runs the fixed queries and the model only replies with fact ids, which completed live runs. Trade-off: the playbook covers one question shape |
| 6 | **H-MEM reference vendored unmodified** (`engines/hmem`) | Implementing the spec from scratch was not feasible in the time; the organisers' guidance was that the reference is acceptable if it is a real working feature and documented here. Rejected: Markovian, because it would own the loop that the evaluator already controls. Trade-off: the reference's embedder is a hash, so recall relies on tags and keywords |
| 7 | **Memory gives hints, never evidence** | A recalled claim can never be cited; every number still needs its own SQL result. Trade-off: memory adds continuity and an audit trail, but does not make an answer more accurate by itself |
| 8 | **Remembered claims are upserted**: the same value is reinforced, a new value supersedes the old record | The first memory demo stored 22 duplicate records. Trade-off: matching is by claim text |
| 9 | **Model route: Meterless Nightshift to Nemotron 3.5 Lightning**, at most 2 concurrent calls, retries with waits of 3 to 24 seconds | Four parallel calls caused 502 errors. Every response carries a receipt naming the model, which the CLI prints |
| 10 | **Query results are cached in memory per run, by exact SQL text** | Saves rate-limit quota and makes a run repeatable. Rejected: a disk cache (could hide a real data change) and an openFDA key in `.env` (the tables call openFDA from inside Zetaris, so the key would have to be in each table's URL). Trade-off: a citation is checked against the stored result, not re-queried |
| 11 | **Tests first, with expected values from independent openFDA calls** | A bug in a Zetaris view cannot hide behind its own output. Trade-off: live checks need Zetaris and the model, so they skip themselves when settings are missing |
| 12 | **Settings are checked before a run** | A missing variable used to end in a stack trace; it now names the variable and points at `.env.example` |

**Failures and what they taught us**

- **Two models in Nightshift's catalog returned `410 Gone`** (`nvidia/nemotron-3-nano-30b-a3b` and the Llama 3.3 Nemotron Super 49B). The catalog was stale; Lightning and Ultra worked. The agent now surfaces such errors as a failed round instead of crashing.
- **The exploratory mode has not produced a verified answer.** Earlier runs reached 12 to 16 model turns and then hit Nightshift `502` errors (`empty_response`, `provider_deadline`), and Zetaris rejects a statement that references the same view twice. Fixes helped (a larger token budget, batched queries, a one-view-per-statement rule): the latest live run finished all 3 rounds, but the model wrote SQL that Zetaris rejected, a share figure whose cited SQL returned a different value, and numbers in the text that no claim backed, so the evaluator rejected every draft and the run ended with no answer. That is the gate working as designed, and the documented verified evidence is playbook mode.
- **`/no_think` did not stop the model reasoning**: turns used 1.4k to 4.4k completion tokens, so a small token cap returned an empty reply. Every turn now asks for 8,000 tokens.
- **The evaluator was described as re-running SQL, but the cache means it checks the stored result.** The wording was corrected here and in the write-up.
- **Model choice through Nightshift changed between versions.** On Nightshift 1.2.1, requesting `nvidia/nemotron-3-ultra-550b-a55b`, `meterless/deep`, `meterless/fast` or `meterless/fusion` all produced receipts with `requested_mode: meterless/auto` and `routed_model` Lightning. On 1.2.2 an explicit model ID is honoured (the same request is routed to Ultra, per its receipt), while the `meterless/deep`, `fast` and `fusion` modes still route to the default lane. Only explicit model IDs are used here.
- **Some model replies come from Nightshift's cache** (reported as 0.0s), so receipt latencies are not a measure of model speed.

## AI and Model Usage

**Models (through Meterless Nightshift, the primary AI provider)**
- `nvidia/nemotron-3.5-lightning-30b-a3b`: writes the explanation from numbered facts in playbook mode, and in exploratory mode drives the investigation with read-only SQL and writes cited claims. Nightshift's receipt for every call names the model that answered (`routed_model`); the command line prints it.
- No other model is used. There is no fine-tuning, and no embeddings model: the H-MEM reference uses a deterministic hash embedder.

**Agent framework:** none; the loop, evaluator and drafter are custom code.

**Sponsor technology:** Zetaris is the data layer (read-only SQL over openFDA tables), Meterless H-MEM is the engine used in a working feature (see "Meterless Engine: H-MEM"), and NVIDIA Nemotron answers through Nightshift.

**Development tools:** built with Claude Code, an AI coding agent, which was used for implementation, tests, debugging and documentation. The tests for each module were specified and approved before its implementation, and the work is in small commits that can be reviewed one by one.

**Data:** real, public openFDA recall and classification records. There is no patient data and no synthetic data. Source: https://open.fda.gov/ (terms: https://open.fda.gov/terms/). The tests use small made-up values for the fake-service checks and say so.

**Fallback and limits:** provider failures are retried and recorded as a failed round; see "Known limits" for what is not covered.

## Known limits

- Playbook mode answers one question shape; the exploratory mode is general but has not yet produced a verified answer in live runs (the evaluator rejected each draft).
- Within one run, identical SQL is served from an in-memory cache, so a citation is checked against the stored result of that query rather than a second round-trip to Zetaris. A new run fetches again.
- Model turns are slow (tens of seconds each) and the provider sometimes times out; the loop records a failed round and continues.
- The scope check is a word check on the claim text, not a semantic one.
- The reference H-MEM embedder is a deterministic hash, so recall relies on tags and keywords.
- The agent can only investigate what the registered tables cover.

## Third-party code and licences

| Component | Licence | Use |
|---|---|---|
| Meterless H-MEM reference implementation and conformance suite | Apache-2.0 | Vendored unmodified in `engines/hmem/` (see `NOTICE.md`) |
| `openai` (Python SDK) | Apache-2.0 | OpenAI-compatible client for Nightshift |
| `requests` | Apache-2.0 | Zetaris REST calls |
| `python-dotenv` | BSD-3-Clause | Loads `.env` |
| `tsx`, `vitest` | MIT | Dev tools inside the vendored engine |
| Zetaris getting-started recipes | see https://github.com/zetaris/getting-started | The REST-table pattern in `sql/` follows its public recipes |

## License

MIT for this repository's code (see `LICENSE`). `engines/hmem/` is the Meterless H-MEM reference
implementation under Apache-2.0 (see `engines/hmem/LICENSE` and `engines/hmem/NOTICE.md`).
