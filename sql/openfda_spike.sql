-- openFDA spike: run ONE statement at a time in the Zetaris SQL Editor,
-- waiting for each to finish. Modeled on 02_pokeapi_create.sql.
-- Source: https://open.fda.gov/apis/device/recall/ (no API key needed).
-- Goal: confirm a REST-backed table can return clean, citable rows.

-- Step 0: database for the source.
CREATE LIGHTNING DATABASE OPENFDA_REST DESCRIBE BY "openFDA device REST source";

-- Step 1: container for the flattened views. Run once.
CREATE SCHEMASTORE CONTAINER openfda;

-- Step 2: root-cause counts across all device recalls (no brackets in the URL).
CREATE LIGHTNING REST TABLE recall_root_cause_counts FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?count=root_cause_description.exact&limit=10",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

-- Step 3: flatten the results array (one row per root cause).
CREATE SCHEMASTORE VIEW recall_root_cause_counts_table WITH CONTAINER openfda AS
SELECT
    r.`term` AS root_cause,
    r.`count` AS recall_count
FROM openfda_rest.recall_root_cause_counts
LATERAL VIEW explode(results) AS r;

-- Check: expect up to 10 rows, e.g. "Under Investigation by firm", "Process control".
-- SELECT * FROM openfda.recall_root_cause_counts_table;

-- Step 4 (only after the check passes): same count with a date filter.
-- The brackets and "+" in the URL may be re-encoded by the connector; if this
-- fails or returns different numbers than a direct curl, note it and stop.
-- CREATE LIGHTNING REST TABLE recall_root_cause_2024 FROM OPENFDA_REST REQUEST(
--     endpoint "https://api.fda.gov/device/recall.json?search=event_date_initiated:[2024-01-01+TO+2024-12-31]&count=root_cause_description.exact&limit=10",
--     method "get",
--     response_type "json",
--     http_encoding "URLENCODED"
-- ) HEADER (
-- ) BODY ();
