-- openFDA registration: run ONE statement at a time in the Zetaris SQL Editor,
-- waiting ~15 seconds between statements (keyless openFDA rate-limits).
-- Note: recall_root_cause_2024 (top 10 only) already exists from the spike and cannot be
--       dropped via SQL; the *_rootcause_all_* tables below hold all terms (limit=100).
-- Prerequisite: openfda_spike.sql steps 0-3 already ran (database OPENFDA_REST
-- and container openfda exist). Do NOT re-run those.
--
-- Join key across tables: product_code (verified present in recall,
-- classification, 510k and event endpoints).
-- Notes from probing the API:
--   * recall records carry NO device class or specialty; classification does.
--   * count=product_code works; count=product_code.exact returns 404.
--   * count=root_cause_description.exact works.
--   * If a date-filtered URL fails through the connector, replace [ and ] with
--     %5B and %5D and retry once.

-- 1. Cardiovascular device types (349 rows, fits one response).
CREATE LIGHTNING REST TABLE classification_cardiovascular FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/classification.json?search=medical_specialty_description:Cardiovascular&limit=1000",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW classification_cardiovascular_table WITH CONTAINER openfda AS
SELECT
    r.product_code,
    r.device_name,
    r.device_class,
    r.medical_specialty_description AS specialty,
    r.regulation_number
FROM openfda_rest.classification_cardiovascular
LATERAL VIEW explode(results) AS r;

-- 2. Recalls per product code, by year of recall initiation (2023, 2024).
CREATE LIGHTNING REST TABLE recall_by_product_2023 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=event_date_initiated:[2023-01-01+TO+2023-12-31]&count=product_code&limit=1000",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_by_product_2023_table WITH CONTAINER openfda AS
SELECT
    r.`term` AS product_code,
    r.`count` AS recall_count
FROM openfda_rest.recall_by_product_2023
LATERAL VIEW explode(results) AS r;

CREATE LIGHTNING REST TABLE recall_by_product_2024 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=event_date_initiated:[2024-01-01+TO+2024-12-31]&count=product_code&limit=1000",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_by_product_2024_table WITH CONTAINER openfda AS
SELECT
    r.`term` AS product_code,
    r.`count` AS recall_count
FROM openfda_rest.recall_by_product_2024
LATERAL VIEW explode(results) AS r;

-- 3. Root causes by year (the "why" evidence).
CREATE LIGHTNING REST TABLE recall_rootcause_all_2023 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=event_date_initiated:[2023-01-01+TO+2023-12-31]&count=root_cause_description.exact&limit=100",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_rootcause_all_2023_table WITH CONTAINER openfda AS
SELECT
    r.`term` AS root_cause,
    r.`count` AS recall_count
FROM openfda_rest.recall_rootcause_all_2023
LATERAL VIEW explode(results) AS r;

CREATE LIGHTNING REST TABLE recall_rootcause_all_2024 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=event_date_initiated:[2024-01-01+TO+2024-12-31]&count=root_cause_description.exact&limit=100",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_rootcause_all_2024_table WITH CONTAINER openfda AS
SELECT
    r.`term` AS root_cause,
    r.`count` AS recall_count
FROM openfda_rest.recall_rootcause_all_2024
LATERAL VIEW explode(results) AS r;

-- 4. Join check (the two-source query): cardiovascular codes, recalls 2023 vs 2024.
-- SELECT c.product_code, c.device_name, c.device_class,
--        COALESCE(a.recall_count, 0) AS recalls_2023,
--        COALESCE(b.recall_count, 0) AS recalls_2024
-- FROM openfda.classification_cardiovascular_table c
-- LEFT JOIN openfda.recall_by_product_2023_table a ON a.product_code = c.product_code
-- LEFT JOIN openfda.recall_by_product_2024_table b ON b.product_code = c.product_code
-- ORDER BY recalls_2024 DESC;
