-- openFDA drill-down: record-level recalls (2024) for the three codes that drove
-- the cardiovascular increase. Run ONE statement at a time, ~15 s apart.
-- Prerequisite: openfda_register.sql objects exist.
--
-- Oracle (direct openFDA, DSY only, checked before writing this file):
--   54 recall records, 1 distinct res_event_number (94614), 54 distinct cfres_id,
--   1 recalling firm (Atrium Medical Corporation), 1 root cause (Equipment maintenance).
-- DQO and OEZ use the same endpoint shape; their numbers are NOT yet verified.
-- Metric wording: "recall records" are product-level; report distinct events too.

-- DSY (Prosthesis, Vascular Graft, 6mm+)
CREATE LIGHTNING REST TABLE recall_dsy_2024 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=product_code:DSY+AND+event_date_initiated:[2024-01-01+TO+2024-12-31]&limit=100",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_dsy_2024_table WITH CONTAINER openfda AS
SELECT
    r.cfres_id,
    r.res_event_number,
    r.product_code,
    r.event_date_initiated,
    r.recall_status,
    r.recalling_firm,
    r.root_cause_description,
    r.reason_for_recall
FROM openfda_rest.recall_dsy_2024
LATERAL VIEW explode(results) AS r;

-- DQO (Catheter, Intravascular, Diagnostic)
CREATE LIGHTNING REST TABLE recall_dqo_2024 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=product_code:DQO+AND+event_date_initiated:[2024-01-01+TO+2024-12-31]&limit=100",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_dqo_2024_table WITH CONTAINER openfda AS
SELECT
    r.cfres_id,
    r.res_event_number,
    r.product_code,
    r.event_date_initiated,
    r.recall_status,
    r.recalling_firm,
    r.root_cause_description,
    r.reason_for_recall
FROM openfda_rest.recall_dqo_2024
LATERAL VIEW explode(results) AS r;

-- OEZ (Cardiovascular Procedure Kit)
CREATE LIGHTNING REST TABLE recall_oez_2024 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=product_code:OEZ+AND+event_date_initiated:[2024-01-01+TO+2024-12-31]&limit=100",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_oez_2024_table WITH CONTAINER openfda AS
SELECT
    r.cfres_id,
    r.res_event_number,
    r.product_code,
    r.event_date_initiated,
    r.recall_status,
    r.recalling_firm,
    r.root_cause_description,
    r.reason_for_recall
FROM openfda_rest.recall_oez_2024
LATERAL VIEW explode(results) AS r;

-- Union of the three (run after all three views exist).
CREATE SCHEMASTORE VIEW recall_drilldown_2024_table WITH CONTAINER openfda AS
SELECT * FROM openfda.recall_dsy_2024_table
UNION ALL
SELECT * FROM openfda.recall_dqo_2024_table
UNION ALL
SELECT * FROM openfda.recall_oez_2024_table;

-- Check (expect DSY: 54 records, 1 event, 1 firm, 1 root cause):
-- SELECT product_code, COUNT(*) AS records, COUNT(DISTINCT res_event_number) AS events,
--        COUNT(DISTINCT recalling_firm) AS firms, COUNT(DISTINCT root_cause_description) AS causes
-- FROM openfda.recall_drilldown_2024_table GROUP BY product_code ORDER BY records DESC;
