-- openFDA drill-down, part 2: record-level 2024 recalls for seven more product codes.
-- Same pattern as openfda_drilldown.sql. Loaded by `python -m agent.setup_data`, or run one statement
-- at a time in the Zetaris SQL Editor, about 15 seconds apart (keyless openFDA rate-limits).
-- Prerequisite: openfda_register.sql objects exist. Each code has under 100 records in 2024, so one request holds them all.
-- Metric wording: "recall records" are product-level; report distinct events too.

-- DQK (Programmable diagnostic computer)
CREATE LIGHTNING REST TABLE recall_dqk_2024 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=product_code:DQK+AND+event_date_initiated:[2024-01-01+TO+2024-12-31]&limit=100",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_dqk_2024_table WITH CONTAINER openfda AS
SELECT
    r.cfres_id,
    r.res_event_number,
    r.product_code,
    r.event_date_initiated,
    r.recall_status,
    r.recalling_firm,
    r.root_cause_description,
    r.reason_for_recall
FROM openfda_rest.recall_dqk_2024
LATERAL VIEW explode(results) AS r;

-- DPT (Extravascular blood-flow probe)
CREATE LIGHTNING REST TABLE recall_dpt_2024 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=product_code:DPT+AND+event_date_initiated:[2024-01-01+TO+2024-12-31]&limit=100",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_dpt_2024_table WITH CONTAINER openfda AS
SELECT
    r.cfres_id,
    r.res_event_number,
    r.product_code,
    r.event_date_initiated,
    r.recall_status,
    r.recalling_firm,
    r.root_cause_description,
    r.reason_for_recall
FROM openfda_rest.recall_dpt_2024
LATERAL VIEW explode(results) AS r;

-- DRE (Vessel dilator for percutaneous catheterization)
CREATE LIGHTNING REST TABLE recall_dre_2024 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=product_code:DRE+AND+event_date_initiated:[2024-01-01+TO+2024-12-31]&limit=100",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_dre_2024_table WITH CONTAINER openfda AS
SELECT
    r.cfres_id,
    r.res_event_number,
    r.product_code,
    r.event_date_initiated,
    r.recall_status,
    r.recalling_firm,
    r.root_cause_description,
    r.reason_for_recall
FROM openfda_rest.recall_dre_2024
LATERAL VIEW explode(results) AS r;

-- DWF (Cardiopulmonary bypass catheter, cannula and tubing)
CREATE LIGHTNING REST TABLE recall_dwf_2024 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=product_code:DWF+AND+event_date_initiated:[2024-01-01+TO+2024-12-31]&limit=100",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_dwf_2024_table WITH CONTAINER openfda AS
SELECT
    r.cfres_id,
    r.res_event_number,
    r.product_code,
    r.event_date_initiated,
    r.recall_status,
    r.recalling_firm,
    r.root_cause_description,
    r.reason_for_recall
FROM openfda_rest.recall_dwf_2024
LATERAL VIEW explode(results) AS r;

-- OES (Cardiac catheterization kit)
CREATE LIGHTNING REST TABLE recall_oes_2024 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=product_code:OES+AND+event_date_initiated:[2024-01-01+TO+2024-12-31]&limit=100",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_oes_2024_table WITH CONTAINER openfda AS
SELECT
    r.cfres_id,
    r.res_event_number,
    r.product_code,
    r.event_date_initiated,
    r.recall_status,
    r.recalling_firm,
    r.root_cause_description,
    r.reason_for_recall
FROM openfda_rest.recall_oes_2024
LATERAL VIEW explode(results) AS r;

-- DYF (Vascular graft prosthesis, under 6mm)
CREATE LIGHTNING REST TABLE recall_dyf_2024 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=product_code:DYF+AND+event_date_initiated:[2024-01-01+TO+2024-12-31]&limit=100",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_dyf_2024_table WITH CONTAINER openfda AS
SELECT
    r.cfres_id,
    r.res_event_number,
    r.product_code,
    r.event_date_initiated,
    r.recall_status,
    r.recalling_firm,
    r.root_cause_description,
    r.reason_for_recall
FROM openfda_rest.recall_dyf_2024
LATERAL VIEW explode(results) AS r;

-- OEQ (Angiography and angioplasty kit)
CREATE LIGHTNING REST TABLE recall_oeq_2024 FROM OPENFDA_REST REQUEST(
    endpoint "https://api.fda.gov/device/recall.json?search=product_code:OEQ+AND+event_date_initiated:[2024-01-01+TO+2024-12-31]&limit=100",
    method "get",
    response_type "json",
    http_encoding "URLENCODED"
) HEADER (
) BODY ();

CREATE SCHEMASTORE VIEW recall_oeq_2024_table WITH CONTAINER openfda AS
SELECT
    r.cfres_id,
    r.res_event_number,
    r.product_code,
    r.event_date_initiated,
    r.recall_status,
    r.recalling_firm,
    r.root_cause_description,
    r.reason_for_recall
FROM openfda_rest.recall_oeq_2024
LATERAL VIEW explode(results) AS r;
