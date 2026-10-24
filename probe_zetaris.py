# probe_zetaris.py: one real call to learn the shape of Zetaris's SQL response.
# Run:  .venv/bin/python probe_zetaris.py
# Prints key names, types and a truncated sample. Never prints the API key.
import json
import os
import uuid

import requests
from dotenv import load_dotenv

load_dotenv()

SQL = "SELECT 1 AS one, 'x' AS label"

response = requests.post(
    os.environ["ZETARIS_REST_URL"].rstrip("/") + "/sql-editor/sqls/run",
    headers={
        "Authorization": "Bearer " + os.environ["ZETARIS_API_KEY"],
        "X-Org-ID": os.environ["ZETARIS_ORG_ID"],
        "X-Request-ID": str(uuid.uuid4()),
    },
    json={"queryId": str(uuid.uuid4()), "sql": SQL, "source": "SqlEditor", "limit": 5},
    timeout=60,
)
print("status      :", response.status_code)
payload = response.json()
print("top keys    :", {k: type(v).__name__ for k, v in payload.items()})
print("sample      :", json.dumps(payload, default=str)[:1200])
