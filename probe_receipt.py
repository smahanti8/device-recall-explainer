# probe_receipt.py: one real call, then find where the Meterless receipt sits.
# Run:  .venv/bin/python probe_receipt.py
# Self-test (no network):  .venv/bin/python probe_receipt.py --selftest   (add --full to dump all receipt fields)
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

WANTED = ("receipt", "routed_model", "provider_model", "egress")
SECRET_HINTS = ("authorization", "key", "token", "cookie", "secret")


def find_paths(node, path=""):
    """Return (path, value) for every dict key whose name is in WANTED."""
    hits = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in WANTED:
                hits.append((here, value))
            hits += find_paths(value, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            hits += find_paths(value, f"{path}[{i}]")
    return hits


def redact(headers):
    return {
        k: ("[redacted]" if any(h in k.lower() for h in SECRET_HINTS) else v)
        for k, v in headers.items()
    }


def summarize(body):
    """Pull the award evidence out of the top-level `meterless` block."""
    meterless = body.get("meterless") or {}
    if not meterless:
        return {}
    receipt = meterless.get("receipt") or {}
    return {
        "provider_model": meterless.get("provider_model"),
        "routed_model": meterless.get("routed_model"),
        "egress": (meterless.get("egress") or {}).get("destination"),
        "latency_ms": (receipt.get("usage") or {}).get("latency_ms"),
        "receipt_id": receipt.get("receipt_id"),
    }


def selftest():
    sample = {"meterless": {"routed_model": "nvidia/n", "egress": {"destination": "nvidia"}}}
    assert summarize(sample)["routed_model"] == "nvidia/n"
    assert summarize({}) == {}
    body = {"model": "x", "choices": [], "meterless_receipt": {"routed_model": "nvidia/n"}}
    assert find_paths(body) == [("meterless_receipt.routed_model", "nvidia/n")]
    assert redact({"Authorization": "Bearer abc", "content-type": "j"}) == {
        "Authorization": "[redacted]",
        "content-type": "j",
    }
    print("selftest PASS")


def main():
    load_dotenv()
    client = OpenAI(
        base_url=os.environ["MODEL_BASE_URL"],
        api_key=os.environ["MODEL_API_KEY"],
        default_headers={"meterless-app": "hackathon-probe", "meterless-scope": "user"},
    )
    raw = client.chat.completions.with_raw_response.create(
        model=os.environ["NEMOTRON_MODEL"],
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=200,
    )
    body = raw.parse().model_dump()
    print("top-level keys :", sorted(body))
    print("model field    :", body.get("model"))
    print("headers        :", json.dumps(redact(dict(raw.headers)), indent=2))
    evidence = summarize(body)
    if evidence:
        for key, value in evidence.items():
            print(f"{key:<15}: {value}")
    else:
        print("receipt        : none in body; check headers or Nightshift's run details")
    if "--full" in sys.argv:
        print(json.dumps(find_paths(body), indent=2, default=str))


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
