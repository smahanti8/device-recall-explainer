"""What a run spent and where: Zetaris queries fetched versus served from cache or recordings, calls per model,
retries and failures. Counts come from the query log and the model-call log, so they describe what actually ran.
"""


def summarize(queries, calls):
    counts = {"fetched": 0, "cache_hit": 0, "replayed": 0, "errors": 0}
    names = {"success": "fetched", "cache_hit": "cache_hit", "replayed": "replayed", "error": "errors"}
    for entry in queries:
        if entry.get("status") in names:
            counts[names[entry["status"]]] += 1
    models, retries, failed = {}, 0, 0
    for call in calls:
        status = call.get("status")
        if status in ("success", "replayed"):
            models[call["model"]] = models.get(call["model"], 0) + 1
        elif status == "retry":
            retries += 1
        elif status == "failed":
            failed += 1
    return {"queries": counts, "models": models, "retries": retries, "failed": failed}


def parts(summary):
    """The same numbers as short phrases, shared by the printed line and the report panel."""
    q = summary["queries"]
    zetaris = []
    if q["fetched"]:
        zetaris.append(f"{q['fetched']} queries fetched")
    if q["cache_hit"]:
        zetaris.append(f"{q['cache_hit']} served from cache")
    if q["replayed"]:
        zetaris.append(f"{q['replayed']} results replayed from recordings")
    if q["errors"]:
        zetaris.append(f"{q['errors']} failed")
    out = []
    if zetaris:
        out.append("Zetaris: " + ", ".join(zetaris))
    if summary["models"]:
        out.append("models: " + ", ".join(f"{model} x{n}" for model, n in summary["models"].items()))
    if summary["retries"]:
        out.append(f"retries: {summary['retries']}")
    if summary["failed"]:
        out.append(f"failed calls: {summary['failed']}")
    return out


def format_line(summary):
    found = parts(summary)
    return "Observability: " + "; ".join(found) if found else ""
