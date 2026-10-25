"""Memory for the agent: remembers verified claims across sessions via the vendored H-MEM engine.

Memory is a hint, never evidence. A claim must still cite fresh SQL, and the
evaluator never accepts a memory as a citation. Every record's `source` is the
SQL that proved it. Talks to engines/hmem through agent/memory_cli.mts (JSON in, JSON out).
"""
import json
import os
import re
import subprocess
from dataclasses import dataclass

from agent.evaluator import product_codes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_DIR = os.path.join(ROOT, "engines", "hmem", "reference")
CLI = os.path.join(ROOT, "agent", "memory_cli.mts")
FEEDBACK_KINDS = {"helpful", "not_helpful", "wrong"}


@dataclass(frozen=True)
class Recollection:
    id: str
    content: str
    relevance: float
    source: str


def _node_run(request):
    done = subprocess.run(["npx", "tsx", CLI], input=json.dumps(request), capture_output=True,
                          text=True, cwd=ENGINE_DIR, timeout=60)
    try:
        reply = json.loads(done.stdout or "{}")
    except ValueError:
        reply = {}
    if done.returncode != 0 or "error" in reply:
        raise RuntimeError(f"H-MEM: {reply.get('error') or done.stderr[-300:]}")
    return reply


def _tags(claim):
    tags = [f"product_code:{code}" for code in product_codes(claim.text)]
    tags += [f"year:{year}" for year in re.findall(r"\b(?:19|20)\d{2}\b", claim.text)]
    if "cardiovascular" in claim.text.lower():
        tags.append("cardiovascular")
    return tags


class Memory:
    def __init__(self, persist_dir, run=None):
        self.persist_dir = os.path.abspath(str(persist_dir))  # Node runs from engines/hmem, so never pass a relative path
        self._run = run or _node_run
        self.last_learn = {"new": 0, "reinforced": 0, "changed": []}

    def _send(self, **request):
        return self._run({"persistDir": self.persist_dir, **request})

    @staticmethod
    def _record(claim):
        if claim.citation is None:
            raise ValueError("a claim without a citation cannot be remembered (H-MEM needs a source)")
        return {"content": f"{claim.text}: {claim.value}", "type": "factual", "tags": _tags(claim),
                "confidence": 0.9, "source": claim.citation.sql}

    def remember(self, claim):
        return self._send(op="add", record=self._record(claim))["id"]

    def recall(self, question, top_n=5):
        reply = self._send(op="query", query=question, topN=top_n)
        return [Recollection(m["id"], m["content"], m["relevance"], m["source"])
                for m in reply.get("memories", [])]

    def context_for(self, question):
        hits = self.recall(question)
        if not hits:
            return ""
        lines = ["Previously verified in earlier sessions (hints, not evidence; "
                 "cite fresh SQL for every number you state):"]
        lines += [f"- {hit.content} (source: {hit.source})" for hit in hits]
        return "\n".join(lines)

    def feedback(self, memory_id, kind):
        if kind not in FEEDBACK_KINDS:
            raise ValueError(f"feedback kind must be one of {sorted(FEEDBACK_KINDS)}")
        self._send(op="feedback", id=memory_id, kind=kind)

    def mark_wrong(self, memory_id):
        self.feedback(memory_id, "wrong")

    def learn(self, claims, rejections):
        """Upsert the claims that passed the evaluator; last_learn reports new, reinforced and changed."""
        rejected = [rejection.claim for rejection in rejections]
        report = {"new": 0, "reinforced": 0, "changed": []}
        ids = []
        for claim in claims:
            if claim.citation is None or claim in rejected:
                continue
            reply = self._send(op="upsert", key=claim.text, record=self._record(claim))
            ids.append(reply["id"])
            status = reply.get("status", "new")
            if status == "changed":
                report["changed"].append({"text": claim.text, "was": reply.get("previous", ""),
                                          "now": f"{claim.text}: {claim.value}"})
            else:
                report["reinforced" if status == "reinforced" else "new"] += 1
        self.last_learn = report
        return ids
