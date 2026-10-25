// One JSON request on stdin, one JSON reply on stdout. Bridges the Python agent to the
// vendored H-MEM engine. Run from engines/hmem/reference so tsx resolves from there.
//   {"op":"add","persistDir":D,"record":{content,type,tags,confidence,source}} -> {"id"}
//   {"op":"query","persistDir":D,"query":Q,"topN":N}                           -> {"memories":[...],"context"}
//   {"op":"feedback","persistDir":D,"id":I,"kind":K}                           -> {"ok":true}
//   {"op":"upsert","persistDir":D,"key":T,"record":{...}}                      -> {"id","status":"new|reinforced|changed","previous"?}
//     key is the claim text; records are "<key>: <value>". Same value -> reinforce (helpful feedback);
//     new value -> add a record that supersedes the current one and mark the old one wrong; none -> add.
import { HMEM } from "../engines/hmem/reference/src/index.ts";

const chunks: Buffer[] = [];
for await (const chunk of process.stdin) chunks.push(chunk as Buffer);

try {
  const request = JSON.parse(Buffer.concat(chunks).toString("utf-8"));
  const hmem = new HMEM({ persistDir: request.persistDir, actor: "hackathon-agent" });
  let reply: unknown;
  if (request.op === "add") {
    reply = { id: hmem.add(request.record).id };
  } else if (request.op === "query") {
    const result = hmem.query(request.query, { topN: request.topN });
    reply = {
      memories: result.memories.map((m) => ({
        id: m.memory.id, content: m.memory.content, relevance: m.relevance, source: m.memory.source,
      })),
      context: result.context,
    };
  } else if (request.op === "upsert") {
    const prefix = `${request.key}: `;
    const all = hmem.store.all();
    const replaced = new Set(all.map((r) => r.supersedes).filter(Boolean));
    const current = all
      .filter((r) => r.content.startsWith(prefix) && !replaced.has(r.id))
      .sort((a, b) => Number(b.id.split("-")[1]) - Number(a.id.split("-")[1]))[0];
    if (current && current.content === request.record.content) {
      hmem.feedback(current.id, "helpful");
      reply = { id: current.id, status: "reinforced" };
    } else if (current) {
      const created = hmem.add({ ...request.record, supersedes: current.id });
      hmem.feedback(current.id, "wrong");
      reply = { id: created.id, status: "changed", previous: current.content };
    } else {
      const created = hmem.add(request.record);
      hmem.feedback(created.id, "helpful");
      reply = { id: created.id, status: "new" };
    }
  } else if (request.op === "feedback") {
    hmem.feedback(request.id, request.kind);
    reply = { ok: true };
  } else {
    throw new Error(`unknown op: ${request.op}`);
  }
  process.stdout.write(JSON.stringify(reply));
} catch (error) {
  process.stdout.write(JSON.stringify({ error: (error as Error).message }));
  process.exit(1);
}
