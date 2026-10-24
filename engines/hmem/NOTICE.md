# Vendored: H-MEM reference implementation and conformance suite

Source: https://github.com/Meterless/Meterless (engines/hmem), commit 1742dec56a66e58eef9dda2d2b865823758ca8cd
License: Apache-2.0 (see LICENSE in this folder)

`reference/` and `conformance/` are copied unmodified. Our own code lives outside this
folder (`agent/memory.py`, `agent/memory_cli.mts`) and talks to H-MEM through its public API
(`add`, `query`, `feedback`).

Verify the copy: `cd engines/hmem/reference && npm ci --ignore-scripts && npx tsx ../conformance/runner.ts`
