---
description: Assign the model each orch role runs (planner, implement, review, fix, repair)
agent: orch
---

Assign the model each orch role runs.

**Roles are exactly:** `planner`, `implement`, `review`, `fix`, `repair`. A role
that is not pinned runs the config's top-level `model` default.

**Which config:** a `--config` file *replaces* `orch.config.yaml` rather than
layering on it, so the file that matters is the one a run is handed. Ask the human
which config to edit if it is not obvious — for the shopdash pilot it is
`pilots/shopdash-phase-3.yaml`, and the answer is never "both". Pass the same
`--config` to every command below.

1. Show the current assignments:

   `orch models list --config <path>`

2. Ask which role should change and which model it should use. The models already
   in use are the likely answers; if the human wants to see what exists, run
   `opencode models`.

3. Apply each choice — one command per role, and repeat the whole set rather than
   editing the file by hand:

   `orch models set <role> <model> --config <path>`

4. Re-run `orch models list --config <path>` and show the result so the human sees
   what actually landed.

Never hand-edit the YAML: `orch models` is what preserves the file's comments, and
those comments are the recorded reason for every pin. Report the final table
verbatim rather than paraphrasing it.
