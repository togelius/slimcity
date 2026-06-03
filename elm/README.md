# ELM: code-genome MAP-Elites with an LLM operator

A second QD harness for slimcity that is **not** based on float vectors /
CMA-ME (that's the top-level `qd_train.py`). Here the genome is **Python
source code** — a closed-loop tile-placement policy — and the variation
operator is **Claude** (mutation & crossover). The MAP-Elites archive uses the
same behavior measures and `[0,1]²` / 20×20 binning as the CMA-ME runs, so the
two are directly comparable.

## The genome contract

A genome is a source string defining one function:

```python
def act(obs, state):
    # called once per action step; sim advances `ticks_per_action` ticks between calls
    return None            # place nothing this step
    # or
    return (tool, x, y)    # place `tool` at tile column x, row y
```

`act` is called `n_actions` times per episode. `state` is a dict that persists
across all steps of one episode (the policy's memory). This is the **same
cadence as the closed-loop policies in `policy.py`** — one decision per call,
simulation between calls — but with an explicit no-op and arbitrary code
instead of an argmax over net logits. Full contract + available primitives:
`elm/sandbox.py` → `PRIMITIVES_CARD`.

## Files

| file | role |
|------|------|
| `sandbox.py` | `Obs`, restricted exec namespace, `compile_policy`, `parse_action`, `PRIMITIVES_CARD` |
| `evaluate_code.py` | `eval_code(source, …)` → fitness/measures (reuses `evaluate.py` shaping + `tile_descriptors`) |
| `archive.py` | `MapElitesArchive` — dict of code elites, same bins as `qd_train.py`; save/load `.npz` |
| `seeds.py` | hand-written starters: `plan` (working ~400-pop city) and `random` (baseline) |
| `operator.py` | `LLMOperator` (Claude) + `MockOperator` (offline) |
| `elm_train.py` | the driver / CLI |

## Running

Use the **engine's interpreter** (`/usr/bin/python3`, Python 3.9 — the built
`_micropolisengine` extension is compiled against it).

Offline smoke (no API key, mock operator, inline eval):

```bash
/usr/bin/python3 -m elm.elm_train --operator mock --iters 30 --no-sandbox \
    --seeds plan,random --out results/elm_smoke.npz
```

Real run (Claude as operator):

```bash
/usr/bin/python3 -m pip install anthropic          # once
export ANTHROPIC_API_KEY=sk-...
/usr/bin/python3 -m elm.elm_train --operator claude --model claude-sonnet-4-6 \
    --iters 200 --measures res_ind --fitness dense \
    --out results/elm_resind.npz
```

Each iteration makes **one Claude call** (the cost driver). The archive saves
every `--save-every` iters and dumps the champion source to `<out>_best.py`.

### Key flags

- `--iters` LLM variation steps
- `--operator {claude,mock}`, `--model`, `--temperature`
- `--seeds plan,random` which starters to inject
- `--crossover-rate` (default 0.25), `--weighted-parents`
- `--n-actions` / `--ticks-per-action` episode shape (defaults 120 / 100)
- `--fitness {pop,dense,varied,growth}` (default `dense` — gives signal even
  before zones grow; pure `pop` is near-binary)
- `--measures {road_ind,res_ind,density,entropy_count}` (default `res_ind`)
- `--archive-dims` (default `20,20`)
- `--timeout` per-eval wall-clock budget; `--no-sandbox` to eval inline

## Safety

Evolved code is LLM-authored. Guards (verified):

- restricted exec namespace — no `import`, file, `eval`/`exec` (e.g. `import os`
  raises `__import__ not found`);
- per-step `act` errors are caught and treated as no-ops (the run continues);
- each evaluation runs in a **spawned subprocess with a wall-clock timeout**
  (`--timeout`), so an infinite loop or engine crash is killed and the genome
  is flagged invalid rather than hanging the run.

This is a guard-rail, not a hard security boundary — only run on policies you
generate yourself.
