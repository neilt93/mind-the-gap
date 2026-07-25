# Mind the Gap

**This robot knows the item will not fit. It tries anyway.**

Mind the Gap exposes hidden beliefs inside robot policies and turns ignored
knowledge into a live safety intervention.

Vision-language-action (VLA) policies can carry an internal, linearly-decodable
belief about the scene — *"this item cannot fit in that carton"* — and still
plan the unsafe action anyway. The belief is present; the action head just
doesn't use it. This demo makes that gap visible, live:

1. **Belief readout** — a frozen linear probe on the policy's own hidden state,
   computed on a live forward pass per request, shown as a gauge.
2. **Contradiction view** — the same forward pass produces the policy's planned
   7-DOF action. When the belief says *cannot fit* and the plan still moves
   toward the carton, that's the gap, on screen.
3. **Judge-operated carton flip** — swap the deep carton for a too-shallow one
   and watch the belief flip in real time.
4. **The gate** — press "USE THE ROBOT'S KNOWLEDGE": a runtime safety monitor
   reads the same frozen probe and vetoes (HOLD) the placement the policy
   would otherwise attempt. Zero retraining: the belief the policy ignores
   becomes the safety check.
5. **Validated Isaac replays** — the physical consequence of each condition
   (successful placement / unsafe jam-and-drop / vetoed hold), recorded in
   the simulator under the same scene and gate condition and clearly labeled
   as replays. The VLA forward pass, hidden-state readout, planned action,
   and GO/HOLD decision are computed live on every click; only the slower
   simulator execution is replayed.

## Architecture

```
browser (frontend/) ──► proxy backend (proxy/) ──► private engine endpoint
                        public, this repo          (VLA + frozen probe + scenes,
                        forwards JSON only          stays on the lab server)
```

The engine URL is supplied via the `ENGINE_URL` environment variable — no
server addresses or credentials live in this repo.

## Context — what existed before tonight, and why

I'm a co-founder at **Cybernetic**, where we build **RoboEval** — an
evaluation platform and benchmark suite for embodied / VLA manipulation
policies. One of its benchmarks is **PackingBench**: a robot arm packs
household items into target cartons in simulation, with carton depth as a
controlled hazard variable (a too-shallow carton means the item cannot be
inserted; forcing it jams and drops the item).

In the weeks before this hackathon, our interpretability work on a fine-tuned
OpenVLA policy for PackingBench found a specific, reproducible gap:

- The policy's hidden state carries a **linearly-decodable belief** about
  whether the item can fit the carton — a frozen linear probe
  (StandardScaler + logistic regression) reads it with wide margins.
- That belief is **behaviorally inert where it matters**: steering it doesn't
  change behavior, gradient training at the recipes we tried couldn't wire it
  into the actions, and the action head plans the same placement motion
  whether or not the belief says *cannot fit*.
- Wired in as a **frozen runtime monitor**, the same probe cleanly gates
  behavior: in the pre-hack validation run, hazardous placement attempts went
  4/4 → 0/4 with the gate on while fit placements stayed 4/4, and the gate's
  decisions were 10/10 with margins of 7+ logits.

Tonight's hack is the debugger that makes that pre-existing result **visible
and interactive** — the belief, the contradicting plan, and the gate, live,
with the carton flip in a judge's hands.

## Honest provenance split

**Existed before Night Hack (private, referenced over HTTP, not in this repo):**
- The VLA policy and its fine-tuned weights.
- The interpretability engine and capture tooling that found the belief
  direction, and the frozen linear probe (`mean/scale/w/b`) that reads it.
- The captured demo scenes (deep / medium / shallow carton episode starts) and
  the research showing the belief exists, the action head ignores it, and the
  probe can gate placements.

**Built tonight (this repo + a thin private wrapper):**
- The debugger UI (`frontend/`): belief gauge, planned-vs-executed action view,
  contradiction banner, judge carton-flip, gate toggle.
- The proxy backend (`proxy/`) and the client interface contract
  (`client/gate_client.py`).
- The scene manifest (`scenes/demo_scenes.json`).
- On the server: a minimal HTTP wrapper exposing the pre-existing gate as
  `POST /gate → {belief_logit, GO/HOLD, planned_action}`. No new science —
  an HTTP surface over what already existed.

## Run

```bash
pip install -r proxy/requirements.txt
ENGINE_URL=http://<engine-host>:<port> uvicorn proxy.server:app --port 8790
# open http://localhost:8790
```

`client/gate_client.py` documents the full request/response contract if you
want to call the engine from your own code.

## What you should see

| Carton  | Belief gauge        | Planned action | Gate ON   |
|---------|---------------------|----------------|-----------|
| Deep    | fits (logit ≪ 0)    | moves to pack  | GO        |
| Shallow | cannot fit (logit > 0) | moves to pack anyway ⚠ | HOLD — vetoed |

The belief flips when the judge flips the carton; the plan doesn't. Minding
that gap is the demo.
