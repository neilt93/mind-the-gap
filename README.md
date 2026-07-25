# Mind the Gap

**This robot knows the item will not fit. It tries anyway.**

Mind the Gap exposes hidden beliefs inside robot policies and turns ignored
knowledge into a live safety intervention.

Vision-language-action (VLA) policies can carry an internal, linearly-decodable
belief about the scene — *"this item cannot fit in that carton"* — and still
plan the unsafe action anyway. The belief is present; the action head just
doesn't use it. This demo makes that gap visible, live:

1. **Time-synchronised cognition viewer** — footage of a validated Isaac
   episode plays beside a live robot trace: what the policy **Sees**,
   **Believes**, **Plans**, and **Executes** at the current timestep.
   *Believes* is the frozen linear probe decoded from a real forward pass on
   the exact frame the policy saw at the run's decision point; *Executes* is
   the controller's measured phase from the run's per-step trace. Every label
   is a measured quantity — nothing is scripted to match the video.
2. **The freeze** — mid-carry on the shallow carton, playback halts on the
   contradiction: *the model knows the item will not fit, and it is still
   moving toward the carton.*
3. **CONNECT BELIEF TO ACTION** — one press switches to the recorded run of
   the same scene with the probe wired as a GO/HOLD veto: the arm never
   touches the hazard. Zero retraining: the belief the policy ignores becomes
   the safety check.
4. **Judge controls** — flip deep ↔ shallow carton, native policy ↔ safety
   wire, scrub the timeline. Raw per-step trace JSONs are linked under
   *Engineering details*, and the **? ABOUT THIS PROJECT** side tab explains
   every mechanism, the validation numbers, and the honest live-vs-recorded
   split.

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
- The cognition viewer (`frontend/`): time-synced belief/plan/execute trace,
  freeze-frame contradiction moment, CONNECT-the-wire flow, scene/mode
  switching, side explainer tab.
- The proxy backend (`proxy/`) and the client interface contract
  (`client/gate_client.py`).
- The scene manifest (`scenes/demo_scenes.json`) and the replay/trace
  plumbing that pairs each clip with its measured per-step trace.
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

| You do | You see |
|---|---|
| Shallow carton · native policy | Trace reads **BELIEVES: will not fit** (probe logit ≈ +7) while **EXECUTES: carrying toward carton** — playback freezes on the contradiction |
| Press **CONNECT BELIEF TO ACTION** | The recorded safety-wire run: HOLD fires at the decision step, the arm never picks up the item |
| Deep carton | Belief reads *fits* (logit ≈ −14), belief and action agree, clean placement |

The belief is decoded from the policy's own hidden state; the behavior
contradicts it until the wire connects them. Minding that gap is the demo.
