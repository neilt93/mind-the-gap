# Mind the Gap

**This robot knows the item will not fit. It tries anyway.**

Mind the Gap exposes hidden beliefs inside robot policies and turns ignored
knowledge into a safety intervention.

Vision-language-action (VLA) policies can carry an internal, linearly
decodable belief about the scene, such as *"this item cannot fit in that
carton"*, and still plan the unsafe action. The belief is present; the action
head just does not use it. This viewer makes the gap visible:

1. **Cognition viewer.** Footage of a validated simulator run plays beside a
   live robot trace: what the policy **Sees**, **Believes**, **Plans**, and
   **Executes** at the current timestep. *Believes* is a frozen linear probe
   decoded from a real forward pass on the exact frame the policy saw.
   *Executes* comes from the run's per-step trace. Every label is a measured
   quantity; nothing is scripted to match the video.
2. **The freeze.** Mid-carry on the shallow carton, playback halts on the
   contradiction: the model knows the item will not fit, and it is still
   moving toward the carton.
3. **Connect belief to action.** One press switches to the recorded run of
   the same scene with the probe wired as a GO/HOLD veto. The arm never
   touches the hazard. Zero retraining: the belief the policy ignores becomes
   the safety check.
4. **Real model runs.** A LIBERO pick-and-place by OpenVLA-OFT, a genuine
   fine-tuned VLA recorded on this rig, honestly labeled (no probe is fit on
   that domain yet). `tools/simpler_trace_adapter.py` converts LIBERO or
   SimplerEnv rollouts into the viewer's trace contract.

## Architecture

```
browser (frontend/) ──► proxy backend (proxy/) ──► private engine endpoint
                        public, this repo          (VLA + frozen probe + scenes,
                        forwards JSON only          stays on the lab server)
```

The engine URL is supplied via the `ENGINE_URL` environment variable. No
server addresses or credentials live in this repo.

## Context: what existed before tonight, and why

I'm a co-founder at **Cybernetic**, where we build **RoboEval**, an
evaluation platform and benchmark suite for embodied and VLA manipulation
policies. One of its benchmarks is **PackingBench**: a robot arm packs
household items into target cartons in simulation, with carton depth as a
controlled hazard variable (a too-shallow carton means the item cannot be
inserted; forcing it jams and drops the item).

In the weeks before this hackathon, our interpretability work on a fine-tuned
OpenVLA policy for PackingBench found a specific, reproducible gap:

- The policy's hidden state carries a **linearly decodable belief** about
  whether the item can fit the carton. A frozen linear probe (standardize
  plus logistic regression) reads it with wide margins.
- That belief is **behaviorally inert where it matters**: steering it does
  not change behavior, fine-tuning at the recipes we tried could not wire it
  into the actions, and the action head plans the same placement motion
  whether or not the belief says *cannot fit*.
- Wired in as a **frozen runtime monitor**, the same probe cleanly gates
  behavior: in the pre-hack validation run, hazardous placement attempts went
  4/4 to 0/4 with the gate on while correct placements stayed 4/4, and the
  gate's decisions were 10/10 with margins of 7+ logits.

Tonight's hack is the debugger that makes that pre-existing result visible
and interactive: the belief, the contradicting plan, and the gate, in a
judge's hands.

**Concept coverage today:** one validated concept, *fit hazard* ("will this
item fit safely?"), the probe behind everything above. The UI also lists six
experimental candidate concepts (grasp success, target location, collision
risk, task progress, wrong object, goal completion), clearly labeled: these
are preliminary probe outputs, not yet validated across held-out scenes.
Motion labels in the trace (approach, grasp, carry, release, GO vs HOLD) are
interpretations of the policy's action commands, not decoded beliefs.

## Honest provenance split

**Existed before Night Hack (private, referenced over HTTP, not in this repo):**
- The VLA policy and its fine-tuned weights.
- The interpretability engine and capture tooling that found the belief
  direction, and the frozen linear probe (`mean/scale/w/b`) that reads it.
- The captured demo scenes and the research showing the belief exists, the
  action head ignores it, and the probe can gate placements.

**Built tonight (this repo plus a thin private wrapper):**
- The cognition viewer (`frontend/`): time-synced belief/plan/execute trace,
  freeze-frame contradiction moment, connect-the-wire flow, real-model tab,
  side explainer panel.
- The proxy backend (`proxy/`) and the client interface contract
  (`client/gate_client.py`).
- The scene manifest (`scenes/demo_scenes.json`), the replay/trace plumbing,
  and the rollout adapter (`tools/simpler_trace_adapter.py`).
- On the server: a minimal HTTP wrapper exposing the pre-existing gate. No
  new science, just an HTTP surface over what already existed.

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
| Play the shallow-carton native run | Trace reads **Believes: item will not fit** (probe logit about +7) while **Executes** walks through grasp and carry; playback freezes on the contradiction |
| Press **CONNECT BELIEF TO ACTION** | The recorded safety-wire run: HOLD at the decision step, the arm never moves |
| Switch to the deep carton | Belief reads *fits* (logit about -14), belief and action agree |
| Open the real-model tab | OpenVLA-OFT completing a LIBERO pick-and-place, recorded on this rig |

The belief is decoded from the policy's own hidden state; the behavior
contradicts it until the wire connects them. Minding that gap is the demo.
