# Mind the Gap

**A live debugger for the gap between what a robot policy believes and what it does.**

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
4. **The gate** — toggle a runtime safety monitor that reads the same frozen
   probe and vetoes (HOLD) the placement the policy would otherwise attempt.
   Zero retraining: the belief the policy ignores becomes the safety check.

## Architecture

```
browser (frontend/) ──► proxy backend (proxy/) ──► private engine endpoint
                        public, this repo          (VLA + frozen probe + scenes,
                        forwards JSON only          stays on the lab server)
```

The engine URL is supplied via the `ENGINE_URL` environment variable — no
server addresses or credentials live in this repo.

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
