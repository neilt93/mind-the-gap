"""Interface contract for the mind-the-gap belief/gate engine.

This module defines the request/response shape for calling the gate
and a tiny helper client. It contains NO engine code — the
interpretability engine, trained policies, and frozen probe live on
a private server and are reached over HTTP.

Contract
--------
POST {ENGINE_URL}/gate      body: {"scene": str, "seed": int}
  -> {
       "scene": str,            # scene id (e.g. "control", "shallow")
       "seed": int,             # which captured episode variant
       "belief_logit": float,   # probe logit; > 0 = model believes
                                #   the item CANNOT fit (hazard)
       "hold": bool,            # gate fired
       "decision": "GO"|"HOLD",
       "planned_action": {axis: float},   # what the policy plans
       "executed_action": {axis: float},  # after the gate (HOLD
                                          #   zeroes the 6 EE deltas)
       "recorded_logit": float|null,      # provenance: the logit
                                          #   this episode logged when
                                          #   originally captured
       "latency_ms": int
     }

GET {ENGINE_URL}/scenes          -> [{"id", "label", "description", "seeds"}]
GET {ENGINE_URL}/frame/{scene}/{seed} -> PNG (the exact model-input image)
GET {ENGINE_URL}/health          -> {"ok": bool, "model_loaded": bool, ...}

Viewer additions (all read-only):
GET {ENGINE_URL}/replay/{key}        -> MP4, validated Isaac episode for
                                        key in {go_valid, unsafe_attempt,
                                        hold_gated}
GET {ENGINE_URL}/replay_trace/{key}  -> {"decision": {belief_logit, decision},
                                         "steps": [{"phase": str}, ...]}
                                        per-step trace time-aligned to the
                                        clip (30 steps/s)
GET {ENGINE_URL}/trace/{arm}         -> raw model's-eye per-step trace JSON
GET {ENGINE_URL}/live/status         -> {"state": RESETTING|READY|INFERRING|
                                         HOLD|EXECUTING|COMPLETED|FAILED|
                                         OFFLINE, ...}
GET {ENGINE_URL}/live/frame          -> JPEG, current live-sim camera frame
POST {ENGINE_URL}/live/start         -> body {"scene": str,
                                         "safety_enabled": bool}
"""

from typing import Any, Dict, List

import httpx

ACTION_AXES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


def get_scenes(engine_url: str) -> List[Dict[str, Any]]:
    r = httpx.get(f"{engine_url}/scenes", timeout=10.0)
    r.raise_for_status()
    return r.json()


def get_gate_decision(engine_url: str, scene: str,
                      seed: int = 13) -> Dict[str, Any]:
    r = httpx.post(f"{engine_url}/gate",
                   json={"scene": scene, "seed": seed}, timeout=60.0)
    r.raise_for_status()
    return r.json()
