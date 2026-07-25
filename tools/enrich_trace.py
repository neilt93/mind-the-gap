"""Enrich a viewer trace with per-step fields computed ONLY from the
recorded rollout actions: plain-language action command, command
strength, and gripper state. Runs after simpler_trace_adapter.py and
augments its <key>.json in place.

Usage:
  enrich_trace.py --trace replays/<key>/actor_trace.json \
      --rollout actions.json [--video-fps 30] [--control-hz 20]
"""

import argparse
import json
from pathlib import Path

import numpy as np

AXES = [("dx", "move right", "move left"),
        ("dy", "move forward", "move back"),
        ("dz", "move up", "move down")]


def command_text(a):
    trans = np.asarray(a[:3], dtype=float)
    rot = np.asarray(a[3:6], dtype=float)
    if np.abs(trans).max() < 1e-3 and np.abs(rot).max() < 1e-3:
        return "hold position", 0.0
    if np.abs(rot).max() > np.abs(trans).max():
        return "rotate wrist", float(np.abs(rot).max())
    i = int(np.abs(trans).argmax())
    _, pos, neg = AXES[i]
    return (pos if trans[i] > 0 else neg), float(np.abs(trans).max())


def strength_text(mag):
    if mag < 1e-3:
        return "none"
    if mag < 0.2:
        return "slight"
    if mag < 0.6:
        return "moderate"
    return "strong"


def gripper_states(acts, close_dir="above", thresh=0.0):
    """open / closing / holding / releasing from the gripper channel."""
    closed = [(a[6] > thresh) if close_dir == "above" else (a[6] < thresh)
              for a in acts]
    out, state = [], "open"
    for t, c in enumerate(closed):
        prev = closed[t - 1] if t else False
        if c and not prev:
            state = "closing"
        elif c and prev:
            state = "holding"
        elif not c and prev:
            state = "releasing"
        elif not c and state == "releasing":
            state = "open"
        out.append(state)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--rollout", required=True, type=Path)
    ap.add_argument("--video-fps", type=float, default=30.0)
    ap.add_argument("--control-hz", type=float, default=20.0)
    args = ap.parse_args()

    tr = json.loads(args.trace.read_text())
    roll = json.loads(args.rollout.read_text())
    acts = roll["actions"]
    grip = gripper_states(acts)
    hnorms = roll.get("h_norms")   # optional: internal activity

    for s in tr["steps"]:
        t = s["step"] / args.video_fps
        j = min(len(acts) - 1, int(t * args.control_hz))
        cmd, mag = command_text(acts[j])
        s["command"] = cmd
        s["strength"] = strength_text(mag)
        s["strength_value"] = round(mag, 3)
        s["gripper"] = grip[j]
        if hnorms:
            k = min(len(hnorms) - 1, int(t * args.control_hz))
            s["h_norm"] = round(float(hnorms[k]), 1)

    tr["enriched"] = ("per-step command/strength/gripper computed from "
                      "the recorded rollout actions"
                      + ("; h_norm = layer-9 activation norm per model "
                         "query" if hnorms else ""))
    args.trace.write_text(json.dumps(tr))
    print(f"[enrich] {args.trace}: {len(tr['steps'])} steps enriched"
          f"{' (+h_norm)' if hnorms else ''}")


if __name__ == "__main__":
    main()
