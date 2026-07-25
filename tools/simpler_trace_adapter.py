#!/usr/bin/env python3
"""Convert a LIBERO / SimplerEnv (or any end-to-end policy) rollout
into the mind-the-gap viewer's replay contract.

LIBERO + an official OpenVLA (or OpenVLA-OFT) checkpoint is the
primary target: `--env libero` sets the robosuite gripper convention
(+1 closes, -1 opens) and ~20 Hz control. OpenVLA-OFT's chunked /
parallel decoding changes nothing here — pass the flattened sequence
of EXECUTED per-step actions; the adapter never assumes how they were
decoded.

The viewer consumes two artifacts per condition key:
  <key>.mp4   — H.264/yuv420p clip (Chrome cannot decode the mpeg4
                Simple Profile that sim writers often emit; this tool
                always re-encodes).
  <key>.json  — {"decision": {...}, "steps": [{"step", "phase"}, ...]}
                with steps aligned to the VIDEO timeline at
                --video-fps (default 30): steps[floor(t * fps)].

End-to-end policies (OpenVLA in SimplerEnv) have no controller phase
machine, so phases here are derived from MEASURED per-step signals
only — the gripper channel and the dominant motion direction of the
end-effector deltas. Vocabulary (already understood by the frontend):
  IDLE APPROACH GRASP LIFT CARRY PLACE RELEASE RETREAT HOLD

Input rollout format (--rollout, .json or .npz):
  JSON: {"actions": [[dx,dy,dz,droll,dpitch,dyaw,gripper], ...],
         optional "control_hz": float}
  or a bare JSON list of 7-vectors.
  NPZ:  an "actions" array of shape (T, 7).

The belief decision is NOT derived here — pass the probe's step-0
readout explicitly (--logit/--decision), from the same
forward-parity capture recipe used to fit the probe.

Example (LIBERO):
  python tools/simpler_trace_adapter.py --env libero \
      --rollout rollout.json --video raw.mp4 --key native_shallow \
      --scene shallow --seed 13 --logit 7.49 --decision HOLD \
      --out-dir serve/
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

VOCAB = ["IDLE", "APPROACH", "GRASP", "LIFT", "CARRY", "PLACE",
         "RELEASE", "RETREAT", "HOLD"]


def load_actions(path: Path):
    if path.suffix == ".npz":
        data = np.load(path)
        acts = np.asarray(data["actions"], dtype=float)
        hz = float(data["control_hz"]) if "control_hz" in data else None
        return acts, hz
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        return np.asarray(raw["actions"], dtype=float), raw.get("control_hz")
    return np.asarray(raw, dtype=float), None


def derive_phases(acts: np.ndarray, close_thresh: float, open_thresh: float,
                  close_dir: str, move_eps: float) -> list:
    """One phase per CONTROL step, from measured signals only.

    ``close_dir`` handles gripper conventions: "below" = a LOW gripper
    value commands close (bridge/SimplerEnv WidowX); "above" = a HIGH
    value commands close (LIBERO / robosuite: +1 close, -1 open).
    """
    T = len(acts)
    g = acts[:, 6]
    xyz = acts[:, :3]
    speed = np.linalg.norm(xyz, axis=1)

    if close_dir == "below":
        closed_mask, open_mask = g < close_thresh, g > open_thresh
    else:
        closed_mask, open_mask = g > close_thresh, g < open_thresh

    # A gated/held rollout: gripper never commands a close and the arm
    # barely moves.
    if float(speed.max(initial=0.0)) < move_eps and not closed_mask.any():
        return ["HOLD"] * T

    c = int(np.argmax(closed_mask)) if closed_mask.any() else None
    reopened = None
    if c is not None:
        after = open_mask[c:]
        if after.any():
            reopened = c + int(np.argmax(after))

    phases = []
    for i in range(T):
        if c is None or i < c:
            phases.append("APPROACH" if speed[i] >= move_eps else "IDLE")
        elif i <= c + 1:
            phases.append("GRASP")
        elif reopened is not None and i >= reopened:
            phases.append("RELEASE" if i <= reopened + 1 else "RETREAT")
        else:
            dx, dy, dz = xyz[i]
            lateral = float(np.hypot(dx, dy))
            if dz > move_eps and dz >= lateral:
                phases.append("LIFT")
            elif dz < -move_eps and -dz >= lateral:
                phases.append("PLACE")
            elif lateral >= move_eps:
                phases.append("CARRY")
            else:
                phases.append(phases[-1] if phases else "CARRY")
    # Light smoothing: kill single-step flickers.
    for i in range(1, T - 1):
        if phases[i - 1] == phases[i + 1] != phases[i]:
            phases[i] = phases[i - 1]
    return phases


def video_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
         str(path)], capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def reencode(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "quiet", "-y", "-i", str(src),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(dst)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout", required=True, type=Path)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--key", required=True,
                    help="condition key, e.g. native_shallow / hold_gated")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logit", type=float, required=True,
                    help="probe step-0 logit (forward-parity capture)")
    ap.add_argument("--decision", choices=["GO", "HOLD"], required=True)
    ap.add_argument("--env", choices=["simpler", "libero", "custom"],
                    default="custom",
                    help="preset: simpler = bridge gripper (low value "
                         "closes), ~5 Hz; libero = robosuite gripper "
                         "(+1 closes, -1 opens), ~20 Hz")
    ap.add_argument("--control-hz", type=float, default=None,
                    help="policy control rate; default = env preset, "
                         "else infer so the rollout spans the video")
    ap.add_argument("--video-fps", type=float, default=30.0)
    ap.add_argument("--gripper-close-thresh", type=float, default=None)
    ap.add_argument("--gripper-open-thresh", type=float, default=None)
    ap.add_argument("--gripper-close-dir", choices=["below", "above"],
                    default=None)
    ap.add_argument("--move-eps", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, default=Path("serve"))
    args = ap.parse_args()

    presets = {
        "simpler": {"dir": "below", "close": 0.5, "open": 0.5, "hz": 5.0},
        "libero": {"dir": "above", "close": 0.5, "open": -0.5, "hz": 20.0},
        "custom": {"dir": "below", "close": 0.5, "open": 0.5, "hz": None},
    }[args.env]
    close_dir = args.gripper_close_dir or presets["dir"]
    close_thresh = (args.gripper_close_thresh
                    if args.gripper_close_thresh is not None
                    else presets["close"])
    open_thresh = (args.gripper_open_thresh
                   if args.gripper_open_thresh is not None
                   else presets["open"])

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg/ffprobe required", file=sys.stderr)
        return 2

    acts, hz_from_file = load_actions(args.rollout)
    if acts.ndim != 2 or acts.shape[1] < 7:
        print(f"actions must be (T, 7+); got {acts.shape}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = args.out_dir / f"{args.key}.mp4"
    reencode(args.video, out_mp4)
    dur = video_duration(out_mp4)

    control_hz = (args.control_hz or hz_from_file or presets["hz"]
                  or (len(acts) / dur))
    control_phases = derive_phases(acts, close_thresh, open_thresh,
                                   close_dir, args.move_eps)

    # Resample control-rate phases onto the video timeline.
    n_video = int(round(dur * args.video_fps))
    steps = []
    for i in range(n_video):
        t = i / args.video_fps
        j = min(len(control_phases) - 1, int(t * control_hz))
        steps.append({"step": i, "phase": control_phases[j]})

    trace = {
        "decision": {
            "scene": args.scene, "seed": args.seed,
            "belief_logit": args.logit, "decision": args.decision,
            "hold": args.decision == "HOLD",
        },
        "env": args.env,
        "fps": args.video_fps,
        "control_hz": control_hz,
        "phase_source": "derived from measured per-step actions "
                        "(gripper channel + motion direction) — no "
                        "controller phase machine exists for an "
                        "end-to-end policy",
        "steps": steps,
    }
    out_json = args.out_dir / f"{args.key}.json"
    out_json.write_text(json.dumps(trace, indent=1))

    seen = []
    for s in steps:
        if not seen or seen[-1][0] != s["phase"]:
            seen.append((s["phase"], s["step"]))
    print(f"[adapter] {args.key}: {len(acts)} control steps @ "
          f"{control_hz:.2f} Hz -> {n_video} video steps @ "
          f"{args.video_fps:.0f} fps ({dur:.1f}s)")
    print("[adapter] timeline: " + " -> ".join(
        f"{p}@{i / args.video_fps:.1f}s" for p, i in seen))
    print(f"[adapter] wrote {out_mp4} and {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
