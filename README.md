# Mind the Gap

**A real robot policy, traced step by step.**

The demo plays a genuine OpenVLA-OFT episode (a fine-tuned
vision-language-action model) completing a LIBERO pick-and-place task in
simulation (robosuite/MuJoCo), beside a live trace panel that updates with
the video:

| Row | Source |
|---|---|
| Scene | the task setup (object + basket) |
| Phase | derived from the recorded per-step actions (approach, grasp, lift, carry, release) |
| Command | the executed action, in plain language, with magnitude |
| Gripper | open / closing / holding / releasing, from the gripper channel |
| Outcome | "in progress" until the episode actually ends |
| Internal | layer-9 activation norm, labeled experimental and not semantically interpreted |

A timeline of key moments appears live as the data reaches them, including
the frozen safety wire's check at the first model step (its live logit is
reported verbatim; the probe is out-of-domain on LIBERO and is not
presented as a belief).

Every label is a measured quantity from the recorded rollout. Nothing is
scripted to match the video.

## How it works

```
browser (frontend/) --> proxy backend (proxy/) --> private engine endpoint
                        public, this repo         (models + probe + clips,
                        forwards JSON only         stay on the lab server)
```

- `frontend/index.html`: the viewer (video + synced trace + timeline).
- `proxy/server.py`: serves the page and forwards `/api/*` to the engine
  (`ENGINE_URL` env var; no server addresses or credentials in this repo).
- `tools/simpler_trace_adapter.py`: converts any end-to-end policy rollout
  (per-step 7-dim actions + video) into the viewer's trace contract, with
  phases derived only from measured signals.
- `tools/enrich_trace.py`: adds per-step command / strength / gripper
  (and optional activation norms) computed from the recorded actions.
- `client/gate_client.py`: the engine's request/response contract.

## Provenance

- **Existed before the hackathon** (private, referenced over HTTP): the
  research stack that motivated the project. On a packing benchmark
  (PackingBench, Isaac Sim) we found a fine-tuned OpenVLA carries a
  linearly-decodable belief ("this item cannot fit") that its action head
  ignores; a frozen linear probe wired in as a runtime GO/HOLD monitor
  took hazardous placement attempts from 4/4 to 0/4 while preserving all
  fit placements, with 10/10 gate decisions at 7+ logit margins.
- **Built during the hackathon**: this viewer, the proxy, the trace
  adapter/enricher, and the LIBERO runs. The OpenVLA-OFT episodes were
  recorded on our rig during the hackathon (10/10 task successes, wire in
  the loop reading GO at every episode with zero interference; the veto
  path was demonstrated with a forced HOLD, labeled as such).
- Model checkpoints: `moojink/openvla-7b-oft-finetuned-libero-object`
  (official OFT LIBERO checkpoint); LIBERO benchmark and robosuite are
  open source.

## Run

```bash
pip install -r proxy/requirements.txt
ENGINE_URL=http://<engine-host>:<port> uvicorn proxy.server:app --port 8790
# open http://localhost:8790
```
