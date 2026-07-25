"""mind-the-gap proxy backend.

Serves the static debugger UI and forwards /api/* to the private
engine endpoint (ENGINE_URL env var). The engine — VLA policy,
frozen probe, interpretability hook — never leaves its server;
this proxy only relays JSON and scene images.

Run:
  ENGINE_URL=http://127.0.0.1:8791 uvicorn proxy.server:app --port 8790
"""

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

ENGINE_URL = os.environ.get("ENGINE_URL", "http://127.0.0.1:8791")
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="mind-the-gap")


class GateReq(BaseModel):
    scene: str
    seed: int = 13


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/scenes")
def scenes():
    r = httpx.get(f"{ENGINE_URL}/scenes", timeout=10.0)
    return Response(r.content, media_type="application/json",
                    status_code=r.status_code)


@app.post("/api/gate")
def gate(req: GateReq):
    try:
        r = httpx.post(f"{ENGINE_URL}/gate", json=req.model_dump(),
                       timeout=60.0)
    except httpx.ConnectError:
        raise HTTPException(502, "engine unreachable")
    return Response(r.content, media_type="application/json",
                    status_code=r.status_code)


@app.get("/api/frame/{scene}/{seed}")
def frame(scene: str, seed: int):
    r = httpx.get(f"{ENGINE_URL}/frame/{scene}/{seed}", timeout=30.0)
    return Response(r.content, media_type="image/png",
                    status_code=r.status_code)


@app.get("/api/health")
def health():
    try:
        r = httpx.get(f"{ENGINE_URL}/health", timeout=5.0)
        return Response(r.content, media_type="application/json")
    except httpx.ConnectError:
        return {"ok": False, "engine": "unreachable"}
