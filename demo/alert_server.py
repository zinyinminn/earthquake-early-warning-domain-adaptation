# alert_server.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import asyncio, json, os, time
from typing import Any, Dict, List

app = FastAPI(title="EEW Demo", docs_url=None, redoc_url=None)

# Allow LAN access (phones on same Wi-Fi)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI from ./static
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

subscribers: List[asyncio.Queue] = []
RECENT_LIMIT = 20
recent_alerts: List[Dict[str, Any]] = []


def pack_sse(ev_type: str, payload: Dict[str, Any]) -> str:
    return f"event: {ev_type}\n" + "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


async def sse_gen(req: Request):
    q: asyncio.Queue = asyncio.Queue()
    subscribers.append(q)
    try:
        # hello on connect
        yield pack_sse("hello", {"ts": time.time()})
        while True:
            if await req.is_disconnected():
                break
            try:
                item = await asyncio.wait_for(q.get(), timeout=15.0)
                yield pack_sse("alert", item)
            except asyncio.TimeoutError:
                # heartbeat
                yield pack_sse("ping", {"ts": time.time()})
    finally:
        try:
            subscribers.remove(q)
        except ValueError:
            pass


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/admin")
def admin():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/stream")
async def stream(req: Request):
    return StreamingResponse(
        sse_gen(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/send")
async def send_alert(payload: Dict[str, Any]):
    """Broadcast an alert to all connected clients."""
    payload = dict(payload)
    payload["server_ts"] = time.time()

    recent_alerts.append(payload)
    if len(recent_alerts) > RECENT_LIMIT:
        del recent_alerts[: len(recent_alerts) - RECENT_LIMIT]

    dead = []
    for q in subscribers:
        try:
            q.put_nowait(payload)
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            subscribers.remove(q)
        except ValueError:
            pass

    return JSONResponse({"status": "ok", "delivered_to": len(subscribers)})


@app.get("/recent")
def get_recent():
    return JSONResponse({"items": recent_alerts[::-1]})


# Optional: placeholder metrics endpoint if you later wire your model here
@app.get("/metrics")
def metrics():
    return JSONResponse(
        dict(
            accuracy=0.994, precision=1.000, recall=0.988, f1=0.994,
            n_eq=10020, n_noise=9880
        )
    )
