"""
Unified entry point: FastAPI (PRIMARY) + Gradio UI on /ui (SECONDARY).

The API is the main surface. On startup the model is preloaded so the first
/OpenAI request is already warm. The Gradio chat is mounted under /ui and
shares the same engine + SQLite history.

Run: python3 app.py   (listens on 0.0.0.0:$PORT, default 8000)
"""

import os
import json
import time
import uuid
import socket
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from engine import engine
from webui import PAGE

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3-0.6B")
MODEL_ID = (os.getenv("CHAT_MODEL_PATH") or os.getenv("CHAT_MODEL_FILE")
            or "Qwen3-0.6B.Q4_K_M.gguf")

# Unique marker returned by ``next(gen, marker)`` when a stream is exhausted.
_STREAM_END = object()



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the model in the startup so the first API request is fast.
    # Do not crash the server if loading fails: /health reports the error and
    # a retry is attempted on the first request via engine._try_load().
    ok = engine.preload()
    if ok:
        print(f"[startup] model preloaded: {MODEL_ID}", flush=True)
    else:
        print(f"[startup] model NOT loaded yet: {engine.last_error}", flush=True)
    yield


app = FastAPI(
    title=f"{MODEL_NAME} Local API",
    description="Local LLM API (OpenAI-compatible) with optional Gradio UI on /ui",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def root():
    # Single-page UI (Chat / Historique / Config). The OpenAI API stays the
    # primary surface; this page is just a thin client over it.
    return PAGE


@app.get("/ui", response_class=HTMLResponse)
def ui_page():
    return PAGE


@app.get("/health")
def health():
    return {
        "status": "healthy" if engine.model_ready else "loading",
        "model": MODEL_ID,
        "model_ready": engine.model_ready,
        "error": engine.last_error,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completion. Supports stream=True (SSE)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"error": "'messages' must be a non-empty list"}, status_code=400)

    model = body.get("model", MODEL_NAME)
    temperature = float(body.get("temperature", 0.7))
    max_tokens = int(body.get("max_tokens", 512))
    top_p = float(body.get("top_p", 0.9))
    repeat_penalty = float(body.get("repeat_penalty", 1.1))
    stream = bool(body.get("stream", False))
    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if not stream:
        try:
            content = await run_in_threadpool(
                engine.chat, messages, temperature, max_tokens, top_p, repeat_penalty
            )
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    async def event_stream():
        gen = engine.chat_stream(messages, temperature, max_tokens, top_p, repeat_penalty)
        disconnected = False
        since_check = 0
        try:
            while True:
                # Detect a dropped client so the engine can abort generation
                # instead of running to the full max_tokens budget on a dead
                # socket. Probing every token costs an extra receive() per
                # token; every 16 (~1.5s at 10 tok/s) is responsive enough.
                if since_check <= 0:
                    if await request.is_disconnected():
                        disconnected = True
                        break
                    since_check = 16
                since_check -= 1
                token = await run_in_threadpool(next, gen, _STREAM_END)
                if token is _STREAM_END:
                    break
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": token}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            if not disconnected:
                final = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
        except GeneratorExit:
            disconnected = True
            raise  # client disconnected: finally below closes `gen`
        except Exception as e:
            err = {"error": {"message": str(e)}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # `gen` holds the generation lock: close it deterministically so the
            # engine aborts the native loop and releases the lock. Relying on GC
            # leaks the lock when a client disconnects early.
            try:
                await run_in_threadpool(gen.close)
            except Exception:
                pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---- Legacy simple endpoints (kept for existing clients) ----

@app.post("/chat")
def chat_simple(request: Request):
    try:
        body = request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    message = body.get("message", "")
    conversation_id = body.get("conversation_id")
    if conversation_id is None:
        conversation_id = engine.new_conversation()
    temp = float(body.get("temperature", 0.7))
    max_tok = int(body.get("max_tokens", 512))
    top_p = float(body.get("top_p", 0.9))
    rep = float(body.get("repeat_penalty", 1.1))
    full = ""
    for chunk in engine.stream_chat(conversation_id, message, temp, max_tok, top_p, rep):
        full += chunk
    return {"conversation_id": conversation_id, "answer": full}


@app.get("/conversations")
def list_conversations():
    rows = engine.list_conversations()
    return [
        {"id": r[0], "title": r[1], "created_at": r[2]}
        for r in rows
    ]


@app.delete("/conversation/{conversation_id}")
def delete_conversation(conversation_id: int):
    engine.delete_conversation(conversation_id)
    return {"status": "deleted", "id": conversation_id}


# ---- UI JSON/SSE endpoints (thin client for webui.PAGE) ----

def _lan_ip() -> str:
    """Best-effort LAN address of this device (works inside the chroot).

    ``request.headers['host']`` is useless here: the VPS gateway rewrites the
    Host header to the loopback address, so we ask the kernel which source IP
    it would use to reach the internet instead.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return "127.0.0.1"


@app.get("/ui/config")
def ui_config(request: Request):
    """Everything the Config tab needs, including the LAN API URL to copy."""
    port = int(os.environ.get("PORT", "8000"))
    lan_ip = _lan_ip()
    return {
        "model_name": MODEL_NAME,
        "model_path": MODEL_ID,
        "model_ready": engine.model_ready,
        "error": engine.last_error,
        # Direct LAN endpoint, usable from any OpenAI client on the network:
        # http://<phone-ip>:<port>/v1  (the client derives the proxy URL).
        "lan_ip": lan_ip,
        "port": port,
        "endpoints": ["/v1/chat/completions", "/v1/models", "/health"],
        "n_threads": int(os.getenv("N_THREADS", "4")),
        "n_ctx": int(os.getenv("N_CTX", "4096")),
    }


@app.get("/ui/conversation/{conversation_id}")
def ui_get_conversation(conversation_id: int):
    msgs = engine.load_conversation(conversation_id)
    return [{"role": m.role, "content": m.content} for m in msgs if m.role in ("user", "assistant")]


@app.post("/ui/chat")
async def ui_chat(request: Request):
    """SSE chat that persists to SQLite (so it shows up in Historique)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    message = (body.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)
    cid = body.get("conversation_id")
    cid = int(cid) if cid else engine.new_conversation()
    temp = float(body.get("temperature", 0.7))
    max_tok = int(body.get("max_tokens", 512))
    top_p = float(body.get("top_p", 0.9))
    rep = float(body.get("repeat_penalty", 1.1))

    gen = engine.stream_chat(cid, message, temp, max_tok, top_p, rep)

    async def es():
        yield f"data: {json.dumps({'conversation_id': cid})}\n\n"
        since_check = 0
        try:
            while True:
                if since_check <= 0:
                    if await request.is_disconnected():
                        break
                    since_check = 16
                since_check -= 1
                tok = await run_in_threadpool(next, gen, _STREAM_END)
                if tok is _STREAM_END:
                    break
                yield f"data: {json.dumps({'token': tok})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            try:
                await run_in_threadpool(gen.close)
            except Exception:
                pass

    return StreamingResponse(es(), media_type="text/event-stream")


@app.post("/ui/model")
async def ui_load_model(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    path = (body.get("model") or "").strip()
    if not path:
        return JSONResponse({"error": "empty model"}, status_code=400)
    try:
        engine.set_model(path)
        os.environ["CHAT_MODEL_PATH"] = path
        return {"ok": True, "model": path}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---- Optional legacy Gradio UI (heavy; disabled unless UI_GRADIO=1) ----

def _mount_ui():
    if os.environ.get("UI_GRADIO", "0") not in ("1", "true", "True"):
        return None
    try:
        import gradio as gr
        from ui import demo
        return gr.mount_gradio_app(app, demo, path="/ui/legacy")
    except Exception as e:
        print(f"[ui] failed to mount Gradio on /ui/legacy: {e}", flush=True)
        return None


_mount_ui()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
