"""
Unified entry point: FastAPI (PRIMARY) + Gradio UI on /ui (SECONDARY).

The API is the main surface. On startup the model is preloaded so the first
/OpenAI request is already warm. The Gradio chat is mounted under /ui and
shares the same engine + SQLite history.

Run: python3 app.py   (listens on 0.0.0.0:$PORT, default 8000)
"""

import os
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from engine import engine

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3-0.6B")
MODEL_ID = os.getenv("CHAT_MODEL_PATH") or "Qwen3-0.6B.Q4_K_M.gguf"


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
    title="Qwen3-0.6B Local API",
    description="Local LLM API (OpenAI-compatible) with optional Gradio UI on /ui",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def root():
    return (
        "<html><body style='font-family:sans-serif;background:#0f172a;"
        "color:#fff;padding:2rem'>"
        "<h1>Qwen3-0.6B Local API</h1>"
        "<p>OpenAI-compatible endpoint: <code>POST /v1/chat/completions</code></p>"
        "<p>Other endpoints: <code>/v1/models</code>, <code>/health</code>, "
        "<code>/chat</code>, <code>/conversations</code></p>"
        "<p>Chat UI (secondary): <a href='/ui' style='color:#7dd3fc'>/ui</a></p>"
        "<p>Docs: <a href='/docs' style='color:#7dd3fc'>/docs</a></p>"
        "</body></html>"
    )


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
            content = engine.chat(messages, temperature, max_tokens, top_p, repeat_penalty)
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

    def event_stream():
        try:
            for token in engine.chat_stream(messages, temperature, max_tokens, top_p, repeat_penalty):
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": token}, "finish_reason": None}
                    ],
                }
                import json
                yield f"data: {json.dumps(chunk)}\n\n"
            final = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            import json
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            import json
            err = {"error": {"message": str(e)}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

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


# ---- Mount the Gradio UI (SECONDARY) on /ui ----

def _mount_ui():
    try:
        import gradio as gr
        from ui import demo
        gr_app = gr.mount_gradio_app(app, demo, path="/ui")
        return gr_app
    except Exception as e:
        print(f"[ui] failed to mount Gradio on /ui: {e}", flush=True)
        return None


_mount_ui()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
