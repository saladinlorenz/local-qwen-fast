"""
FastAPI REST API server.

- POST /chat : réponse complète (inchangé)
- POST /chat/stream : SSE streaming (nouveau)
- GET /conversations, DELETE /conversation/{id}, GET /health
Run with: python api.py
"""

import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from engine import engine

app = FastAPI(
    title="Qwen3-0.6B Local API",
    description="REST API for local Qwen3-0.6B chat model",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    conversation_id: Optional[int] = None
    message: str
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    repeat_penalty: float = 1.1


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    conversation_id: int
    answer: str


class ConversationInfo(BaseModel):
    """Conversation metadata."""
    id: int
    title: str
    created_at: str


@app.get("/")
def root():
    return {"name": "Qwen3-0.6B Local API", "version": "1.1.0", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Send a message and get a response (non-streaming)."""
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message vide")
    cid = req.conversation_id
    if cid is None:
        cid = engine.new_conversation()
    try:
        full_answer = ""
        for chunk in engine.stream_chat(
            conversation_id=cid,
            user_text=req.message,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            repeat_penalty=req.repeat_penalty,
        ):
            full_answer += chunk
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"erreur modèle : {e}")
    return ChatResponse(conversation_id=cid, answer=full_answer)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Streaming SSE : curl -N POST /chat/stream. Events: data: {\"chunk\": ...} puis data: [DONE]."""
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message vide")
    cid = req.conversation_id
    if cid is None:
        cid = engine.new_conversation()

    def gen():
        yield f"data: {json.dumps({'conversation_id': cid})}\n\n"
        try:
            for chunk in engine.stream_chat(
                conversation_id=cid,
                user_text=req.message,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                top_p=req.top_p,
                repeat_penalty=req.repeat_penalty,
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/chat/regenerate", response_model=ChatResponse)
def regenerate(req: ChatRequest) -> ChatResponse:
    """Régénère la dernière réponse de la conversation."""
    if req.conversation_id is None:
        raise HTTPException(status_code=400, detail="conversation_id requis")
    full = ""
    for chunk in engine.regenerate_last(
        conversation_id=req.conversation_id,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        top_p=req.top_p,
        repeat_penalty=req.repeat_penalty,
    ):
        full += chunk
    return ChatResponse(conversation_id=req.conversation_id, answer=full)


@app.get("/conversations", response_model=List[ConversationInfo])
def list_conversations():
    rows = engine.list_conversations()
    return [ConversationInfo(id=r[0], title=r[1], created_at=r[2]) for r in rows]


@app.get("/conversation/{conversation_id}/export")
def export_conversation(conversation_id: int):
    """Export markdown d'une conversation."""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(engine.export_text(conversation_id), media_type="text/markdown")


@app.delete("/conversation/{conversation_id}")
def delete_conversation(conversation_id: int):
    engine.delete_conversation(conversation_id)
    return {"status": "deleted", "id": conversation_id}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": getattr(engine, "_model_path", "?")}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
