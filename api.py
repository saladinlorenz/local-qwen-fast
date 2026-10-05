"""
FastAPI REST API server.

Provides OpenAI-compatible endpoints for chat completions.
Run with: python src/api.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from engine import engine

app = FastAPI(
    title="Qwen3-0.6B Local API",
    description="REST API for local Qwen3-0.6B chat model",
    version="1.0.0",
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
    """Root endpoint with API info."""
    return {
        "name": "Qwen3-0.6B Local API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Send a message and get a response.

    If no conversation_id is provided, a new conversation is created.
    """
    cid = req.conversation_id
    if cid is None:
        cid = engine.new_conversation()

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

    return ChatResponse(conversation_id=cid, answer=full_answer)


@app.get("/conversations", response_model=List[ConversationInfo])
def list_conversations():
    """List all conversations."""
    rows = engine.list_conversations()
    return [
        ConversationInfo(id=r[0], title=r[1], created_at=r[2])
        for r in rows
    ]


@app.delete("/conversation/{conversation_id}")
def delete_conversation(conversation_id: int):
    """Delete a conversation by ID."""
    engine.delete_conversation(conversation_id)
    return {"status": "deleted", "id": conversation_id}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
