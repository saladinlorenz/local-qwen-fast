"""
Core chat engine with SQLite persistence.

This module handles:
- Model loading (llama-cpp-python) — lazy, non-blocking at import, with an
  explicit preload() for API-first deployments (fast first request).
- Conversation management (SQLite)
- Streaming chat completions

Performance notes (Android / CPU-only llama.cpp):
- n_threads defaults to 4: a 0.6B model does not scale past ~4-6 threads and
  using all 8 cores oversubscribes the scheduler (slower + hotter). Override
  with the N_THREADS env var.
- A generation lock serialises calls: llama-cpp-python is NOT safe to call
  concurrently on a single Llama instance, so concurrent API requests are
  queued instead of crashing.
"""

import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, List, Tuple

DB_PATH = Path(os.getenv("CHAT_DB_PATH", "chat_history.db"))
MODEL_PATH = os.getenv("CHAT_MODEL_PATH", "")
DEFAULT_REPO = "MaziyarPanahi/Qwen3-0.6B-GGUF"
DEFAULT_FILE = "Qwen3-0.6B.Q4_K_M.gguf"

# CPU tuning: sensible defaults for a mid-range Android SoC, all overridable.
N_THREADS = int(os.getenv("N_THREADS", "4"))
N_CTX = int(os.getenv("N_CTX", "4096"))
N_BATCH = int(os.getenv("N_BATCH", "512"))
USE_MMAP = os.getenv("USE_MMAP", "1") not in ("0", "false", "False")


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ChatEngine:

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path or MODEL_PATH
        self.llm = None
        self._load_error = None
        self._load_lock = threading.Lock()   # guards lazy load
        self._gen_lock = threading.Lock()    # serialises generation
        self._init_db()

    def _try_load(self) -> bool:
        """Attempt to load the model. Returns True on success. Thread-safe."""
        if self.llm is not None:
            return True
        with self._load_lock:
            if self.llm is not None:
                return True
            try:
                from llama_cpp import Llama
                kwargs = dict(n_ctx=N_CTX, n_threads=N_THREADS, n_batch=N_BATCH,
                              use_mmap=USE_MMAP, verbose=False)
                if self._model_path:
                    self.llm = Llama(model_path=self._model_path, **kwargs)
                else:
                    self.llm = Llama.from_pretrained(
                        repo_id=DEFAULT_REPO,
                        filename=DEFAULT_FILE,
                        **kwargs,
                    )
                self._load_error = None
                return True
            except Exception as e:
                self._load_error = str(e)
                return False

    def preload(self) -> bool:
        """Eagerly load the model so the first API request is fast."""
        return self._try_load()

    @property
    def model_ready(self) -> bool:
        return self.llm is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._load_error

    def set_model(self, model_path: str):
        with self._load_lock:
            self._model_path = model_path
            self.llm = None
            os.environ["CHAT_MODEL_PATH"] = model_path
        if not self._try_load():
            raise RuntimeError(self._load_error or "Model loading failed")

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, created_at TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER, role TEXT, content TEXT, timestamp TEXT, FOREIGN KEY (conversation_id) REFERENCES conversations(id))""")
        conn.commit()
        conn.close()

    def new_conversation(self, title: str = "New Conversation") -> int:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO conversations (title, created_at) VALUES (?, ?)", (title, datetime.utcnow().isoformat()))
        cid = cur.lastrowid
        conn.commit()
        conn.close()
        return cid

    def load_conversation(self, conversation_id: int) -> List[Message]:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,))
        rows = cur.fetchall()
        conn.close()
        return [Message(role=r[0], content=r[1], timestamp=r[2]) for r in rows]

    def save_message(self, conversation_id: int, message: Message):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (conversation_id, message.role, message.content, message.timestamp))
        conn.commit()
        conn.close()

    def list_conversations(self) -> List[Tuple[int, str, str]]:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, title, created_at FROM conversations ORDER BY id DESC")
        rows = cur.fetchall()
        conn.close()
        return rows

    def delete_conversation(self, conversation_id: int):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cur.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        conn.close()

    def update_conversation_title(self, conversation_id: int, title: str):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
        conn.commit()
        conn.close()

    def chat(self, messages: List[dict], temperature: float = 0.7,
             max_tokens: int = 512, top_p: float = 0.9,
             repeat_penalty: float = 1.1) -> str:
        """Non-streaming completion (fast path for the REST API).

        Returns the full assistant text. Thread-safe via the generation lock.
        """
        if not self._try_load():
            raise RuntimeError(self._load_error or "Model not loaded")
        with self._gen_lock:
            completion = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=False,
            )
        return completion["choices"][0]["message"]["content"] or ""

    def chat_stream(self, messages: List[dict], temperature: float = 0.7,
                    max_tokens: int = 512, top_p: float = 0.9,
                    repeat_penalty: float = 1.1) -> Generator[str, None, None]:
        """Streaming completion over a plain message list (used by the API).

        Yields token strings. The generation lock is held for the duration of
        the stream so concurrent requests are serialised, and released on
        client disconnect (GeneratorExit) via try/finally.
        """
        if not self._try_load():
            yield f"Model not loaded: {self._load_error}"
            return
        self._gen_lock.acquire()
        try:
            completion = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=True,
            )
            for part in completion:
                delta = part["choices"][0].get("delta", {})
                content = delta.get("content", "") or ""
                if content:
                    yield content
        finally:
            self._gen_lock.release()

    def stream_chat(self, conversation_id: int, user_text: str,
                    temperature: float = 0.7, max_tokens: int = 512,
                    top_p: float = 0.9, repeat_penalty: float = 1.1) -> Generator[str, None, None]:
        """Streaming chat bound to a stored conversation (used by the UI)."""
        if not self._try_load():
            yield f"Model not loaded: {self._load_error}\n"
            return

        history = self.load_conversation(conversation_id)
        messages = [{"role": m.role, "content": m.content} for m in history]
        user_msg = Message(role="user", content=user_text)
        messages.append({"role": "user", "content": user_text})
        self.save_message(conversation_id, user_msg)

        if len(history) == 0:
            title = user_text[:50] + ("..." if len(user_text) > 50 else "")
            self.update_conversation_title(conversation_id, title)

        full_answer = ""
        for content in self.chat_stream(messages, temperature, max_tokens, top_p, repeat_penalty):
            full_answer += content
            yield content

        assistant_msg = Message(role="assistant", content=full_answer)
        self.save_message(conversation_id, assistant_msg)


engine = ChatEngine()
