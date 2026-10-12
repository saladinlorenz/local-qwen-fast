"""
Core chat engine with SQLite persistence.

This module handles:
- Model loading (llama-cpp-python) — lazy, non-blocking at import (VPS/proot safe)
- Conversation management (SQLite WAL + index)
- Streaming chat completions (system prompt FR + sliding window)
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, List, Tuple, Dict

import os

DB_PATH = Path(os.getenv("CHAT_DB_PATH", "chat_history.db"))

# Modèle : chemin local / repo HF via CHAT_MODEL_PATH, sinon fichier bundlé, sinon HF.
# (nom avec points = layout réel du repo MaziyarPanahi, tirets = ancien nom local)
_LOCAL_CANDIDATES = [
    Path("Qwen3-0.6B-Q4_K_M.gguf"),
    Path("models/Qwen3-0.6B-Q4_K_M.gguf"),
    Path("Qwen3-0.6B.Q4_K_M.gguf"),
    Path("models/Qwen3-0.6B.Q4_K_M.gguf"),
]
MODEL_PATH = os.getenv("CHAT_MODEL_PATH", "") or next(
    (str(p) for p in _LOCAL_CANDIDATES if p.is_file()), ""
)
DEFAULT_REPO = os.getenv("CHAT_MODEL_REPO", "MaziyarPanahi/Qwen3-0.6B-GGUF")
DEFAULT_FILE = os.getenv("CHAT_MODEL_FILE", "Qwen3-0.6B.Q4_K_M.gguf")

# --- Réglages faciles (env overridable) ---
N_CTX = int(os.getenv("CHAT_N_CTX", "2048"))  # 2048 ≈ 2x plus rapide que 4096, suffisant pour 0.6B
N_THREADS = int(os.getenv("CHAT_THREADS", str(os.cpu_count() or 4)))
N_BATCH = int(os.getenv("CHAT_BATCH", "256"))
MAX_HISTORY_MESSAGES = int(os.getenv("CHAT_HISTORY", "20"))  # fenêtre glissante anti-dépassement
SYSTEM_PROMPT = os.getenv(
    "CHAT_SYSTEM",
    "Tu es Qwen3, un assistant utile, concis et amical. Réponds en français sauf si "
    "l'utilisateur écrit dans une autre langue. Réponses courtes par défaut, détaillées si demandé.",
)


@dataclass
class Message:
    """Represents a single chat message."""
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def _llama_kwargs() -> Dict:
    return dict(n_ctx=N_CTX, n_threads=N_THREADS, n_batch=N_BATCH, verbose=False)


class ChatEngine:

    def __init__(self, model_path: Optional[str] = None):
        # Lazy : ne charge PAS le modèle à l'import (pas de crash si absent, démarrage instantané).
        self._model_path = model_path or MODEL_PATH
        self.llm = None
        self._load_error = None
        self._init_db()

    def _try_load(self) -> bool:
        """Attempt to load the model. Returns True on success."""
        if self.llm is not None:
            return True
        try:
            from llama_cpp import Llama
            kw = _llama_kwargs()
            if self._model_path:
                p = str(self._model_path)
                if "/" in p and not Path(p).exists():
                    # Repo ID Hugging Face
                    try:
                        self.llm = Llama.from_pretrained(repo_id=p, filename=DEFAULT_FILE, **kw)
                    except Exception:
                        self.llm = Llama.from_pretrained(
                            repo_id=p, filename=DEFAULT_FILE.replace(".", "-"), **kw
                        )
                else:
                    self.llm = Llama(model_path=p, **kw)
            else:
                self.llm = Llama.from_pretrained(
                    repo_id=DEFAULT_REPO,
                    filename=DEFAULT_FILE,
                    **kw,
                )
            self._model_path = self._model_path or f"{DEFAULT_REPO}/{DEFAULT_FILE}"
            self._load_error = None
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    @property
    def model_ready(self) -> bool:
        return self.llm is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._load_error

    def set_model(self, model_path: str):
        """Change the model at runtime."""
        self._model_path = model_path.strip()
        self.llm = None
        os.environ["CHAT_MODEL_PATH"] = self._model_path
        if not self._try_load():
            raise RuntimeError(self._load_error or "Model loading failed")

    # ---------- DB ----------
    def _connect(self):
        conn = sqlite3.connect(DB_PATH)
        # WAL = lectures non bloquantes (Gradio + API en parallèle)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        return conn

    def _init_db(self):
        """Initialize SQLite database with required tables."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                created_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conv
            ON messages(conversation_id, id)
        """)
        conn.commit()
        conn.close()

    def new_conversation(self, title: str = "New Conversation") -> int:
        """Create a new conversation."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversations (title, created_at) VALUES (?, ?)",
            (title, datetime.utcnow().isoformat()),
        )
        cid = cur.lastrowid
        conn.commit()
        conn.close()
        return cid

    def load_conversation(self, conversation_id: int, limit: Optional[int] = None) -> List[Message]:
        """Load messages from a conversation (image refs filtered, text-only model)."""
        conn = self._connect()
        cur = conn.cursor()
        if limit:
            cur.execute(
                "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            )
            rows = list(reversed(cur.fetchall()))
        else:
            cur.execute(
                "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            )
            rows = cur.fetchall()
        conn.close()
        out = []
        for r in rows:
            content = r[1] if len(r) > 1 else ""
            if content and (".png" in content or ".jpg" in content or content.startswith("http")):
                continue
            out.append(Message(role=r[0], content=r[1], timestamp=r[2] if len(r) > 2 else ""))
        return out

    def save_message(self, conversation_id: int, message: Message):
        """Save a message to the database (image refs ignored)."""
        if message.content and (".png" in message.content or ".jpg" in message.content or message.content.startswith("http")):
            return
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (conversation_id, message.role, message.content, message.timestamp),
        )
        conn.commit()
        conn.close()

    def list_conversations(self) -> List[Tuple[int, str, str]]:
        """List all conversations."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT id, title, created_at FROM conversations ORDER BY id DESC")
        rows = cur.fetchall()
        conn.close()
        return rows

    def delete_conversation(self, conversation_id: int):
        """Delete a conversation and all its messages."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cur.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        conn.close()

    def update_conversation_title(self, conversation_id: int, title: str):
        """Update the title of a conversation."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
        conn.commit()
        conn.close()

    def export_text(self, conversation_id: int) -> str:
        """Export a conversation as Markdown."""
        msgs = self.load_conversation(conversation_id)
        lines = [f"# Conversation {conversation_id}", ""]
        for m in msgs:
            lines.append(f"**{m.role}** ({m.timestamp}):\n{m.content}\n")
        return "\n".join(lines)

    # ---------- Inference ----------
    def _build_messages(self, history: List[Message], user_text: str) -> List[Dict]:
        """System FR + sliding window (never exceeds n_ctx)."""
        window = history[-MAX_HISTORY_MESSAGES:] if len(history) > MAX_HISTORY_MESSAGES else history
        messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in window:
            if m.role in ("user", "assistant", "system") and m.content and m.content.strip():
                messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": user_text})
        return messages

    def stream_chat(
        self,
        conversation_id: int,
        user_text: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
    ) -> Generator[str, None, None]:
        """Stream a chat completion."""
        if not self._try_load():
            yield f"❌ Modèle non chargé : {self._load_error}\n"
            return

        history = self.load_conversation(conversation_id)
        messages = self._build_messages(history, user_text)

        self.save_message(conversation_id, Message(role="user", content=user_text))
        if len(history) == 0:
            title = user_text[:50] + ("..." if len(user_text) > 50 else "")
            self.update_conversation_title(conversation_id, title)

        try:
            completion = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=True,
            )
        except Exception as e:
            yield f"\n❌ Erreur modèle : {e}"
            return

        full_answer = ""
        try:
            for part in completion:
                delta = part["choices"][0].get("delta", {})
                content = delta.get("content", "") or ""
                if content:
                    full_answer += content
                    yield content
        except Exception as e:
            err = f"\n\n⚠️ Génération interrompue : {e}"
            full_answer += err
            yield err

        self.save_message(conversation_id, Message(role="assistant", content=full_answer))

    def regenerate_last(
        self,
        conversation_id: int,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
    ) -> Generator[str, None, None]:
        """Delete the last assistant reply and re-stream it (reuses last user message)."""
        history = self.load_conversation(conversation_id)
        if not history:
            yield ""
            return
        last_user = None
        for m in reversed(history):
            if m.role == "user":
                last_user = m.content
                break
        if last_user is None:
            yield ""
            return
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM messages WHERE id = (SELECT MAX(id) FROM messages WHERE conversation_id = ? AND role = 'assistant')",
            (conversation_id,),
        )
        cur.execute(
            "DELETE FROM messages WHERE id = (SELECT MAX(id) FROM messages WHERE conversation_id = ? AND role = 'user' AND content = ?)",
            (conversation_id, last_user),
        )
        conn.commit()
        conn.close()
        yield from self.stream_chat(conversation_id, last_user, temperature, max_tokens, top_p, repeat_penalty)


engine = ChatEngine()
