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
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, List, Tuple

DB_PATH = Path(os.getenv("CHAT_DB_PATH", "chat_history.db"))
MODEL_PATH = os.getenv("CHAT_MODEL_PATH", "")
DEFAULT_REPO = os.getenv("CHAT_MODEL_REPO", "MaziyarPanahi/Qwen3-0.6B-GGUF")
DEFAULT_FILE = os.getenv("CHAT_MODEL_FILE", "Qwen3-0.6B.Q4_K_M.gguf")

# CPU tuning: sensible defaults for a mid-range Android SoC, all overridable.
N_THREADS = int(os.getenv("N_THREADS", "4"))
N_CTX = int(os.getenv("N_CTX", "4096"))
N_BATCH = int(os.getenv("N_BATCH", "512"))
USE_MMAP = os.getenv("USE_MMAP", "1") not in ("0", "false", "False")
# big.LITTLE pinning: run the LLM on the fastest cores only. Off by default so
# desktops keep the full machine; enabled on Android via PIN_BIG=1.
PIN_BIG = os.getenv("PIN_BIG", "0") not in ("0", "false", "False")
USE_FLASH_ATTN = os.getenv("USE_FLASH_ATTN", "0") not in ("0", "false", "False")
# Qwen3 "thinking" mode burns hundreds of tokens before answering; for a local
# 0.6B assistant it is pure latency. Off by default, re-enable with ENABLE_THINKING=1.
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "0") not in ("0", "false", "False")

# Qwen3 occasionally emits an empty ``<think>\n\n</think>`` preamble even with
# the ``/no_think`` soft switch; drop it so neither the UI nor the API shows it.
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_leading_think(text: str) -> str:
    """Remove a leading Qwen3 ``<think>...</think>`` block from a full answer."""
    return _THINK_BLOCK_RE.sub("", text, count=1)


class _ThinkFilter:
    """Streaming filter that drops a leading ``<think>...</think>`` block.

    Only ever touches the very start of the answer; everything else passes
    through untouched. Whitespace/partial prefixes are buffered only until the
    stream can be classified, so normal answers lose no latency.
    """

    def __init__(self, strip: bool):
        self.done = not strip
        self.decided = not strip
        self.in_think = False
        self.drop_ws = False
        self.buf = ""

    def feed(self, chunk: str) -> str:
        if self.done:
            if self.drop_ws:
                chunk = chunk.lstrip()
                if not chunk:
                    return ""
                self.drop_ws = False
            return chunk
        self.buf += chunk
        head = self.buf.lstrip()
        if not self.decided:
            if head.startswith("<think"):
                self.decided = True
                self.in_think = True
            elif "<think".startswith(head):
                return ""  # ambiguous prefix ("", "<", "<th", ...): keep buffering
            else:
                self.decided = True
                self.in_think = False
        if self.in_think:
            if "</think>" in self.buf:
                after = self.buf.split("</think>", 1)[1]
                self.buf = ""
                self.in_think = False
                self.done = True
                after = after.lstrip()
                self.drop_ws = not after
                return after
            return ""
        out, self.buf = self.buf, ""
        self.done = True
        return out

    def flush(self) -> str:
        if self.done:
            return ""
        self.done = True
        if self.in_think:  # unterminated think block at EOF: drop it
            return ""
        out, self.buf = self.buf, ""
        return out



def _pin_to_fastest_cores() -> int:
    """Pin the process to the fastest CPU cluster; return that cluster's size.

    big.LITTLE aware: CPUs are grouped by ``cpuinfo_max_freq`` and only the top
    cluster is kept. This matters because llama.cpp barriers make every thread
    wait on the slowest one, so adding two LITTLE cores to two BIG cores is
    slower than using the two BIG cores alone (measured: 13.7 vs 12.1 tok/s),
    and oversubscribing the BIG cluster (4 threads on 2 cores) collapses to
    ~3 tok/s. Returns 0 when frequencies are unreadable (affinity untouched).
    """
    try:
        import os as _os
        def _maxfreq(c: int) -> int:
            try:
                with open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq") as f:
                    return int(f.read().strip())
            except Exception:
                return 0
        cpus = sorted(_os.sched_getaffinity(0))
        freqs = {c: _maxfreq(c) for c in cpus}
        top = max(freqs.values()) if freqs else 0
        if top <= 0:
            print("[engine] cpu max freq unreadable; affinity left as-is", flush=True)
            return 0
        chosen = sorted(c for c in cpus if freqs[c] == top)
        if chosen:
            _os.sched_setaffinity(0, set(chosen))
            print(f"[engine] pinned to fastest cluster {chosen} ({top // 1000} MHz)", flush=True)
        return len(chosen)
    except Exception as e:
        print(f"[engine] affinity not set: {e}", flush=True)
        return 0



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
            threads = N_THREADS
            if PIN_BIG:
                cluster = _pin_to_fastest_cores()
                if cluster:
                    threads = max(1, min(N_THREADS, cluster))
            try:
                from llama_cpp import Llama
                kwargs = dict(n_ctx=N_CTX, n_threads=threads, n_batch=N_BATCH,
                              use_mmap=USE_MMAP, verbose=False)
                if USE_FLASH_ATTN:
                    kwargs["flash_attn"] = True
                try:
                    self.llm = self._create_llama(Llama, kwargs)
                except Exception:
                    # flash_attn unsupported on this build: retry without it.
                    if "flash_attn" in kwargs:
                        kwargs.pop("flash_attn", None)
                        self.llm = self._create_llama(Llama, kwargs)
                    else:
                        raise
                self._load_error = None
                return True
            except Exception as e:
                self._load_error = str(e)
                return False

    def _create_llama(self, Llama, kwargs: dict):
        if self._model_path:
            return Llama(model_path=self._model_path, **kwargs)
        return Llama.from_pretrained(
            repo_id=DEFAULT_REPO,
            filename=DEFAULT_FILE,
            **kwargs,
        )

    def preload(self) -> bool:
        """Eagerly load the model so the first API request is fast."""
        return self._try_load()

    def _completion_kwargs(self) -> dict:
        """Extra kwargs shared by chat()/chat_stream().

        Kept for builds that honour ``chat_template_kwargs``; the reliable
        switch on llama-cpp-python 0.3.x is the ``/no_think`` soft token that
        ``_prepare_messages`` appends to the last user turn.
        """
        if ENABLE_THINKING:
            return {}
        return {"chat_template_kwargs": {"enable_thinking": False}}

    def _prepare_messages(self, messages: List[dict]) -> List[dict]:
        """Disable the Qwen3 reasoning preamble unless ENABLE_THINKING is set.

        Qwen3's chat template emits a long ``<think>`` block before answering;
        for a local 0.6B model that is pure latency. The official soft switch is
        the ``/no_think`` token, honoured by the template when it appears in the
        system turn, so we make sure a system message carries it (creating one
        when the caller sent none). ``_create_completion`` also passes the
        template kwarg for newer llama-cpp-python builds.
        """
        if ENABLE_THINKING or not messages:
            return messages
        msgs = [dict(m) for m in messages]
        if msgs and msgs[0].get("role") == "system":
            content = msgs[0].get("content") or ""
            if "/no_think" not in content:
                msgs[0]["content"] = content + " /no_think"
        else:
            msgs.insert(0, {"role": "system", "content": "You are a helpful assistant. /no_think"})
        return msgs

    def _create_completion(self, **kwargs):
        """create_chat_completion with the thinking switch, with a safe retry."""
        try:
            return self.llm.create_chat_completion(**kwargs)
        except TypeError:
            kwargs.pop("chat_template_kwargs", None)
            return self.llm.create_chat_completion(**kwargs)

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
        if not self._gen_lock.acquire(timeout=120):
            raise RuntimeError("Model busy: another generation is still running, retry shortly")
        try:
            completion = self._create_completion(
                messages=self._prepare_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=False,
                **self._completion_kwargs(),
            )
        finally:
            self._gen_lock.release()
        result = completion["choices"][0]["message"]["content"] or ""
        if not ENABLE_THINKING:
            result = strip_leading_think(result)
        return result

    def chat_stream(self, messages: List[dict], temperature: float = 0.7,
                    max_tokens: int = 512, top_p: float = 0.9,
                    repeat_penalty: float = 1.1) -> Generator[str, None, None]:
        """Streaming completion over a plain message list (used by the API).

        Generation runs in a dedicated producer thread that owns the generation
        lock and feeds a queue. A client disconnect (GeneratorExit on the
        consumer generator) sets an abort flag which stops the native loop
        within one token and ALWAYS releases the lock in the producer's
        ``finally``. This removes the previous failure mode where a dropped SSE
        connection left ``_gen_lock`` held forever and every later request got
        "Model busy".
        """
        if not self._try_load():
            yield f"Model not loaded: {self._load_error}"
            return
        if not self._gen_lock.acquire(timeout=120):
            yield "Model busy: another generation is still running, retry shortly"
            return

        import queue as _queue
        q: "_queue.Queue" = _queue.Queue()
        abort = threading.Event()
        sentinel = object()

        def _producer():
            stream = None
            try:
                stream = self._create_completion(
                    messages=self._prepare_messages(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stream=True,
                    **self._completion_kwargs(),
                )
                for part in stream:
                    if abort.is_set():
                        break
                    delta = part["choices"][0].get("delta", {})
                    content = delta.get("content", "") or ""
                    if content:
                        q.put(content)
            except BaseException as e:  # noqa: BLE001 - forwarded to the consumer
                q.put(e)
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                q.put(sentinel)
                self._gen_lock.release()

        producer = threading.Thread(target=_producer, name="llama-gen", daemon=True)
        producer.start()
        think = _ThinkFilter(strip=not ENABLE_THINKING)
        try:
            while True:
                item = q.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                out = think.feed(item)
                if out:
                    yield out
            tail = think.flush()
            if tail:
                yield tail
        except GeneratorExit:
            # Client disconnected: tell the producer to stop at the next token
            # boundary. It releases the lock itself; never release it here.
            abort.set()
            raise
        except Exception:
            abort.set()
            raise

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
print(
    f"[engine] ready pin_big={PIN_BIG} flash_attn={USE_FLASH_ATTN} "
    f"thinking={ENABLE_THINKING} threads={N_THREADS} ctx={N_CTX} batch={N_BATCH}",
    flush=True,
)
