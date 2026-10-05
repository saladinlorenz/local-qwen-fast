# Qwen3-0.6B Local Chat

A lightweight, 100% Python chat application with Qwen3-0.6B running locally. Features a modern Gradio interface, persistent chat history (SQLite), and an optional FastAPI REST API.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **100% Local & Private**: No external APIs, no Ollama, no cloud dependencies
- **Lightweight**: Uses quantized GGUF model (~400MB RAM)
- **Modern Chat Interface**: Gradio-based UI with streaming responses (typewriter effect)
- **Persistent History**: All conversations saved in SQLite database
- **Multi-Conversation**: Create, switch, and delete multiple chat sessions
- **REST API**: Optional FastAPI server with OpenAI-compatible endpoints
- **Advanced Controls**: Temperature, max tokens, top-p, repeat penalty
- **Easy Setup**: Install and run in minutes

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/qwen3-local-chat.git
cd qwen3-local-chat
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Chat Interface Only

```bash
python src/app.py
```

Then open your browser to: **http://127.0.0.1:7860**

### Option 2: Chat Interface + REST API

Terminal 1 (API Server):
```bash
python src/api.py
```

Terminal 2 (Chat Interface):
```bash
python src/app.py
```

## Usage

### Chat Interface

1. Open http://127.0.0.1:7860 in your browser
2. Start typing in the message box
3. Press Enter or click Send
4. Use the sidebar to:
   - Switch between conversations
   - Create new discussions
   - Delete old conversations
5. Adjust advanced parameters in the "Advanced Settings" accordion

### REST API

The API server runs on **http://127.0.0.1:8000** by default.

#### Endpoints

**POST /chat** - Send a message and get a response

```bash
curl http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, explain what is an API in one sentence",
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

Response:
```json
{
  "conversation_id": 1,
  "answer": "An API (Application Programming Interface) is..."
}
```

**GET /conversations** - List all conversations

```bash
curl http://127.0.0.1:8000/conversations
```

**DELETE /conversation/{id}** - Delete a conversation

```bash
curl -X DELETE http://127.0.0.1:8000/conversation/1
```

#### Interactive API Docs

Open **http://127.0.0.1:8000/docs** for Swagger UI interactive documentation.

## Configuration

### Model Selection

By default, the app downloads `qwen3-0.6b-q4_k_m.gguf` from Hugging Face. You can change this in `src/engine.py`:

```python
self.llm = Llama.from_pretrained(
    repo_id="MaziyarPanahi/Qwen3-0.6B-GGUF",
    filename="qwen3-0.6b-q4_k_m.gguf",  # Change this
    n_ctx=4096,
    verbose=False,
)
```

Available quantizations:
- `q4_k_m` - Best balance (recommended)
- `q5_k_m` - Better quality, slightly larger
- `q8_0` - Highest quality, largest size
- `q2_k` - Smallest, fastest, lower quality

### Custom Model Path

If you already have a GGUF file locally:

```python
self.llm = Llama(
    model_path="/path/to/your/model.gguf",
    n_ctx=4096,
    verbose=False,
)
```

### Port Configuration

Change ports in `src/app.py` and `src/api.py`:

```python
demo.launch(server_name="127.0.0.1", server_port=7860)  # Gradio
uvicorn.run(app, host="127.0.0.1", port=8000)           # FastAPI
```

## Project Structure

```
qwen3-local-chat/
├── src/
│   ├── __init__.py
│   ├── engine.py       # Core chat engine + SQLite
│   ├── app.py          # Gradio chat interface
│   └── api.py          # FastAPI REST server
├── requirements.txt
├── README.md
├── LICENSE
├── example_client.py
├── setup.sh
├── setup.bat
└── .gitignore
```

## Technical Details

### Dependencies

- **llama-cpp-python**: Python bindings for llama.cpp (GGUF inference)
- **gradio**: Modern chat UI
- **fastapi**: REST API framework
- **uvicorn**: ASGI server
- **aiosqlite**: Async SQLite support

### Database Schema

Two tables in `chat_history.db`:

**conversations**
- id (INTEGER PRIMARY KEY)
- title (TEXT)
- created_at (TEXT)

**messages**
- id (INTEGER PRIMARY KEY)
- conversation_id (INTEGER, FK)
- role (TEXT: user/assistant/system)
- content (TEXT)
- timestamp (TEXT)

### Memory Usage

- Model (Q4_K_M): ~400MB RAM
- Base Python + deps: ~200MB RAM
- **Total: ~600MB RAM** (varies by system)

## Example: Python Client

```python
import requests

# Send a message
response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={
        "message": "What is machine learning?",
        "temperature": 0.7,
        "max_tokens": 256
    }
)

data = response.json()
print(f"Conversation ID: {data['conversation_id']}")
print(f"Answer: {data['answer']}")
```

## Troubleshooting

### Issue: Model download fails

**Solution**: Manually download the GGUF file from Hugging Face and place it in the project directory, then update `engine.py` to use `model_path` instead of `from_pretrained`.

### Issue: Out of memory

**Solution**: Use a smaller quantization (e.g., `q2_k` or `q3_k_m`) or reduce `n_ctx` in `engine.py`.

### Issue: Slow responses

**Solution**: 
- Use GPU if available (llama-cpp-python supports CUDA/Metal)
- Reduce `max_tokens` parameter
- Use a smaller quantization

### Issue: Gradio port already in use

**Solution**: Change the port in `src/app.py`:
```python
demo.launch(server_name="127.0.0.1", server_port=7861)  # Use different port
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen3) for the Qwen3-0.6B model
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF format and inference
- [Gradio](https://gradio.app/) for the chat interface
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API

## Contact

For issues, questions, or contributions, please open a GitHub issue or pull request.

---

**Built with ❤️ for local AI enthusiasts**
