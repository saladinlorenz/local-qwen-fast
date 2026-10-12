# Documentation — Local Qwen Fast (Qwen3-0.6B Local Chat)

> Chat 100 % local, privé, léger : modèle Qwen3-0.6B quantifié GGUF (~400 Mo RAM),
> interface Gradio avec streaming + API REST FastAPI utilisable **sans UI**.
> Fonctionne sur PC comme sur VPS Debian / proot Android.

---

## 1. C'est quoi ?

- **Inférence locale** via `llama-cpp-python` (format GGUF, pas d'Ollama, pas de cloud).
- **2 façons d'utiliser le même moteur** (`engine.py`) :
  1. **UI Gradio** (`app.py`, port 7860) : chat multi-conversations, streaming effet machine à écrire.
  2. **API REST** (`api.py`, port 8000) : fonctionne **sans lancer l'UI**, pour scripts, apps, curl.
- **Mémoire persistante** SQLite (`chat_history.db`) : conversations + messages conservés après redémarrage.
- **Modèle interchangeable à chaud** : repo Hugging Face ou fichier `.gguf` local, depuis l'UI ou `CHAT_MODEL_PATH`.

## 2. Architecture

```
┌────────────┐      ┌────────────┐
│  app.py    │      │  api.py    │   ← 2 frontends indépendants
│  Gradio UI │      │  FastAPI   │
└─────┬──────┘      └─────┬──────┘
      │                   │
      └───────┬───────────┘
              ▼
        ┌────────────┐     ┌──────────────┐
        │ engine.py  │────▶│ Qwen3-0.6B   │
        │ ChatEngine │     │ GGUF (llama) │
        └─────┬──────┘     └──────────────┘
              ▼
        ┌────────────┐
        │ SQLite WAL │
        │ chat_*.db  │
        └────────────┘
```

| Fichier | Rôle |
|---|---|
| `engine.py` | Chargement modèle, `stream_chat()`, `regenerate_last()`, CRUD SQLite, system prompt FR, fenêtre glissante, `engine` global partagé |
| `app.py` | UI Gradio : dropdown conversations, chatbot, sliders, boutons Envoyer / Stop / Régénérer / Exporter, chargement modèle |
| `api.py` | REST : `POST /chat`, `POST /chat/stream` (SSE), `POST /chat/regenerate`, `GET /conversations`, `GET /conversation/{id}/export`, `DELETE /conversation/{id}`, `GET /health`, `GET /docs` |
| `example_client.py` | Client Python d'exemple (`chat()` + `chat_stream()` SSE) |
| `models/Qwen3-0.6B-Q4_K_M.gguf` | Modèle local auto-détecté au démarrage |
| `requirements.txt` | `llama-cpp-python`, `gradio`, `fastapi`, `uvicorn`, `pydantic`, `huggingface_hub` |
| `setup.bat` / `setup.sh` | Création venv + `pip install -r requirements.txt` |
| `chat_history.db` | Base SQLite (créée auto, ignorée par git) |

## 3. Installation

```bash
git clone https://github.com/saladinlorenz/local-qwen-fast.git
cd local-qwen-fast

# Windows :
setup.bat
# Linux / Mac / Debian proot :
bash setup.sh

# Ou manuel :
python -m venv venv
# Windows : venv\Scripts\activate | Linux : source venv/bin/activate
pip install -r requirements.txt
```

Prérequis : Python 3.10+, ~600 Mo RAM (modèle Q4_K_M ~400 Mo + ~200 Mo Python).

## 4. Utilisation UI (Gradio)

```bash
python app.py
# → http://127.0.0.1:7860 (ou http://0.0.0.0:7860 sur VPS, port via PORT=7861 python app.py)
```

- Écrire → `Entrée` ou **Envoyer**. Réponse en streaming.
- Sidebar : **Select Conversation** (switch), **+ New Conversation**, **Delete Conversation**.
- Boutons chat : **⏹ Stop** (annule le streaming), **🔄 Régénérer** (rejoue la dernière réponse sans retaper), **⬇ Exporter** (télécharge `.md`).
- **Advanced Settings** : Temperature (0–2, défaut 0.7), Max Tokens (64–2048, défaut 512), Top P (défaut 0.9), Repeat Penalty (défaut 1.1).
- **Modèle personnalisé** : saisir repo HF (`user/modele`) ou chemin `.gguf` → **Charger le modèle**. Sauvegardé dans `CHAT_MODEL_PATH`.
- Les messages contenant `.png` / `.jpg` / `http...` sont ignorés (modèle texte uniquement).

## 5. Utilisation API sans UI (headless)

L'API **ne nécessite pas** `app.py`. Un seul processus suffit :

```bash
python api.py
# → http://127.0.0.1:8000/docs (Swagger), http://127.0.0.1:8000/health
```

### 5.1 POST /chat (simple)

```bash
curl http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\":\"Bonjour, qui es-tu ?\"}"
# → {"conversation_id":1,"answer":"..."}
# Suite de la discussion :
curl http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"conversation_id\":1,\"message\":\"suite\",\"temperature\":0.7,\"max_tokens\":512}"
```

### 5.2 POST /chat/stream (SSE, temps réel)

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d "{\"message\":\"Raconte une histoire courte\"}"
# data: {"conversation_id": 1}
# data: {"chunk": "Il "}
# ...
# data: [DONE]
```

```python
import requests, json
with requests.post("http://127.0.0.1:8000/chat/stream", json={"message": "Bonjour"}, stream=True) as r:
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("data:"):
            print(line)
```

Voir aussi `example_client.py` : `chat()` et `chat_stream()`.

### 5.3 Autres endpoints

| Méthode | Route | Effet |
|---|---|---|
| `POST` | `/chat/regenerate` | `{"conversation_id": N, ...}` → régénère la dernière réponse |
| `GET` | `/conversations` | Liste `[{id, title, created_at}]` |
| `GET` | `/conversation/{id}/export` | Export Markdown |
| `DELETE` | `/conversation/{id}` | Supprime conversation + messages |
| `GET` | `/health` | `{"status":"healthy","model":"..."}` |

## 6. Configuration (variables d'environnement)

| Variable | Défaut | Effet |
|---|---|---|
| `CHAT_MODEL_PATH` | `Qwen3-0.6B-Q4_K_M.gguf` ou `models/...` si présent | Chemin `.gguf` local ou repo HF (`user/repo`). Cherché aussi via `CHAT_MODEL_FILE` pour le filename HF |
| `CHAT_DB_PATH` | `chat_history.db` | Chemin base SQLite (ex. `/data/chat_history.db` sur VPS) |
| `CHAT_N_CTX` | `2048` | Fenêtre contexte (2048 ≈ 2x plus rapide que 4096, suffisant pour 0.6B) |
| `CHAT_THREADS` | nb CPU | Threads d'inférence |
| `CHAT_BATCH` | `256` | Taille batch llama.cpp |
| `CHAT_HISTORY` | `20` | Nb max de messages envoyés au modèle (fenêtre glissante anti-dépassement) |
| `CHAT_SYSTEM` | prompt FR concis/amical | System prompt (ex. `"Tu es un expert Python..."`) |
| `PORT` | `7860` | Port Gradio (`app.py` uniquement) |

Exemples :

```bash
# Modèle local précis + DB persistante :
export CHAT_MODEL_PATH="/data/models/mistral-7b-q4.gguf"
export CHAT_DB_PATH="/data/chat_history.db"
python api.py   # ou python app.py

# VPS peu puissant :
CHAT_N_CTX=1024 CHAT_THREADS=2 CHAT_HISTORY=12 python app.py

# Persona :
CHAT_SYSTEM="Tu es un tuteur Python qui répond par des exemples courts." python app.py
```

## 7. Base de données

Créée automatiquement (`engine.py:_init_db`, mode WAL + index `idx_messages_conv` pour UI+API en parallèle) :

- `conversations(id INTEGER PK, title TEXT, created_at TEXT)`
- `messages(id INTEGER PK, conversation_id INTEGER FK, role TEXT, content TEXT, timestamp TEXT)`

Le titre = 50 premiers caractères du 1er message. Export via UI (**Exporter**) ou `GET /conversation/{id}/export`.

## 8. Déploiement VPS / proot Android

```bash
export CHAT_MODEL_PATH="MaziyarPanahi/Qwen3-0.6B-GGUF"
export CHAT_DB_PATH="/data/chat_history.db"
PORT=7860 nohup python app.py > app.log 2>&1 &
nohup python api.py > api.log 2>&1 &
```

- `app.py` écoute `0.0.0.0:$PORT` (adapté VPS), `api.py` `127.0.0.1:8000` (mettre `0.0.0.0` + reverse-proxy si exposition publique).
- Si OOM : quantification plus petite (`q2_k`) ou `CHAT_N_CTX=1024`. Si lent : baisser `max_tokens`, augmenter `CHAT_THREADS`.

## 9. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `Value: 1 is not in the list of choices: []` | Dropdown Gradio recevait `value` sans `choices` (ancien bug, corrigé : update atomique `gr.Dropdown(choices, value)`) | Mettre à jour `app.py` |
| Téléchargement modèle échoue | Réseau / espace disque | Télécharger le `.gguf` à la main → `CHAT_MODEL_PATH=/chemin/modele.gguf` |
| OOM / crash | `n_ctx` trop grand, quantification lourde | `CHAT_N_CTX=1024`, `q2_k`, fermer autres process |
| Port occupé | 7860/8000 déjà pris | `PORT=7861 python app.py` / éditer `uvicorn.run(..., port=8001)` dans `api.py` |
| Réponses vides/coupées | Historique > contexte | Baisser `CHAT_HISTORY`, augmenter `max_tokens` |
| `llama-cpp-python` ne compile pas (proot) | Dépendances C manquantes | `apt update && apt install -y build-essential cmake`, puis `pip install llama-cpp-python` |

## 10. Structure du dépôt

```
local-qwen-fast/
├── app.py               # UI Gradio
├── api.py               # API FastAPI (headless OK)
├── engine.py            # Moteur + SQLite + llama.cpp
├── example_client.py    # Client REST + SSE d'exemple
├── models/              # GGUF local (Qwen3-0.6B-Q4_K_M.gguf)
├── requirements.txt
├── setup.sh / setup.bat
├── DOCUMENTATION.md     # ce fichier
├── README.md
├── LICENSE
└── .gitignore           # *.db, *.gguf, venv/ ignorés
```
