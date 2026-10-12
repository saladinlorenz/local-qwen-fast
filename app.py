"""
Gradio chat interface.

Run with: python app.py
"""

import gradio as gr
from engine import engine
import os
import tempfile


def load_model_from_url(model_url: str):
    """Load a model from Hugging Face repo ID or local path."""
    if not model_url or not model_url.strip():
        return "⚠️ Veuillez entrer un repo ID Hugging Face ou un chemin de modèle.", ""
    try:
        engine.set_model(model_url.strip())
        os.environ["CHAT_MODEL_PATH"] = model_url.strip()
        return f"✅ Modèle chargé : {model_url.strip()}", model_url.strip()
    except Exception as e:
        return f"❌ Erreur lors du chargement : {str(e)}", ""


def load_conversation_list():
    """Load conversation list for dropdown. Returns (choices, selected_id)."""
    rows = engine.list_conversations()
    choices = [(f"{r[1]} (ID: {r[0]})", r[0]) for r in rows]
    if not choices:
        cid = engine.new_conversation()
        rows = engine.list_conversations()
        choices = [(f"{r[1]} (ID: {r[0]})", r[0]) for r in rows]
        if not choices:
            return [], None
    return choices, choices[0][1]


def dropdown_update(choices, cid):
    """Atomic choices+value update (avoids 'Value not in choices')."""
    values = [v for _, v in choices]
    if cid not in values:
        cid = values[0] if values else None
    return gr.Dropdown(choices=choices, value=cid)


def on_select_conversation(cid, history):
    """Load conversation history when selecting from dropdown."""
    if cid is None:
        return []
    try:
        cid = int(cid)
    except (TypeError, ValueError):
        return history
    try:
        messages = engine.load_conversation(cid)
    except Exception as e:
        return history + [{"role": "assistant", "content": f"⚠️ Impossible de charger : {e}"}]
    return [
        {"role": m.role, "content": m.content}
        for m in messages
        if m.role in ("user", "assistant")
    ]


def refresh_conv_list():
    """Refresh conversation dropdown. Returns (Dropdown update, history)."""
    choices, cid = load_conversation_list()
    hist = on_select_conversation(cid, [])
    return dropdown_update(choices, cid), hist


def create_new_conv():
    """Create a new conversation."""
    engine.new_conversation()
    return refresh_conv_list()


def delete_current_conv(cid):
    """Delete current conversation."""
    if cid is not None:
        try:
            engine.delete_conversation(int(cid))
        except (TypeError, ValueError):
            pass
    return refresh_conv_list()


def _ensure_cid(cid):
    if cid is None:
        return engine.new_conversation()
    try:
        return int(cid)
    except (TypeError, ValueError):
        return engine.new_conversation()


def on_submit(message, history, cid, temp, max_tok, top_p, rep):
    """Handle message submission (streaming generator)."""
    if history is None:
        history = []
    if not message or not message.strip():
        choices, current = load_conversation_list()
        if cid in [v for _, v in choices]:
            current = cid
        yield history, dropdown_update(choices, current), ""
        return
    if ".png" in message or ".jpg" in message or message.startswith("http"):
        choices, current = load_conversation_list()
        if cid in [v for _, v in choices]:
            current = cid
        yield history, dropdown_update(choices, current), ""
        return

    cid = _ensure_cid(cid)
    history = history + [{"role": "user", "content": message}]
    choices, _ = load_conversation_list()
    if cid not in [v for _, v in choices]:
        choices, cid = load_conversation_list()
        cid = _ensure_cid(cid)
    yield history, dropdown_update(choices, cid), ""

    full = ""
    try:
        for chunk in engine.stream_chat(
            conversation_id=cid,
            user_text=message,
            temperature=temp,
            max_tokens=max_tok,
            top_p=top_p,
            repeat_penalty=rep,
        ):
            full += chunk
            yield history + [{"role": "assistant", "content": full}], dropdown_update(choices, cid), ""
    except Exception as e:
        full += f"\n\n❌ Erreur : {e}"
        yield history + [{"role": "assistant", "content": full}], dropdown_update(choices, cid), ""
        return

    # Title may have changed after the first exchange: refresh labels.
    choices, _ = load_conversation_list()
    yield history + [{"role": "assistant", "content": full}], dropdown_update(choices, cid), ""


def on_regen(history, cid, temp, max_tok, top_p, rep):
    """Regenerate the last reply without retyping the prompt."""
    if history is None:
        history = []
    if cid is None:
        yield history, dropdown_update(*load_conversation_list())
        return
    try:
        cid = int(cid)
    except (TypeError, ValueError):
        yield history, dropdown_update(*load_conversation_list())
        return
    base = list(history)
    if base and base[-1].get("role") == "assistant":
        base = base[:-1]
    yield base, dropdown_update(*load_conversation_list())
    full = ""
    try:
        for chunk in engine.regenerate_last(
            conversation_id=cid,
            temperature=temp,
            max_tokens=max_tok,
            top_p=top_p,
            repeat_penalty=rep,
        ):
            full += chunk
            yield base + [{"role": "assistant", "content": full}], dropdown_update(*load_conversation_list())
    except Exception as e:
        yield base + [{"role": "assistant", "content": f"❌ Erreur régénération : {e}"}], dropdown_update(*load_conversation_list())


def on_export(cid):
    """Export the conversation as a downloadable .md file."""
    if cid is None:
        return None
    try:
        cid = int(cid)
        text = engine.export_text(cid)
    except Exception as e:
        text = f"Erreur export : {e}"
    fd, path = tempfile.mkstemp(prefix=f"conv_{cid}_", suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


CSS = """
    .gradio-container {
        max-width: 100% !important;
    }
    @media (max-width: 768px) {
        .gr .wrap > .gradio-space {
            flex-direction: column-reverse !important;
        }
    }
"""

# Pre-compute initial state BEFORE building components so the Dropdown
# never starts with choices=[] and an out-of-list value
# (that race caused "Value: 1 is not in the list of choices: []").
init_choices, init_cid = load_conversation_list()
init_hist = on_select_conversation(init_cid, []) if init_cid is not None else []

with gr.Blocks(title="Qwen3-0.6B Local Chat") as demo:
    gr.Markdown("# Qwen3-0.6B Local Chat")
    gr.Markdown("A lightweight, 100% Python chat application with persistent history and streaming responses.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Conversations")
            conv_dropdown = gr.Dropdown(
                label="Select Conversation",
                choices=init_choices,
                value=init_cid,
                interactive=True,
            )
            new_conv_btn = gr.Button("+ New Conversation", variant="primary")
            del_conv_btn = gr.Button("Delete Conversation", variant="stop")

            gr.Markdown("---")
            gr.Markdown("**Model**: Qwen3-0.6B (GGUF)")
            gr.Markdown("**Backend**: llama-cpp-python")
            gr.Markdown("**Storage**: SQLite")

            gr.Markdown("**Modèle personnalisé**")
            model_url_input = gr.Textbox(
                label="Repo ID Hugging Face ou chemin de modèle",
                placeholder="ex: utilisateur/nom-modele ou chemin/local.gguf",
                value="",
                scale=4,
            )
            load_model_btn = gr.Button("Charger le modèle", variant="secondary")
            model_status = gr.Textbox(
                label="État du modèle",
                value="⏳ Modèle chargé à la demande (lazy) — instantané au démarrage.",
                interactive=False,
                scale=2,
            )

            gr.Markdown("---")
            gr.Markdown("**Modèle par défaut** : MaziyarPanahi/Qwen3-0.6B-GGUF · "
                        "<span style='font-size: 0.7em; color: #666;'>Env: Debian proot VPS Android</span>")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=500, value=init_hist)
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=2,
                show_label=True,
            )
            with gr.Row():
                send_btn = gr.Button("Envoyer", variant="primary")
                stop_btn = gr.Button("⏹ Stop", variant="stop")
                regen_btn = gr.Button("🔄 Régénérer", variant="secondary")
                export_btn = gr.Button("⬇ Exporter", variant="secondary")
            export_file = gr.File(label="Export .md", interactive=False)

            with gr.Accordion("Advanced Settings", open=False):
                temp_slider = gr.Slider(
                    minimum=0, maximum=2, value=0.7, step=0.1,
                    label="Temperature", info="Higher = more creative, Lower = more focused"
                )
                max_tok_slider = gr.Slider(
                    minimum=64, maximum=2048, value=512, step=64,
                    label="Max Tokens", info="Maximum response length"
                )
                top_p_slider = gr.Slider(
                    minimum=0.1, maximum=1, value=0.9, step=0.05,
                    label="Top P", info="Nucleus sampling parameter"
                )
                rep_slider = gr.Slider(
                    minimum=1, maximum=2, value=1.1, step=0.05,
                    label="Repeat Penalty", info="Higher = less repetition"
                )

    load_model_btn.click(
        fn=load_model_from_url,
        inputs=model_url_input,
        outputs=[model_status, model_url_input],
    )

    demo.load(
        fn=refresh_conv_list,
        inputs=[],
        outputs=[conv_dropdown, chatbot],
    )

    conv_dropdown.change(
        fn=on_select_conversation,
        inputs=[conv_dropdown, chatbot],
        outputs=[chatbot],
    )

    new_conv_btn.click(fn=create_new_conv, inputs=[], outputs=[conv_dropdown, chatbot])
    del_conv_btn.click(fn=delete_current_conv, inputs=[conv_dropdown], outputs=[conv_dropdown, chatbot])

    submit_event = msg.submit(
        fn=on_submit,
        inputs=[msg, chatbot, conv_dropdown, temp_slider, max_tok_slider, top_p_slider, rep_slider],
        outputs=[chatbot, conv_dropdown, msg],
    )
    send_event = send_btn.click(
        fn=on_submit,
        inputs=[msg, chatbot, conv_dropdown, temp_slider, max_tok_slider, top_p_slider, rep_slider],
        outputs=[chatbot, conv_dropdown, msg],
    )
    # Stop: cancels the running stream (no function needed).
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[submit_event, send_event])

    regen_btn.click(
        fn=on_regen,
        inputs=[chatbot, conv_dropdown, temp_slider, max_tok_slider, top_p_slider, rep_slider],
        outputs=[chatbot, conv_dropdown],
    )
    export_btn.click(fn=on_export, inputs=[conv_dropdown], outputs=[export_file])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft(), css=CSS)
