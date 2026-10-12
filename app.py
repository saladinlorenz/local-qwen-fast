"""
Gradio chat interface.

Run with: python src/app.py
"""

import gradio as gr
from engine import engine
import os


def load_model_from_url(model_url: str):
    """Load a model from Hugging Face repo ID or local path."""
    if not model_url or not model_url.strip():
        return "⚠️ Veuillez entrer un repo ID Hugging Face ou un chemin de modèle.", ""
    try:
        engine.set_model(model_url.strip())
        # Save path to env var for persistence across restarts
        os.environ["CHAT_MODEL_PATH"] = model_url.strip()
        return f"✅ Modèle chargé : {model_url.strip()}", model_url.strip()
    except Exception as e:
        return f"❌ Erreur lors du chargement : {str(e)}", ""


def get_choices():
    """Return (choices, current_id). Choices are [(label, value), ...]."""
    rows = engine.list_conversations()
    if not rows:
        cid = engine.new_conversation()
        rows = engine.list_conversations()
    choices = [(f"{r[1]} (ID: {r[0]})", r[0]) for r in rows]
    cid = choices[0][1]
    return choices, cid


def load_conversation_list():
    """Load conversation list for dropdown (legacy helper)."""
    return get_choices()


def on_select_conversation(cid, history):
    """Load conversation history when selecting from dropdown."""
    if cid is None:
        return []
    try:
        cid_int = int(cid)
    except (TypeError, ValueError):
        return []
    messages = engine.load_conversation(cid_int)
    return [
        {"role": m.role, "content": m.content}
        for m in messages
        if m.role in ("user", "assistant")
    ]


def chat_stream(message: str, history: list, cid, temp: float, max_tok: int, top_p: float, rep: float):
    """Stream chat response token by token."""
    if cid is None:
        cid = engine.new_conversation()

    full = ""
    for chunk in engine.stream_chat(
        conversation_id=cid,
        user_text=message,
        temperature=temp,
        max_tokens=max_tok,
        top_p=top_p,
        repeat_penalty=rep,
    ):
        full += chunk
        yield history + [{"role": "assistant", "content": full}], cid


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
# never starts with choices=[] and a value that is not in the list
# (that race caused "Value: 1 is not in the list of choices: []").
init_choices, init_cid = get_choices()
init_hist = on_select_conversation(init_cid, [])

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

            # Modèle Hugging Face personnalisé
            gr.Markdown("**Modèle personnalisé**")
            model_url_input = gr.Textbox(
                label="Repo ID Hugging Face ou chemin de modèle",
                placeholder="ex: utilisateur/nom-modele ou chemin/local.gguf",
                value="",
                scale=4,
            )
            load_model_btn = gr.Button("Charger le modèle", variant="secondary")
            model_status = gr.Textbox(label="État du modèle", interactive=False, scale=2)

            gr.Markdown("---")
            gr.Markdown("**Modèle par défaut** : MaziyarPanahi/Qwen3-0.6B-GGUF · "
                        "<span style='font-size: 0.7em; color: #666;'>Env: Debian proot VPS Android</span>")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                value=init_hist,
            )
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=2,
                show_label=True,
            )

            with gr.Accordion("Advanced Settings", open=False):
                temp_slider = gr.Slider(
                    minimum=0, maximum=2, value=0.7, step=0.1,
                    label="Temperature",
                    info="Higher = more creative, Lower = more focused"
                )
                max_tok_slider = gr.Slider(
                    minimum=64, maximum=2048, value=512, step=64,
                    label="Max Tokens",
                    info="Maximum response length"
                )
                top_p_slider = gr.Slider(
                    minimum=0.1, maximum=1, value=0.9, step=0.05,
                    label="Top P",
                    info="Nucleus sampling parameter"
                )
                rep_slider = gr.Slider(
                    minimum=1, maximum=2, value=1.1, step=0.05,
                    label="Repeat Penalty",
                    info="Higher = less repetition"
                )

    load_model_btn.click(
        fn=load_model_from_url,
        inputs=model_url_input,
        outputs=[model_status, model_url_input],
    )

    def refresh_conv_list():
        """Refresh conversation dropdown. Returns (Dropdown update, history)."""
        choices, cid = get_choices()
        hist = on_select_conversation(cid, [])
        return gr.Dropdown(choices=choices, value=cid), hist

    def create_new_conv():
        """Create a new conversation."""
        cid = engine.new_conversation()
        choices, _ = get_choices()
        # get_choices guarantees the new cid is present
        return gr.Dropdown(choices=choices, value=cid), []

    def delete_current_conv(cid):
        """Delete current conversation."""
        if cid is not None:
            try:
                engine.delete_conversation(int(cid))
            except (TypeError, ValueError):
                pass
        choices, new_cid = get_choices()
        hist = on_select_conversation(new_cid, [])
        return gr.Dropdown(choices=choices, value=new_cid), hist

    def on_submit(message, history, cid, temp, max_tok, top_p, rep):
        """Handle message submission (streaming)."""
        if not message or not message.strip():
            yield history or [], cid
            return
        history = (history or []) + [{"role": "user", "content": message}]
        # If no conversation selected yet, create one now so the
        # Dropdown value stays valid for the whole stream.
        if cid is None:
            new_cid = engine.new_conversation()
            choices, _ = get_choices()
            dd_static = gr.Dropdown(choices=choices, value=new_cid)
            cid_eff = new_cid
        else:
            dd_static = cid
            cid_eff = cid
        full = ""
        yielded_any = False
        for chunk in engine.stream_chat(
            conversation_id=int(cid_eff),
            user_text=message,
            temperature=temp,
            max_tokens=max_tok,
            top_p=top_p,
            repeat_penalty=rep,
        ):
            full += chunk
            yielded_any = True
            yield history + [{"role": "assistant", "content": full}], dd_static
        if not yielded_any:
            # stream_chat yielded nothing (e.g. model error message is
            # yielded as one chunk, so this is just a safety net)
            yield history, dd_static
            return
        # Title may have changed after the first exchange: refresh labels.
        try:
            rows = engine.list_conversations()
            if rows:
                choices = [(f"{r[1]} (ID: {r[0]})", r[0]) for r in rows]
                dd_final = gr.Dropdown(choices=choices, value=int(cid_eff))
            else:
                dd_final = dd_static
        except Exception:
            dd_final = dd_static
        yield history + [{"role": "assistant", "content": full}], dd_final

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

    new_conv_btn.click(
        fn=create_new_conv,
        inputs=[],
        outputs=[conv_dropdown, chatbot],
    )

    del_conv_btn.click(
        fn=delete_current_conv,
        inputs=[conv_dropdown],
        outputs=[conv_dropdown, chatbot],
    )

    msg.submit(
        fn=on_submit,
        inputs=[msg, chatbot, conv_dropdown, temp_slider, max_tok_slider, top_p_slider, rep_slider],
        outputs=[chatbot, conv_dropdown],
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8000"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(),
        css=CSS,
    )
