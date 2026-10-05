"""
Gradio chat interface.

Run with: python src/app.py
"""

import gradio as gr
from engine import engine


def load_conversation_list():
    """Load conversation list for dropdown."""
    rows = engine.list_conversations()
    choices = [(f"{r[1]} (ID: {r[0]})", r[0]) for r in rows]
    if not choices:
        cid = engine.new_conversation()
        return [(f"New Conversation (ID: {cid})", cid)], cid
    return choices, choices[0][1]


def on_select_conversation(cid: int, history):
    """Load conversation history when selecting from dropdown."""
    messages = engine.load_conversation(cid)
    return [
        {"role": m.role, "content": m.content}
        for m in messages
        if m.role in ("user", "assistant")
    ]


def chat_stream(message: str, history: list, cid: int, temp: float, max_tok: int, top_p: float, rep: float):
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


with gr.Blocks(title="Qwen3-0.6B Local Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Qwen3-0.6B Local Chat")
    gr.Markdown("A lightweight, 100% Python chat application with persistent history and streaming responses.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Conversations")
            conv_dropdown = gr.Dropdown(label="Select Conversation", choices=[], interactive=True)
            new_conv_btn = gr.Button("+ New Conversation", variant="primary")
            del_conv_btn = gr.Button("Delete Conversation", variant="stop")

            gr.Markdown("---")
            gr.Markdown("**Model**: Qwen3-0.6B (GGUF)")
            gr.Markdown("**Backend**: llama-cpp-python")
            gr.Markdown("**Storage**: SQLite")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                type="messages",
                label="Chat",
                height=500,
                show_copy_button=True,
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

    def refresh_conv_list():
        """Refresh conversation dropdown."""
        choices, cid = load_conversation_list()
        hist = on_select_conversation(cid, [])
        return choices, cid, hist

    def create_new_conv():
        """Create a new conversation."""
        cid = engine.new_conversation()
        choices, _ = load_conversation_list()
        return choices, cid, []

    def delete_current_conv(cid: int):
        """Delete current conversation."""
        if cid is not None:
            engine.delete_conversation(cid)
        return refresh_conv_list()

    def on_submit(message, history, cid, temp, max_tok, top_p, rep):
        """Handle message submission."""
        if not message.strip():
            return history, cid
        history = history + [{"role": "user", "content": message}]
        return chat_stream(message, history, cid, temp, max_tok, top_p, rep)

    init_choices, init_cid = load_conversation_list()
    init_hist = on_select_conversation(init_cid, [])

    demo.load(
        fn=lambda: (init_choices, init_cid, init_hist),
        inputs=[],
        outputs=[conv_dropdown, conv_dropdown, chatbot],
    )

    conv_dropdown.change(
        fn=on_select_conversation,
        inputs=[conv_dropdown, chatbot],
        outputs=[chatbot],
    )

    new_conv_btn.click(
        fn=create_new_conv,
        inputs=[],
        outputs=[conv_dropdown, conv_dropdown, chatbot],
    )

    del_conv_btn.click(
        fn=delete_current_conv,
        inputs=[conv_dropdown],
        outputs=[conv_dropdown, conv_dropdown, chatbot],
    )

    msg.submit(
        fn=on_submit,
        inputs=[msg, chatbot, conv_dropdown, temp_slider, max_tok_slider, top_p_slider, rep_slider],
        outputs=[chatbot, conv_dropdown],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
