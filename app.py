import litert_lm
import gradio as gr
import time

# Configuration
MODEL_PATH = "/home/moriarty4k/.litert-lm/models/gemma-e2b/model.litertlm"

SYSTEM_INSTRUCTION = "You are a helpful AI assistant."

# Initialize the Engine once (text-only)
engine = litert_lm.Engine(MODEL_PATH, backend=litert_lm.Backend.CPU)

# ---------------------------------------------------------------------------
# Persistent conversation — keeps the KV cache alive across turns so the
# model only prefills the NEW tokens on each message, not the entire history.
# ---------------------------------------------------------------------------
_conversation = None

def _get_conversation():
    """Return the live conversation, creating one if needed."""
    global _conversation
    if _conversation is None:
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}],
        }
        ctx = engine.create_conversation(messages=[system_message])
        _conversation = ctx.__enter__()
        print("[INFO] New conversation started.")
    return _conversation

def _reset_conversation():
    """Close the current conversation and clear the KV cache."""
    global _conversation
    if _conversation is not None:
        try:
            _conversation.__exit__(None, None, None)
        except Exception:
            pass
        _conversation = None
    print("[INFO] Conversation reset.")


def chat_response(message, history):
    """
    Handles a text message and streams the model response.
    The conversation object is kept alive between calls so the KV cache
    is reused — only the new user tokens are prefilled each turn.
    """
    conversation = _get_conversation()
    current_user_content = [{"type": "text", "text": message}]

    # Update Gradio history for display
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    yield history, gr.update(value="")

    try:
        partial_message = ""
        start_time = time.perf_counter()
        first_token_received = False

        for chunk in conversation.send_message_async({"role": "user", "content": current_user_content}):
            if not first_token_received:
                ttft = time.perf_counter() - start_time
                print(f"\n[METRICS] Time to First Token (TTFT): {ttft:.3f}s")
                first_token_received = True

            for item in chunk.get("content", []):
                if item.get("type") == "text":
                    partial_message += item["text"]
                    print(item["text"], end="", flush=True)
                    history[-1]["content"] = partial_message
                    yield history, gr.update()

        print()  # newline after generation completes

    except Exception as e:
        history[-1]["content"] = f"Error: {str(e)}"
        yield history, gr.update()


def new_chat():
    """Reset the persistent conversation and clear the UI."""
    _reset_conversation()
    return [], gr.update(value="")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="Gemma Chat") as demo:
    gr.Markdown("# Gemma Chat")
    gr.Markdown("Chat with Gemma. The conversation context is preserved across turns for fast responses.")

    chatbot = gr.Chatbot()

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type a message...",
            show_label=False,
            scale=4,
        )
        send_btn = gr.Button("Send", scale=1, variant="primary")
        new_chat_btn = gr.Button("New Chat", scale=1, variant="secondary")

    def submit(message, history):
        if not message.strip():
            return history, ""
        yield from chat_response(message, history)

    msg.submit(submit, inputs=[msg, chatbot], outputs=[chatbot, msg])
    send_btn.click(submit, inputs=[msg, chatbot], outputs=[chatbot, msg])
    new_chat_btn.click(new_chat, inputs=[], outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch(theme=theme)
