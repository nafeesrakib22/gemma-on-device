import litert_lm
import gradio as gr
import time
import os
from dotenv import load_dotenv
from prompts import PROMPTS

# Load .env file
load_dotenv()

# Configuration — override MODEL_PATH env var for Docker deployments
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/moriarty4k/.litert-lm/models/gemma-e2b/model.litertlm",
)

# Select System Instruction based on PROMPT_TYPE
PROMPT_TYPE = os.environ.get("PROMPT_TYPE", "survey")
SYSTEM_INSTRUCTION = PROMPTS.get(PROMPT_TYPE, PROMPTS["survey"])

print(f"[INFO] Using Prompt Type: {PROMPT_TYPE}")


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
    current_user_content = message

    # Update Gradio history for display (messages format)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    # Timing / Metrics
    start_time = time.perf_counter()
    first_token_time = None
    full_response = ""

    for chunk in conversation.send_message_async(current_user_content):
        if first_token_time is None:
            first_token_time = time.perf_counter()
            ttft = first_token_time - start_time
            print(f"\\n[METRICS] Time to First Token (TTFT): {ttft:.3f}s")

        # Extract text from the dictionary chunk
        if isinstance(chunk, dict) and "content" in chunk:
            text = chunk["content"][0].get("text", "")
            full_response += text
        else:
            full_response += chunk

        history[-1]["content"] = full_response
        yield history

    total_time = time.perf_counter() - start_time
    print(f"[METRICS] Total Generation Time: {total_time:.3f}s")


def clear_chat():
    """Reset the Gradio UI and the internal model conversation."""
    _reset_conversation()
    return None


# Build Gradio UI
with gr.Blocks(title="Exentec Survey Agent (Gemma-2b)") as demo:
    gr.Markdown("# 🤖 Exentec Survey Agent")
    gr.Markdown("Survey powered by Gemma-2b-it with persistent KV cache.")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Type a message..", label="User Input")
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")

    # Link events
    msg.submit(chat_response, [msg, chatbot], [chatbot])
    submit_btn.click(chat_response, [msg, chatbot], [chatbot])
    clear_btn.click(clear_chat, None, [chatbot], queue=False)

    # Automatically clear textbox after submission
    submit_btn.click(lambda: "", None, [msg], queue=False)
    msg.submit(lambda: "", None, [msg], queue=False)

if __name__ == "__main__":
    # Launch Gradio (standard 7860 port for Docker)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
