import httpx
import gradio as gr
import time
import os
import json
import uuid
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configuration
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:7860/chat")
SESSION_ID = str(uuid.uuid4())

print(f"[INFO] Connecting to Backend: {SERVER_URL}")
print(f"[INFO] Session ID: {SESSION_ID}")

# ---------------------------------------------------------------------------
# Client Logic
# ---------------------------------------------------------------------------

def chat_response(message, history):
    """
    Sends request to the proxy server and streams the response.
    """
    # Update Gradio history for display
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    start_time = time.perf_counter()
    full_response = ""

    # Send streaming request to our proxy server
    try:
        with httpx.stream("POST", SERVER_URL, json={"message": message, "session_id": SESSION_ID}, timeout=None) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data["type"] == "content":
                    text = data["text"]
                    full_response += text
                    history[-1]["content"] = full_response
                    yield history
                
                elif data["type"] == "metrics":
                    if "ttft" in data:
                        print(f"[METRICS] TTFT: {data['ttft']:.3f}s")
                    if "total_time" in data:
                        print(f"[METRICS] Total Time: {data['total_time']:.3f}s")
                
                elif data["type"] == "error":
                    history[-1]["content"] = f"Error: {data['message']}"
                    yield history
                    break

    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        history[-1]["content"] = f"Connection error: {e}"
        yield history

def clear_chat():
    """Reset the Gradio UI. Note: In this version, we'd need a server endpoint to clear session."""
    return None

# Build Gradio UI
with gr.Blocks(title="Exentec Survey Agent (Optimized)") as demo:
    gr.Markdown("# 🤖 Exentec Survey Agent")
    gr.Markdown("Survey powered by Gemma-2b-it (Optimized Inference Architecture)")

    chatbot = gr.Chatbot(height=500, type="messages")
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
    # Launch Gradio on a different port than the server
    demo.launch(server_name="0.0.0.0", server_port=7861)
