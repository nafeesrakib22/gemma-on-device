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

# Robust Version detection
GR_VERSION = gr.__version__.split(".")
GR_MAJOR = int(GR_VERSION[0])

print(f"[INFO] Connecting to Backend: {SERVER_URL}")
print(f"[INFO] Gradio Version: {gr.__version__} (Major: {GR_MAJOR})")

# ---------------------------------------------------------------------------
# Client Logic
# ---------------------------------------------------------------------------

def chat_response(message, history):
    """
    Sends request to the proxy server and streams the response.
    """
    # Use dictionary format for Gradio 4.0 and above
    if GR_MAJOR >= 4:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
    else:
        history.append([message, ""])

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
                    if GR_MAJOR >= 4:
                        history[-1]["content"] += text
                    else:
                        history[-1][1] += text
                    yield history
                
                elif data["type"] == "metrics":
                    if "ttft" in data:
                        ttft = data['ttft']
                        print(f"[METRICS] TTFT: {ttft:.3f}s")
                        # Attach TTFT to the assistant message metadata
                        if GR_MAJOR >= 4:
                            history[-1]["ttft"] = ttft
                        else:
                            # For older Gradio versions, we might need a different approach 
                            # but let's assume we stick to the dict structure if possible or just log it
                            pass
                
                elif data["type"] == "error":
                    err_msg = f"Error: {data['message']}"
                    if GR_MAJOR >= 4:
                        history[-1]["content"] = err_msg
                    else:
                        history[-1][1] = err_msg
                    yield history
                    break

    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        err_msg = f"Connection error: {e}"
        if GR_MAJOR >= 4:
            history[-1]["content"] = err_msg
        else:
            history[-1][1] = err_msg
        yield history

def export_conversation(history):
    """
    Saves the current conversation history to a JSON file and returns the file path.
    """
    if not history:
        return None
    
    os.makedirs("conversations", exist_ok=True)
    filename = f"conversations/conversation_{int(time.time())}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    return filename

def clear_chat():
    """Reset the Gradio UI."""
    return None, None

# Build Gradio UI
with gr.Blocks(title="Exentec Survey Agent (Optimized)") as demo:
    gr.Markdown("# 🤖 Exentec Survey Agent")
    gr.Markdown("Survey powered by Gemma-2b-it (Optimized Inference Architecture)")

    # Only Gradio 4 needs type="messages"
    # Gradio 5+ has it as default and removed the argument
    # Gradio 3 does not support it
    chatbot_kwargs = {"height": 500}
    if GR_MAJOR == 4:
        chatbot_kwargs["type"] = "messages"
    
    chatbot = gr.Chatbot(**chatbot_kwargs)
    msg = gr.Textbox(placeholder="Type a message..", label="User Input")
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")
        export_btn = gr.Button("Export JSON")

    export_output = gr.File(label="Download Conversation")

    # Link events
    msg.submit(chat_response, [msg, chatbot], [chatbot])
    submit_btn.click(chat_response, [msg, chatbot], [chatbot])
    clear_btn.click(clear_chat, None, [chatbot, export_output], queue=False)
    export_btn.click(export_conversation, [chatbot], [export_output])

    # Automatically clear textbox after submission
    submit_btn.click(lambda: "", None, [msg], queue=False)
    msg.submit(lambda: "", None, [msg], queue=False)
if __name__ == "__main__":
    # Launch Gradio on a different port than the server
    demo.launch(server_name="0.0.0.0", server_port=7861)
