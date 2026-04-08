import litert_lm
import gradio as gr
import time
from tools.web_search import web_search

# Configuration
MODEL_PATH = "/home/moriarty4k/.litert-lm/models/gemma-e2b/model.litertlm"

# Initialize the Engine (text-only — no vision/audio backends needed)
engine = litert_lm.Engine(MODEL_PATH, backend=litert_lm.Backend.CPU)


def chat_response(message, history):
    """
    Handles text messages from Gradio and returns a streaming response.
    """
    # 1. Build system prompt
    system_instruction = (
        "You are a helpful AI assistant. "
        "You can call web_search for requests that require current internet data: "
        "news, live scores, stock prices, recent events, etc."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]}
    ]

    # 2. Process chat history
    for entry in history:
        role = entry.get("role")
        content = entry.get("content")
        if isinstance(content, str):
            formatted_content = [{"type": "text", "text": content}]
        else:
            formatted_content = [{"type": "text", "text": str(content)}]
        messages.append({"role": role, "content": formatted_content})

    # 3. Build current user content
    current_user_content = [{"type": "text", "text": message}]

    # 4. Append user message and placeholder assistant turn to history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    # Yield immediately so the UI shows the user message
    yield history, gr.update(value="")

    # 5. Run generation with tool support
    try:
        with engine.create_conversation(messages=messages, tools=[web_search]) as conversation:
            partial_message = ""
            # Start timing for TTFT — after context setup, right before generation
            start_time = time.perf_counter()
            first_token_received = False

            for chunk in conversation.send_message_async({"role": "user", "content": current_user_content}):
                # Record TTFT on first chunk
                if not first_token_received:
                    ttft = time.perf_counter() - start_time
                    print(f"\n[METRICS] Time to First Token (TTFT): {ttft:.3f}s")
                    first_token_received = True

                for item in chunk.get("content", []):
                    if item.get("type") == "text":
                        partial_message += item["text"]
                        history[-1]["content"] = partial_message
                        yield history, gr.update()
    except Exception as e:
        history[-1]["content"] = f"Error: {str(e)}"
        yield history, gr.update()


# Build the UI
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="Gemma Chat") as demo:
    gr.Markdown("# Gemma Chat")
    gr.Markdown("Chat with Gemma using text. The model can search the web for current information.")

    chatbot = gr.Chatbot(type="messages")

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type a message...",
            show_label=False,
            scale=4,
        )
        send_btn = gr.Button("Send", scale=1, variant="primary")

    def submit(message, history):
        if not message.strip():
            return history, ""
        yield from chat_response(message, history)

    msg.submit(submit, inputs=[msg, chatbot], outputs=[chatbot, msg])
    send_btn.click(submit, inputs=[msg, chatbot], outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch(theme=theme)
