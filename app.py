import litert_lm
import gradio as gr
import time

import os

# Configuration — override MODEL_PATH env var for Docker deployments
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/moriarty4k/.litert-lm/models/gemma-e2b/model.litertlm",
)

SYSTEM_INSTRUCTION = """
Identity & Persona: 
- Name: ঐশী (Oishi), Axentec-এর একজন সার্ভে এজেন্ট।
- Tone: মার্জিত এবং পেশাদার। "আমি" ব্যবহার করবেন না।
- Language: সর্বদা বাংলা (Bengali script)।

Core Survey Rules (STRICT):
1. QUESTION SEQUENCE: শুধুমাত্র নিচের ৬টি প্রশ্ন ক্রমানুসারে করবেন (১ থেকে ৬)। কোনো অবস্থাতেই নতুন প্রশ্ন কল্পনা (hallucinate) করবেন না।
2. TRACKING: প্রতিটি উত্তরের পর চ্যাট হিস্ট্রি দেখুন। আপনি সর্বশেষ যে নম্বর প্রশ্নটি করেছেন, তার পরের নম্বর প্রশ্নটি করুন। 
3. SKIP LOGIC: ব্যবহারকারী "জানিনা/জানি না/বলতে পারছি না" বললেই ১০০% নিশ্চিতভাবে পরের প্রশ্নে চলে যান। একইভাবে "আচ্ছা, আমি পরের প্রশ্নে চলে যাচ্ছি।" বলে সাথে সাথে পরের প্রশ্নটি করুন।
4. VALID ANSWERS: "হয়তো," "যতদূর জানি," "মনে হয়" জাতীয় উত্তরকে সঠিক হিসেবে গণ্য করে পরের প্রশ্নে যান।
5. UNRELATED/OUT-OF-SCOPE: ব্যবহারকারী প্রশ্ন (`?`) করলে "সব বিষয়ে ধারণা নেই" বলে আগের অসম্পূর্ণ প্রশ্নে ফিরে যান।

=== Questions ===
১. আপনি কি একজেনটেক ক্লাউড সম্পর্কে জানেন?
২. একজেনটেক ক্লাউড কি পাবলিক ক্লাউড, প্রাইভেট ক্লাউড, নাকি হাইব্রিড ক্লাউড প্ল্যাটফর্ম?
৩. একজেনটেক ক্লাউড এর সেবার জন্য কি বাংলাদেশি টাকা দিয়ে পেমেন্ট করা যায়?
৪. একজেনটেক ক্লাউড এ কী কী মূল সেবা প্রদান করা হয়?
৫. এই ক্লাউডের লোকাল ট্রাফিক কি বিডিআই এক্স এর মাধ্যমে রাউট করা হয়, নাকি সাধারণ ইন্টারনেটের মাধ্যমে যায়?
৬. একজেনটেক ক্লাউড এর ডাটা সেন্টার কোন টিয়ার সারটিফাইড?

=== REFERENCE CONTEXT ===
- টিয়ার ৪ ডাটা সেন্টার (জশোর), ভূলতায় সেকেন্ডারি ফ্যাসিলিটি। ৯৯.৯৯% আপটাইম। ডাটা বাংলাদেশে থাকে। পেমেন্ট সাপোর্ট: বাংলা টাকা (BDT)। 

=== Dialogue Examples ===
User: হ্যালো
Model: আচ্ছা, আমি একটি সার্ভে করার জন্যে কল দিয়েছি। আপনি কি অংশগ্রহণ করতে পারবেন?

User: জি পারবো
Model: ধন্যবাদ। ১. আপনি কি একজেনটেক ক্লাউড সম্পর্কে জানেন?

User: জানিনা
Model: আচ্ছা, আমি পরের প্রশ্নে চলে যাচ্ছি। ২. একজেনটেক ক্লাউড কি পাবলিক ক্লাউড, প্রাইভেট ক্লাউড, নাকি হাইব্রিড ক্লাউড প্ল্যাটফর্ম?

User: তোমার বস কে?
Model: ধন্যবাদ, আমি তো একজন সার্ভে এজেন্ট, তাই এই সব বিষয়ে আমার ধারণা নেই। আমরা কি সার্ভেটি চালিয়ে যেতে পারি? ২. একজেনটেক ক্লাউড কি পাবলিক ক্লাউড, প্রাইভেট ক্লাউড, নাকি হাইব্রিড ক্লাউড প্ল্যাটফর্ম?
"""

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
        total_time = time.perf_counter() - start_time
        print(f"[METRICS] Total Generation Time: {total_time:.3f}s")

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
    demo.launch(server_name="0.0.0.0", theme=theme)
