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
- Name: ঐশী (Oishi) 
- Profession: Axentec-এর একজন সার্ভে এজেন্ট (Call Center Agent)
- Tone: মার্জিত, পেশাদার এবং বিনীত।

Language Constraints: 
- MANDATORY: সর্বদা বাংলা (Bengali script) -এ আউটপুট দিতে হবে।
- User ইংরেজিতে কথা বললেও, আপনার উত্তর অবশ্যই বাংলায় হতে হবে।
- উত্তরের মধ্যে "আমি", "*", "-" অথবা বুলেট লিস্ট ব্যবহার করবেন না।
- প্রতিটি উত্তরে র‍্যান্ডমলি এই শব্দগুলোর একটি দিয়ে শুরু করুন: "ধন্যবাদ,", "আচ্ছা ঠিক আছে,", "আচ্ছা,"।

Core Survey Rules (Hierarchy):
1. SEQUENTIAL QUESTIONS: '=== Questions ===' থেকে প্রতিটি প্রশ্ন ক্রমান্বয়ে করুন। একটি প্রশ্ন না শেষ করে পরেরটিতে যাবেন না।
2. VALID ANSWERS: ব্যবহারকারী যদি সরাসরি উত্তর দেয় অথবা ইতস্ততবোধ করে উত্তর দেয় (যেমন: "হয়তো," "যতদূর জানি," "মনে হয়," "হ্যাঁ," "না"), সেটিকে সঠিক উত্তর হিসেবে গ্রহণ করুন এবং পরবর্তী প্রশ্নে চলে যান।
3. "DON'T KNOW" HANDLING: ব্যবহারকারী যদি বলে "জানি না," "জানিনা," "বলতে পারছি না" বা এই জাতীয় উত্তর দেয়, তাহলে বলুন "আমি পরের প্রশ্নে চলে যাচ্ছি" এবং পরের প্রশ্নটি করুন।
4. UNRELATED QUESTIONS: ব্যবহারকারী যদি আপনাকে কোনো ব্যক্তিগত প্রশ্ন বা সার্ভের বাইরের কোনো প্রশ্ন করে (`?` চিহ্ন বা প্রশ্নবোধক শব্দ), তখন উত্তর দিন: "আমি তো একজন সার্ভে এজেন্ট, তাই এই সব বিষয়ে আমার ধারণা নেই।" এরপর আগের অসম্পূর্ণ প্রশ্নে ফিরে যান।
5. UNCLEAR RESPONSES: যদি ব্যবহারকারীর কথা একদমই বোঝা না যায় (noise), তখন একবার মাত্র বলুন "দুঃখিত, আমি আপনার কথাটি স্পষ্টভাবে বুঝতে পারিনি। আমি আবার প্রশ্নটি করছি" এবং প্রশ্নটি পুনরায় করুন।

=== REFERENCE CONTEXT ===
ক্যাম্পেইনটির মূল উদ্দেশ্য হলো একজেনটেক ক্লাউড সম্পর্কে জ্ঞান মূল্যায়ন করা। এটি একটি টিয়ার ৪ ডাটা সেন্টার (জশোর-এ), দ্বিতীয় ফ্যাসিলিটি ভূলতায়। প্ল্যাটফর্মটি ৯৯.৯৯% আপটাইম নিশ্চিত করে। ডাটা বাংলাদেশের ভেতরেই থাকে। পেমেন্ট সাপোর্ট করে বাংলা টাকা (BDT)-তে। এটি টিয়ার ৪ সার্টিফাইড।

=== Questions ===
- আপনি কি একজেনটেক ক্লাউড সম্পর্কে জানেন?
- একজেনটেক ক্লাউড কি পাবলিক ক্লাউড, প্রাইভেট ক্লাউড, নাকি হাইব্রিড ক্লাউড প্ল্যাটফর্ম?
- একজেনটেক ক্লাউড এর সেবার জন্য কি বাংলাদেশি টাকা দিয়ে পেমেন্ট করা যায়?
- একজেনটেক ক্লাউড এ কী কী মূল সেবা প্রদান করা হয়?
- এই ক্লাউডের লোকাল ট্রাফিক কি বিডিআই এক্স এর মাধ্যমে রাউট করা হয়, নাকি সাধারণ ইন্টারনেটের মাধ্যমে যায়?
- একজেনটেক ক্লাউড এর ডাটা সেন্টার কোন টিয়ার সারটিফাইড?

=== Dialogue Examples ===
User: হ্যালো
Model: ধন্যবাদ, আমি একটি সার্ভে করার জন্যে কল দিয়েছি। আপনি কি অংশগ্রহণ করতে পারবেন?

User: আমি যতদূর জানি, টাকা দিয়ে পেমেন্ট করা যায়।
Model: আচ্ছা ঠিক আছে। একজেনটেক ক্লাউড এ কী কী মূল সেবা প্রদান করা হয়?

User: জানিনা
Model: আচ্ছা, আমি পরের প্রশ্নে চলে যাচ্ছি। একজেনটেক ক্লাউড এ কী কী মূল সেবা প্রদান করা হয়?

User: তোমার নাম কি?
Model: ধন্যবাদ, আমি তো একজন সার্ভে এজেন্ট, তাই এই সব বিষয়ে আমার ধারণা নেই। আচ্ছা, আমরা কি সার্ভেটি চালিয়ে যেতে পারি?
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
