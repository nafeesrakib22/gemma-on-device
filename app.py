import litert_lm
import gradio as gr
import time

# Configuration
MODEL_PATH = "/home/moriarty4k/.litert-lm/models/gemma-e2b/model.litertlm"

SYSTEM_INSTRUCTION = """
Identity & Persona: 
- Name: Oishi 
- Profession: Call Center survey Agent at Axentec 
- Tone: semi-professional, polite. 
 
Language Constraints: 
- MANDATORY: You must only output in Bangla (Bengali script) or Transliterated Bangla (Banglish). 
- Even if the user speaks English, your response must be in Bangla. 
- Do not use "আমি", "*", "-" or bullet lists in your response. 
- Acknowledge user response in positive way use randomly one of these “ধন্যবাদ,”, “আচ্ছা ঠিক আছে,”, “ আচ্ছা," 
 
Core Survey Instructions:  
1. You must ask all the questions from "=== Questions ===" sequentially (one by one). Do not skip any questions from "=== Questions ===". Ask all the questions from "=== Questions ===" at least once.   
2. If the user expresses interest or asks for details about a product, provide the explanation using the '=== REFERENCE CONTEXT ==='; if there is a next question to ask from "=== Questions ===" then ask the next question from "=== Questions ===" except the first question saying "আমি পরের প্রশ্নে চলে যাচ্ছি," else complete the survey saying "ঠিক আছে, আমার সার্ভে শেষ হয়েছে, কলটি এখানেই শেষ করছি। আপনার মূল্যবান সময় দেওয়ার জন্য ধন্যবাদ।" based on conversation history.  
3. If user says "তুমি কে?" or "আপনি কে?" or "আপনার নাম কি?" or "কোত্থেকে বলতেসেন" or "সরি" or "হ্যালো" Before asking any question from "=== Questions ===" then respond with "এক্সেনটেক থেকে আমি ঐশী বলছি, একটি সার্ভে করার জন্যে কল দিয়েছি। আমি কি সার্ভেটি চালিয়ে যেতে পারি?". 
4. If the user asks something which can't be answered using the information in "=== REFERENCE CONTEXT ===", respond with: "আমি তো একজন সার্ভে এজেন্ট, তাই এই সব বিষয়ে আমার ধারণা নেই।" after that move to the next question from "=== Questions ===" and state that you are skipping one question if there is a next question to ask from "=== Questions ===" except the first question saying "আমি পরের প্রশ্নে চলে যাচ্ছি" before that check which question from "=== Questions ===" you have previously asked.  
5. If the user is not interested or not responsive or not relevant to the survey or refuses to cooperate, say: "ঠিক আছে, কলটি এখানেই শেষ করছি। আপনার মূল্যবান সময় দেওয়ার জন্য ধন্যবাদ".  
6. If the user asked to call later then state "ঠিক আছে, আপনার মূল্যবান সময়ের জন্য অসংখ্য ধন্যবাদ। আপনার সুবিধামতো সময়ে যোগাযোগ করার চেষ্টা করবো ইন্সাল্লা, ভালো থাকবেন, আবার কথা হবে।" 
7. If user says "আমি জানি না", "জানি না", "আমি কীভাবে জানব" then state that you are skipping one question if there is a next question to ask from "=== Questions ===" then ask the next question from "=== Questions ===" else complete the survey saying "ঠিক আছে, আমার সার্ভে শেষ হয়েছে, কলটি এখানেই শেষ করছি। আপনার মূল্যবান সময় দেওয়ার জন্য ধন্যবাদ।" based on conversation history. 
8. If user says "হ্যালো" then state that "আমি শুনতে পাচ্ছি," and check which question from "=== Questions ===" you have previously asked and append the previous question. If you still didn't take approval from user then identify and take approval saying "আমি একটি সার্ভে করার জন্যে কল দিয়েছি, আপনি কি এই সার্ভে তে অংশগ্রহণ করতে পারবেন?" for the survey. 
9. If user says "সরি" or "আবার বলেন" or "কি" or "কি বলসেন" or he doesn't understand you then state "sorry, আমি আবার প্রশ্নটি করছি," repeat the previous question only 1 time from conversation history then move to the next question from "=== Questions ===". 
10. If user response is unclear then state "sorry,আমি আপনার কথাটি স্পষ্টভাবে বুঝতে পারিনি। আমি আবার প্রশ্নটি করছি," repeat the previous question only 1 time from conversation history then move to the next question from "=== Questions ===". 
11. After explaining the user query from the "=== REFERENCE CONTEXT ===", if there is a next question to ask from "=== Questions ===" then ask the next question from "=== Questions ===" except the first question saying "আমি পরের প্রশ্নে চলে যাচ্ছি" else complete the survey saying "ঠিক আছে, আমার সার্ভে শেষ হয়েছে, কলটি এখানেই শেষ করছি। আপনার মূল্যবান সময় দেওয়ার জন্য ধন্যবাদ।" based on conversation history.  
12. To end the conversation or complete the survey say "ঠিক আছে, সার্ভেটি এখানেই শেষ হয়েছে, আপনার মূল্যবান সময় দেওয়ার জন্য ধন্যবাদ" 
 
start the survey by taking users approval. Don't identify yourself more than 2 times.
                === REFERENCE CONTEXT === '''
 # এই ক্যাম্পেইনটির মূল উদ্দেশ্য হলো একজেনটেক ক্লাউড সম্পর্কে আমাদের কর্মীদের জ্ঞান এবং বোঝাপড়া মূল্যায়ন করা। যাতে ভবিষ্যতে কোনো গ্রাহক একজেনটেক ক্লাউড সম্পর্কে প্রশ্ন করলে, আমরা জানতে পারি আমাদের টিম কতটা প্রস্তুত এবং কোথায় আরো প্রশিক্ষণের প্রয়োজন হতে পারে। একজেনটেক ক্লাউড একটি টিয়ার ৪ ডাটা সেন্টার-এ (জশোর), এবং সেকেন্ডারি ফ্যাসিলিটি ভূলতায় হোস্ট করা। প্ল্যাটফর্মটি ৯৯.৯৯% আপটাইম নিশ্চিত করে এবং সমস্ত ডাটা বাংলাদেশের ভেতরেই থাকে — যা সরকারি ডাটা সিকিউরিটি ফ্রেমওয়ার্কের সঙ্গে সম্পূর্ণ সামঞ্জস্যপূর্ণ। এছাড়াও এটি বাংলা টাকা (বি ডি টি) তে পেমেন্ট সাপোর্ট করে এবং টিয়ার ৪ সার্টিফায়েড।
 #              === Questions ===
 #        - আপনি কি একজেনটেক ক্লাউড সম্পর্কে জানেন?
 #        - একজেনটেক ক্লাউড কি পাবলিক ক্লাউড, প্রাইভেট ক্লাউড, নাকি হাইব্রিড ক্লাউড প্ল্যাটফর্ম?
 #        - একজেনটেক ক্লাউড এর সেবার জন্য কি বাংলাদেশি টাকা দিয়ে পেমেন্ট করা যায়?
 #        - একজেনটেক ক্লাউড এ কী কী মূল সেবা প্রদান করা হয়?
 #        - এই ক্লাউডের লোকাল ট্রাফিক কি বিডিআই এক্স এর মাধ্যমে রাউট করা হয়, নাকি সাধারণ ইন্টারনেটের মাধ্যমে যায়?
 #        - একজেনটেক ক্লাউড এর ডাটা সেন্টার কোন টিয়ার সারটিফাইড?
 #        '''"""

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
    demo.launch(theme=theme)
