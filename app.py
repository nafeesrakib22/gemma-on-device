import litert_lm
import gradio as gr
import os
from tools.web_search import web_search

# Configuration
MODEL_PATH = "/home/moriarty4k/.litert-lm/models/gemma-e2b/model.litertlm"

# Initialize the Engine once with vision and audio backends
engine = litert_lm.Engine(
    MODEL_PATH,
    audio_backend=litert_lm.Backend.CPU,
    vision_backend=litert_lm.Backend.CPU,
)

import re

def get_mime_info(file_path):
    """Detects if a file is an image or audio based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']:
        return "image"
    elif ext in ['.wav', '.mp3', '.flac', '.ogg', '.webm', '.m4a', '.aac', '.mp4']:
        return "audio"
    return None

def format_content(text, files):
    """Formats text and files into LiteRT-LM's content structure."""
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for f in files:
        if isinstance(f, str):
            path = f
        elif isinstance(f, dict):
            path = f.get("path")
        else:
            continue
        mtype = get_mime_info(path)
        if mtype:
            content.append({"type": mtype, "path": path})
    return content

def chat_response(message, history, audio_recording):
    """
    Handles multi-modal messages from Gradio and returns a streaming response.
    """
    # 1. Initialize LiteRT-LM history with a strict multimodal system prompt
    system_instruction = (
        "You are a helpful multimodal AI assistant. "
        "You have native built-in capabilities to directly perceive and understand "
        "audio and images — you do NOT need any external tools for this.\n"
        "STRICT RULES:\n"
        "- If the user provides AUDIO: listen to it using your built-in audio perception "
        "and answer directly. NEVER call web_search to transcribe, process, or look up audio.\n"
        "- If the user provides an IMAGE: look at it using your built-in vision and answer "
        "directly. NEVER call web_search to describe or analyse images.\n"
        "- ONLY call web_search for requests that genuinely require current internet data: "
        "news, live scores, stock prices, recent events, etc.\n"
        "- When audio or an image is present in the conversation, process it directly first "
        "before considering any tool use."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]}
    ]
    
    # 2. Process chat history for LiteRT-LM
    # In Gradio 6 Messages type, history entries have 'role' and 'content'.
    # content can be a list of Dicts.
    for entry in history:
        role = entry.get("role")
        content = entry.get("content")
        
        # Debug: raw entry
        print(f"DEBUG: Raw History Entry: {entry}")
        
        formatted_content = []
        if isinstance(content, str):
            formatted_content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        formatted_content.append({"type": "text", "text": item["text"]})
                    elif "path" in item:
                        path = item["path"]
                        # Robust mtype detection
                        mtype = item.get("mime_type", "").split("/")[0] if item.get("mime_type") else None
                        if mtype not in ["image", "audio"]:
                            mtype = get_mime_info(path)
                        if not mtype and item.get("type") in ["image", "audio"]:
                            mtype = item["type"]
                        if mtype == "audio":
                            # LiteRT-LM counts <|audio> tokens across the ENTIRE prompt
                            # (history + current message) and expects one audio stream per
                            # token. Historical audio cannot be re-streamed, so passing the
                            # path triggers INVALID_ARGUMENT: "Provided less audio than
                            # expected." Use a text marker instead.
                            formatted_content.append({"type": "text", "text": "[audio]"})
                        elif mtype == "image":
                            formatted_content.append({"type": "image", "path": path})
                        else:
                            print(f"WARNING: Could not determine mime type for file {path}. Item: {item}")
                    elif "file" in item and isinstance(item["file"], dict):
                        # Gradio audio recorder stores history items as:
                        # {"file": {"path": "...", "mime_type": "audio/wav", ...}, "type": "file"}
                        file_data = item["file"]
                        path = file_data.get("path")
                        mime_type = file_data.get("mime_type", "")
                        mtype = mime_type.split("/")[0] if mime_type else None
                        if mtype not in ["image", "audio"]:
                            mtype = get_mime_info(path) if path else None
                        if mtype == "audio":
                            # Same reason as above — replace with text marker.
                            formatted_content.append({"type": "text", "text": "[audio]"})
                        elif mtype == "image" and path:
                            formatted_content.append({"type": "image", "path": path})
                        else:
                            print(f"WARNING: Could not determine mime type for recorder file {path}. Item: {item}")
                else:
                    formatted_content.append({"type": "text", "text": str(item)})
        else:
            formatted_content = [{"type": "text", "text": str(content)}]
            
        messages.append({"role": role, "content": formatted_content})
    
    # Debug: Print the final history payload summary
    print("\n" + "="*50)
    print("DEBUG: LiteRT-LM History Payload Summary:")
    for msg_idx, msg_payload in enumerate(messages):
        types = [c.get('type') for c in msg_payload['content']]
        print(f"  {msg_idx}. {msg_payload['role']}: {types}")
    print("="*50 + "\n")
    
    # 3. Create current user content
    raw_files = message.get("files", [])
    print(f"DEBUG: message['files'] (raw) = {raw_files}")
    all_files = list(raw_files)
    if audio_recording:
        # Pass the original recorder audio directly — LiteRT-LM resamples internally.
        all_files.append(audio_recording)

    current_user_content_list = format_content(message["text"], all_files)
    print(f"DEBUG: current_user_content_list sent to LiteRT-LM = {current_user_content_list}")
    
    # 4. Append user message to history and update UI
    history.append({"role": "user", "content": current_user_content_list})
    # Add empty assistant response to be filled
    history.append({"role": "assistant", "content": ""})
    
    # Yield history, clear audio, and clear multimodal textbox
    yield history, gr.update(value=None), gr.update(value=None)
    
    # 5. Start LiteRT-LM generation with Tool support.
    # IMPORTANT: Remove web_search when the message contains audio or images.
    # A small model like Gemma 2B will call web_search("transcribe audio") instead
    # of using its native multimodal perception, so we withhold the tool entirely
    # for multimodal inputs to prevent that misuse.
    has_multimodal = any(
        c.get("type") in ("audio", "image") for c in current_user_content_list
    )
    tools = [] if has_multimodal else [web_search]
    try:
        with engine.create_conversation(messages=messages, tools=tools) as conversation:
            partial_message = ""
            # Send the actual structured content list
            for chunk in conversation.send_message_async({"role": "user", "content": current_user_content_list}):
                for item in chunk.get("content", []):
                    if item.get("type") == "text":
                        partial_message += item["text"]
                        # Update the last assistant message in history
                        history[-1]["content"] = partial_message
                        yield history, gr.update(), gr.update()
    except Exception as e:
        history[-1]["content"] = f"Error: {str(e)}"
        yield history, gr.update(), gr.update()

# Build the UI

# Modern aesthetics: Use a Soft theme with rounded corners and a premium indigo color palette
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="Gemma 2b Voice Chat") as demo:
    gr.Markdown("# Gemma 2b Voice-Enabled Chat")
    gr.Markdown("Interact with Gemma using text, images, or audio (files and live recording).")
    
    chatbot = gr.Chatbot()
    
    with gr.Row():
        with gr.Column(scale=4):
            # MultimodalTextbox for text and file uploads
            msg = gr.MultimodalTextbox(
                placeholder="Type a message or upload files...",
                file_types=["image", "audio"],
                show_label=False,
            )
        with gr.Column(scale=1):
            # Dedicated Audio Recorder
            audio_recorder = gr.Audio(
                sources=["microphone"], 
                type="filepath", 
                label="Record Audio",
            )

    # Submission logic: pass both inputs to the function
    msg.submit(
        chat_response, 
        inputs=[msg, chatbot, audio_recorder], 
        outputs=[chatbot, audio_recorder, msg]
    )
    # Also clear the voice recorder after submission is handled in chat_response via yield None

if __name__ == "__main__":
    # In Gradio 6.0+, 'theme' is passed to launch() instead of the constructor
    demo.launch(share=True, theme=theme)
