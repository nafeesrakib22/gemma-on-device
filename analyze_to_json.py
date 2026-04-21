import litert_lm
import json
import os
import time
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/home/moriarty4k/.litert-lm/models/gemma-e4b/model.litertlm")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INPUT_FOLDER = "/home/moriarty4k/Documents/Gemma/transcriptions"
INPUT_FILE = os.path.join(INPUT_FOLDER, "Results.json")
OUTPUT_FILE = os.path.join(INPUT_FOLDER, "AnalysisResults.json")

# Engine selection: "gemma" or "gemini"
ENGINE = os.getenv("ENGINE", "gemma") 

# Refined System Prompt
SYSTEM_PROMPT = """You are an expert call center analyst. Your task is to analyze the following conversation transcript and produce a structured output in JSON format.

**Instructions:**
1. **Diarization:** The text may or may not be diarized. If it is not diarized, you must diarize it with proper speaker identification (customer/agent) by fully understanding the context.
2. **Rating:** Rate each dialogue turn from 0 (very negative) to 9 (very positive).
3. **Title:** Provide a suitable title focusing on the main topic of the conversation.
4. **Summary:** Provide a concise summary of the conversation.
5. **Tags:** 
    - Identify relevant tags for the conversation.
    - Prefer using these tags: GPS, Payment, Scheduling, Network, Support.
    - Introduce new tags only if absolutely necessary.

**Output Format:**
You MUST return ONLY a valid JSON object with the following structure:
{{
  "title": "Conversation Title",
  "summary": "Conversation Summary",
  "tags": ["tag1", "tag2"],
  "dialogues": [
    {{
      "text": "dialogue text",
      "speaker": "customer or agent",
      "rating": 0-9
    }},
    ...
  ]
}}
"""

def analyze_with_gemma(engine, conversation_data):
    dialogues = get_dialogues(conversation_data)
    prompt_input = format_input_data(conversation_data, dialogues)
    user_message = f"**Input Data:**\n- **Transcription:** {prompt_input['transcription']}\n- **Agent Reference:** {prompt_input['agent_reference']}\n- **Call Meta ID:** {prompt_input['call_meta_id']}"

    ctx = engine.create_conversation(messages=[
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ])
    
    with ctx as convo:
        response_text = ""
        for chunk in convo.send_message_async({"role": "user", "content": [{"type": "text", "text": user_message}]}):
            for item in chunk.get("content", []):
                if item.get("type") == "text":
                    response_text += item["text"]
        return parse_json_response(response_text, dialogues)

def analyze_with_gemini(client, conversation_data):
    dialogues = get_dialogues(conversation_data)
    prompt_input = format_input_data(conversation_data, dialogues)
    user_message = f"**Input Data:**\n- **Transcription:** {prompt_input['transcription']}\n- **Agent Reference:** {prompt_input['agent_reference']}\n- **Call Meta ID:** {prompt_input['call_meta_id']}"

    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite-preview',
            config={
                'system_instruction': SYSTEM_PROMPT,
                'response_mime_type': 'application/json'
            },
            contents=user_message
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"[ERROR] Gemini API failed: {e}")
        return {
            "title": "Error",
            "summary": "Gemini API call failed",
            "tags": ["Error"],
            "dialogues": dialogues
        }

def get_dialogues(conversation_data):
    try:
        dialogues_raw = conversation_data.get("dialogues", "[]")
        if isinstance(dialogues_raw, str):
            return json.loads(dialogues_raw)
        return dialogues_raw
    except:
        return []

def format_input_data(conversation_data, dialogues):
    return {
        "transcription": json.dumps(dialogues, ensure_ascii=False, indent=2),
        "agent_reference": conversation_data.get("agent_reference", "N/A"),
        "call_meta_id": conversation_data.get("call_meta_id", "N/A")
    }

def parse_json_response(response_text, original_dialogues):
    try:
        json_str = response_text.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:-3].strip()
        elif json_str.startswith("```"):
            json_str = json_str[3:-3].strip()
        return json.loads(json_str)
    except Exception as e:
        print(f"[ERROR] Failed to parse model response: {e}")
        return {
            "title": "Error",
            "summary": "Model output parsing failed",
            "tags": ["Error"],
            "dialogues": original_dialogues
        }

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return

    print(f"[INFO] Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    # Initialize Engine/Client
    engine = None
    client = None
    if ENGINE == "gemma":
        print(f"[INFO] Initializing Gemma Engine ({MODEL_PATH})...")
        engine = litert_lm.Engine(MODEL_PATH, backend=litert_lm.Backend.CPU)
    elif ENGINE == "gemini":
        print(f"[INFO] Initializing Gemini Client...")
        if not GEMINI_API_KEY:
            print("[ERROR] GEMINI_API_KEY not found in .env")
            return
        client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        print(f"[ERROR] Invalid ENGINE selection: {ENGINE}")
        return

    all_results = []
    
    # Debug limit for testing
    DEBUG_LIMIT = 2
    data_to_process = data[:DEBUG_LIMIT] if DEBUG_LIMIT else data

    for i, entry in enumerate(data_to_process):
        print(f"[INFO] Analyzing conversation {i+1} of {len(data_to_process)} using {ENGINE}...")
        
        start_time = time.time()
        if ENGINE == "gemma":
            analysis = analyze_with_gemma(engine, entry)
        else:
            analysis = analyze_with_gemini(client, entry)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"[INFO] Analysis took {duration:.2f} seconds")
        
        # Add metadata to the result
        analysis["metadata"] = {
            "engine": ENGINE,
            "generation_time_seconds": round(duration, 2)
        }
        
        all_results.append(analysis)
        
        if (i + 1) % 5 == 0:
            save_results(all_results)

    save_results(all_results)
    print(f"[INFO] Analysis complete. Output saved to {OUTPUT_FILE}")

def save_results(results):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
